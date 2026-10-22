# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Settle the edits made outside a run between its executions, at `agent6 resume`.

Each end records the tree the run left (`session.end.tree`). At the next start, the paths
whose content differs from that tree changed outside the run: they are the operator's and
follow `[git].dirty_tree`. "Differs from the chain tip" is not the test, because a run's own
work can be uncommitted after a clean end: an operator stop skips the final checkpoint, and
`commit_per_turn = false`, `git.control = "model"` and a failed commit all leave it in the
tree. A run killed before it could record a tree (SIGKILL, a crash) leaves only the chain tip
to go by: what differs from it and no tool of the run wrote follows the policy too, since
nothing says whose it is.
"""

from __future__ import annotations

import dataclasses
import pathlib
import shlex
from collections.abc import Callable, Collection, Sequence

from agent6 import errors, git_ops, terminal
from agent6.app import preflight
from agent6.app import reporter as app_reporter
from agent6.tools import schema

_LISTED = 10


def outside_edits(
    cwd: pathlib.Path,
    end_tree: str,
    seed: str | None,
    exclude: Collection[str],
    *,
    written: Collection[str] = (),
) -> list[str]:
    """Return the paths whose content changed since the run's last execution ended.

    Args:
        cwd: The checkout.
        end_tree: The tree the run left: the one its last end recorded, or HEAD's once its
            work landed there; "" when none was recorded.
        seed: The commit that tree was staged over: the chain tip, else the run's base.
        exclude: The operator's untracked files, left out of both trees.
        written: The paths the run's own tools wrote, which stay its own when no tree was
            recorded.

    Returns:
        The differing paths; with no recorded tree, what differs from the seed that no tool
        of the run wrote; empty when nothing changed.
    """
    tree = git_ops.worktree_tree(cwd, seed, exclude)
    if end_tree:
        return git_ops.tree_diff_paths(cwd, end_tree, tree)
    if seed is None:
        return []
    return [p for p in git_ops.tree_diff_paths(cwd, seed, tree) if p not in written]


def _description(end_tree: str) -> str:
    """Return what the listed paths are, by whether the run's last end recorded a tree."""
    if end_tree:
        return "changed outside this run since it last ended"
    return (
        "changed since this run's last commit, which no tool of the run wrote; its last end"
        " recorded no tree, so they may be yours or its own unfinished edits"
    )


def _also_the_runs(cwd: pathlib.Path, left: str, outside: Sequence[str]) -> list[str]:
    """Return the outside paths the run left changed against HEAD.

    A stash returns a path to HEAD's content, so it would take the run's work with the
    operator's edit.

    Args:
        cwd: The checkout.
        left: The tree the run left: its recorded end tree, else its chain tip; "" for neither.
        outside: The paths that changed outside the run.

    Returns:
        The paths the run also changed; empty when `left` is "".
    """
    if not left:
        return []
    changed = set(git_ops.tree_diff_paths(cwd, "HEAD", left))
    return [p for p in outside if p in changed]


def _listing(paths: Sequence[str], *, description: str) -> str:
    n = len(paths)
    head = f"{n} {'file' if n == 1 else 'files'} {description}:"
    lines = [f"- {terminal.printable_name(p)}" for p in paths[:_LISTED]]
    if n > _LISTED:
        lines.append(f"- ... {n - _LISTED} more")
    return "\n".join([head, *lines])


def _shared_refusal(
    cwd: pathlib.Path, shared: Sequence[str], *, end_tree: str, run_branch: str | None
) -> str:
    source = end_tree
    if (
        run_branch
        and git_ops.branch_exists(cwd, run_branch)
        and not git_ops.tree_diff_paths(cwd, run_branch, end_tree)
    ):
        source = run_branch
    # Literal pathspecs: a name holding `*`, `?` or `[` would restore other files too. A name
    # that does not print as it is (a newline in it would forge a line) is left to the operator.
    if all(terminal.printable_name(p) == p for p in shared):
        names = (f":(top,literal){p}" for p in shared)
        restore = shlex.join(["git", "restore", "--source", source, "--", *names])
    else:
        restore = (
            f"git restore --source {source} -- <path>, the path typed as the listing spells it"
        )
    description = (
        "blocks stashing (the run also changed it)"
        if len(shared) == 1
        else "block stashing (the run also changed them)"
    )
    return (
        f"{_listing(shared, description=description)}\n"
        "Stashing would also set aside the run's work.\n"
        "To discard only your edits and keep the run's last tree:\n"
        f"    {restore}\n"
        'Or set [git].dirty_tree = "include" so the run\'s next commit records your edits,'
        " then resume."
    )


def refusal(
    cwd: pathlib.Path,
    *,
    policy: str,
    outside: Sequence[str],
    end_tree: str,
    seed: str | None,
    run_branch: str | None,
    answerable: bool,
) -> str | None:
    """Return why the resume cannot settle the outside edits, before it has a live worker.

    Args:
        cwd: The checkout.
        policy: `[git].dirty_tree`.
        outside: The paths that changed outside the run.
        end_tree: The tree id the run's last end recorded; "" when none was.
        seed: The chain tip, else the run's base: what the run left when no tree was recorded.
        run_branch: The run's branch, or None.
        answerable: Whether a question would reach anyone.

    Returns:
        The refusal text, or None when the resume goes on.
    """
    if not outside or policy == "include":
        return None
    if policy == "ask":
        if answerable:
            return None
        return (
            f"{_listing(outside, description=_description(end_tree))}\n"
            "This resume has no terminal and no front-end to ask how to treat them. Stash"
            ' them first, or decide in config: [git].dirty_tree = "stash" stashes them for the'
            ' run; "include" lets the run\'s next commit record them.'
        )
    left = end_tree or seed or ""
    shared = _also_the_runs(cwd, left, outside)
    return _shared_refusal(cwd, shared, end_tree=left, run_branch=run_branch) if shared else None


@dataclasses.dataclass(frozen=True, slots=True)
class Settled:
    """What a resume did with the outside edits.

    Attributes:
        stashed: Whether it stashed them.
        pop: Whether the operator chose the stash at the question, so the end applies it back.
    """

    stashed: bool = False
    pop: bool = False


def settle(
    cwd: pathlib.Path,
    *,
    policy: str,
    outside: Sequence[str],
    end_tree: str,
    seed: str | None,
    run_branch: str | None,
    session_id: str,
    ask: Callable[[schema.UserQuestion], str],
    reporter: app_reporter.Reporter,
) -> Settled | None:
    """Apply `[git].dirty_tree` to the outside edits, asking when the policy says so.

    Args:
        cwd: The checkout.
        policy: `[git].dirty_tree`.
        outside: The paths that changed outside the run.
        end_tree: The tree id the run's last end recorded; "" when none was.
        seed: The chain tip, else the run's base: what the run left when no tree was recorded.
        run_branch: The run's branch, or None.
        session_id: The run, which names the stash.
        ask: Asks the operator one question and returns the answer.
        reporter: Receives a refusal.

    Returns:
        What was done, or None when the resume is refused or cancelled (already reported).
    """
    if not outside or policy == "include":
        return Settled()
    asked = policy == "ask"
    if asked:
        choice = preflight.dirty_tree_choice(
            ask(
                schema.UserQuestion(
                    question=(
                        f"{_listing(outside, description=_description(end_tree))}\n"
                        "How should this run treat them?\n"
                        "stash: set them aside for the run (applied back over the run's work"
                        " at the end; a failed apply keeps the stash and prints the restore"
                        " command)\n"
                        "include: the run's next commit records them with its own work\n"
                        "cancel: do not resume; the tree stays as it is"
                    ),
                    options=preflight.DIRTY_TREE_OPTIONS,
                )
            )
        )
        if choice == "cancel":
            reporter.refuse("not resuming: the working tree is left as it is")
            return None
        if choice == "include":
            return Settled()
        left = end_tree or seed or ""
        if shared := _also_the_runs(cwd, left, outside):
            reporter.refuse(_shared_refusal(cwd, shared, end_tree=left, run_branch=run_branch))
            return None
    try:
        git_ops.stash_tracked_changes(cwd, git_ops.auto_stash_message(session_id), outside)
    except errors.GitError as exc:
        reporter.refuse(f"stashing the edits made outside the run failed: {exc}")
        return None
    names = ", ".join(terminal.printable_name(p) for p in outside[:4])
    shown = names + (f", +{len(outside) - 4} more" if len(outside) > 4 else "")
    since = "the run last ended" if end_tree else "the run's last commit"
    reporter.note(f"stashed for this execution as yours: {shown} (changed since {since})")
    return Settled(stashed=True, pop=asked)
