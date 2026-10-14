# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Settle the edits made outside a run between its executions, at `agent6 resume`.

Each end records the tree the run left (`session.end.tree`). At the next start, the paths
whose content differs from that tree changed outside the run: they are the operator's and
follow `[git].dirty_tree`. "Differs from the chain tip" is not the test, because a run's own
work can be uncommitted after a clean end: an operator stop skips the final checkpoint, and
`commit_per_step = false`, `git.control = "model"` and a failed commit all leave it in the
tree.
"""

from __future__ import annotations

import dataclasses
import pathlib
from collections.abc import Callable, Collection, Sequence

from agent6 import git_ops
from agent6.app import preflight
from agent6.app import reporter as app_reporter
from agent6.tools import schema

_LISTED = 10


def outside_edits(
    cwd: pathlib.Path, end_tree: str, seed: str | None, exclude: Collection[str]
) -> list[str]:
    """Return the paths whose content changed since the run's last execution ended.

    Args:
        cwd: The checkout.
        end_tree: The tree id the last end recorded; "" when it recorded none.
        seed: The commit the end's tree was staged over: the chain tip, else the run's base.
        exclude: The operator's untracked files, left out of both trees.

    Returns:
        The differing paths; empty when no tree was recorded or nothing changed.
    """
    if not end_tree:
        return []
    now = git_ops.worktree_tree(cwd, seed, exclude)
    return [] if now == end_tree else git_ops.tree_diff_paths(cwd, end_tree, now)


def _also_the_runs(cwd: pathlib.Path, end_tree: str, outside: Sequence[str]) -> list[str]:
    """Return the outside paths the run left changed against HEAD.

    A stash returns a path to HEAD's content, so it would take the run's work with the
    operator's edit.
    """
    changed = set(git_ops.tree_diff_paths(cwd, "HEAD", end_tree))
    return [p for p in outside if p in changed]


def _listing(paths: Sequence[str]) -> str:
    n = len(paths)
    head = f"{n} {'file' if n == 1 else 'files'} changed outside this run since it last ended:"
    lines = [f"- {p}" for p in paths[:_LISTED]]
    if n > _LISTED:
        lines.append(f"- ... {n - _LISTED} more")
    return "\n".join([head, *lines])


def _shared_refusal(shared: Sequence[str]) -> str:
    return (
        f"{_listing(shared)}\n"
        "Stashing them would take this run's own work with yours: the run also changed"
        f" {'it' if len(shared) == 1 else 'them'}. Undo your edit, or set"
        ' [git].dirty_tree = "include" so the run\'s next commit records it, then resume.'
    )


def refusal(
    cwd: pathlib.Path, *, policy: str, outside: Sequence[str], end_tree: str, answerable: bool
) -> str | None:
    """Return why the resume cannot settle the outside edits, before it has a live worker.

    Args:
        cwd: The checkout.
        policy: `[git].dirty_tree`.
        outside: The paths that changed outside the run.
        end_tree: The tree id the run's last end recorded.
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
            f"{_listing(outside)}\n"
            "This resume has no terminal and no front-end to ask how to treat them. Stash"
            ' them first, or decide in config: [git].dirty_tree = "stash" stashes them for the'
            ' run; "include" lets the run\'s next commit record them.'
        )
    shared = _also_the_runs(cwd, end_tree, outside)
    return _shared_refusal(shared) if shared else None


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
    session_id: str,
    ask: Callable[[schema.UserQuestion], str],
    reporter: app_reporter.Reporter,
) -> Settled | None:
    """Apply `[git].dirty_tree` to the outside edits, asking when the policy says so.

    Args:
        cwd: The checkout.
        policy: `[git].dirty_tree`.
        outside: The paths that changed outside the run.
        end_tree: The tree id the run's last end recorded.
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
                        f"{_listing(outside)}\n"
                        "How should this run treat them?\n"
                        "stash: set them aside for the run (applied back at the end when the"
                        " tree is clean, else the `git stash apply` line is printed)\n"
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
        if shared := _also_the_runs(cwd, end_tree, outside):
            reporter.refuse(_shared_refusal(shared))
            return None
    try:
        git_ops.stash_tracked_changes(cwd, git_ops.auto_stash_message(session_id), outside)
    except git_ops.GitError as exc:
        reporter.refuse(f"stashing the edits made outside the run failed: {exc}")
        return None
    shown = ", ".join(outside[:4]) + (f", +{len(outside) - 4} more" if len(outside) > 4 else "")
    reporter.note(
        f"stashed for this execution as yours: {shown} (changed since the run last ended)"
    )
    return Settled(stashed=True, pop=asked)
