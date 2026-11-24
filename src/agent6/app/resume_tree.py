# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Settle the edits made outside a run between its executions, at `agent6 resume`.

Each end records the tree the run left (`session.stop.tree`). At the next start, the paths
whose content differs from that tree changed outside the run: they are the operator's and
follow `[git].dirty_tree`. "Differs from the chain tip" is not the test, because a run's own
work can be uncommitted after a clean end: an operator stop skips the final checkpoint, and
`commit_per_turn = false`, `git.control = "model"` and a failed commit all leave it in the
tree. A run killed before it could record a tree (SIGKILL, a crash) leaves only the chain tip
to go by, or under the model's git the HEAD its last snapshot recorded: what differs from it
and no tool of the run wrote follows the policy too, since nothing says whose it is.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import shlex
from collections.abc import Callable, Collection, Sequence

from agent6 import errors, kinds, terminal
from agent6.app import preflight, resume_chain
from agent6.app import reporter as app_reporter
from agent6.config import Config
from agent6.harness import loop_snapshot
from agent6.sessions import layout as sessions_layout
from agent6.sessions import manifest as sessions_manifest
from agent6.tools import schema
from agent6.vcs.git import ops as git_ops
from agent6.vcs.git import read as git_read
from agent6.viewmodel.session import log as session_log

_LISTED = 10


def visible_changes(
    cwd: pathlib.Path, before: str, after: str, exclude: Collection[str]
) -> list[str]:
    """List changed paths the run can take, even when the trees have different seeds.

    Args:
        cwd: The checkout.
        before: The tree or commit compared from.
        after: The tree compared to.
        exclude: Paths left at their seed's content in each tree.

    Returns:
        The differing paths outside the exclusions.
    """
    return [
        p for p in git_read.tree_diff_paths(cwd, before, after) if not git_read.excluded(p, exclude)
    ]


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
        end_tree: The tree the run left, or HEAD's tree after a recorded merge into HEAD.
            An empty id means no tree was recorded.
        seed: The recorded index or commit whose tracked paths define the working tree.
        exclude: The operator's untracked files and the run's hidden paths, left out of
            both trees.
        written: The paths the run's own tools wrote, which stay its own when no tree was
            recorded.

    Returns:
        The differing paths; with no recorded tree, what differs from the seed that no tool
        of the run wrote; empty when nothing changed.
    """
    tree = git_read.worktree_tree(cwd, seed, exclude)
    if end_tree:
        return visible_changes(cwd, end_tree, tree, exclude)
    if seed is None:
        return []
    return [p for p in git_read.tree_diff_paths(cwd, seed, tree) if p not in written]


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
    changed = set(git_read.tree_diff_paths(cwd, "HEAD", left))
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
        and git_read.branch_exists(cwd, run_branch)
        and not git_read.tree_diff_paths(cwd, run_branch, end_tree)
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
        seed: The chain tip, else the HEAD the last snapshot recorded, else the run's base:
            what the run left when no tree was recorded.
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


def paths_the_run_wrote(logs_path: pathlib.Path) -> frozenset[str]:
    """Collect every path a `tool.result` of the run names as written.

    Args:
        logs_path: The run's event log.

    Returns:
        The paths over every execution; a torn line contributes none.
    """
    paths: set[str] = set()
    try:
        with logs_path.open(encoding="utf-8", errors="replace") as lines:
            for line in lines:
                paths.update(_written_by(line))
    except OSError:
        return frozenset()
    return frozenset(paths)


def _written_by(line: str) -> list[str]:
    """Return the paths one journal line names as written; none unless it is a `tool.result`."""
    try:
        event = json.loads(line)
    except ValueError:
        return []
    if not isinstance(event, dict) or event.get("type") != "tool.result":
        return []
    named = event.get("paths", ())
    return [str(p) for p in named] if isinstance(named, list) else []


def first_four(paths: Collection[str]) -> str:
    """Return the first four paths in order, then how many more there are."""
    named = sorted(paths)
    more = f", +{len(named) - 4} more" if len(named) > 4 else ""
    return ", ".join(map(terminal.printable_name, named[:4])) + more


@dataclasses.dataclass(frozen=True, slots=True)
class TreeVerdict:
    """Whose the working tree's changes are, as a resume reads them.

    Attributes:
        untracked_at_start: The operator's untracked files, with those that arrived between
            executions; every commit leaves them out.
        hidden: The run's hidden paths inside the repository.
        outside: The paths that changed outside the run, which follow `[git].dirty_tree`.
        carried: The run's own edits that its last execution left uncommitted under the
            model's git.
        end_tree: The tree the run left: the one its last end recorded, or HEAD's after a
            merge into HEAD; "" when none was recorded.
        seed: The chain tip, else the HEAD the last snapshot recorded, else the run's base:
            what the run left when no tree was recorded; None without any.
    """

    untracked_at_start: frozenset[str]
    hidden: frozenset[str]
    outside: tuple[str, ...] = ()
    carried: frozenset[str] = frozenset()
    end_tree: str = ""
    seed: str | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class _Record:
    """What the run's record says it left, read before the working tree is.

    Attributes:
        died_in_turn: Whether the last execution died in the turn the snapshot resumes at.
        model_end: Whether the run's last end recorded a tree under the model's git.
        record_ref: What holds the run's files: the chain, or under the model's git the
            recorded index, else the snapshot's HEAD, else the end tree.
        end_tree: The tree the run left; see `TreeVerdict`.
        seed: What the run left when no tree was recorded; see `TreeVerdict`.
        written: The paths the run's tools wrote.
    """

    died_in_turn: bool
    model_end: bool
    record_ref: str
    end_tree: str
    seed: str | None
    written: frozenset[str]


def _record(
    cwd: pathlib.Path,
    layout: sessions_layout.SessionLayout,
    manifest: sessions_manifest.SessionManifest,
    *,
    snapshot: loop_snapshot.SessionSnapshot,
    scan: session_log.LogScan,
    merged: resume_chain.MergedStart | None,
) -> _Record:
    """Return what the run left, read from its journal, its snapshot and its chain."""
    try:
        flight = loop_snapshot.read_turn_marker(
            layout.session_dir / loop_snapshot.TURN_IN_FLIGHT_NAME
        )
    except ValueError:
        flight = None  # the loop names and drops it when it resumes
    chain_ref = git_read.chain_ref_for(layout.session_id)
    end_tree = scan.end_tree
    model_end = manifest.git_control == "model" and bool(end_tree)
    record_ref = (scan.index_tree or snapshot.head_sha or end_tree) if model_end else chain_ref
    seed = (
        record_ref
        if model_end
        else git_read.chain_tip(cwd, chain_ref)
        or snapshot.head_sha  # the model's git keeps no chain
        or manifest.base_sha
        or None
    )
    written = paths_the_run_wrote(layout.logs_path)
    if merged is not None:
        # The run's work is in HEAD, so what differs from HEAD is the operator's.
        end_tree, seed = f"{merged.head}^{{tree}}", merged.head
    return _Record(
        died_in_turn=flight is not None and flight.turn == snapshot.next_turn,
        model_end=model_end,
        record_ref=record_ref,
        end_tree=end_tree,
        seed=seed,
        written=written,
    )


def _arrived(cwd: pathlib.Path, record: _Record, untracked: frozenset[str]) -> frozenset[str]:
    """Find the untracked files that arrived between executions.

    Args:
        cwd: The checkout.
        record: What the run left.
        untracked: The operator's untracked files at the run's start.

    Returns:
        The files the run did not leave, its record does not hold and none of its tools wrote.
    """
    stopped_paths = git_read.tree_paths(cwd, record.end_tree) if record.end_tree else frozenset()
    arrived = (
        git_read.untracked_paths(cwd)
        - untracked
        - git_read.tree_paths(cwd, record.record_ref)
        - stopped_paths
        - record.written
    )
    if not arrived:
        return arrived
    # A file replacing a recorded directory is a change to that tree.
    return arrived - {str(p) for name in stopped_paths for p in pathlib.PurePosixPath(name).parents}


def _leave_out_the_operators(
    layout: sessions_layout.SessionLayout,
    untracked: frozenset[str],
    arrived: frozenset[str],
    *,
    died_in_turn: bool,
    reporter: app_reporter.Reporter,
) -> frozenset[str]:
    """Say whose the arrived files are, and record the operator's as left out.

    After a death mid-turn a call of that turn may have written a file with no result to say
    so, and it stays the run's.

    Args:
        layout: The session's directory layout.
        untracked: The operator's untracked files at the run's start.
        arrived: The untracked files that arrived between executions.
        died_in_turn: Whether the last execution died in the turn the snapshot resumes at.
        reporter: Receives the note.

    Returns:
        The operator's untracked files, with the arrived ones that are theirs.
    """
    if not arrived:
        return untracked
    shown = first_four(arrived)
    if died_in_turn:
        them = "it" if len(arrived) == 1 else "them"
        reporter.note(
            f"taken as this run's: {shown} (untracked since the turn the last"
            f" execution died in, whose calls may have written {them}; the run's"
            f" next commit holds {them})"
        )
        return untracked
    untracked |= arrived
    sessions_layout.write_untracked_at_start(layout.session_dir, untracked)
    reporter.note(
        f"left out of this run's commits as yours: {shown} (arrived"
        " between executions, unwritten by any tool of the run)"
    )
    return untracked


def _under_the_models_git(
    cwd: pathlib.Path,
    outside: list[str],
    *,
    end_tree: str,
    head_sha: str,
    index_tree: str,
    hidden: frozenset[str],
    exclude: Collection[str],
    policy: str,
    reporter: app_reporter.Reporter,
) -> tuple[list[str], frozenset[str]]:
    """Add the edits staged outside the run to the outside ones, and find the run's own.

    Args:
        cwd: The checkout.
        outside: The paths whose working-tree content changed outside the run.
        end_tree: The tree the run's last end recorded.
        head_sha: The HEAD the last snapshot recorded; "" for none.
        index_tree: The staged tree the last stop recorded; "" for none.
        hidden: The run's hidden paths inside the repository.
        exclude: The paths left out of both trees.
        policy: `[git].dirty_tree`.
        reporter: Receives a refusal and the note.

    Returns:
        The outside paths, and the run's own edits that no commit holds.

    Raises:
        preflight.SessionRefusedError: `.agent6` holds a tracked edit, or staged edits have no
            record to tell them from the run's and the policy does not include them.
    """
    committed = head_sha or git_read.empty_tree(cwd)
    staged = git_read.index_tree(cwd)
    if any(
        code != "??"
        and code[1] != " "
        and git_read.excluded(p, {".agent6"})
        and not git_read.excluded(p, hidden)
        for code, p in git_read.porcelain_entries(cwd)
    ):
        reporter.refuse(
            "a tracked file under .agent6 has working-copy edits."
            " The run did not measure this directory."
        )
        raise preflight.SessionRefusedError(2)
    staged_changes = visible_changes(cwd, index_tree or committed, staged, hidden)
    if staged_changes and not index_tree:
        note = (
            "the run's last stop has no staged-tree record."
            f" Unconfirmed staged edits: {first_four(staged_changes)}"
        )
        if policy != "include":
            reporter.refuse(note)
            raise preflight.SessionRefusedError(2)
        reporter.note(note)
    else:
        outside = sorted(set(outside).union(staged_changes))
    carried = (
        frozenset(visible_changes(cwd, committed, end_tree, exclude))
        .union(visible_changes(cwd, committed, staged, exclude))
        .difference(outside, staged_changes)
    )
    return outside, carried


def classify(
    cwd: pathlib.Path,
    layout: sessions_layout.SessionLayout,
    manifest: sessions_manifest.SessionManifest,
    cfg: Config,
    *,
    mode: kinds.LoopMode,
    snapshot: loop_snapshot.SessionSnapshot,
    scan: session_log.LogScan,
    merged: resume_chain.MergedStart | None,
    answerable: bool,
    reporter: app_reporter.Reporter,
) -> TreeVerdict:
    """Tell the run's files from the operator's, before the resume has a live worker.

    An untracked file the run did not leave, the chain does not hold and no tool wrote arrived
    between executions, so it is the operator's. Tracked content that changed since the last
    end is the operator's too. A git failure refuses, since this decides what the run may
    commit. Only a run commits, so every other mode reads the two path sets and stops.

    Args:
        cwd: The checkout.
        layout: The session's directory layout.
        manifest: The session's manifest.
        cfg: The execution's config, for the hidden paths and `[git].dirty_tree`.
        mode: The session mode.
        snapshot: The run's last snapshot.
        scan: The scan of the run's journal.
        merged: Where the run continues from when HEAD holds its merged work.
        answerable: Whether a question would reach anyone.
        reporter: Receives the notes and a refusal.

    Returns:
        The verdict.

    Raises:
        preflight.SessionRefusedError: git could not tell the files apart, or the resume
            cannot settle the outside edits.
    """
    untracked = sessions_layout.read_untracked_at_start(layout.session_dir)
    hidden = git_read.hidden_in_repo(cwd, cfg.sandbox.hide_paths)
    if mode != "run":
        return TreeVerdict(untracked_at_start=untracked, hidden=hidden)
    record = _record(cwd, layout, manifest, snapshot=snapshot, scan=scan, merged=merged)
    carried: frozenset[str] = frozenset()
    try:
        arrived = _arrived(cwd, record, untracked)
        untracked = _leave_out_the_operators(
            layout, untracked, arrived, died_in_turn=record.died_in_turn, reporter=reporter
        )
        # A file taken as the run's above stays its own.
        runs_own = record.written | arrived if record.died_in_turn else record.written
        exclude = untracked | hidden | {".agent6"}
        outside = outside_edits(cwd, record.end_tree, record.seed, exclude, written=runs_own)
        if record.model_end:
            outside, carried = _under_the_models_git(
                cwd,
                outside,
                end_tree=record.end_tree,
                head_sha=snapshot.head_sha,
                index_tree=scan.index_tree,
                hidden=hidden,
                exclude=exclude,
                policy=cfg.git.dirty_tree,
                reporter=reporter,
            )
        unsettled = refusal(
            cwd,
            policy=cfg.git.dirty_tree,
            outside=outside,
            end_tree=record.end_tree,
            seed=record.seed,
            run_branch=manifest.run_branch,
            answerable=answerable,
        )
    except (errors.GitError, OSError) as exc:
        reporter.error(f"cannot tell the run's files from the operator's: {exc}")
        raise preflight.SessionRefusedError(2) from exc
    if unsettled is not None:
        reporter.refuse(unsettled)
        raise preflight.SessionRefusedError(2)
    return TreeVerdict(
        untracked_at_start=untracked,
        hidden=hidden,
        outside=tuple(outside),
        carried=carried,
        end_tree=record.end_tree,
        seed=record.seed,
    )


@dataclasses.dataclass(frozen=True, slots=True)
class Settled:
    """What a resume did with the outside edits.

    Attributes:
        stashed: Whether it stashed them.
        pop: Whether the operator chose the stash at the question, so the end applies it back.
        kept: The paths left in the tree for the run; the model hears their names at its
            first turn, so it rereads them instead of writing over the operator's change.
    """

    stashed: bool = False
    pop: bool = False
    kept: tuple[str, ...] = ()


def settle(
    cwd: pathlib.Path,
    tree: TreeVerdict,
    *,
    policy: str,
    run_branch: str | None,
    session_id: str,
    ask: Callable[[schema.UserQuestion], str],
    reporter: app_reporter.Reporter,
) -> Settled:
    """Apply `[git].dirty_tree` to the outside edits, asking when the policy says so.

    Args:
        cwd: The checkout.
        tree: Whose the working tree's changes are.
        policy: `[git].dirty_tree`.
        run_branch: The run's branch, or None.
        session_id: The run, which names the stash.
        ask: Asks the operator one question and returns the answer.
        reporter: Receives a refusal.

    Returns:
        What was done.

    Raises:
        preflight.SessionRefusedError: The resume is refused or cancelled, already reported.
    """
    outside, end_tree = tree.outside, tree.end_tree
    if not outside or policy == "include":
        return Settled(kept=outside)
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
            raise preflight.SessionRefusedError(2)
        if choice == "include":
            return Settled(kept=outside)
        left = end_tree or tree.seed or ""
        if shared := _also_the_runs(cwd, left, outside):
            reporter.refuse(_shared_refusal(cwd, shared, end_tree=left, run_branch=run_branch))
            raise preflight.SessionRefusedError(2)
    try:
        git_ops.stash_tracked_changes(cwd, git_ops.auto_stash_message(session_id), outside)
    except errors.GitError as exc:
        reporter.refuse(f"stashing the edits made outside the run failed: {exc}")
        raise preflight.SessionRefusedError(2) from exc
    names = ", ".join(terminal.printable_name(p) for p in outside[:4])
    shown = names + (f", +{len(outside) - 4} more" if len(outside) > 4 else "")
    since = "the run last ended" if end_tree else "the run's last commit"
    reporter.note(f"stashed for this execution as yours: {shown} (changed since {since})")
    return Settled(stashed=True, pop=asked)
