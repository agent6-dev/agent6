# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Rewind a session for `/undo`: fork it at the checkpoint before its last message.

The fork keeps the undone session's checkout, which is put back to the checkpoint's tree.
Nothing is lost: the tree as it stands is committed onto the undone session's ref first, so
the later commits stay there. The operator's untracked-at-start files, HEAD and the index are
left alone.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent6.app.fork import Checkout, create_fork, resolve_source
from agent6.app.reporter import STDIO_REPORTER, Reporter
from agent6.app.resume import commits_note
from agent6.config import ConfigError
from agent6.config.layer import load_effective
from agent6.git_ops import (
    CommitIdentity,
    GitError,
    chain_commit,
    chain_ref_for,
    chain_tip,
    sync_worktree,
    tree_diff_paths,
    worktree_tree,
)
from agent6.graph.storage import (
    list_checkpoint_turns,
)
from agent6.harness._snapshot import SessionSnapshot, load_session_snapshot
from agent6.paths import state_dir
from agent6.sessions.ipc import read_worker_pid, worker_is_alive
from agent6.sessions.layout import (
    SessionLayout,
    read_untracked_at_start,
)
from agent6.sessions.lock import (
    acquire_repo_writer,
    release_single_writer,
    repo_writer_holder,
)
from agent6.sessions.manifest import (
    ManifestError,
    model_git_refusal,
    read_manifest,
)
from agent6.task_text import operator_task_text

_STEER_NOTICE = "OPERATOR STEERING"


def _text_of(content: object) -> str:
    """Return the plain text of a message content, a string or content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            str(block.get("text", ""))
            for block in content  # pyright: ignore[reportUnknownVariableType]
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "\n".join(part for part in parts if part)
    return ""


def _operator_messages(messages: list[dict[str, Any]]) -> list[str]:
    """Return the operator's words in a restored conversation: the task, then each steer.

    A steer loses its notice line; tool results and other harness notices stay out.
    """
    out: list[str] = []
    for i, message in enumerate(messages):
        if message.get("role") != "user":
            continue
        text = _text_of(message.get("content"))
        if not text:
            continue
        if i == 0:
            out.append(text)
        elif text.startswith(_STEER_NOTICE):
            body = text.partition("\n")[2].strip()
            out.append(body or text)
    return out


@dataclass(frozen=True, slots=True)
class _Checkpoint:
    """Locate a checkpoint.

    Attributes:
        session_id: The session holding it.
        at_turn: The file it sits in, what `fork --at-turn` addresses.
        turn: The turn its conversation stands before, the number every surface prints; a
            fork's seed differs, file 0 holding its source's turn N.
    """

    session_id: str
    at_turn: int
    turn: int


@dataclass(frozen=True, slots=True)
class UndoTarget:
    """Say where `/undo` forks a session.

    Attributes:
        session: The session being undone.
        source_session_id: The session holding the checkpoint: itself or a fork ancestor.
        at_turn: The checkpoint file.
        turn: The turn the checkpoint stands before.
        undone_text: The message taken back, refilled into the composer.
    """

    session: SessionLayout
    source_session_id: str
    at_turn: int
    turn: int
    undone_text: str


def _snapshot_at(layout: SessionLayout, at_turn: int) -> SessionSnapshot | None:
    """Return the checkpoint in the file, or None when it is unreadable."""
    try:
        return load_session_snapshot(layout.checkpoint_path(at_turn))
    except (OSError, ValueError):
        return None


def undo_target(  # noqa: PLR0911 - each refusal names its own reason
    state_dir: Path, session_id: str, *, reporter: Reporter = STDIO_REPORTER
) -> UndoTarget | None:
    """Resolve the checkpoint `/undo` forks a session at.

    The newest checkpoint, in the session or up its fork lineage, whose conversation ends
    before the last operator message. With only the opening task, the earliest checkpoint.

    Args:
        state_dir: The repo's state directory.
        session_id: The session to undo.
        reporter: Where a refusal is printed.

    Returns:
        The target, or None with the reason printed.
    """
    src = resolve_source(state_dir, session_id, reporter=reporter)
    if src is None:
        return None
    turns = sorted(list_checkpoint_turns(src))
    if not turns:
        reporter.err(f"nothing to undo: {src.session_id} has no checkpoints.")
        return None
    newest = turns[-1]
    try:
        snap = load_session_snapshot(src.checkpoint_path(newest))
    except (OSError, ValueError) as exc:
        reporter.error(f"cannot read checkpoint {newest} of {src.session_id}: {exc}")
        return None
    ops = _operator_messages(snap.messages)
    if len(ops) <= 1:
        if len(turns) < 2:
            reporter.err(f"nothing to undo: {src.session_id} is at its opening message.")
            return None
        first = _snapshot_at(src, turns[0])
        if first is None:
            reporter.error(f"cannot read checkpoint {turns[0]} of {src.session_id}.")
            return None
        try:
            task = read_manifest(src.session_dir).user_task
        except ManifestError:
            task = operator_task_text(ops[0]) if ops else ""
        return UndoTarget(src, src.session_id, turns[0], first.next_iteration, task)
    target = _newest_checkpoint_below(src, len(ops), ops[-1])
    if target is None:
        reporter.err(f"nothing to undo: no state before the last message of {src.session_id}.")
        return None
    return UndoTarget(src, target.session_id, target.at_turn, target.turn, ops[-1])


def _newest_checkpoint_below(
    layout: SessionLayout,
    current_ops: int,
    last_message: str,
    *,
    seen: frozenset[str] = frozenset(),
) -> _Checkpoint | None:
    """Return the checkpoint before the last message first appeared, following fork lineage.

    Compaction shrinks conversations, so the operator-message count is not monotonic: the
    append transition decides first, the count second. A fork carries one seed checkpoint, so
    walking past it resolves in the parent.

    Args:
        layout: The session to search.
        current_ops: The operator-message count at the newest checkpoint.
        last_message: The message being taken back.
        seen: The sessions already walked; a revisited id (a corrupt lineage) ends the walk.

    Returns:
        The checkpoint, or None when no ancestor holds one.
    """
    snapshots = [
        (at_turn, snap, _operator_messages(snap.messages))
        for at_turn in sorted(list_checkpoint_turns(layout))
        if (snap := _snapshot_at(layout, at_turn)) is not None
    ]
    for previous, current in zip(reversed(snapshots[:-1]), reversed(snapshots[1:]), strict=True):
        if current[2].count(last_message) > previous[2].count(last_message):
            return _Checkpoint(layout.session_id, previous[0], previous[1].next_iteration)
    for at_turn, snap, messages in reversed(snapshots):
        if len(messages) < current_ops:
            return _Checkpoint(layout.session_id, at_turn, snap.next_iteration)
    try:
        parent = read_manifest(layout.session_dir).parent_session_id
    except ManifestError:
        return None
    if not parent or parent in seen:
        return None
    parent_layout = SessionLayout(
        state_dir=layout.state_dir, session_id=parent, subdir=layout.subdir
    )
    if not parent_layout.session_dir.is_dir():
        return None
    return _newest_checkpoint_below(
        parent_layout, current_ops, last_message, seen=seen | {layout.session_id}
    )


def _rewind_checkout(checkout: Path, *, tip: str, sha: str, exclude: frozenset[str]) -> list[str]:
    """Put the checkout back to a commit's tree for every tracked path that differs.

    The current content is staged as a tree first, so the two-tree sync moves only the paths
    that differ. HEAD and the shared index stay untouched.

    Args:
        checkout: The checkout to rewind.
        tip: The chain commit holding the current content.
        sha: The commit to rewind to.
        exclude: The session's untracked-at-start files, left alone.

    Returns:
        The paths put back.
    """
    current = worktree_tree(checkout, tip, exclude)
    paths = tree_diff_paths(checkout, sha, current)
    if paths:
        sync_worktree(checkout, current, sha)
    return paths


def _checkout_writer_lock(
    state: Path, checkout: Path, undone: SessionLayout
) -> tuple[int | None, str]:
    """Take the checkout's writer lock for the commit and the rewind.

    The undone session's own live worker is checked directly: a plan or ask worker takes no
    writer lock.

    Args:
        state: The repo's state directory.
        checkout: The checkout to lock.
        undone: The session being undone.

    Returns:
        The held fd and "", or None and the refusal; None and "" when this process is the
        undone session's worker, whose lock is no obstacle.
    """
    if read_worker_pid(undone.session_dir) == os.getpid():
        return None, ""
    if worker_is_alive(undone.session_dir):
        return None, (
            f"run {undone.session_id!r} is still live; /undo would put the tree back"
            " under it. Stop it first:\n"
            f"    agent6 stop {undone.session_id}"
        )
    lock_fd = acquire_repo_writer(state, checkout, undone.session_id)
    if lock_fd is None:
        holder = repo_writer_holder(state, checkout) or "another run"
        return None, (
            f"run {holder!r} is driving this checkout, and /undo would put the tree back"
            " under it. Stop it first:\n"
            f"    agent6 stop {holder}"
        )
    return lock_fd, ""


def undo_fork(  # noqa: PLR0911 - each refusal names its own reason
    config_path: Path | None,
    session_id: str,
    *,
    cwd: Path,
    reporter: Reporter = STDIO_REPORTER,
) -> tuple[str, str] | None:
    """Commit the tree as it stands, fork the session at its undo target, and rewind the checkout.

    The fork is unstarted, in the session's own checkout: its worktree when it has one, else
    the repository. The writer lock is held across the commit and the rewind unless this
    process is the session's live worker; any other live run driving the checkout refuses.

    Args:
        config_path: An explicit config file, else the effective one.
        session_id: The session to undo.
        cwd: The repository.
        reporter: Where refusals and the notice go.

    Returns:
        The fork's id and the text taken back, or None with the reason printed.
    """
    state = state_dir(cwd)
    target = undo_target(state, session_id, reporter=reporter)
    if target is None:
        return None
    undone = target.session
    try:
        manifest = read_manifest(undone.session_dir)
    except ManifestError as exc:
        reporter.error(f"cannot read the manifest of {undone.session_id}: {exc}")
        return None
    refusal = model_git_refusal(manifest, "undo")
    if refusal is not None:
        # A model-controlled run has no chain; a chain ref written here is one auto_merge lands.
        reporter.error(refusal)
        return None
    checkout = manifest.worktree or cwd
    if manifest.worktree is not None and manifest.worktree_git_dir is None:
        reporter.error(
            f"cannot undo {undone.session_id}: its manifest names a worktree but not the"
            f" repository git dir it points into; `agent6 fork {undone.session_id}` continues"
            " its commits in a new worktree."
        )
        return None
    if manifest.worktree is not None and not (checkout / ".git").exists():
        reporter.error(
            f"cannot undo {undone.session_id}: its worktree {checkout} is gone (pruned or"
            f" removed); {commits_note(cwd, manifest)}; `agent6 fork {undone.session_id}`"
            " continues it in a new worktree."
        )
        return None
    try:
        cfg = load_effective(cwd, config_path).config
    except ConfigError as exc:
        reporter.error(str(exc))
        return None
    lock_fd, refusal = _checkout_writer_lock(state, checkout, undone)
    if refusal:
        reporter.refuse(refusal)
        return None
    ref = chain_ref_for(undone.session_id)
    where = manifest.run_branch or ref
    exclude = read_untracked_at_start(undone.session_dir)
    try:
        try:
            kept = chain_commit(
                checkout,
                f"agent6 undo: the tree before turn {target.turn} was taken back",
                ref=ref,
                fallback_parent=manifest.base_sha or None,
                identity=CommitIdentity(name=cfg.git.commit.name, email=cfg.git.commit.email),
                also_branch=manifest.run_branch,
                exclude=exclude,
            )
        except GitError as exc:
            reporter.error(f"the tree as it stands could not be committed onto {where}: {exc}")
            return None
        child, rc = create_fork(
            config_path,
            target.source_session_id,
            at_turn=target.at_turn,
            cwd=cwd,
            worktree=False,
            checkout=(
                Checkout(manifest.worktree, manifest.worktree_git_dir)
                if manifest.worktree is not None and manifest.worktree_git_dir is not None
                else None
            ),
            checkout_untracked=exclude,
            reporter=reporter,
        )
        if rc != 0:
            return None
        child_layout = SessionLayout(state_dir=state, session_id=child, subdir=undone.subdir)
        sha = read_manifest(child_layout.session_dir).forked_from_sha or ""
        turn = f"turn {target.turn} ({sha[:12]})"
        try:
            paths = _rewind_checkout(
                checkout, tip=chain_tip(checkout, ref) or sha, sha=sha, exclude=exclude
            )
        except GitError as exc:
            reporter.error(
                f"the checkout was not put back to {turn}: {exc}."
                f" Fork {child} continues from the tree as it is."
            )
            return child, target.undone_text
    finally:
        release_single_writer(lock_fd)
    _report_rewind(reporter, paths, turn=turn, kept=kept, where=where)
    return child, target.undone_text


def _report_rewind(
    reporter: Reporter, paths: list[str], *, turn: str, kept: str | None, where: str
) -> None:
    """Print the undo notice: the paths put back and where the earlier tree and commits live."""
    if paths:
        shown = ", ".join(paths[:10]) + (f", +{len(paths) - 10} more" if len(paths) > 10 else "")
        reporter.note(
            f"put {len(paths)} path(s) back to {turn}: {shown} (HEAD and the index are untouched)"
        )
    else:
        reporter.note(f"the checkout already matches {turn}")
    stood = f"the tree as it stood is commit {kept[:12]} on {where}; " if kept else ""
    reporter.note(f"{stood}the later commits stay on {where}")
