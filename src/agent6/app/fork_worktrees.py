# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Manage a fork's linked git worktree after the fork.

The dirt check before one is removed, the sessions that still own one, and the sweep
`sessions prune` runs over the worktrees of forks whose tips landed.
"""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from pathlib import Path

from agent6.git_ops import (
    GitError,
    chain_dirty,
    chain_dirty_paths,
    chain_ref_for,
    chain_tip,
    merge_stamp_holds,
    remove_worktree,
)
from agent6.paths import state_dir
from agent6.sessions.ipc import worker_is_alive
from agent6.sessions.lock import (
    checkout_lock_path,
)
from agent6.sessions.manifest import (
    ManifestError,
    SessionManifest,
    read_manifest,
)
from agent6.viewmodel import session_dirs


def remove_fork_worktree(repo: Path, worktree: Path, tips: tuple[str, ...]) -> tuple[bool, str]:
    """Delete a fork's worktree and its checkout lock unless it holds unlanded work.

    The dirt check is git's own rule for `worktree remove`: a merged fork's tree can still
    carry an uncommitted edit or a file never added, which `rmtree` would take with no way back.

    Args:
        repo: The repository the worktree is linked to.
        worktree: The worktree to remove.
        tips: The commits its sessions landed.

    Returns:
        Whether it was removed, and the reason when not ("" on success).
    """
    dirt = uncommitted_in_worktree(worktree, tips)
    if dirt:
        return False, dirt
    lock_path = checkout_lock_path(state_dir(worktree), worktree)
    if not remove_worktree(repo, worktree):
        return False, (
            "could not be removed: not a linked worktree of this repository,"
            " or a file in it would not delete"
        )
    lock_path.unlink(missing_ok=True)
    return True, ""


def uncommitted_in_worktree(worktree: Path, tips: tuple[str, ...]) -> str:
    """Return what the worktree holds that none of the tips does, as one phrase.

    A fork's worktree stays detached at its fork point while its run commits to the chain,
    so `git status` there reports the whole run as dirt: the comparison is against the run's
    own tips (HEAD when it has none).

    Args:
        worktree: The worktree to inspect.
        tips: The commits its sessions landed.

    Returns:
        The keep-line phrase, or "" when a tip covers it or it is unreadable.
    """
    if not worktree.is_dir():
        return ""
    tips = tips or ("HEAD",)
    try:
        for tip in tips:
            if not chain_dirty(worktree, tip, None):
                return ""
        held = chain_dirty_paths(worktree, tips[-1], None, 5)
    except (GitError, OSError):
        # git runs with cwd=worktree, so a directory that vanished is an OSError: not dirt.
        return ""
    if not held:
        return ""
    named = ", ".join(held[:4]) + (", ..." if len(held) > 4 else "")
    return f"holds work no commit has: {named}"


def worktree_owners(state_dir: Path) -> dict[Path, list[tuple[Path, SessionManifest]]]:
    """Return every worktree a session manifest names, with the sessions naming it.

    The manifests are the only record of which directories are agent6's: a path no manifest
    names is never touched. An `/undo` fork shares its source's worktree.
    """
    owners: dict[Path, list[tuple[Path, SessionManifest]]] = {}
    for session_dir in session_dirs(state_dir):
        with contextlib.suppress(ManifestError):
            manifest = read_manifest(session_dir)
            if manifest.worktree is not None:
                owners.setdefault(manifest.worktree, []).append((session_dir, manifest))
    return owners


def _still_needs_worktree(repo: Path, session_dir: Path, manifest: SessionManifest) -> str:
    """Return why the session still needs its worktree ("live", "unmerged"), or "".

    The merge stamp is the test of "merged": the branch still points where the merge left it.
    """
    if worker_is_alive(session_dir):
        return "live"
    merged = manifest.merged is not None and merge_stamp_holds(
        repo, session_dir.name, manifest.run_branch or "", manifest.merged.tip
    )
    return "" if merged else "unmerged"


def _landed_tips(repo: Path, sessions: Sequence[tuple[Path, SessionManifest]]) -> tuple[str, ...]:
    """Return the commits the sessions landed: each chain tip, else its merge stamp's tip.

    `--delete-squashed` deletes the ref in the same sweep, and the commit outlives it.
    """
    tips = (
        chain_tip(repo, chain_ref_for(d.name)) or (m.merged.tip if m.merged else "")
        for d, m in sessions
    )
    return tuple(tip for tip in tips if tip)


def sweep_fork_worktrees(repo: Path, state: Path) -> tuple[list[str], list[tuple[str, str]]]:
    """Remove every fork worktree whose sessions all landed their work; keep the rest.

    Args:
        repo: The repository the worktrees are linked to.
        state: The repo's state directory.

    Returns:
        The removed session ids, and (kept id, why) per kept session.
    """
    removed: list[str] = []
    kept: list[tuple[str, str]] = []
    for worktree, sessions in worktree_owners(state).items():
        if not worktree.exists():
            continue
        needs = {d.name: why for d, m in sessions if (why := _still_needs_worktree(repo, d, m))}
        if needs:
            first = next(iter(needs))
            kept.extend((d.name, needs.get(d.name, f"shared with {first}")) for d, _ in sessions)
            continue
        gone, note = remove_fork_worktree(repo, worktree, _landed_tips(repo, sessions))
        if gone:
            removed.extend(d.name for d, _ in sessions)
        elif note:
            # Merged, but the tree holds work no commit has or would not delete: keep it, say why.
            kept.extend((d.name, note) for d, _ in sessions)
    return removed, kept
