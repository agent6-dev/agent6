# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Manage a session's linked git worktree: a fork's or a `run --worktree` run's.

Adding one, the dirt check before one is removed, the sessions that still own one, and the
sweep `sessions prune` runs over the worktrees of sessions with no work left to land.
"""

from __future__ import annotations

import contextlib
import dataclasses
import pathlib
from collections.abc import Sequence

from agent6 import errors, git_ops, paths, terminal
from agent6.sessions import lock, process
from agent6.sessions import manifest as sessions_manifest
from agent6.viewmodel import session_dirs


@dataclasses.dataclass(frozen=True, slots=True)
class Checkout:
    """A session's own checkout, recorded so its jail never reads the worktree's `.git` pointer.

    Attributes:
        worktree: The linked worktree.
        git_dir: The repository git dir the worktree points into.
        set_up: Whether the worktree ran its setup command; a new one has not.
    """

    worktree: pathlib.Path
    git_dir: pathlib.Path
    set_up: bool = False


def add_session_worktree(repo: pathlib.Path, path: pathlib.Path, sha: str) -> Checkout:
    """Add a session's linked worktree, detached at the sha, and return it with its git dir.

    Args:
        repo: The repository the worktree is linked to.
        path: Where the worktree goes.
        sha: The commit it starts at.

    Returns:
        The worktree and the git dir the session's jail grants from.

    Raises:
        GitError: Git refused the worktree.
    """
    checkout = Checkout(path, git_ops.git_common_dir(repo))
    git_ops.add_worktree(repo, path, sha)
    return checkout


def remove_fork_worktree(
    repo: pathlib.Path, worktree: pathlib.Path, tips: tuple[str, ...]
) -> tuple[bool, str]:
    """Delete a fork's worktree and its checkout lock unless it holds unlanded work.

    The dirt check is git's own rule for `worktree remove`: a merged fork's tree can still
    carry an uncommitted edit or a file never added, which `rmtree` would take with no way back.
    A commit made by hand on the worktree's detached HEAD has no ref but that HEAD, which the
    removal deletes: it stays too.

    Args:
        repo: The repository the worktree is linked to.
        worktree: The worktree to remove.
        tips: The commits its sessions landed, and the ones they started from.

    Returns:
        Whether it was removed, and the reason when not ("" on success).
    """
    try:
        head = git_ops.status(worktree).head_sha if worktree.is_dir() else ""
    except (errors.GitError, OSError):
        head = ""  # a vanished directory, or no git there: nothing of its own to keep
    if (
        head
        and head not in tips
        and not any(git_ops.is_ancestor(repo, head, tip) for tip in tips)
        and not git_ops.commit_is_reachable(repo, head)
    ):
        return False, "holds a commit no branch or chain has"
    dirt = uncommitted_in_worktree(worktree, tips)
    if dirt:
        return False, dirt
    lock_path = lock.checkout_lock_path(paths.state_dir(worktree), worktree)
    if not git_ops.remove_worktree(repo, worktree):
        return False, (
            "could not be removed: not a linked worktree of this repository,"
            " or a file in it would not delete"
        )
    lock_path.unlink(missing_ok=True)
    return True, ""


def uncommitted_in_worktree(worktree: pathlib.Path, tips: tuple[str, ...]) -> str:
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
            if not git_ops.chain_dirty(worktree, tip, None):
                return ""
        held = git_ops.chain_dirty_paths(worktree, tips[-1], None, 5)
    except (errors.GitError, OSError):
        # git runs with cwd=worktree, so a directory that vanished is an OSError: not dirt.
        return ""
    if not held:
        return ""
    named = ", ".join(map(terminal.printable_name, held[:4])) + (", ..." if len(held) > 4 else "")
    return f"holds work no commit has: {named}"


def worktree_owners(
    state_dir: pathlib.Path,
) -> dict[pathlib.Path, list[tuple[pathlib.Path, sessions_manifest.SessionManifest]]]:
    """Return every worktree a session manifest names, with the sessions naming it.

    The manifests are the only record of which directories are agent6's: a path no manifest
    names is never touched. An `/undo` fork shares its source's worktree.

    Args:
        state_dir: The repo's state directory.

    Returns:
        Each worktree, mapped to the directory and manifest of every session naming it.
    """
    owners: dict[pathlib.Path, list[tuple[pathlib.Path, sessions_manifest.SessionManifest]]] = {}
    for session_dir in session_dirs(state_dir):
        with contextlib.suppress(sessions_manifest.ManifestError):
            manifest = sessions_manifest.read_manifest(session_dir)
            if manifest.worktree is not None:
                owners.setdefault(manifest.worktree, []).append((session_dir, manifest))
    return owners


def sweep_checkout(cwd: pathlib.Path) -> pathlib.Path:
    """Return the checkout `sessions prune` and `sessions rm` run git from.

    Args:
        cwd: The directory the verb runs in.

    Returns:
        The repository's own checkout when `cwd` is inside a session's worktree, else `cwd`.
        Those verbs remove the worktree, and git cannot run in a directory that is gone.
    """
    owned = {worktree.resolve() for worktree in worktree_owners(paths.state_dir(cwd))}
    return paths.project_root(cwd) if paths.checkout_root(cwd) in owned else cwd


def _still_needs_worktree(
    repo: pathlib.Path, session_dir: pathlib.Path, manifest: sessions_manifest.SessionManifest
) -> str:
    """Return why the session still needs its worktree ("live", "unmerged"), or "".

    The merge stamp is the test of "merged": the branch still points where the merge left it.
    A session with no chain committed nothing and has no merge to wait for; a fork's chain is
    cut at its fork point, so a fork always has one.

    Args:
        repo: The repository the worktree is linked to.
        session_dir: The session's directory.
        manifest: The session's manifest.

    Returns:
        "live" while its worker runs, "unmerged" until its merge stamp holds, else "".
    """
    if process.worker_is_alive(session_dir):
        return "live"
    chain = git_ops.chain_ref_for(session_dir.name)
    if manifest.merged is None and git_ops.chain_tip(repo, chain) is None:
        return ""
    merged = manifest.merged is not None and git_ops.merge_stamp_holds(
        repo, session_dir.name, manifest.run_branch or "", manifest.merged.tip
    )
    return "" if merged else "unmerged"


def _landed_tips(
    repo: pathlib.Path, sessions: Sequence[tuple[pathlib.Path, sessions_manifest.SessionManifest]]
) -> tuple[str, ...]:
    """Return the commits the sessions landed or started from.

    Each chain tip, else its merge stamp's tip (`--delete-squashed` deletes the ref in the same
    sweep, and the commit outlives it), plus each session's base and fork point: a worktree
    left at one of those holds nothing of its own.

    Args:
        repo: The repository.
        sessions: The directory and manifest of each session sharing the worktree.

    Returns:
        The commits, deduplicated, in session order.
    """
    tips: list[str] = []
    for d, m in sessions:
        landed = git_ops.chain_tip(repo, git_ops.chain_ref_for(d.name)) or (
            m.merged.tip if m.merged else ""
        )
        tips += [tip for tip in (landed, m.base_sha, m.forked_from_sha or "") if tip]
    return tuple(dict.fromkeys(tips))


def sweep_fork_worktrees(
    repo: pathlib.Path, state: pathlib.Path
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Remove every session worktree whose sessions have no work left to land; keep the rest.

    Args:
        repo: The repository the worktrees are linked to.
        state: The repo's state directory.

    Returns:
        (removed id, why) per removed session, and (kept id, why) per kept session.
    """
    removed: list[tuple[str, str]] = []
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
            removed.extend(
                (d.name, "merged" if m.merged else "committed nothing") for d, m in sessions
            )
        elif note:
            # Merged, but the tree holds work no commit has or would not delete: keep it, say why.
            kept.extend((d.name, note) for d, _ in sessions)
    return removed, kept
