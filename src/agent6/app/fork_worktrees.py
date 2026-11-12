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
import shutil
from collections.abc import Sequence
from typing import Literal

from agent6 import errors, paths, terminal
from agent6.sessions import lock, process
from agent6.sessions import manifest as sessions_manifest
from agent6.vcs import git_ops, git_read
from agent6.viewmodel import session_dirs


@dataclasses.dataclass(frozen=True, slots=True)
class Checkout:
    """A session's own worktree, recorded so its jail never reads a linked one's `.git` pointer.

    Attributes:
        worktree: The worktree: a clone of the repository, or a linked worktree of it.
        git_dir: The repository git dir a linked worktree points into; None for a clone, which
            holds its own.
        set_up: Whether the worktree ran its setup command; a new one has not.
    """

    worktree: pathlib.Path
    git_dir: pathlib.Path | None
    set_up: bool = False


def add_session_worktree(
    repo: pathlib.Path, path: pathlib.Path, sha: str, kind: Literal["clone", "linked"]
) -> Checkout:
    """Make a session's own worktree, detached at the sha.

    Args:
        repo: The repository.
        path: Where the worktree goes.
        sha: The commit it starts at.
        kind: `[git.worktree].kind`: a clone of the repository, or a linked worktree of it.

    Returns:
        The worktree, with the git dir the session's jail grants from when it is linked.

    Raises:
        GitError: Git refused the worktree.
    """
    if kind == "clone":
        # A local clone copies the whole object store, so a commit only a hidden ref holds is there.
        git_ops.clone_repo(repo, path)
        git_ops.checkout_detached(path, sha)
        # A command typed in the clone finds the repository's sessions, as in a linked worktree.
        origin = paths.checkout_root(repo)
        (path / ".git" / paths.SESSION_CLONE_ORIGIN).write_text(f"{origin}\n", encoding="utf-8")
        record = paths.session_clone_record(origin, path)
        paths.mkdir_for_real_user(record.parent)
        record.write_text(f"{path.resolve()}\n", encoding="utf-8")
        return Checkout(path, None)
    checkout = Checkout(path, git_read.git_common_dir(repo))
    git_ops.add_worktree(repo, path, sha)
    return checkout


def is_clone(worktree: pathlib.Path) -> bool:
    """Return whether a session's worktree is a clone, which holds its own git data.

    Args:
        worktree: The session's worktree.

    Returns:
        True for a clone; False for a linked worktree, whose `.git` is a file.
    """
    return (worktree / ".git").is_dir()


def mirror_origin(repo: pathlib.Path, worktree: pathlib.Path) -> pathlib.Path | None:
    """Return the repository a session's clone copies its commits into.

    Args:
        repo: The repository the session belongs to.
        worktree: The directory the session works in.

    Returns:
        The repository when the session works in a clone of it; None in the repository itself
        or in a linked worktree, where a commit is the repository's as it is made.
    """
    return repo if worktree.resolve() != repo.resolve() and is_clone(worktree) else None


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
        head = git_read.status(worktree).head_sha if worktree.is_dir() else ""
    except (errors.GitError, OSError):
        head = ""  # a vanished directory, or no git there: nothing of its own to keep
    if (
        head
        and head not in tips
        and not any(git_read.is_ancestor(repo, head, tip) for tip in tips)
        and not git_read.commit_is_reachable(repo, head)
    ):
        return False, "holds a commit no branch or chain has"
    dirt = uncommitted_in_worktree(worktree, tips)
    if dirt:
        return False, dirt
    lock_path = lock.checkout_lock_path(paths.state_dir(repo), worktree)
    if is_clone(worktree):
        record = paths.session_clone_record(paths.checkout_root(repo), worktree)
        try:
            shutil.rmtree(worktree)
        except OSError:
            return False, "could not be removed: a file in it would not delete"
        record.unlink(missing_ok=True)
    elif not git_ops.remove_worktree(repo, worktree):
        return False, (
            "could not be removed: not a linked worktree of this repository,"
            " or a file in it would not delete"
        )
    lock_path.unlink(missing_ok=True)
    return True, ""


def uncommitted_in_worktree(worktree: pathlib.Path, tips: tuple[str, ...]) -> str:
    """Return what the worktree holds that neither HEAD nor any tip does, as one phrase.

    A fork's worktree stays detached at its fork point while its run commits to the chain,
    so `git status` there reports the whole run as dirt: the comparison is against the run's
    own tips and HEAD.

    Args:
        worktree: The worktree to inspect.
        tips: The commits its sessions landed.

    Returns:
        The keep-line phrase, or "" when HEAD or a tip covers it or it is unreadable.
    """
    if not worktree.is_dir():
        return ""
    tips = (*tips, "HEAD")
    try:
        for tip in tips:
            if not git_read.chain_dirty(worktree, tip, None):
                return ""
        # Named against the closest commit: against HEAD every file the run changed differs.
        held = min((git_read.chain_dirty_paths(worktree, tip, None, 5) for tip in tips), key=len)
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
    """Return the checkout `sessions prune` and `sessions remove` run git from.

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
    """Return why the session still needs its worktree, or "".

    HEAD's commits must reach the base branch or be covered by the session's merge stamp.
    A chain also needs a merge stamp that still holds; a fork's chain starts at its fork point.

    Args:
        repo: The repository the worktree is linked to.
        session_dir: The session's directory.
        manifest: The session's manifest.

    Returns:
        "live", HEAD's unmerged commit count, "unmerged" for an unlanded chain, or "".
    """
    if process.worker_is_alive(session_dir):
        return "live"
    stamp = manifest.merged
    merged = stamp is not None and git_read.merge_stamp_holds(
        repo, session_dir.name, manifest.run_branch or "", stamp.tip
    )
    if manifest.worktree is not None and manifest.worktree.is_dir():
        head = git_read.status(manifest.worktree)
        base = git_read.branch_tip_sha(repo, manifest.base_branch) or manifest.base_sha
        if base and head.head_sha:
            count = len(git_read.list_run_commits(repo, base, head.head_sha))
            # Squash and task merges cover the fork point without making it reachable from the base.
            if count and not (
                merged
                and stamp is not None
                and stamp.tip
                and git_read.is_ancestor(repo, head.head_sha, stamp.tip)
            ):
                name = terminal.printable_name(head.branch or "detached HEAD")
                return f"{count} unmerged commit{'s' if count != 1 else ''} on {name}"
    chain = git_read.chain_ref_for(session_dir.name)
    if stamp is None and git_read.chain_tip(repo, chain) is None:
        return ""
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
        landed = git_read.chain_tip(repo, git_read.chain_ref_for(d.name)) or (
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
        head = git_read.status(worktree).head_sha
        gone, note = remove_fork_worktree(repo, worktree, _landed_tips(repo, sessions))
        if gone:
            removed.extend(
                (d.name, "merged" if m.merged or head != m.base_sha else "committed nothing")
                for d, m in sessions
            )
        elif note:
            # Merged, but the tree holds work no commit has or would not delete: keep it, say why.
            kept.extend((d.name, note) for d, _ in sessions)
    return removed, kept
