# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The writer flocks: one per session dir, one per checkout.

A second concurrent writer of one run's shared state (the snapshot, the checkpoints, the
curator DAG, the run branch) is refused, as `machine_lock` refuses one for a machine.
"""

from __future__ import annotations

import contextlib
import os
import pathlib

from agent6 import paths, portable


def acquire_single_writer(session_dir: pathlib.Path) -> int | None:
    """Take a non-blocking exclusive lock on `<session-dir>/worker.lock`.

    A second writer of the same run dir would spawn a second curator whose in-memory cache
    clobbers the first's parent-child links, and interleave commits on the run branch.
    flock releases on process death, so a crashed writer leaves no lock to block a resume.

    Args:
        session_dir: The session directory, created when absent.

    Returns:
        The held fd, passed to `release_single_writer` at teardown; None when another live
        process holds the lock.
    """
    paths.mkdir_for_real_user(session_dir)
    fd = os.open(session_dir / "worker.lock", os.O_CREAT | os.O_RDWR, 0o644)
    try:
        portable.lock_exclusive(fd, blocking=False)
    except OSError:
        os.close(fd)
        return None
    return fd


def release_single_writer(fd: int | None) -> None:
    """Release and close a lock fd; a no-op on None.

    A raw fd does not self-close on GC, and a leaked one keeps the flock held and refuses a
    later same-dir run in the same process.

    Args:
        fd: The fd from `acquire_single_writer` or `acquire_repo_writer`.
    """
    if fd is None:
        return
    with contextlib.suppress(OSError):
        portable.unlock(fd)
    with contextlib.suppress(OSError):
        os.close(fd)


SINGLE_WRITER_BUSY = (
    "REFUSING: run {rid!r} is already being driven by another agent6 process "
    "(its worker.lock is held). Concurrent run/resume of the same run would "
    "corrupt its state (a second curator clobbers the task graph, and commits "
    "interleave on the run branch). Wait for that process to finish; a crashed "
    "one releases the lock automatically."
)


def checkout_lock_path(state_dir: pathlib.Path, checkout: pathlib.Path) -> pathlib.Path:
    """Return the writer lock of the checkout a path is in.

    A repository's checkouts (its working tree, each linked worktree) share one state dir
    and hold one lock each.

    Args:
        state_dir: The repo's state dir.
        checkout: Any path inside the checkout.

    Returns:
        `<state-dir>/locks/<checkout-id>.lock`.
    """
    return state_dir / "locks" / f"{paths.repo_id(paths.checkout_root(checkout))}.lock"


def acquire_repo_writer(
    state_dir: pathlib.Path, checkout: pathlib.Path, session_id: str
) -> int | None:
    """Take a non-blocking exclusive lock on a checkout: one live run-mode worker per checkout.

    Each commit stages the whole working tree, so a second concurrent run would fold the
    other's in-flight edits into its own chain. Plan and ask make no commits and never take
    this lock. The holder stamps its session id into the file so a refusal can name it.

    Args:
        state_dir: The repo's state dir.
        checkout: Any path inside the checkout.
        session_id: The holder's id.

    Returns:
        The held fd, passed to `release_single_writer` at teardown; None when another live
        process holds the lock.
    """
    lock_path = checkout_lock_path(state_dir, checkout)
    paths.mkdir_for_real_user(lock_path.parent)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        portable.lock_exclusive(fd, blocking=False)
    except OSError:
        os.close(fd)
        return None
    os.ftruncate(fd, 0)
    os.write(fd, f"{session_id}\n".encode())
    return fd


def repo_writer_holder(state_dir: pathlib.Path, checkout: pathlib.Path) -> str:
    """Return the session id the checkout lock's holder stamped, or "".

    Advisory, for a refusal message; the flock is the boundary.

    Args:
        state_dir: The repo's state dir.
        checkout: Any path inside the checkout.

    Returns:
        The stamped id, or "" when the file is unreadable.
    """
    try:
        return checkout_lock_path(state_dir, checkout).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def repo_writer_held(state_dir: pathlib.Path, checkout: pathlib.Path) -> bool:
    """Return whether a live worker holds the checkout lock.

    An advisory probe for a front-end preflight: it takes a shared lock, which an exclusive
    holder blocks and a second probe does not, so asking never excludes the writer asked
    about. A race past this probe still parks at `acquire_repo_writer`.

    Args:
        state_dir: The repo's state dir.
        checkout: Any path inside the checkout.

    Returns:
        True when the exclusive lock is held.
    """
    lock_path = checkout_lock_path(state_dir, checkout)
    if not lock_path.exists():
        return False
    try:
        fd = os.open(lock_path, os.O_RDWR)
    except OSError:
        return False
    try:
        portable.lock_shared_nonblocking(fd)
    except OSError:
        os.close(fd)
        return True
    release_single_writer(fd)
    return False
