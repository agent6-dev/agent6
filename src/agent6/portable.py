# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Cross-platform primitives for the few places agent6 touches POSIX-only APIs.

Pure stdlib, no agent6 imports. Keeps the Windows/Unix split contained in one
spot instead of scattering ``sys.platform`` checks through the graph and
machine journals. The sandbox itself remains Linux-only (see
``agent6.detect.sandbox_available``); this module only covers the platform-neutral
plumbing (file locks, durable renames) that must keep working everywhere so the
agent can run unsandboxed on Windows and macOS.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl


def lock_exclusive(fd: int, *, blocking: bool) -> None:
    """Take an exclusive lock on an open file descriptor.

    When ``blocking`` is False and another process already holds the lock this
    raises ``OSError`` immediately. On POSIX this is an advisory whole-file lock
    via ``flock(2)``; on Windows it is a mandatory one-byte range lock via
    ``msvcrt.locking`` (offset 0, which the OS happily locks past EOF).
    """
    if sys.platform == "win32":
        os.lseek(fd, 0, os.SEEK_SET)
        mode = msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK
        msvcrt.locking(fd, mode, 1)
    else:
        flags = fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB
        fcntl.flock(fd, flags)


def unlock(fd: int) -> None:
    """Release a lock previously taken by :func:`lock_exclusive`."""
    if sys.platform == "win32":
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)


def fsync_dir(path: Path) -> None:
    """fsync a directory so a rename into it is durable.

    No-op on Windows, which has no directory file descriptors to fsync; the
    ``MoveFileEx``/``ReplaceFile`` semantics behind ``Path.replace`` already
    make the rename durable there.
    """
    if sys.platform == "win32":
        return
    fd = os.open(path, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
