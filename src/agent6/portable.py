# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Cross-platform primitives for the few places agent6 touches POSIX-only APIs.

Pure stdlib, no agent6 imports. Keeps the platform split contained in one
spot instead of scattering ``sys.platform`` checks through the graph and
machine journals. The sandbox itself remains Linux-only (see
``agent6.detect.sandbox_available``), and native Windows is unsupported
(use WSL); this module keeps the platform-neutral plumbing (file locks,
durable renames) working so the agent can run unsandboxed on macOS.
"""

from __future__ import annotations

import os
import sys
import tempfile
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


def atomic_write(path: Path, data: str | bytes) -> None:
    """Write data via temp file + durable rename.

    The temp file lives beside the target, is fsync'd before the rename, and the
    parent directory is fsync'd after the rename so a crash cannot lose the new
    directory entry on POSIX filesystems.
    """
    _ensure_parent_dirs(path.parent)
    fd = -1
    tmp_name = ""
    try:
        fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        # Preserve an existing target's mode across a re-publish; for a NEW file
        # keep mkstemp's owner-only 0o600 rather than widening to a hardcoded
        # 0o644 (which bypassed the umask). These are per-user run/machine state
        # files; owner-only is the secure default.
        # chmod the fd before writing (the two mode-specific branches below only
        # differ in text vs binary, which pyright needs narrowed for `fh.write`).
        mode = _existing_mode(path)
        if sys.platform != "win32" and mode is not None:
            os.fchmod(fd, mode)
        if isinstance(data, bytes):
            with os.fdopen(fd, "wb") as fh:
                fd = -1
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
        else:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fd = -1
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
        Path(tmp_name).replace(path)
    except Exception:
        if fd >= 0:
            os.close(fd)
        if tmp_name:
            Path(tmp_name).unlink(missing_ok=True)
        raise
    fsync_dir(path.parent)


def _ensure_parent_dirs(parent: Path) -> None:
    missing: list[Path] = []
    cur = parent
    while not cur.exists():
        missing.append(cur)
        if cur.parent == cur:
            break
        cur = cur.parent
    parent.mkdir(parents=True, exist_ok=True)
    for directory in reversed(missing):
        fsync_dir(directory.parent)


def _existing_mode(path: Path) -> int | None:
    """The target's current permission bits, or None if it does not exist yet
    (the caller then leaves the temp file at mkstemp's owner-only 0o600)."""
    try:
        return path.stat().st_mode & 0o777
    except OSError:
        return None
