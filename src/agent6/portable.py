# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Keep the platform split in one place: file locks, durable renames, TOML strings.

Pure stdlib, no agent6 imports. The sandbox is Linux-only and native Windows is
unsupported; this plumbing lets the agent run unsandboxed on macOS.
"""

from __future__ import annotations

import contextlib
import os
import sys
import tempfile
import threading
from collections.abc import Generator
from pathlib import Path
from typing import IO

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl


def lock_shared_nonblocking(fd: int) -> None:
    """Take a shared lock on an open descriptor without waiting.

    A probe asking whether someone is writing takes this one: an exclusive probe
    would exclude the very writer it asks about. Windows has no shared range lock,
    so the probe there takes the exclusive one.

    Raises:
        OSError: An exclusive holder has the lock.
    """
    if sys.platform == "win32":
        lock_exclusive(fd, blocking=False)
        return
    fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)


def lock_exclusive(fd: int, *, blocking: bool) -> None:
    """Take an exclusive lock on an open descriptor.

    An advisory whole-file `flock` on POSIX; a mandatory one-byte range lock at offset 0
    on Windows.

    Args:
        fd: The descriptor.
        blocking: Wait for the lock.

    Raises:
        OSError: Another process holds the lock and the call does not block.
    """
    if sys.platform == "win32":
        os.lseek(fd, 0, os.SEEK_SET)
        mode = msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK
        msvcrt.locking(fd, mode, 1)
    else:
        flags = fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB
        fcntl.flock(fd, flags)


def unlock(fd: int) -> None:
    """Release a lock taken by `lock_exclusive`."""
    if sys.platform == "win32":
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)


def _same_file(fd: int, path: Path) -> bool:
    """Return whether the descriptor and the path name the same inode."""
    try:
        a = os.fstat(fd)
        b = path.stat()
    except OSError:
        return False
    return (a.st_dev, a.st_ino) == (b.st_dev, b.st_ino)


# The lock paths the current thread holds, for reentrancy.
_HELD_LOCKS = threading.local()


def _acquire_lock(lock_path: Path) -> int | None:
    """Open and lock the lock file.

    `O_NOFOLLOW` refuses a planted symlink at the predictable lock path outright. Any
    other failure (a stale root-owned lock a non-root process cannot reopen) also
    fails open: the lock is never a correctness barrier.

    Args:
        lock_path: The lock file.

    Returns:
        The held descriptor, or None when the lock cannot be taken.
    """
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
    if sys.platform != "win32":
        flags |= os.O_NOFOLLOW
    while True:
        try:
            fd = os.open(lock_path, flags, 0o600)
        except OSError:
            return None
        try:
            lock_exclusive(fd, blocking=True)
            if sys.platform == "win32" or _same_file(fd, lock_path):
                return fd
            # The previous holder unlinked this inode after the open; retry on the fresh file.
            unlock(fd)
        except OSError:
            os.close(fd)
            return None
        except BaseException:
            os.close(fd)
            raise
        os.close(fd)


@contextlib.contextmanager
def locked_file(target: Path) -> Generator[bool]:
    """Serialize read-modify-write cycles on a file across processes.

    The lock is a sibling `<name>.lock` file, never the target: `atomic_write`
    replaces the target's inode, and a lock on it would let a waiter on the orphaned
    inode run beside a fresh locker. It fails open, since `atomic_write` already
    keeps each publish all-or-nothing: a lock that cannot be opened or taken runs
    the body unserialized, never wedging or following a symlink. The file is
    unlinked on release and a concurrent unlink is detected by inode and retried;
    on Windows it stays in place. Same-thread reentrant, keyed on the lock path
    with its parent resolved, since a second flock on the same file would
    self-deadlock; other threads block.

    Args:
        target: The file the cycle rewrites.

    Yields:
        Whether the lock is held; a transaction that would restore a whole-file
        snapshot on failure must not do so over an unserialized cycle.
    """
    _ensure_parent_dirs(target.parent)
    lock_path = target.with_name(target.name + ".lock")
    key = str(target.parent.resolve() / lock_path.name)
    held: dict[str, bool] = getattr(_HELD_LOCKS, "paths", {})
    if key in held:
        yield held[key]
        return
    fd = _acquire_lock(lock_path)
    held[key] = fd is not None
    _HELD_LOCKS.paths = held
    try:
        yield fd is not None
    finally:
        held.pop(key, None)
        if fd is not None:
            # Unlink before unlock: unlock first would let a waiter win the orphaned inode.
            if sys.platform != "win32":
                with contextlib.suppress(OSError):
                    lock_path.unlink()
            with contextlib.suppress(OSError):
                unlock(fd)
            os.close(fd)


def fsync_dir(path: Path) -> None:
    """Fsync a directory so a rename into it is durable; a no-op on Windows."""
    if sys.platform == "win32":
        return
    fd = os.open(path, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write(path: Path, data: str | bytes) -> None:
    """Write a file through a temp file beside it and a durable rename.

    The temp file is fsynced before the rename and the parent after it, so a crash
    cannot lose the new entry. An existing target keeps its mode; a new file keeps
    mkstemp's owner-only 0600.

    Args:
        path: The file.
        data: Its content.
    """
    _ensure_parent_dirs(path.parent)
    fd = -1
    tmp_name = ""
    try:
        fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
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


_TOML_BASIC_ESCAPES = {
    "\\": "\\\\",
    '"': '\\"',
    "\b": "\\b",
    "\t": "\\t",
    "\n": "\\n",
    "\f": "\\f",
    "\r": "\\r",
}


def toml_basic_string(value: str) -> str:
    """Return the value as a TOML basic string literal, quotes included.

    The one owner of the escaping: an unescaped control character writes a file
    that fails to parse on read while the write reported success.
    """
    out: list[str] = []
    for ch in value:
        if ch in _TOML_BASIC_ESCAPES:
            out.append(_TOML_BASIC_ESCAPES[ch])
        elif ord(ch) < 0x20 or ch == "\x7f":
            out.append(f"\\u{ord(ch):04X}")
        else:
            out.append(ch)
    return '"' + "".join(out) + '"'


def _ensure_parent_dirs(parent: Path) -> None:
    """Create the directory and its missing ancestors, fsyncing each new entry."""
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
    """Return the target's permission bits, or None when it does not exist yet."""
    try:
        return path.stat().st_mode & 0o777
    except OSError:
        return None


# A child's stderr is bounded: unbounded capture let a hostile MCP server write 1.8 GB in 3 s.
STDERR_KEEP_BYTES = 8192


def drain_stderr(pipe: IO[bytes], keep: list[bytes], *, close: bool = False) -> None:
    """Read a child's stderr to EOF, keeping only the tail.

    A pipe nobody reads stops the writer at 64 KB. Read at the descriptor: a buffered
    read returns only at 4 KB or EOF, so a live child's words would reach a failure
    message only after it died. The drain must be the pipe's only reader.

    Args:
        pipe: The child's stderr.
        keep: Receives the tail, at most `STDERR_KEEP_BYTES`.
        close: Close the pipe at EOF.
    """
    with contextlib.suppress(OSError, ValueError):
        while chunk := os.read(pipe.fileno(), 4096):
            keep.append(chunk)
            if len(keep) > 2:
                keep[:] = [b"".join(keep)[-STDERR_KEEP_BYTES:]]
    if close:
        pipe.close()


def stderr_tail(keep: list[bytes], limit: int = 400) -> str:
    """Return the last of what a child said, cut at a line start and marked when cut."""
    text = b"".join(keep)[-STDERR_KEEP_BYTES:].decode(errors="replace").strip()
    if len(text) <= limit:
        return text
    tail = text[-limit:]
    nl = tail.find("\n")
    if 0 <= nl < len(tail) - 1:
        tail = tail[nl + 1 :]
    return f"…[agent6: {len(text) - len(tail)} earlier chars cut]\n{tail.strip()}"


def has_controlling_tty() -> bool:
    """Return whether a controlling terminal exists, so the operator can be prompted."""
    try:
        fd = os.open("/dev/tty", os.O_RDWR | os.O_NOCTTY)
    except OSError:
        return False
    os.close(fd)
    return True
