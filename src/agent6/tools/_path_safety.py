# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Containment for in-process filesystem access.

Every tool that reads or writes a path outside `agent6.sandbox.jail.run_in_jail` resolves it
here first: an absolute path or a `..` component is refused, and the resolved path must stay
under its base. The fs handlers, the navigation handlers and the symbol index share it.
"""

from __future__ import annotations

import contextlib
import errno
import os
import stat
from dataclasses import dataclass
from pathlib import Path

from agent6.tools.errors import ToolError


class NotRegularFileError(ToolError):
    """The leaf is inside the boundary but is a directory, a FIFO or a device.

    Its own type because callers word it differently from a containment refusal.
    """


@dataclass(frozen=True, slots=True)
class SafePath:
    """A path that passed containment.

    Attributes:
        base: The tree it was contained against; every read and write walks from it.
        rel_path: The path relative to the base.
        abs_path: The base joined with the relative path.
    """

    base: Path
    rel_path: Path
    abs_path: Path


@dataclass(frozen=True, slots=True)
class ContainedEntry:
    """One entry of a contained listing.

    Attributes:
        name: The entry's name.
        is_dir: Whether it is a directory, following a symlink like `Path.is_dir`.
        is_symlink: Whether it is a symlink; a caller that recurses checks it, since the walk
            refuses to traverse one.
    """

    name: str
    is_dir: bool
    is_symlink: bool


def fold_name(name: str) -> str:
    """Return a path component as a case-insensitive filesystem would match it.

    macOS and Windows match names case-insensitively, and macOS runs agent6 unsandboxed, so
    these refusals are all that protects `.git` and the hidden trees there: compared exactly,
    `.GIT/config` opens the real `.git/config`. Folded on every platform, one rule; the cost is
    refusing a path to a distinct `.GIT`, which nobody has.
    """
    return name.lower()


def path_within(target: Path, prefix: Path) -> bool:
    """Return whether the target is the prefix or lies under it, matched by whole components."""
    folded = [fold_name(p) for p in prefix.parts]
    return [fold_name(p) for p in target.parts][: len(folded)] == folded


@dataclass(frozen=True, slots=True)
class Workspace:
    """The file boundary for everything agent6 does in-process.

    The sandbox confines child processes; this is the same policy at the other place an
    untrusted model reaches files, the tools, which ask nobody's approval. It holds at every
    isolation level, `none` included: the boundary follows the operator's config values, never
    the isolation level, so a degradation never widens what the tools may read. A relative path
    is always the workspace's; an absolute one is allowed only inside a grant. The tools that
    read another tree (a skill, the bundled docs, the run's own state) use `contain` instead.

    Attributes:
        root: The workspace root.
        denied: `[sandbox].hide_paths` plus agent6's private dirs, refused for reads and writes
            alike; a denial beats every grant.
        read_roots: `extra_read_paths` plus `extra_write_paths`, the trees the jail mounts too.
        write_roots: `extra_write_paths`.
        read_only: Files inside a write grant that the harness owns: readable, never written.
        exempt: agent6's own carve-outs from `denied`, exactly the per-repo memory dir, which
            is model-writable by design; an exempt path still needs a grant to be reachable.
    """

    root: Path
    denied: tuple[Path, ...] = ()
    read_roots: tuple[Path, ...] = ()
    write_roots: tuple[Path, ...] = ()
    read_only: tuple[Path, ...] = ()
    exempt: tuple[Path, ...] = ()

    def _denying(self, abs_path: Path) -> Path | None:
        """Return the denied root covering the path, or None; the one owner of the verdict."""
        if any(path_within(abs_path, e) for e in self.exempt):
            return None
        for d in self.denied:
            if path_within(abs_path, d):
                return d
        return None

    def is_denied(self, abs_path: Path) -> bool:
        """Return whether the path lies under a denied root and no exemption."""
        return self._denying(abs_path) is not None

    def resolve_read(self, candidate: str) -> SafePath:
        """Contain a path for reading.

        Args:
            candidate: The path the model gave.

        Returns:
            The contained path.

        Raises:
            ToolError: The path escapes the workspace and every read grant, or is denied.
        """
        return self._resolve(candidate, (self.root, *self.read_roots))

    def resolve_write(self, candidate: str) -> SafePath:
        """Contain a path for writing.

        Args:
            candidate: The path the model gave.

        Returns:
            The contained path.

        Raises:
            ToolError: The path escapes the workspace and every write grant, is denied, or is
                harness-owned.
        """
        sp = self._resolve(candidate, (self.root, *self.write_roots))
        if sp.abs_path in self.read_only:
            raise ToolError(f"Path is harness-owned and read-only: {candidate!r}")
        return sp

    def _resolve(self, candidate: str, bases: tuple[Path, ...]) -> SafePath:
        sp = (
            self._in_grant(candidate, bases)
            if candidate.startswith("/")
            else resolve_in_root(self.root, candidate)
        )
        self._refuse_denied(sp, candidate)
        return sp

    def _in_grant(self, candidate: str, bases: tuple[Path, ...]) -> SafePath:
        """Contain an absolute path against the deepest grant holding it.

        Returns:
            The path contained against that grant.

        Raises:
            ToolError: No grant holds the path.
        """
        target = Path(candidate).resolve()
        for base in sorted(bases, key=lambda b: len(b.parts), reverse=True):
            if path_within(target, base):
                return SafePath(base=base, rel_path=target.relative_to(base), abs_path=target)
        raise ToolError(f"Absolute paths are only allowed inside a granted path: {candidate!r}")

    def _refuse_denied(self, sp: SafePath, candidate: str) -> None:
        # Refused, not answered empty: a tool result can carry an error where a jail mask cannot.
        d = self._denying(sp.abs_path)
        if d is not None:
            raise ToolError(f"Path is hidden from this run: {candidate!r} (under {d})")


def contain(base: Path, candidate: str | Path) -> SafePath:
    """Contain a path under a base the caller chose, without resolving symlinks.

    For a skill's own directory or the bundled docs; the descriptor walk in `open_contained`
    enforces the containment, refusing every symlink hop.

    Args:
        base: The tree.
        candidate: The path inside it.

    Returns:
        The contained path.

    Raises:
        ToolError: The path is absolute or contains `..`.
    """
    rel = Path(candidate)
    if rel.is_absolute():
        raise ToolError(f"Absolute paths not allowed: {str(candidate)!r}")
    if ".." in rel.parts:
        raise ToolError(f"Path contains '..': {str(candidate)!r}")
    return SafePath(base=base, rel_path=rel, abs_path=base / rel)


def resolve_in_root(root: Path, candidate: str) -> SafePath:
    """Resolve a path relative to a root and require it to stay inside.

    Args:
        root: The workspace root.
        candidate: The path the model gave.

    Returns:
        The contained path.

    Raises:
        ToolError: The path is absolute, contains `..`, or resolves outside the root.
    """
    if candidate.startswith("/"):
        raise ToolError(f"Absolute paths not allowed: {candidate!r}")
    parts = Path(candidate).parts
    if ".." in parts:
        raise ToolError(f"Path contains '..': {candidate!r}")
    abs_path = (root / candidate).resolve()
    try:
        rel = abs_path.relative_to(root.resolve())
    except ValueError as exc:
        raise ToolError(f"Path escapes repo root: {candidate!r}") from exc
    return SafePath(base=root, rel_path=rel, abs_path=abs_path)


def _open_dir(dir_fd: int, name: str, *, create: bool) -> int:
    """Return a descriptor on a subdirectory, creating it when missing and asked to.

    Raises:
        FileNotFoundError: The subdirectory is missing and `create` is off.
    """
    flags = os.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        return os.open(name, flags, dir_fd=dir_fd)
    except FileNotFoundError:
        if not create:
            raise
    with contextlib.suppress(FileExistsError):
        os.mkdir(name, dir_fd=dir_fd)
    return os.open(name, flags, dir_fd=dir_fd)


def open_contained(sp: SafePath, flags: int, *, create_parents: bool = False) -> int:
    """Open a contained path one component at a time from a descriptor on its base.

    Opening a contained path again by its full name is a second lookup, and a jailed background
    command can swap a component for a symlink out of the workspace in between; for a write the
    host file is truncated before any after-the-fact check. `O_NOFOLLOW` on every hop, the
    parents this creates included, contains the walk by construction. `..` and an absolute path
    are refused here too, so containment holds for a hand-built SafePath. Unless `O_DIRECTORY`
    is asked for, the leaf must be a regular file, checked by `fstat` on the descriptor just
    opened, never by name. `O_NONBLOCK` keeps the open from blocking on a FIFO swapped in for
    the leaf; the flag is cleared before the caller reads or writes.

    Args:
        sp: The contained path.
        flags: The `os.open` flags.
        create_parents: Whether to create missing parent directories along the walk.

    Returns:
        A descriptor the caller owns.

    Raises:
        ToolError: The path is not relative, contains `..`, or a component became a symlink or
            is not a directory.
        NotRegularFileError: The leaf is not a regular file.
        OSError: The open failed for any other reason.
    """
    rel_path = sp.rel_path
    if rel_path.is_absolute():
        raise ToolError(f"Path is not relative to the workspace: {rel_path}")
    if ".." in rel_path.parts:
        raise ToolError(f"Path contains '..': {rel_path}")
    dir_fd = os.open(sp.base, os.O_PATH | os.O_DIRECTORY)
    at = "."  # the component the walk is on, for the error path below
    try:
        for at in rel_path.parts[:-1]:
            child = _open_dir(dir_fd, at, create=create_parents)
            os.close(dir_fd)
            dir_fd = child
        # The root itself is the one path with no leaf to name.
        at = rel_path.name or "."
        if flags & os.O_DIRECTORY:
            return os.open(at, flags | os.O_NOFOLLOW, 0o644, dir_fd=dir_fd)
        fd = os.open(at, flags | os.O_NOFOLLOW | os.O_NONBLOCK, 0o644, dir_fd=dir_fd)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise NotRegularFileError(f"Not a regular file: {rel_path}")
            os.set_blocking(fd, True)
        except BaseException:
            os.close(fd)
            raise
        return fd
    except NotADirectoryError as exc:
        # O_NOFOLLOW|O_DIRECTORY on a symlink is ENOTDIR on Linux, not ELOOP; one lstat names it.
        with contextlib.suppress(OSError):
            if stat.S_ISLNK(os.lstat(at, dir_fd=dir_fd).st_mode):
                raise ToolError(
                    f"Path became a symlink while it was being used: {rel_path}"
                ) from exc
        raise ToolError(f"Path component is not a directory: {rel_path}") from exc
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ToolError(f"Path became a symlink while it was being used: {rel_path}") from exc
        if exc.errno == errno.ENXIO:
            # O_WRONLY|O_NONBLOCK on a reader-less FIFO: the open itself rejects the leaf.
            raise NotRegularFileError(f"Not a regular file: {rel_path}") from exc
        raise
    finally:
        os.close(dir_fd)


def read_contained(sp: SafePath, *, errors: str = "strict", limit_chars: int | None = None) -> str:
    """Read the file's text through a descriptor walked from its base.

    Args:
        sp: The contained path.
        errors: The decode error handler.
        limit_chars: The most characters pulled into memory, so a huge file cannot OOM the
            unsandboxed agent; a caller detects truncation by reading one more. None reads
            the whole file.

    Returns:
        The text.

    Raises:
        UnicodeDecodeError: The file is not valid text under the handler.
    """
    fd = open_contained(sp, os.O_RDONLY)
    with os.fdopen(fd, encoding="utf-8", errors=errors) as handle:
        return handle.read() if limit_chars is None else handle.read(limit_chars)


def read_bytes_contained(sp: SafePath) -> bytes:
    """Return the file's bytes, read through a descriptor walked from its base.

    For a reader that indexes by byte offset (tree-sitter), which a text read's newline
    translation would shift.
    """
    fd = open_contained(sp, os.O_RDONLY)
    with os.fdopen(fd, "rb") as handle:
        return handle.read()


def list_contained(sp: SafePath) -> list[ContainedEntry]:
    """Return the directory's entries, listed through a descriptor walked from its base.

    A listing taken by full path is a second lookup, so it can be a host directory's.
    """
    fd = open_contained(sp, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with os.scandir(fd) as entries:
            return [ContainedEntry(e.name, e.is_dir(), e.is_symlink()) for e in entries]
    finally:
        os.close(fd)


def unlink_contained(sp: SafePath) -> None:
    """Remove the file by name relative to a descriptor walked to its parent.

    Raises:
        ToolError: The path names the base itself, or a component is not a directory.
    """
    if not sp.rel_path.name:
        raise ToolError(f"Not a file: {sp.rel_path}")
    parent = SafePath(sp.base, sp.rel_path.parent, sp.abs_path.parent)
    fd = open_contained(parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.unlink(sp.rel_path.name, dir_fd=fd)
    finally:
        os.close(fd)


def disk_bytes(content: str, *, like: bytes | None) -> bytes:
    """Encode the content as a write puts it on disk, keeping the file's line ending.

    A text read translates CRLF to LF, so when the file's first line ending is CRLF every line
    ending in the content is written as CRLF (one already there is not doubled).

    Args:
        content: The text to write.
        like: The file's bytes before the write, or None for a new file.

    Returns:
        The bytes to write.
    """
    if like is not None:
        i = like.find(b"\n")
        if i > 0 and like[i - 1 : i] == b"\r":
            content = content.replace("\r\n", "\n").replace("\n", "\r\n")
    return content.encode("utf-8")


def write_contained(sp: SafePath, content: str) -> int:
    """Replace the file's text through a descriptor walked from its base.

    Missing parent directories are created along the same walk.

    Args:
        sp: The contained path.
        content: The new text.

    Returns:
        The number of bytes written.
    """
    like = None
    with contextlib.suppress(OSError, ToolError):
        like = read_bytes_contained(sp)
    data = disk_bytes(content, like=like)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    fd = open_contained(sp, flags, create_parents=True)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
    return len(data)
