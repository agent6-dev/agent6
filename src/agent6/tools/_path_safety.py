# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Containment for in-process filesystem access.

Every tool that reads/writes a path in-process (outside
``agent6.sandbox.jail.run_in_jail``) resolves it through here first: reject an
absolute path or a ``..`` component, then require the resolved path to still
be under *root*. Shared by the fs handlers (read_file / list_dir / grep /
apply_edit / apply_patch), the navigation handlers (outline / find_*) -- which
all take an untrusted ``path`` argument -- and the symbol index they query.
"""

from __future__ import annotations

import contextlib
import errno
import os
from dataclasses import dataclass
from pathlib import Path

from agent6.tools.errors import ToolError


@dataclass(frozen=True, slots=True)
class SafePath:
    abs_path: Path
    rel_path: Path


@dataclass(frozen=True, slots=True)
class ContainedEntry:
    """One entry of a contained listing. ``is_dir`` follows a symlink, like
    ``Path.is_dir``; a caller that recurses checks ``is_symlink`` too, because
    the walk refuses to traverse one."""

    name: str
    is_dir: bool
    is_symlink: bool


def resolve_in_root(root: Path, candidate: str) -> SafePath:
    """Resolve *candidate* relative to *root* and ensure it stays inside *root*."""
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
    return SafePath(abs_path=abs_path, rel_path=rel)


def _open_dir(dir_fd: int, name: str, *, create: bool) -> int:
    """A descriptor for subdirectory *name* of *dir_fd*, created when it is
    missing and *create*."""
    flags = os.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        return os.open(name, flags, dir_fd=dir_fd)
    except FileNotFoundError:
        if not create:
            raise
    with contextlib.suppress(FileExistsError):
        os.mkdir(name, dir_fd=dir_fd)
    return os.open(name, flags, dir_fd=dir_fd)


def open_contained(root: Path, rel_path: Path, flags: int, *, create_parents: bool = False) -> int:
    """Open *rel_path* one component at a time from a descriptor on *root*,
    each hop relative to the one before it. Returns an fd the caller owns.

    :func:`resolve_in_root` resolves and contains a path; opening it again by
    its full path is a second lookup, and a jailed ``run_background`` loop can
    swap a component for a symlink out of the workspace in between (the
    workspace is writable, a symlink needs no access to its target, and these
    tools run IN-PROCESS, outside the jail, as the operator). For a write
    (``O_CREAT|O_TRUNC``) the host file is already truncated by the time any
    after-the-fact check can reject it.

    ``O_NOFOLLOW`` on every component, including the parents this creates,
    contains the walk by construction: no hop can traverse a symlink. ``..``
    and an absolute path are refused here rather than trusted to the caller,
    so containment is a property of this function, not of nine call sites.
    Honest callers are unaffected, including one working through an in-repo
    symlink, whose resolved path names the real target.
    """
    if rel_path.is_absolute():
        raise ToolError(f"Path is not relative to the workspace: {rel_path}")
    if ".." in rel_path.parts:
        raise ToolError(f"Path contains '..': {rel_path}")
    dir_fd = os.open(root, os.O_PATH | os.O_DIRECTORY)
    try:
        for name in rel_path.parts[:-1]:
            child = _open_dir(dir_fd, name, create=create_parents)
            os.close(dir_fd)
            dir_fd = child
        # The root itself is the one path with no leaf to name.
        return os.open(rel_path.name or ".", flags | os.O_NOFOLLOW, 0o644, dir_fd=dir_fd)
    except NotADirectoryError as exc:
        raise ToolError(f"Path component is not a directory: {rel_path}") from exc
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ToolError(f"Path became a symlink while it was being used: {rel_path}") from exc
        raise
    finally:
        os.close(dir_fd)


def read_contained(root: Path, rel_path: Path, *, errors: str = "strict") -> str:
    """The file's text, read through a descriptor walked from *root*.
    ``UnicodeDecodeError`` still reaches the caller, which reports it."""
    fd = open_contained(root, rel_path, os.O_RDONLY)
    with os.fdopen(fd, encoding="utf-8", errors=errors) as handle:
        return handle.read()


def read_bytes_contained(root: Path, rel_path: Path) -> bytes:
    """The file's bytes, read through a descriptor walked from *root*. For a
    reader that indexes into the source by byte offset (tree-sitter), which the
    newline translation of a text read would shift."""
    fd = open_contained(root, rel_path, os.O_RDONLY)
    with os.fdopen(fd, "rb") as handle:
        return handle.read()


def list_contained(root: Path, rel_path: Path) -> list[ContainedEntry]:
    """The directory's entries, listed through a descriptor walked from *root*.

    The same containment as :func:`read_contained`, for the tools that read a
    directory rather than a file: a name resolved a second time is a second
    lookup, so a listing taken by full path can be a host directory's.
    """
    fd = open_contained(root, rel_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with os.scandir(fd) as entries:
            return [ContainedEntry(e.name, e.is_dir(), e.is_symlink()) for e in entries]
    finally:
        os.close(fd)


def write_contained(root: Path, rel_path: Path, content: str) -> None:
    """Replace the file's text through a descriptor walked from *root*, adding
    any missing parent directories along the same walk."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    fd = open_contained(root, rel_path, flags, create_parents=True)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(content)
