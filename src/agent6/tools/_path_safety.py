# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Containment for in-process filesystem access.

Every tool that reads/writes a path in-process (outside
``agent6.sandbox.jail.run_in_jail``) resolves it through here first: reject an
absolute path or a ``..`` component, then require the resolved path to still
be under *root*. Shared by the fs handlers (read_file / list_dir / grep /
apply_edit / apply_patch) and the navigation handlers (outline / find_*),
which all take an untrusted ``path`` argument.
"""

from __future__ import annotations

import errno
import os
from dataclasses import dataclass
from pathlib import Path

from agent6.tools.errors import ToolError


@dataclass(frozen=True, slots=True)
class SafePath:
    abs_path: Path
    rel_path: Path


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


def open_contained(root: Path, abs_path: Path, flags: int, rel: str) -> int:
    """Open the checked path and PROVE the descriptor is the file that was
    checked. Returns an fd the caller owns.

    :func:`resolve_in_root` resolves and contains a path; every caller then
    re-opened it BY PATH, which is a second lookup with a window in between. A
    jailed ``run_background`` loop can swap the leaf for a symlink inside that
    window -- the workspace is writable and a symlink needs no access to its
    target -- and these tools run IN-PROCESS, outside the jail, as the
    operator. Probed against the unguarded write: content landed outside the
    workspace on the 7th attempt.

    ``O_NOFOLLOW`` refuses a leaf that BECAME a symlink (a resolved path has
    none, so an honest call is unaffected -- including one through an in-repo
    symlink, which resolved to its real target already). The ``/proc/self/fd``
    readback then re-checks containment against what the kernel actually
    opened, which also covers a parent directory swapped in the same window.
    """
    try:
        fd = os.open(abs_path, flags | os.O_NOFOLLOW, 0o644)
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.EMLINK):
            raise ToolError(f"Path became a symlink while it was being used: {rel!r}") from exc
        raise
    try:
        opened = Path(f"/proc/self/fd/{fd}").readlink()
        root_real = root.resolve()
        if opened != root_real and root_real not in opened.parents:
            raise ToolError(f"Path escapes repo root: {rel!r}")
    except Exception:
        os.close(fd)
        raise
    return fd


def read_contained(root: Path, abs_path: Path, rel: str, *, errors: str = "strict") -> str:
    """The file's text, read through a descriptor proven to be the checked
    file. ``UnicodeDecodeError`` still reaches the caller, which reports it."""
    fd = open_contained(root, abs_path, os.O_RDONLY, rel)
    with os.fdopen(fd, encoding="utf-8", errors=errors) as handle:
        return handle.read()


def write_contained(root: Path, abs_path: Path, rel: str, content: str) -> None:
    """Replace the file's text through a descriptor proven to be the checked
    file."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    with os.fdopen(open_contained(root, abs_path, flags, rel), "w", encoding="utf-8") as handle:
        handle.write(content)
