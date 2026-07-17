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
