# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Per-repo agent memory under ``<state_dir>/memory/``.

One fact per markdown file plus a ``MEMORY.md`` index (one line per entry).
The index is injected into every run's system prompt; the files are read and
edited with the ordinary in-process tools through a narrow path grant, so
recording or correcting a memory is a normal file edit. Model-authored
context: never instructions, never secrets. Repo-only by design; sharing a
memory across repos is the operator copying it.
"""

from __future__ import annotations

import re
from pathlib import Path

from agent6.errors import OperatorError

MEMORY_DIR_NAME = "memory"
INDEX_NAME = "MEMORY.md"
# The index is injected whole; past the cap it is clipped with a pointer so
# a runaway index cannot flood every prompt in the repo.
INDEX_INJECT_CAP = 4_096

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


class MemoryStoreError(OperatorError):
    """Memory-store operation failed (bad name, unreadable store)."""


def memory_dir(state_dir: Path) -> Path:
    return state_dir / MEMORY_DIR_NAME


def index_path(state_dir: Path) -> Path:
    return memory_dir(state_dir) / INDEX_NAME


def index_text(state_dir: Path) -> str:
    """The index body for prompt injection; "" when absent or unreadable
    (memory is context, one stray byte must not kill every run)."""
    try:
        return index_path(state_dir).read_text(encoding="utf-8").strip()
    except (OSError, ValueError):
        return ""


def _check_name(name: str) -> str:
    if not _NAME_RE.match(name):
        raise MemoryStoreError(
            f"bad memory name {name!r}: lowercase letters, digits, and dashes only"
        )
    return name


def add(state_dir: Path, name: str, body: str) -> Path:
    """Operator CLI helper: write ``<name>.md`` and append its index line."""
    body = body.strip()
    if not body:
        raise MemoryStoreError("memory body must be non-empty")
    d = memory_dir(state_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{_check_name(name)}.md"
    if path.exists():
        raise MemoryStoreError(f"memory {name!r} exists; edit {path} or pick another name")
    path.write_text(body + "\n", encoding="utf-8")
    hook = body.splitlines()[0][:120]
    idx = index_path(state_dir)
    existing = index_text(state_dir)
    line = f"- {name}: {hook}"
    idx.write_text((existing + "\n" if existing else "") + line + "\n", encoding="utf-8")
    return path


def remove(state_dir: Path, name: str) -> None:
    """Operator CLI helper: delete ``<name>.md`` and its index line."""
    _check_name(name)
    path = memory_dir(state_dir) / f"{name}.md"
    if not path.is_file():
        raise MemoryStoreError(f"no memory named {name!r}")
    path.unlink()
    idx = index_path(state_dir)
    kept = [
        ln
        for ln in index_text(state_dir).splitlines()
        if not re.match(rf"^\s*[-*]\s*{re.escape(name)}\s*:", ln)
    ]
    idx.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")


def show(state_dir: Path, name: str) -> str:
    _check_name(name)
    path = memory_dir(state_dir) / f"{name}.md"
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise MemoryStoreError(f"no memory named {name!r}") from exc
