# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Format a tree-sitter symbol outline into a compact prompt block.

The repo-priors section of the worker's system prompt can include a per-file
symbol outline (top-level defs/classes) so the model orients without reading
every file. This module renders that block from the dispatcher's outlines,
bounded by file/per-file/char caps. Pure formatting; loop.py decides whether
to include it.
"""

from __future__ import annotations

from pathlib import Path

from agent6.tools.index import Symbol

SYMBOL_OUTLINE_MAX_CHARS = 8000
SYMBOL_OUTLINE_MAX_FILES = 120
SYMBOL_OUTLINE_MAX_PER_FILE = 12
SYMBOL_OUTLINE_KIND_PRIORITY: tuple[str, ...] = (
    "class",
    "struct",
    "enum",
    "trait",
    "interface",
    "function",
    "method",
)


def build_symbol_outline_block(
    outlines: dict[Path, list[Symbol]],
    *,
    root: Path,
) -> str:
    """Format the per-file symbol outline into a compact prompt block.

    Layout::

        path/to/file.py:
          class Foo:12
          function bar:30
          ...
        path/to/other.rs:
          struct Bar:5

    Hard caps keep the block bounded:
      - At most ``SYMBOL_OUTLINE_MAX_PER_FILE`` rows per file (truncated
        with a ``... (+N more)`` line).
      - At most ``SYMBOL_OUTLINE_MAX_FILES`` files (overflow summarised).
      - At most ``SYMBOL_OUTLINE_MAX_CHARS`` characters total; we stop
        emitting files as soon as the budget would be exceeded.

    Returns an empty string when ``outlines`` is empty.
    """
    if not outlines:
        return ""
    root_resolved = root.resolve()
    rel_entries: list[tuple[str, list[Symbol]]] = []
    for path, syms in outlines.items():
        if not syms:
            continue
        try:
            rel = path.resolve().relative_to(root_resolved)
            rel_str = str(rel)
        except ValueError:
            continue
        rel_entries.append((rel_str, syms))
    rel_entries.sort(key=lambda t: t[0])

    rows: list[str] = []
    total = 0
    files_emitted = 0
    for files_emitted, (rel_str, syms) in enumerate(rel_entries):
        if files_emitted >= SYMBOL_OUTLINE_MAX_FILES:
            remaining = len(rel_entries) - files_emitted
            rows.append(f"... ({remaining} more files)")
            break
        kept = sorted(
            syms,
            key=lambda s: (
                SYMBOL_OUTLINE_KIND_PRIORITY.index(s.kind)
                if s.kind in SYMBOL_OUTLINE_KIND_PRIORITY
                else len(SYMBOL_OUTLINE_KIND_PRIORITY),
                s.line,
            ),
        )[:SYMBOL_OUTLINE_MAX_PER_FILE]
        kept.sort(key=lambda s: s.line)
        header = f"{rel_str}:"
        body_lines = [f"  {s.kind} {s.name}:{s.line}" for s in kept]
        if len(syms) > SYMBOL_OUTLINE_MAX_PER_FILE:
            body_lines.append(f"  ... (+{len(syms) - SYMBOL_OUTLINE_MAX_PER_FILE} more)")
        chunk = "\n".join([header, *body_lines])
        added = len(chunk) + 1
        if total + added > SYMBOL_OUTLINE_MAX_CHARS and rows:
            remaining = len(rel_entries) - files_emitted
            rows.append(f"... ({remaining} more files; outline budget exhausted)")
            break
        rows.append(chunk)
        total += added
    return "\n".join(rows)
