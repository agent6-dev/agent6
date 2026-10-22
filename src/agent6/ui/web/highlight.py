# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Highlight source text for the page with the tree-sitter language pack.

Spans per line, each `(start, end, class)` in UTF-16 units as the page slices strings, from
the pack's highlights query for the language a path names. The work is bounded: no spans
over `HIGHLIGHT_MAX_BYTES`, none when the grammar is not cached and cannot be fetched, none
when the parse and query run past `TIMEOUT_S`. The text never depends on it.
"""

from __future__ import annotations

import itertools
import threading

import tree_sitter
import tree_sitter_language_pack

# Text over this size is not highlighted.
HIGHLIGHT_MAX_BYTES = 512 * 1024
# The most the text waits for its spans; a slower parse finishes in its thread, unused.
TIMEOUT_S = 1.0

Span = tuple[int, int, str]

# The capture name's first part to the page's class; a name outside the map is dropped.
_CLASSES = {
    "attribute": "attribute",
    "boolean": "constant",
    "character": "string",
    "comment": "comment",
    "constant": "constant",
    "constructor": "function",
    "delimiter": "punctuation",
    "escape": "string",
    "float": "number",
    "function": "function",
    "keyword": "keyword",
    "label": "keyword",
    "module": "type",
    "namespace": "type",
    "number": "number",
    "operator": "operator",
    "property": "property",
    "punctuation": "punctuation",
    "string": "string",
    "tag": "tag",
    "type": "type",
    "variable": "variable",
}


def language_of(path: str) -> str:
    """Return the pack's language name for a path, or "" when it names none or plain text."""
    if path.lower().endswith(".txt"):
        return ""  # the pack maps `.txt` to vimdoc; a text file is plain
    return tree_sitter_language_pack.detect_language_from_path(path) or ""


def spans(text: str, language: str) -> list[list[Span]]:
    """Return each line's highlight spans, or an empty list when nothing is highlighted.

    Args:
        text: The source; a newline ends a line, and a final one starts no empty line.
        language: The pack's language name, from `language_of`.

    Returns:
        One list per line of `(start, end, class)` in UTF-16 units, the innermost and most
        specific capture kept where captures nest; `[]` without a language, over the cap,
        without the grammar, or past the deadline.
    """
    if not language or len(text.encode()) > HIGHLIGHT_MAX_BYTES:
        return []
    out: list[list[list[Span]]] = []
    worker = threading.Thread(target=lambda: out.append(_compute(text, language)), daemon=True)
    worker.start()
    worker.join(TIMEOUT_S)
    return out[0] if out else []


def _compute(text: str, language: str) -> list[list[Span]]:
    """Parse the text, run the highlights query, and lay the captures out per line.

    Args:
        text: The source.
        language: The pack's language name.

    Returns:
        The spans per line; `[]` when the grammar cannot be loaded or has no query.
    """
    try:
        parser = tree_sitter_language_pack.get_parser(language)
        source = tree_sitter_language_pack.get_highlights_query(language)
        lang = parser.language
        if source is None or lang is None:
            return []
        data = text.encode()
        tree = parser.parse(data)
        matches = tree_sitter.QueryCursor(tree_sitter.Query(lang, source)).matches(tree.root_node)
    except (tree_sitter_language_pack.Error, OSError, ValueError):
        return []
    # Painted outer before inner, and an earlier pattern before a later one, so the
    # innermost and most specific capture is what remains.
    captures = sorted(
        (node.start_byte, -node.end_byte, index, cls)
        for index, groups in matches
        for name, nodes in groups.items()
        if (cls := _CLASSES.get(name.partition(".")[0]))
        for node in nodes
    )
    paint: list[str | None] = [None] * len(data)
    for start, neg_end, _index, cls in captures:
        paint[start:-neg_end] = [cls] * (-neg_end - start)
    lines = data.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    rows: list[list[Span]] = []
    pos = 0
    for line in lines:
        row: list[Span] = []
        col = 0
        for cls, run in itertools.groupby(paint[pos : pos + len(line)]):
            n = len(list(run))
            if cls is not None:
                row.append((col, col + n, cls))
            col += n
        if not line.isascii():
            row = [(_utf16(line, a), _utf16(line, b), c) for a, b, c in row]
        rows.append(row)
        pos += len(line) + 1
    return rows


def _utf16(line: bytes, byte_col: int) -> int:
    """Return the UTF-16 column of a byte offset into a line, as the page counts."""
    return len(line[:byte_col].decode("utf-8", errors="replace").encode("utf-16-le")) // 2
