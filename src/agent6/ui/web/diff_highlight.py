"""Highlight diff inputs only where their bytes match the displayed patch lines."""

from __future__ import annotations

import pathlib
import re
from typing import Literal

from agent6.ui.web import highlight
from agent6.vcs import git_files, git_history

_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_HEADER = re.compile(r'^diff --git ("(?:[^"\\]|\\.)*"|a/.*?) ("(?:[^"\\]|\\.)*"|b/.*)$')


def side_spans(
    patch: str,
    data: bytes | None,
    path: str,
    side: Literal["old", "new"],
    *,
    context: str | None = None,
) -> list[list[highlight.Span]]:
    """Return spans at absolute line indexes, with undisplayed lines left empty.

    Args:
        patch: One file's displayed unified patch, possibly cut at a byte cap.
        data: The complete bytes read with the patch, or None for an unavailable side.
        path: This side's path, which selects its language.
        side: The old or new side.
        context: Displayed file text for a change without text hunks, or None.

    Returns:
        Per-line UTF-16 spans, or an empty list for absent, binary, oversized or
        converted text, an unavailable grammar, or a parse past its deadline.
    """
    if not data or len(data) > highlight.HIGHLIGHT_MAX_BYTES or git_history.looks_binary(data):
        return []
    displayed: dict[int, str] = {}
    old = new = old_left = new_left = 0
    for line in patch.split("\n"):
        if match := _HUNK.match(line):
            old, new = int(match[1]), int(match[3])
            old_left = int(match[2]) if match[2] is not None else 1
            new_left = int(match[4]) if match[4] is not None else 1
        elif (old_left or new_left) and line[:1] in {" ", "+", "-"}:
            mark, text = line[0], line[1:]
            if side == "old" and mark != "+":
                displayed[old] = text
            if side == "new" and mark != "-":
                displayed[new] = text
            if mark != "+":
                old += 1
                old_left -= 1
            if mark != "-":
                new += 1
                new_left -= 1
    if context is not None:
        displayed = dict(enumerate(context.removesuffix("\n").split("\n"), 1)) if context else {}
    if not displayed:
        return []
    text = data.decode("utf-8", errors="replace")
    lines = text.split("\n")
    if any(n < 1 or n > len(lines) or lines[n - 1] != row for n, row in displayed.items()):
        return []
    rows = highlight.spans(text, highlight.language_of(path))
    return [row if i + 1 in displayed else [] for i, row in enumerate(rows)]


def revision_spans(
    repo: pathlib.Path, base: str, head: str, patch: str
) -> tuple[dict[str, list[list[highlight.Span]]], dict[str, list[list[highlight.Span]]]]:
    """Return path-keyed old and new spans for a patch between immutable revisions.

    Args:
        repo: The repository.
        base: The resolved base commit or empty tree.
        head: The resolved head commit.
        patch: The displayed patch, possibly truncated.

    Returns:
        The old and new span maps, each indexed by its side's path and line minus one.
    """
    old_spans: dict[str, list[list[highlight.Span]]] = {}
    new_spans: dict[str, list[list[highlight.Span]]] = {}
    for chunk in re.split(r"(?m)(?=^diff --git )", patch):
        header = chunk.partition("\n")[0]
        if not (match := _HEADER.match(header)):
            continue
        old_path, new_path = (git_files.unquote_git_path(match[i])[2:] for i in (1, 2))
        old = git_files.blob_at(repo, base, old_path, max_bytes=highlight.HIGHLIGHT_MAX_BYTES)
        new = git_files.blob_at(repo, head, new_path, max_bytes=highlight.HIGHLIGHT_MAX_BYTES)
        old_spans[old_path] = side_spans(chunk, old.data if old else None, old_path, "old")
        new_spans[new_path] = side_spans(chunk, new.data if new else None, new_path, "new")
    return old_spans, new_spans
