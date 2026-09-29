# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Render a folded `TranscriptItem` as styled lines for the CLI stream and the TUI view.

Each line is a list of `(text, style)` spans, the style a semantic name each
front-end maps to its own output, so the two skins cannot drift. Relative indent
is in the line text; each front-end adds its own left margin.
"""

from __future__ import annotations

from typing import Literal

from agent6.viewmodel.transcript import (
    CALL,
    COMMIT,
    DONE,
    OPERATOR,
    RESULT,
    THINK,
    TranscriptItem,
)

StyleName = Literal[
    "thinking",
    "think-marker",
    "text",
    "call",
    "verify",
    "arg",
    "ok",
    "fail",
    "detail",
    "tail",
    "more",
    "commit",
    "marker",
    "done-ok",
    "done-fail",
    "done-neutral",
    "body",
    "done-detail",
    "operator",
]
Span = tuple[str, StyleName]
Line = list[Span]

# hidden omits thinking and tool items; collapsed clips them; expanded shows them in full.
DetailLevel = Literal["hidden", "collapsed", "expanded"]

TAIL_CLIP = 120
DETAIL_CLIP = 120


def _tool_lines(item: TranscriptItem, *, expanded: bool) -> list[Line]:
    """Render a tool call: the head, then the result under it.

    The result glyph carries the pass or fail colour; the detail is its own neutral
    span.

    Args:
        item: The tool item.
        expanded: Show the whole detail and tail rather than a clipped first line.

    Returns:
        The lines; an in-flight call is its head alone, marked running.
    """
    head_style: StyleName = "verify" if item.name == "run_verify_command" else "call"
    head: Line = [(f"{CALL} {item.name}", head_style)]
    if item.arg:
        head.append((f"  {item.arg}", "arg"))
    if item.ok is None:
        head.append((f"  · {item.detail or 'running'}", "more"))
        return [head]
    glyph: StyleName = "ok" if item.ok else "fail"
    detail_lines = item.detail.split("\n")
    long = len(detail_lines) > 1 or len(detail_lines[0]) > DETAIL_CLIP
    if expanded and long:
        lines: list[Line] = [head, [(f"  {RESULT} ", glyph), (detail_lines[0], "detail")]]
        lines.extend([(f"      {ln}", "detail")] for ln in detail_lines[1:])
    else:
        reason = detail_lines[0]
        if len(reason) > DETAIL_CLIP:
            reason = reason[: DETAIL_CLIP - 1] + "…"
        result: Line = [(f"  {RESULT} ", glyph), (reason, "detail")]
        extra = len(detail_lines) - 1
        if extra:
            result.append((f"  (+{extra} more line{'' if extra == 1 else 's'})", "more"))
        lines = [head, result]
    if item.tail:
        if expanded:
            lines.extend([(f"    {ln}", "tail")] for ln in item.tail.split("\n"))
        else:
            flat = " ".join(item.tail.split())
            clip = flat[:TAIL_CLIP] + ("…" if len(flat) > TAIL_CLIP else "")
            lines.append([(f"    {clip}", "tail")])
    return lines


def _thinking_lines(item: TranscriptItem, *, expanded: bool) -> list[Line]:
    """Render a reasoning block.

    Args:
        item: The thinking item.
        expanded: One line per body line, the contract every consumer's line
            arithmetic relies on; collapsed gives the first non-empty line and a count.

    Returns:
        The lines.
    """
    if expanded:
        body_lines = item.body.split("\n")
        out: list[Line] = [[(f"{THINK} ", "think-marker"), (body_lines[0], "thinking")]]
        out.extend([(f"  {ln}", "thinking")] for ln in body_lines[1:])
        return out
    n = item.body.count("\n") + 1
    first = next((ln.strip() for ln in item.body.split("\n") if ln.strip()), "")
    if len(first) > DETAIL_CLIP:
        first = first[: DETAIL_CLIP - 1] + "…"
    line: Line = [(f"{THINK} ", "think-marker"), (first, "thinking")]
    if n > 1:
        line.append((f"  (+{n - 1} more line{'' if n == 2 else 's'})", "more"))
    return [line]


def item_lines(item: TranscriptItem, *, detail: DetailLevel) -> list[Line]:
    """Render one folded conversation item as styled lines.

    Args:
        item: The item.
        detail: The detail level the TUI cycles.

    Returns:
        The lines; none for a thinking or tool item at the hidden level.
    """
    if detail == "hidden" and item.kind in ("thinking", "tool"):
        return []
    lines: list[Line] = []
    if item.kind == "thinking":
        lines.extend(_thinking_lines(item, expanded=detail == "expanded"))
    elif item.kind == "text":
        lines.extend([(ln, "text")] for ln in item.body.split("\n"))
    elif item.kind == "tool":
        lines.extend(_tool_lines(item, expanded=detail == "expanded"))
    elif item.kind == "operator":
        body_lines = item.body.split("\n")
        lines.append([(f"{OPERATOR} ", "operator"), (body_lines[0], "operator")])
        lines.extend([(f"  {ln}", "operator")] for ln in body_lines[1:])
    elif item.kind == "commit":
        lines.append([(f"{COMMIT} commit  {item.detail}", "commit")])
    elif item.kind == "marker":
        body_lines = item.body.split("\n")
        lines.append([(f"── {body_lines[0]} ──", "marker")])
        lines.extend([(f"   {ln}", "marker")] for ln in body_lines[1:])
    elif item.kind == "done":
        # Neutral for an end no gate judged: a gateless finish, a stop, an undo.
        badge: Line = [
            (
                f"{DONE} {item.name}",
                "done-ok" if item.ok else "done-fail" if item.ok is False else "done-neutral",
            )
        ]
        body_lines = item.body.split("\n") if item.body else []
        if body_lines:
            badge.append((f"  {body_lines[0]}", "body"))
        lines.append([])
        lines.append(badge)
        lines.extend([(f"  {ln}", "body")] for ln in body_lines[1:])
        lines.append([(item.detail, "done-detail")])
    return lines
