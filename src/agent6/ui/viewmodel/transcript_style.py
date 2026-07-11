# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""One renderer for a folded ``TranscriptItem``, shared by the CLI stream and the
TUI conversation view.

``item_lines`` produces medium-agnostic styled lines: each line is a list of
``(text, style)`` spans, where *style* is a SEMANTIC name (not an ANSI code or a
Rich style). Each front-end maps those names to its own output -- the CLI to ANSI,
the TUI to a Rich ``Text`` -- so the structure and the styling decisions live in
ONE place and the two skins can't drift (they used to: the tool-output tail was
clipped at 100 chars in the CLI vs 160 in the TUI, its colour was ``dim`` vs
``dim red``, and the TUI painted a failed tool's whole detail red by appending it
to the red result marker).

Relative indent is baked into the line text (a tool's result/tail sit under its
call); each front-end adds its own left margin (the CLI a two-space gutter, the
TUI its CSS padding).
"""

from __future__ import annotations

from typing import Literal

from agent6.ui.viewmodel.transcript import (
    CALL,
    COMMIT,
    DONE,
    RESULT,
    THINK,
    TranscriptItem,
)

StyleName = Literal[
    "thinking",
    "text",
    "call",
    "arg",
    "ok",
    "fail",
    "detail",
    "tail",
    "commit",
    "marker",
    "done-ok",
    "done-fail",
    "body",
    "done-detail",
]
Span = tuple[str, StyleName]
Line = list[Span]

TAIL_CLIP = 120  # chars of a failed tool's captured output tail shown inline


def item_lines(item: TranscriptItem, *, show_thinking: bool) -> list[Line]:
    """The styled lines for one folded conversation item (both skins render these)."""
    lines: list[Line] = []
    if item.kind == "thinking":
        if show_thinking:
            lines.append([(f"{THINK} {item.body}", "thinking")])
    elif item.kind == "text":
        lines.extend([(ln, "text")] for ln in item.body.split("\n"))
    elif item.kind == "tool":
        head: Line = [(f"{CALL} {item.name}", "call")]
        if item.arg:
            head.append((f"  {item.arg}", "arg"))
        lines.append(head)
        # RESULT glyph carries the pass/fail colour; the detail is its OWN neutral
        # span (this is the #1 fix -- it no longer inherits the fail colour).
        lines.append([(f"  {RESULT} ", "ok" if item.ok else "fail"), (item.detail, "detail")])
        if item.tail:
            lines.append([(f"    {' '.join(item.tail.split())[:TAIL_CLIP]}", "tail")])
    elif item.kind == "commit":
        lines.append([(f"{COMMIT} commit  {item.detail}", "commit")])
    elif item.kind == "marker":
        lines.append([(f"── {item.body} ──", "marker")])
    elif item.kind == "done":
        badge: Line = (
            [(f"{DONE} done", "done-ok")]
            if item.ok
            else [(f"{DONE} {item.name or 'stopped'}", "done-fail")]
        )
        if item.body:
            badge.append((f"  {item.body}", "body"))
        lines.append([])  # a blank line sets the verdict apart
        lines.append(badge)
        lines.append([(item.detail, "done-detail")])
    return lines
