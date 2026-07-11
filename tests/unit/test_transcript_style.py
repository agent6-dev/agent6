# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The shared transcript renderer (item_lines): structure + the #1 neutral-detail fix."""

from __future__ import annotations

from agent6.ui.viewmodel.transcript import TranscriptItem
from agent6.ui.viewmodel.transcript_style import TAIL_CLIP, item_lines


def test_failed_tool_detail_is_a_neutral_span_not_the_fail_colour() -> None:
    item = TranscriptItem("tool", name="apply_edit", arg="x.py", ok=False, detail="err " * 100)
    result = item_lines(item, show_thinking=True)[1]
    assert result[0][1] == "fail"  # the RESULT glyph carries the fail colour
    assert result[1][1] == "detail"  # the detail is its OWN neutral span (the #1 fix)
    assert "err" in result[1][0]


def test_tool_tail_clipped_to_one_length_and_neutral() -> None:
    item = TranscriptItem("tool", name="run_command", ok=False, detail="d", tail="x" * 500)
    tail = item_lines(item, show_thinking=True)[-1]
    assert tail[0][1] == "tail"
    assert len(tail[0][0]) <= TAIL_CLIP + 4  # tail + the 4-space relative indent


def test_thinking_hidden_when_off_else_one_span() -> None:
    hidden = item_lines(TranscriptItem("thinking", body="hmm"), show_thinking=False)
    assert hidden == []
    shown = item_lines(TranscriptItem("thinking", body="hmm"), show_thinking=True)
    assert shown[0][0][1] == "thinking"


def test_tool_head_is_one_call_span_plus_arg() -> None:
    item = TranscriptItem("tool", name="read_file", arg="a.py", ok=True, detail="ok")
    head = item_lines(item, show_thinking=True)[0]
    assert head[0][1] == "call" and "read_file" in head[0][0]
    assert head[1][1] == "arg" and "a.py" in head[1][0]
