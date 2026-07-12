# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The shared transcript renderer (item_lines): structure, the #1 neutral-detail fix,
and the collapsed/expanded/hidden detail levels."""

from __future__ import annotations

from agent6.ui.viewmodel.transcript import TranscriptItem
from agent6.ui.viewmodel.transcript_style import TAIL_CLIP, item_lines


def test_failed_tool_detail_is_a_neutral_span_not_the_fail_colour() -> None:
    item = TranscriptItem("tool", name="apply_edit", arg="x.py", ok=False, detail="err " * 100)
    result = item_lines(item, detail="collapsed")[1]
    assert result[0][1] == "fail"  # the RESULT glyph carries the fail colour
    assert result[1][1] == "detail"  # the detail is its OWN neutral span (the #1 fix)
    assert "err" in result[1][0]


def test_long_multiline_detail_clips_to_reason_plus_more_note_when_collapsed() -> None:
    detail = "old_string not found\n" + "\n".join(f"line {i}" for i in range(50))
    item = TranscriptItem("tool", name="apply_edit", ok=False, detail=detail)
    result = item_lines(item, detail="collapsed")[1]
    assert result[1] == ("old_string not found", "detail")  # first line only, neutral
    assert result[2][1] == "more" and "50 more lines" in result[2][0]


def test_expanded_shows_the_full_tool_detail() -> None:
    detail = "old_string not found\n" + "\n".join(f"line {i}" for i in range(50))
    item = TranscriptItem("tool", name="apply_edit", ok=False, detail=detail)
    lines = item_lines(item, detail="expanded")
    assert lines[1] == [("  └ ", "fail"), ("old_string not found", "detail")]
    assert all(span[1] == "detail" for line in lines[2:52] for span in line)  # every line neutral
    assert "line 49" in lines[51][0][0]  # the last detail line is present, no "+N more"


def test_short_single_line_detail_is_not_clipped_and_has_no_more_note() -> None:
    item = TranscriptItem("tool", name="read_file", ok=True, detail="2440 bytes")
    result = item_lines(item, detail="collapsed")[1]
    assert result[1] == ("2440 bytes", "detail")
    assert len(result) == 2  # no "more" span


def test_tool_tail_clipped_to_one_length_and_neutral() -> None:
    item = TranscriptItem("tool", name="run_command", ok=False, detail="d", tail="x" * 500)
    tail = item_lines(item, detail="collapsed")[-1]
    assert tail[0][1] == "tail"
    assert len(tail[0][0]) <= TAIL_CLIP + 4  # tail + the 4-space relative indent


def test_thinking_detail_levels() -> None:
    item = TranscriptItem("thinking", body="plan the fix\nb\nc")
    assert item_lines(item, detail="hidden") == []  # omitted entirely
    collapsed = item_lines(item, detail="collapsed")
    # Collapsed = the FIRST LINE of the reasoning as a summary + a more-count,
    # so it still says what the model is thinking about.
    assert collapsed[0][0][1] == "thinking" and "plan the fix" in collapsed[0][0][0]
    assert "b" not in collapsed[0][0][0].split("plan the fix")[-1]  # only the first line
    assert collapsed[0][1][1] == "more" and "+2 more lines" in collapsed[0][1][0]
    # A single-line thought has no more-count; a long first line is clipped.
    single = item_lines(TranscriptItem("thinking", body="only line"), detail="collapsed")
    assert len(single[0]) == 1 and "only line" in single[0][0][0]
    long = item_lines(TranscriptItem("thinking", body="x" * 400), detail="collapsed")
    assert long[0][0][0].endswith("…") and len(long[0][0][0]) < 200
    expanded = item_lines(item, detail="expanded")
    assert expanded[0][0][1] == "thinking" and expanded[0][0][0].endswith("plan the fix\nb\nc")


def test_tool_head_is_one_call_span_plus_arg() -> None:
    item = TranscriptItem("tool", name="read_file", arg="a.py", ok=True, detail="ok")
    head = item_lines(item, detail="collapsed")[0]
    assert head[0][1] == "call" and "read_file" in head[0][0]
    assert head[1][1] == "arg" and "a.py" in head[1][0]
