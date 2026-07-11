# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The conversation view copies through the clipboard toolkit, and its chrome
(title + live pane) is non-selectable so a drag only grabs transcript text."""

from __future__ import annotations

from pathlib import Path

from agent6.ui.tui.conversation import ConversationScreen, _ChromeStatic


def test_copy_text_emits_the_osc52_sequence_via_the_seam(tmp_path: Path) -> None:
    logs = tmp_path / "logs.jsonl"
    logs.write_text("", encoding="utf-8")
    screen = ConversationScreen(logs, title="t")
    written: list[str] = []
    screen._emit = written.append  # substitute the raw-write seam (no running app)
    status = screen._copy_text("hello", method="osc52")
    assert written == ["\x1b]52;c;aGVsbG8=\x07"]  # base64("hello") == "aGVsbG8="
    assert "osc" in status.lower()


def test_copy_prefers_the_current_selection_else_whole_transcript(tmp_path: Path) -> None:
    logs = tmp_path / "logs.jsonl"
    logs.write_text("", encoding="utf-8")
    screen = ConversationScreen(logs, title="t")
    screen._content.append("line one\nline two")
    # No body selection (unmounted, so no #conv-body) -> whole transcript.
    text, what = screen._selected_or_all()
    assert text == "line one\nline two" and what == "whole transcript"
    # A body selection -> that selection (footer/chrome never contribute).
    screen._body_selection = lambda: "one"  # type: ignore[method-assign]
    text, what = screen._selected_or_all()
    assert text == "one" and what == "selection"


def test_chrome_static_is_not_selectable() -> None:
    assert _ChromeStatic.ALLOW_SELECT is False
