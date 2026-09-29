# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The conversation view copies through the clipboard toolkit, and its chrome is non-selectable."""

from __future__ import annotations

import pathlib

from agent6.ui.tui import conversation


def test_copy_text_emits_the_osc52_sequence_via_the_seam(tmp_path: pathlib.Path) -> None:
    logs = tmp_path / "logs.jsonl"
    logs.write_text("", encoding="utf-8")
    screen = conversation.ConversationScreen(logs, title=lambda _ctx: "t")
    written: list[str] = []
    screen._emit = written.append  # type: ignore[method-assign]  # sub the raw-write seam
    status = screen._copy_text("hello", method="osc52")
    assert written == ["\x1b]52;c;aGVsbG8=\x07"]  # base64("hello") == "aGVsbG8="
    assert "osc" in status.lower()


def test_copy_prefers_the_current_selection_else_whole_transcript(tmp_path: pathlib.Path) -> None:
    logs = tmp_path / "logs.jsonl"
    logs.write_text("", encoding="utf-8")
    screen = conversation.ConversationScreen(logs, title=lambda _ctx: "t")
    screen._content.append("line one\nline two")
    # No body selection (unmounted, so no #conv-body) -> whole transcript.
    text, what = screen._selected_or_all()
    assert text == "line one\nline two" and what == "whole transcript"
    # A body selection -> that selection (footer/chrome never contribute).
    screen._body_selection = lambda: "one"  # type: ignore[method-assign]
    text, what = screen._selected_or_all()
    assert text == "one" and what == "selection"


def test_get_selected_text_gathers_body_only(tmp_path: pathlib.Path) -> None:
    # get_selected_text is Textual's Ctrl+C path; gathering the body only keeps footer keys out.
    logs = tmp_path / "logs.jsonl"
    logs.write_text("", encoding="utf-8")
    screen = conversation.ConversationScreen(logs, title=lambda _ctx: "t")
    screen._body_selection = lambda: "body text"  # type: ignore[method-assign]
    assert screen.get_selected_text() == "body text"
    screen._body_selection = lambda: None  # type: ignore[method-assign]
    assert screen.get_selected_text() is None


def test_chrome_static_is_not_selectable() -> None:
    assert conversation._ChromeStatic.ALLOW_SELECT is False
