# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The framework-agnostic clipboard primitives for the TUI copy toolkit."""

from __future__ import annotations

import pytest

from agent6.ui.tui import clipboard as cb


def test_osc52_bare_wraps_nothing() -> None:
    assert cb.osc52_sequence("hi", wrap="") == "\x1b]52;c;aGk=\x07"  # base64("hi") == "aGk="


def test_osc52_tmux_doubles_esc_and_wraps_in_dcs() -> None:
    seq = cb.osc52_sequence("hi", wrap="tmux")
    assert seq.startswith("\x1bPtmux;") and seq.endswith("\x1b\\")
    assert "\x1b\x1b]52;c;aGk=\x07" in seq  # inner OSC 52 with every ESC doubled


def test_osc52_screen_wraps_in_dcs() -> None:
    seq = cb.osc52_sequence("hi", wrap="screen")
    assert seq.startswith("\x1bP") and seq.endswith("\x1b\\")


def test_resolve_auto_prefers_tmux_inside_tmux(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TMUX", "/tmp/tmux-x,1,0")
    monkeypatch.delenv("STY", raising=False)
    assert cb.resolve_method("auto") == "tmux-buffer"


def test_resolve_auto_bare_osc52_without_multiplexer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.delenv("STY", raising=False)
    assert cb.resolve_method("auto") == "osc52"


def test_resolve_explicit_passes_through() -> None:
    assert cb.resolve_method("osc52-tmux") == "osc52-tmux"


def test_emit_osc52_calls_write_with_the_sequence() -> None:
    written: list[str] = []
    status = cb.emit_clipboard("hi", "osc52", written.append)
    assert written and written[0] == "\x1b]52;c;aGk=\x07"
    assert "osc" in status.lower()


def test_emit_osc52_tmux_wraps() -> None:
    written: list[str] = []
    cb.emit_clipboard("hi", "osc52-tmux", written.append)
    assert written and written[0].startswith("\x1bPtmux;")


def test_write_transcript_file_roundtrips() -> None:
    path = cb.write_transcript_file("hello world")
    try:
        assert path.read_text(encoding="utf-8") == "hello world"
    finally:
        path.unlink()


def test_resolve_auto_uses_screen_wrap_inside_screen(monkeypatch: pytest.MonkeyPatch) -> None:
    """GNU screen cannot decode tmux's `tmux;`-prefixed doubled-ESC DCS; auto
    must route to the screen passthrough or every copy silently fails while
    the toast says copied."""
    monkeypatch.setenv("STY", "1234.pts-0.host")
    monkeypatch.delenv("TMUX", raising=False)
    assert cb.resolve_method("auto") == "osc52-screen"


def test_emit_osc52_screen_wraps_in_plain_dcs() -> None:
    written: list[str] = []
    status = cb.emit_clipboard("hi", "osc52-screen", written.append)
    assert written[0].startswith("\x1bP") and not written[0].startswith("\x1bPtmux;")
    assert written[0].endswith("\x1b\\")
    assert "screen" in status.lower() and "tmux" not in status.lower()
    assert "osc52-screen" in cb.COPY_METHODS
