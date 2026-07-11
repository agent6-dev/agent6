# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The fish-style pause-menu line reader: Tab previews and cycles commands,
typing steers, history recalls; input()'s EOF/interrupt contract holds."""

from __future__ import annotations

import os

import pytest

from agent6.ui.cli._menu_input import (
    _read_key,  # pyright: ignore[reportPrivateUsage]
    menu_input,
)
from agent6.ui.cli._steer_menu import COMMANDS


def _chars(text: str) -> list[str]:
    return [f"char:{c}" for c in text]


def _run(keys: list[str], history: list[str] | None = None) -> tuple[str, str]:
    """Drive menu_input with scripted keys; returns (line, everything written)."""
    out: list[str] = []
    it = iter(keys)
    line = menu_input(
        "P> ",
        COMMANDS,
        history if history is not None else [],
        read_key=lambda: next(it),
        write=out.append,
    )
    return line, "".join(out)


def test_typed_line_returns_verbatim_and_lands_in_history() -> None:
    history: list[str] = []
    line, _ = _run([*_chars("focus on tests"), "enter"], history)
    assert line == "focus on tests"
    assert history == ["focus on tests"]
    # Accepting the same line again does not duplicate the history entry.
    line, _ = _run([*_chars("focus on tests"), "enter"], history)
    assert history == ["focus on tests"]


def test_tab_on_empty_line_previews_all_commands_and_cycles() -> None:
    line, out = _run(["tab", "enter"])
    assert line == "/status"  # first candidate selected
    # The menu rendered every command with its description.
    assert "/detach" in out and COMMANDS["/detach"] in out
    assert "\x1b[7m" in out  # the selection is highlighted
    line, _ = _run(["tab", "tab", "enter"])
    assert line == "/tasks"  # second candidate
    line, _ = _run(["tab", "backtab", "enter"])
    assert line == "/help"  # backwards wraps to the last


def test_tab_prefix_filters_and_arrows_move_selection() -> None:
    line, out = _run([*_chars("/st"), "tab", "enter"])
    assert line == "/status"
    assert "/stop" in out  # both matches were previewed
    assert "/tasks" not in out  # non-matches stay out of the menu
    line, _ = _run([*_chars("/st"), "tab", "down", "enter"])
    assert line == "/stop"
    line, _ = _run([*_chars("/st"), "tab", "down", "up", "enter"])
    assert line == "/status"


def test_unique_prefix_completes_without_a_menu() -> None:
    line, out = _run([*_chars("/sta"), "tab", "enter"])
    assert line == "/status"
    assert "\x1b[7m" not in out  # no menu, no highlight


def test_esc_restores_the_typed_stem_and_typing_keeps_the_candidate() -> None:
    line, _ = _run([*_chars("/st"), "tab", "esc", "enter"])
    assert line == "/st"
    # Typing after cycling keeps the selected candidate and edits from there.
    line, _ = _run(["tab", *_chars("x"), "enter"])
    assert line == "/statusx"


def test_tab_is_inert_inside_steer_text() -> None:
    line, out = _run([*_chars("fix it"), "tab", "enter"])
    assert line == "fix it"
    assert "\a" in out  # rang the bell instead of opening a menu
    line, out = _run([*_chars("q"), "tab", "enter"])
    assert line == "q"  # a non-slash word never completes
    assert "\a" in out


def test_editing_keys() -> None:
    line, _ = _run([*_chars("ab"), "left", *_chars("X"), "enter"])
    assert line == "aXb"
    line, _ = _run([*_chars("ab"), "backspace", "enter"])
    assert line == "a"
    line, _ = _run([*_chars("ab"), "kill-line", *_chars("c"), "enter"])
    assert line == "c"
    line, _ = _run([*_chars("keep this word"), "kill-word", "enter"])
    assert line == "keep this "
    line, _ = _run([*_chars("ab"), "home", "delete", "enter"])
    assert line == "b"


def test_history_recall_with_draft() -> None:
    history = ["first", "second"]
    line, _ = _run(["up", "enter"], history)
    assert line == "second"
    line, _ = _run(["up", "up", "enter"], history)
    assert line == "first"
    # Down past the newest entry restores the unsubmitted draft.
    line, _ = _run([*_chars("dra"), "up", "down", *_chars("ft"), "enter"], history)
    assert line == "draft"


def test_eof_contract() -> None:
    with pytest.raises(EOFError):
        _run(["eof"])
    # Ctrl-D on a non-empty line is a bell, not EOF.
    line, out = _run([*_chars("x"), "eof", "enter"])
    assert line == "x" and "\a" in out


def test_interrupt_raises_keyboard_interrupt() -> None:
    with pytest.raises(KeyboardInterrupt):
        _run([*_chars("half typed"), "interrupt"])


def test_menu_rows_clamp_to_narrow_terminals(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rows stay one terminal row wide (wrapping breaks the cursor-up math):
    descriptions truncate, the command labels survive."""
    monkeypatch.setattr("agent6.ui.cli._menu_input._width", lambda: 20)
    _, out = _run(["tab", "enter"])
    assert "/detach" in out  # labels intact
    for row in out.split("\r\n")[1:]:
        visible = row.split("\x1b[2m")[-1].split("\x1b[22m")[0]
        assert len(visible) < 20  # descriptions clamped


def test_read_key_decodes_bytes_from_a_pipe() -> None:
    """The raw decoder: control keys, CSI sequences, bare Esc, UTF-8 text."""
    r, w = os.pipe()
    try:
        cases = [
            (b"\t", "tab"),
            (b"\r", "enter"),
            (b"\x7f", "backspace"),
            (b"\x03", "interrupt"),
            (b"\x1b[A", "up"),
            (b"\x1b[Z", "backtab"),
            (b"\x1b[3~", "delete"),
            (b"q", "char:q"),
            ("é".encode(), "char:é"),
        ]
        for raw, expected in cases:
            os.write(w, raw)
            assert _read_key(r) == expected, raw
        os.write(w, b"\x1b")  # bare Esc: resolved by the 30ms poll timing out
        assert _read_key(r) == "esc"
    finally:
        os.close(r)
        os.close(w)
