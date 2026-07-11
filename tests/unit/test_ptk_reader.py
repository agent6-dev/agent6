# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""prompt_toolkit inline input widgets, driven deterministically with a pipe input
+ create_app_session (no real PTY), so the arrow-key logic is unit-tested."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from agent6.ui.cli import _ptk_reader


def _drive(monkeypatch: pytest.MonkeyPatch, keys: str, fn: Callable[[], Any]) -> Any:
    """Run an Application-based reader with scripted keystrokes fed through a pipe."""
    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input.defaults import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    monkeypatch.setattr(_ptk_reader, "on_tty", lambda: True)
    with create_pipe_input() as inp:
        inp.send_text(keys)
        with create_app_session(input=inp, output=DummyOutput()):
            return fn()


def test_radio_select_arrows_to_option(monkeypatch: pytest.MonkeyPatch) -> None:
    # down moves to the 2nd option, enter selects it.
    got = _drive(monkeypatch, "\x1b[B\r", lambda: _ptk_reader.radio_select("pick", ["a", "b", "c"]))
    assert got == "b"


def test_radio_select_ctrl_c_cancels(monkeypatch: pytest.MonkeyPatch) -> None:
    # ctrl-c -> None so the caller falls back to the plain prompt.
    assert _drive(monkeypatch, "\x03", lambda: _ptk_reader.radio_select("pick", ["a", "b"])) is None


def test_ask_navigate_answers_each_then_submits(monkeypatch: pytest.MonkeyPatch) -> None:
    # enter answers q0 (answer() -> "A0"), auto-advances to q1, enter answers it, then
    # auto-advances to the submit row; enter submits.
    result = _drive(
        monkeypatch,
        "\r\r\r",
        lambda: _ptk_reader.ask_navigate(["q0", "q1"], lambda i: f"A{i}"),
    )
    assert result == ["A0", "A1"]


def test_ask_navigate_back_nav_then_submit_leaves_unanswered_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # answer q0 (auto-advance to q1), go BACK up to q0, ctrl-c submits as-is -> q1 empty.
    result = _drive(
        monkeypatch,
        "\r\x1b[A\x03",
        lambda: _ptk_reader.ask_navigate(["q0", "q1"], lambda _i: "X"),
    )
    assert result == ["X", ""]


def test_radio_select_offers_free_text(monkeypatch: pytest.MonkeyPatch) -> None:
    # arrow past the choices to "Type your own answer...", enter, then type a custom
    # answer -- every choice question always allows free text.
    got = _drive(
        monkeypatch,
        "\x1b[B\x1b[B\rmy own answer\r",
        lambda: _ptk_reader.radio_select("pick", ["a", "b"]),
    )
    assert got == "my own answer"


def test_ptk_prompt_reads_a_line(monkeypatch: pytest.MonkeyPatch) -> None:
    # the line editor returns the typed line (used for free-text answers + steer/REPL).
    assert (
        _drive(monkeypatch, "hello world\r", lambda: _ptk_reader.ptk_prompt("> ")) == "hello world"
    )


def test_ptk_prompt_ctrl_c_aborts_only_when_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    # The steer prompt passes interrupt_aborts=True so Ctrl-C aborts the run rather
    # than being swallowed as a no-op continue; the REPL leaves it off (Ctrl-C -> None).
    with pytest.raises(KeyboardInterrupt):
        _drive(monkeypatch, "\x03", lambda: _ptk_reader.ptk_prompt("> ", interrupt_aborts=True))
    assert _drive(monkeypatch, "\x03", lambda: _ptk_reader.ptk_prompt("> ")) is None


def test_widgets_fall_back_without_a_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    # No controlling terminal: both return None so the caller uses the plain path.
    monkeypatch.setattr(_ptk_reader, "on_tty", lambda: False)
    assert _ptk_reader.radio_select("q", ["a"]) is None
    assert _ptk_reader.ask_navigate(["q"], lambda _i: "X") is None
    assert _ptk_reader.ptk_prompt("q> ") is None
