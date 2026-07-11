# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""prompt_toolkit inline input widgets, driven deterministically with a pipe input
+ create_app_session (no real PTY), so the arrow-key logic is unit-tested."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from agent6.ui.cli import _ptk_reader
from agent6.ui.cli._ptk_reader import RadioNav


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


def test_radio_select_initial_preselects(monkeypatch: pytest.MonkeyPatch) -> None:
    # revisiting an answered question: the cursor starts on the previous answer.
    got = _drive(monkeypatch, "\r", lambda: _ptk_reader.radio_select("pick", ["a", "b"], initial=1))
    assert got == "b"


def test_radio_select_ctrl_c_cancels(monkeypatch: pytest.MonkeyPatch) -> None:
    # ctrl-c -> CANCEL so the caller treats the question as unanswered.
    got = _drive(monkeypatch, "\x03", lambda: _ptk_reader.radio_select("pick", ["a", "b"]))
    assert got is RadioNav.CANCEL


def test_radio_select_series_back_and_skip(monkeypatch: pytest.MonkeyPatch) -> None:
    # in a series (allow_back): left goes back a question, esc skips this one.
    back = _drive(
        monkeypatch, "\x1b[D", lambda: _ptk_reader.radio_select("pick", ["a"], allow_back=True)
    )
    assert back is RadioNav.BACK
    skip = _drive(
        monkeypatch, "\x1b", lambda: _ptk_reader.radio_select("pick", ["a"], allow_back=True)
    )
    assert skip is RadioNav.SKIP


def test_radio_select_esc_cancels_outside_a_series(monkeypatch: pytest.MonkeyPatch) -> None:
    got = _drive(monkeypatch, "\x1b", lambda: _ptk_reader.radio_select("pick", ["a", "b"]))
    assert got is RadioNav.CANCEL


def test_radio_select_offers_free_text(monkeypatch: pytest.MonkeyPatch) -> None:
    # arrow past the choices to "Type your own answer...", enter, then type a custom
    # answer -- every choice question always allows free text.
    got = _drive(
        monkeypatch,
        "\x1b[B\x1b[B\rmy own answer\r",
        lambda: _ptk_reader.radio_select("pick", ["a", "b"]),
    )
    assert got == "my own answer"


def test_radio_select_free_text_backout_returns_to_radio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Ctrl-C in the free-text editor backs out to the radio (cursor kept on the
    # free-text entry) instead of cancelling; up + enter then picks "b".
    got = _drive(
        monkeypatch,
        "\x1b[B\x1b[B\r\x03\x1b[A\r",
        lambda: _ptk_reader.radio_select("pick", ["a", "b"]),
    )
    assert got == "b"


def test_radio_select_no_free_text_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    # free_text=False (the review screen): the last entry is a real option.
    got = _drive(
        monkeypatch,
        "\x1b[B\r",
        lambda: _ptk_reader.radio_select("go?", ["Submit", "Revise"], free_text=False),
    )
    assert got == "Revise"


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
    # No controlling terminal: both report it so the caller uses the plain path.
    monkeypatch.setattr(_ptk_reader, "on_tty", lambda: False)
    assert _ptk_reader.radio_select("q", ["a"]) is RadioNav.CANCEL
    assert _ptk_reader.ptk_prompt("q> ") is None
