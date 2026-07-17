# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Steering a run that has no controlling terminal (detached spawn).

Regression: make_steer_state used to return a null steer without /dev/tty, so
a run spawned from the TUI hub or the web UI never polled steer.request and
every front-end steer was silently dropped.
"""

from __future__ import annotations

import builtins
from pathlib import Path

import pytest

from agent6.events import EventSink
from agent6.runs.bridge import (
    request_steer,
    steer_request_pending,
    write_steer_answer,
)
from agent6.ui.cli._steer import file_bridge_steer, install_steer_sigint, make_steer_state


def test_prompt_consumes_bridged_answer(tmp_path: Path) -> None:
    steer = file_bridge_steer(tmp_path)
    assert steer.requested() is False
    write_steer_answer(tmp_path, "focus on the tests")
    request_steer(tmp_path)
    assert steer.requested() is True
    assert steer.interrupt() is True  # a typed front-end steer injects now
    assert steer.prompt() == "focus on the tests"
    steer.clear()
    assert steer.requested() is False


def test_prompt_without_answer_clears_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A dead/abandoned front-end yields None; the request marker must go with
    # it or the next loop boundary re-triggers another blocking read forever.
    def no_answer(run_dir: Path) -> str | None:
        return None

    monkeypatch.setattr("agent6.ui.cli._steer.read_steer_answer", no_answer)
    request_steer(tmp_path)
    steer = file_bridge_steer(tmp_path)
    assert steer.prompt() is None
    assert steer_request_pending(tmp_path) is False


def test_make_steer_state_without_tty_uses_bridge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_open = builtins.open

    def fake_open(file: object, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        if file == "/dev/tty":
            raise OSError("no controlling terminal")
        return real_open(file, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("builtins.open", fake_open)
    events = EventSink(tmp_path / "logs.jsonl")
    steer = make_steer_state(events, tmp_path)
    # The old null steer answered False here even with a request pending.
    request_steer(tmp_path)
    assert steer.requested() is True


def test_steer_answer_is_abort_peeks_without_consuming(tmp_path: Path) -> None:
    """The non-blocking stop peek: True only for abort/stop, and it never consumes
    the answer (the between-step boundary still handles it)."""
    from agent6.runs.bridge import steer_answer_is_abort

    assert not steer_answer_is_abort(tmp_path)  # no answer file yet
    write_steer_answer(tmp_path, "focus on the parser")
    assert not steer_answer_is_abort(tmp_path)  # a steering instruction is not a stop
    write_steer_answer(tmp_path, "stop")
    # Even "stop" is a steer instruction, not a stop -- the Stop button writes
    # "abort", and the between-step boundary stops only on "abort". Consistency.
    assert not steer_answer_is_abort(tmp_path)
    write_steer_answer(tmp_path, "  ABORT  ")
    assert steer_answer_is_abort(tmp_path)  # exactly the Stop contract, case/space-insensitive
    assert (tmp_path / "steer.answer").exists()  # peek did not consume it
    # A non-UTF-8 answer must read as "no abort", never raise -- a raising peek
    # would kill the streaming watchdog thread (and its idle-hang detection).
    (tmp_path / "steer.answer").write_bytes(b"\xff\xfe not utf8")
    assert not steer_answer_is_abort(tmp_path)


def _silent_banner(text: str) -> None:
    """tty_message stand-in: keep test output off the developer's terminal."""


def test_sigint_escalates_boundary_interrupt_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Ctrl-C stages: 1st pauses at the next between-step boundary (the
    in-flight call finishes), 2nd interrupts the in-flight call, 3rd stops."""
    import signal

    monkeypatch.setattr("agent6.ui.cli._steer.tty_message", _silent_banner)
    events = EventSink(tmp_path / "logs.jsonl")
    steer = install_steer_sigint(events, tmp_path)
    try:
        assert steer.requested() is False
        signal.raise_signal(signal.SIGINT)  # 1st: graceful pause
        assert steer.requested() is True
        assert steer.interrupt() is False
        signal.raise_signal(signal.SIGINT)  # 2nd: abort the in-flight call
        assert steer.interrupt() is True
        with pytest.raises(KeyboardInterrupt):  # 3rd: stop the run
            signal.raise_signal(signal.SIGINT)
        steer.clear()
        assert steer.requested() is False
        assert steer.interrupt() is False
    finally:
        steer.restore()


def test_sigint_at_the_pause_prompt_stops(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """At the pause prompt itself a Ctrl-C stops the run outright, whatever the
    stage: the banner promised it, and there is nothing in flight to interrupt."""
    import signal

    monkeypatch.setattr("agent6.ui.cli._steer.tty_message", _silent_banner)
    monkeypatch.setattr("agent6.ui.cli._steer.menu_capable", lambda: False)

    def prompt_hit_by_ctrl_c(text: str, **_kw: object) -> str | None:
        signal.raise_signal(signal.SIGINT)
        return ""

    monkeypatch.setattr("agent6.ui.cli._steer.tty_prompt", prompt_hit_by_ctrl_c)
    events = EventSink(tmp_path / "logs.jsonl")
    steer = install_steer_sigint(events, tmp_path)
    try:
        signal.raise_signal(signal.SIGINT)  # stage 1: the boundary pause
        with pytest.raises(KeyboardInterrupt):
            steer.prompt()
    finally:
        steer.restore()


def test_prompt_pauses_the_console_spinner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The pause menu runs inside ConsoleView.pause(): the heartbeat spinner's
    per-tick line-erase otherwise wipes the pause-menu line and its Tab preview."""
    import contextlib
    from collections.abc import Generator
    from typing import cast

    from agent6.ui.cli._console_view import ConsoleView

    calls: list[str] = []

    class FakeView:
        @contextlib.contextmanager
        def pause(self) -> Generator[None]:
            calls.append("pause")
            yield
            calls.append("resume")

    monkeypatch.setattr("agent6.ui.cli._steer.menu_capable", lambda: True)

    def fake_menu(run_dir: Path) -> str | None:
        calls.append("prompt")
        return "steer text"

    monkeypatch.setattr("agent6.ui.cli._steer.pause_menu", fake_menu)
    events = EventSink(tmp_path / "logs.jsonl")
    steer = install_steer_sigint(events, tmp_path, cast(ConsoleView, FakeView()))
    try:
        assert steer.prompt() == "steer text"
    finally:
        steer.restore()
    assert calls == ["pause", "prompt", "resume"]
