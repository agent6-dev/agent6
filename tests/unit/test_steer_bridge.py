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

from agent6.cli._steer import file_bridge_steer, make_steer_state
from agent6.events import EventSink
from agent6.frontend.approval import (
    request_steer,
    steer_request_pending,
    write_steer_answer,
)


def test_prompt_consumes_bridged_answer(tmp_path: Path) -> None:
    steer = file_bridge_steer(tmp_path)
    assert steer.requested() is False
    write_steer_answer(tmp_path, "focus on the tests")
    request_steer(tmp_path)
    assert steer.requested() is True
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

    monkeypatch.setattr("agent6.cli._steer.read_steer_answer", no_answer)
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
    from agent6.frontend.approval import steer_answer_is_abort

    assert not steer_answer_is_abort(tmp_path)  # no answer file yet
    write_steer_answer(tmp_path, "focus on the parser")
    assert not steer_answer_is_abort(tmp_path)  # a steering instruction is not a stop
    write_steer_answer(tmp_path, "abort")
    assert steer_answer_is_abort(tmp_path)
    write_steer_answer(tmp_path, "  STOP  ")
    assert steer_answer_is_abort(tmp_path)
    assert (tmp_path / "steer.answer").exists()  # peek did not consume it
