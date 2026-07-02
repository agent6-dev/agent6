# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the machine `agent` state interactivity bridges.

Answers live in the per-state dir; the liveness gate probes the instance dir
where a front-end registers `frontend.pid`.
"""

from __future__ import annotations

import os
from pathlib import Path

from agent6.cli.machine_agent import _build_machine_bridges  # pyright: ignore[reportPrivateUsage]
from agent6.events import EventSink
from agent6.frontend.approval import (
    write_answer,
    write_frontend_pid,
    write_question_answer,
    write_steer_answer,
)


def _dirs(tmp_path: Path) -> tuple[Path, Path, EventSink]:
    instance = tmp_path / "inst"
    state = instance / "states" / "0000-review"
    state.mkdir(parents=True)
    return instance, state, EventSink(state / "logs.jsonl")


def test_headless_defaults_when_no_frontend(tmp_path: Path) -> None:
    instance, state, events = _dirs(tmp_path)
    b = _build_machine_bridges(instance, state, events)
    # No frontend.pid on the instance dir: deny approvals, empty answers, no steer.
    assert b.approve("run rm -rf?") is False
    assert b.ask("pick", ("a", "b")) == ""
    assert b.steer_requested() is False
    assert b.steer_prompt() is None


def test_approval_answer_read_from_per_state_dir(tmp_path: Path) -> None:
    instance, state, events = _dirs(tmp_path)
    write_frontend_pid(instance, os.getpid())  # a live front-end owns the instance
    write_answer(state, "approval-1", approved=True)  # answer in the PER-STATE dir
    b = _build_machine_bridges(instance, state, events)
    assert b.approve("allow?") is True


def test_question_answer_read_from_per_state_dir(tmp_path: Path) -> None:
    instance, state, events = _dirs(tmp_path)
    write_frontend_pid(instance, os.getpid())
    write_question_answer(state, "question-1", "chosen")
    b = _build_machine_bridges(instance, state, events)
    assert b.ask("which?", ("chosen", "other")) == "chosen"


def test_steer_request_and_answer_bridge(tmp_path: Path) -> None:
    instance, state, events = _dirs(tmp_path)
    write_frontend_pid(instance, os.getpid())
    b = _build_machine_bridges(instance, state, events)
    # A front-end drops a steer.request in the per-state dir.
    from agent6.frontend.approval import request_steer

    request_steer(state)
    assert b.steer_requested() is True
    write_steer_answer(state, "focus on tests")
    assert b.steer_prompt() == "focus on tests"
    b.steer_clear()
    assert b.steer_requested() is False
