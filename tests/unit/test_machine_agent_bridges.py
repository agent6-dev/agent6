# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the machine `agent` state interactivity bridges.

Answers live in the per-state dir; the liveness gate probes the instance dir
where a front-end registers `frontend.pid`.
"""

from __future__ import annotations

import os
from pathlib import Path

from agent6.events import EventSink
from agent6.tools.schema import UserQuestion
from agent6.ui.bridge.approval import (
    write_answer,
    write_frontend_pid,
    write_question_answers,
    write_steer_answer,
)
from agent6.ui.cli.machine_agent import (
    _build_machine_bridges,  # pyright: ignore[reportPrivateUsage]
)


def _dirs(tmp_path: Path) -> tuple[Path, Path, EventSink]:
    instance = tmp_path / "inst"
    state = instance / "states" / "0000-review"
    state.mkdir(parents=True)
    return instance, state, EventSink(state / "logs.jsonl")


def test_stale_answers_cleared_before_state_reexecution(tmp_path: Path) -> None:
    # Crash recovery re-executes the same `<seq>-<state>` dir with fresh prompt-id
    # counters; an answer file left by the aborted attempt must not satisfy this
    # execution's first prompt. Building the bridges drops the stale files.
    instance, state, events = _dirs(tmp_path)
    write_frontend_pid(instance, os.getpid())
    write_answer(state, "approval-1", approved=True)  # stale: from the aborted attempt
    write_question_answers(state, "question-1", ["stale"])
    _build_machine_bridges(instance, state, events)
    assert not (state / "approvals" / "approval-1.answer").exists()
    assert not (state / "questions" / "question-1.answer").exists()
    # The instance-dir front-end registration is untouched (it lives one level up).
    assert (instance / "frontend.pid").exists()


def test_headless_defaults_when_no_frontend(tmp_path: Path) -> None:
    instance, state, events = _dirs(tmp_path)
    b = _build_machine_bridges(instance, state, events)
    # No frontend.pid on the instance dir: deny approvals, empty answers, no steer.
    assert b.approve("run rm -rf?") is False
    assert b.ask((UserQuestion(question="pick", options=("a", "b")),)) == ("",)
    assert b.steer_requested() is False
    assert b.steer_prompt() is None


def test_approval_answer_read_from_per_state_dir(tmp_path: Path) -> None:
    instance, state, events = _dirs(tmp_path)
    write_frontend_pid(instance, os.getpid())  # a live front-end owns the instance
    b = _build_machine_bridges(instance, state, events)  # clears pre-existing answers
    write_answer(state, "approval-1", approved=True)  # answer in the PER-STATE dir
    assert b.approve("allow?") is True


def test_question_answer_read_from_per_state_dir(tmp_path: Path) -> None:
    instance, state, events = _dirs(tmp_path)
    write_frontend_pid(instance, os.getpid())
    b = _build_machine_bridges(instance, state, events)
    write_question_answers(state, "question-1", ["chosen"])
    assert b.ask((UserQuestion(question="which?", options=("chosen", "other")),)) == ("chosen",)


def test_steer_request_and_answer_bridge(tmp_path: Path) -> None:
    instance, state, events = _dirs(tmp_path)
    write_frontend_pid(instance, os.getpid())
    b = _build_machine_bridges(instance, state, events)
    # A front-end drops a steer.request in the per-state dir.
    from agent6.ui.bridge.approval import request_steer

    request_steer(state)
    assert b.steer_requested() is True
    write_steer_answer(state, "focus on tests")
    assert b.steer_prompt() == "focus on tests"
    b.steer_clear()
    assert b.steer_requested() is False
