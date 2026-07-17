# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the machine `agent` state interactivity bridges.

Answers live in the per-state dir; the liveness gate probes the instance dir
where a front-end registers `frontend.pid`.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

from agent6.app.machine_agent import (
    _build_machine_bridges,  # pyright: ignore[reportPrivateUsage]
)
from agent6.events import EventSink
from agent6.runs.ipc import (
    write_answer,
    write_frontend_pid,
    write_question_answers,
    write_steer_answer,
)
from agent6.tools.schema import UserQuestion


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
    # A real front-end writes the answer AFTER approve() emits the prompt (approve
    # clears any premature pre-write first). A writer thread does exactly that;
    # the answer lands in the PER-STATE dir and read_answer picks it up promptly.
    threading.Thread(
        target=lambda: (time.sleep(0.2), write_answer(state, "approval-1", approved=True)),
        daemon=True,
    ).start()
    assert b.approve("allow?") is True


def test_question_answer_read_from_per_state_dir(tmp_path: Path) -> None:
    instance, state, events = _dirs(tmp_path)
    write_frontend_pid(instance, os.getpid())
    b = _build_machine_bridges(instance, state, events)
    threading.Thread(
        target=lambda: (time.sleep(0.2), write_question_answers(state, "question-1", ["chosen"])),
        daemon=True,
    ).start()
    assert b.ask((UserQuestion(question="which?", options=("chosen", "other")),)) == ("chosen",)


def test_machine_approval_ignores_a_premature_answer(tmp_path: Path) -> None:
    # The security property on the machine surface: an answer pre-written before
    # the prompt is emitted (a premature /api/machine/<name>/approve) is cleared
    # and not consumed -- the headless default (deny) applies instead.
    instance, state, events = _dirs(tmp_path)
    write_frontend_pid(instance, os.getpid())
    b = _build_machine_bridges(instance, state, events)
    write_answer(state, "approval-1", approved=True)  # premature: no prompt yet
    # No writer thread: nothing arrives after the prompt, so with the premature
    # answer cleared the approver falls through to the headless deny. Shrink the
    # read timeout so the poll gives up quickly instead of blocking 600s.
    from agent6.app import machine_agent

    orig = machine_agent.read_answer

    def _fast_read(rd: Path, pid: str, **kw: object) -> bool | None:
        return orig(rd, pid, timeout_s=0.3, poll_s=0.05, live_dir=kw.get("live_dir"))  # type: ignore[arg-type]

    machine_agent.read_answer = _fast_read  # type: ignore[assignment]
    try:
        assert b.approve("run rm -rf?") is False
    finally:
        machine_agent.read_answer = orig


def test_steer_request_and_answer_bridge(tmp_path: Path) -> None:
    instance, state, events = _dirs(tmp_path)
    write_frontend_pid(instance, os.getpid())
    b = _build_machine_bridges(instance, state, events)
    # A front-end drops a steer.request in the per-state dir.
    from agent6.runs.ipc import request_steer

    request_steer(state)
    assert b.steer_requested() is True
    write_steer_answer(state, "focus on tests")
    assert b.steer_prompt() == "focus on tests"
    b.steer_clear()
    assert b.steer_requested() is False
