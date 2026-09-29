# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The machine `agent` state's interactivity bridges.

Answers live in the per-state dir; the liveness gate probes the instance dir where a front-end
registers its claim.
"""

from __future__ import annotations

import os
import pathlib
import threading
import time

import pytest

from agent6 import events as agent6_events
from agent6.app import machine_agent as app_machine_agent
from agent6.sessions import ipc
from agent6.tools import schema


def _dirs(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path, agent6_events.EventSink]:
    instance = tmp_path / "inst"
    state = instance / "states" / "0000-review"
    state.mkdir(parents=True)
    return instance, state, agent6_events.EventSink(state / "logs.jsonl")


def test_a_machine_command_grant_does_not_answer_another_scopes_prompt(
    tmp_path: pathlib.Path,
) -> None:
    """The machine approver honours the prompt's scope, as the run approver does."""
    instance, state = tmp_path / "inst", tmp_path / "inst" / "1-agent"
    state.mkdir(parents=True)
    ipc.set_session_allow(state, ipc.COMMAND_SCOPE)
    bridges = app_machine_agent._build_machine_bridges(
        instance, state, agent6_events.EventSink(state / "logs.jsonl")
    )

    assert bridges.prompts.approve("Allow run_command: ls", scope=ipc.COMMAND_SCOPE) is True
    # A headless deny, not the grant.
    assert bridges.prompts.approve("Allow fetch: evil.example /x") is False


def test_stale_answers_cleared_before_state_reexecution(tmp_path: pathlib.Path) -> None:
    # A re-executed `<seq>-<state>` dir drops the aborted attempt's stale answer files.
    instance, state, events = _dirs(tmp_path)
    ipc.register_frontend(instance, os.getpid())
    ipc.write_answer(state, "approval-1", "yes")  # stale: from the aborted attempt
    ipc.write_question_answers(state, "question-1", ["stale"])
    for stale in (
        state / "approvals" / "approval-1.answer",
        state / "questions" / "question-1.answer",
    ):
        os.utime(stale, (time.time() - 60, time.time() - 60))  # the attempt died a minute ago
    app_machine_agent._build_machine_bridges(instance, state, events)
    assert not (state / "approvals" / "approval-1.answer").exists()
    assert not (state / "questions" / "question-1.answer").exists()
    # The instance-dir front-end registration is untouched (it lives one level up).
    assert (instance / "frontends" / str(os.getpid())).exists()


def test_headless_defaults_when_no_frontend(tmp_path: pathlib.Path) -> None:
    instance, state, events = _dirs(tmp_path)
    b = app_machine_agent._build_machine_bridges(instance, state, events)
    # No front-end claim on the instance dir: deny approvals, empty answers, no steer.
    assert b.prompts.approve("run rm -rf?") is False
    assert b.prompts.ask((schema.UserQuestion(question="pick", options=("a", "b")),)).answers == (
        "",
    )
    assert b.steer_requested() is False
    assert b.steer_prompt() is None


def test_approval_answer_read_from_per_state_dir(tmp_path: pathlib.Path) -> None:
    instance, state, events = _dirs(tmp_path)
    ipc.register_frontend(instance, os.getpid())  # a live front-end owns the instance
    b = app_machine_agent._build_machine_bridges(
        instance, state, events
    )  # clears pre-existing answers
    # A writer thread answers after approve() emits the prompt, into the per-state dir.
    threading.Thread(
        target=lambda: (time.sleep(0.2), ipc.write_answer(state, "approval-1", "yes")),
        daemon=True,
    ).start()
    assert b.prompts.approve("allow?") is True


def test_question_answer_read_from_per_state_dir(tmp_path: pathlib.Path) -> None:
    instance, state, events = _dirs(tmp_path)
    ipc.register_frontend(instance, os.getpid())
    b = app_machine_agent._build_machine_bridges(instance, state, events)
    threading.Thread(
        target=lambda: (
            time.sleep(0.2),
            ipc.write_question_answers(state, "question-1", ["chosen"]),
        ),
        daemon=True,
    ).start()
    question = schema.UserQuestion(question="which?", options=("chosen", "other"))
    assert b.prompts.ask((question,)).answers == ("chosen",)


def test_machine_approval_ignores_a_premature_answer(tmp_path: pathlib.Path) -> None:
    # An answer pre-written before the prompt is cleared, not consumed; the headless deny applies.
    instance, state, events = _dirs(tmp_path)
    ipc.register_frontend(instance, os.getpid())
    b = app_machine_agent._build_machine_bridges(instance, state, events)
    ipc.write_answer(state, "approval-1", "yes")  # premature: no prompt yet
    # No writer thread, so the approver falls through to the deny; a short timeout keeps it quick.
    from agent6.app import machine_agent

    orig = machine_agent.read_answer

    def _fast_read(rd: pathlib.Path, pid: str, **kw: object) -> bool | None:
        return orig(rd, pid, timeout_s=0.3, poll_s=0.05, live_dir=kw.get("live_dir"))  # type: ignore[arg-type]

    machine_agent.read_answer = _fast_read  # type: ignore[assignment]
    try:
        assert b.prompts.approve("run rm -rf?") is False
    finally:
        machine_agent.read_answer = orig


def test_steer_request_and_answer_bridge(tmp_path: pathlib.Path) -> None:
    instance, state, events = _dirs(tmp_path)
    ipc.register_frontend(instance, os.getpid())
    b = app_machine_agent._build_machine_bridges(instance, state, events)
    # A front-end drops a steer.request in the per-state dir.

    ipc.request_steer(state)
    assert b.steer_requested() is True
    ipc.write_steer_answer(state, "focus on tests")
    assert b.steer_prompt() == "focus on tests"
    b.steer_clear()
    assert b.steer_requested() is False


def test_machine_agent_wires_the_summariser_seat(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The machine agent wires the reviewer-role summariser and shares one TranscriptSink.

    Compaction side-calls otherwise fell back to the worker provider and were stamped seat="worker".
    """
    from typing import Any

    from agent6.app import machine_agent
    from agent6.harness import _snapshot
    from agent6.machine import engine

    gdir = tmp_path / "g"
    (gdir / "agent6").mkdir(parents=True, exist_ok=True)
    (gdir / "agent6" / "config.toml").write_text(
        '[providers.anthropic]\napi_format = "anthropic"\n'
        '[models.worker]\nprovider = "anthropic"\nmodel = "claude-x"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", str(gdir))

    wf_kwargs: dict[str, Any] = {}
    sinks: dict[str, Any] = {}
    summariser = object()

    class _FakeWf:
        def __init__(self, **kw: Any) -> None:
            wf_kwargs.update(kw)

        def run(self, _prompt: str) -> _snapshot.SessionResult:
            return _snapshot.SessionResult(
                reason="finish_session", completed=True, summary="done", iterations=1, tool_calls=0
            )

    def _fake_role(*_a: Any, **k: Any) -> object:
        sinks["worker"] = k.get("transcript_sink")
        return object()

    def _fake_summariser(*_a: Any, **k: Any) -> object:
        sinks["summariser"] = k.get("transcript_sink")
        return summariser

    monkeypatch.setattr(machine_agent, "Harness", _FakeWf)
    monkeypatch.setattr(machine_agent, "build_role_provider", _fake_role)
    monkeypatch.setattr(machine_agent, "reviewer_seat_provider", _fake_summariser)

    def _fake_dispatcher(**_k: Any) -> object:
        return object()

    monkeypatch.setattr(machine_agent, "ToolDispatcher", _fake_dispatcher)

    req = machine_agent.MachineAgentRequest(
        cwd=tmp_path,
        root=tmp_path,
        overlay={},
        isolation="none",
        transcript_dir=tmp_path / "t",
        request=engine.AgentRequest(
            model="claude-x", prompt="go", timeout_s=5.0, provider="anthropic"
        ),
    )
    out = machine_agent.run_one(req)
    assert out.reason == "finish_session"
    assert wf_kwargs["compaction"].summariser is summariser
    assert sinks["worker"] is sinks["summariser"]  # one sink, one seq counter


def test_away_wait_parks_a_prompt_for_the_frontend(tmp_path: pathlib.Path) -> None:
    """A hub-spawned machine parks approvals and questions for the front-end, whenever it claims.

    The headless answer is never invented; an answer arriving after the prompt fired is honoured.
    """
    instance, state, events = _dirs(tmp_path)
    ipc.set_away_mode(instance, "wait")
    b = app_machine_agent._build_machine_bridges(instance, state, events)

    def _answer_late() -> None:
        time.sleep(0.4)
        ipc.register_frontend(instance, os.getpid())
        ipc.write_answer(state, "approval-1", "yes")

    t = threading.Thread(target=_answer_late)
    t.start()
    assert b.prompts.approve("run ls?", scope="command") is True
    t.join()

    def _answer_question_late() -> None:
        time.sleep(0.4)
        ipc.write_question_answers(state, "question-1", ("blue",))

    t2 = threading.Thread(target=_answer_question_late)
    t2.start()
    q = schema.UserQuestion(question="colour?", options=("blue", "red"))
    assert b.prompts.ask((q,)).answers == ("blue",)
    t2.join()


def test_away_wait_prompt_stops_with_the_run(tmp_path: pathlib.Path) -> None:
    """A parked prompt does not outlive the operator's Stop: the approval resolves to deny."""
    instance, state, events = _dirs(tmp_path)
    ipc.set_away_mode(instance, "wait")
    b = app_machine_agent._build_machine_bridges(instance, state, events)

    def _stop_late() -> None:
        time.sleep(0.4)
        ipc.request_steer(instance)
        ipc.write_steer_answer(instance, "abort")

    t = threading.Thread(target=_stop_late)
    t.start()
    assert b.prompts.approve("run ls?", scope="command") is False
    t.join()


def test_the_agent_seat_journals_under_a_driving_role(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The machine execution's seat carries a role a SessionKind knows, so its replies fold."""
    from unittest import mock

    from agent6 import budget, kinds
    from agent6.app import machine_agent
    from agent6.config import Config
    from agent6.machine import AgentRequest

    def stub_provider(*_args: object, **_kwargs: object) -> mock.MagicMock:
        return mock.MagicMock()

    monkeypatch.setattr(machine_agent, "build_role_provider", stub_provider)
    monkeypatch.setattr(machine_agent, "reviewer_seat_provider", stub_provider)
    req = machine_agent.MachineAgentRequest(
        cwd=tmp_path,
        root=tmp_path,
        overlay={},
        isolation="none",
        transcript_dir=tmp_path / "transcripts",
        request=AgentRequest(prompt="hi", timeout_s=30.0, mode="agent"),
    )
    provider, _summariser, _events = machine_agent._build_agent_providers(  # pyright: ignore[reportPrivateUsage]
        Config(),
        req,
        budget=budget.BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1),
        attach_console=lambda _sink: None,
    )
    assert not kinds.is_side_role(provider.role)
