# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A machine agent state's log opens and closes with its execution.

The state's harness writes the log's `session.start` and `session.stop`, which every surface's
verb gate (`machine_verb_refusals`) reads; `machine create` writes its draft log's own.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
from typing import Any, override

import pytest

from agent6.app import role_providers
from agent6.app.machine import agent
from agent6.machine import AgentRequest
from agent6.machine import journal as machine_journal
from agent6.providers import ProviderResponse
from agent6.sessions import event_log, ipc, journal, process
from agent6.viewmodel.machine import state as machine_state
from tests.unit import test_compaction_writers as writers

JUDGE = """
machine = "judge"
version = 1
initial = "look"

[budget]
max_transitions = 10

[schemas.verdict]
approved = "bool"

[vars.agent]
verdict = { type = "verdict", default = {} }

[states.look]
kind = "agent"
prompt = "Judge the repository."
output_schema = "verdict"
capture = { finish_json = "verdict" }
timeout_s = 60
on = { ok = "done", failed = "done", budget_exhausted = "done", timeout = "done" }

[states.done]
kind = "terminal"
status = "ok"
reason = "judged"
"""


class _Finishes:
    """A provider that finishes the execution, recording an instance's steer refusal first."""

    def __init__(self, instance: pathlib.Path | None) -> None:
        self.instance = instance
        self.steer_refusals: list[str] = []

    def call(self, **_kw: Any) -> ProviderResponse:
        if self.instance is not None:
            self.steer_refusals.append(
                machine_state.machine_verb_refusal(self.instance, self.instance.name, "steer")
            )
        tools = (
            {"type": "tool_use", "id": "t1", "name": "finish_session", "input": {"summary": "ok"}},
        )
        return ProviderResponse(
            text="",
            tool_uses=tools,
            stop_reason="tool_use",
            input_tokens=1,
            output_tokens=1,
            cache_read_tokens=0,
            cache_write_tokens=0,
            raw={"content": list(tools)},
        )


class _SteeredOnce(_Finishes):
    """A provider whose first turn lists the repository while a steer lands, then finishes.

    The steer is written as the web page writes it, with its request, and no front-end claims
    the instance.
    """

    def __init__(self, state: pathlib.Path) -> None:
        super().__init__(None)
        self.state = state
        self.seen: list[str] = []

    @override
    def call(self, **kw: Any) -> ProviderResponse:
        self.seen.append(json.dumps(kw["messages"]))
        if len(self.seen) > 1:
            return super().call(**kw)
        assert ipc.submit_steer(self.state, "Read the tests first.")
        tools: tuple[dict[str, Any], ...] = (
            {"type": "tool_use", "id": "t0", "name": "list_dir", "input": {}},
        )
        return ProviderResponse(
            text="",
            tool_uses=tools,
            stop_reason="tool_use",
            input_tokens=1,
            output_tokens=1,
            cache_read_tokens=0,
            cache_write_tokens=0,
            raw={"content": list(tools)},
        )


def _run(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    owner: pathlib.Path,
    log: pathlib.Path,
    request: AgentRequest,
    prov: _Finishes | None = None,
    *,
    compaction: dict[str, int] | None = None,
) -> _Finishes:
    """Run one execution through `run_one` with the real harness, logging to `log`.

    Args:
        tmp_path: The test's directory.
        monkeypatch: The patcher.
        owner: The machine instance, whose steer refusal each call records, or the draft.
        log: The execution's log.
        request: The execution's request.
        prov: The provider; by default one that finishes at once.
        compaction: Local restart settings for the state.

    Returns:
        The provider, holding the refusals it read.
    """
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setenv("PROBE_KEY", "sk-probe")
    if prov is None:
        prov = _Finishes(owner if request.state_name else None)

    def _factory(*_a: Any, **_k: Any) -> _Finishes:
        return prov

    monkeypatch.setattr(role_providers, "build_role_provider", _factory)
    repo = tmp_path / "repo"
    repo.mkdir()
    for args in (("init", "-q"), ("commit", "-q", "--allow-empty", "-m", "init")):
        git = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-C", str(repo)]
        subprocess.run([*git, *args], check=True)
    req = agent.MachineAgentRequest(
        cwd=repo,
        root=repo,
        overlay={
            "compaction": compaction or {},
            "sandbox": {"isolation": "none"},
            "models": {"default": {"model": "p/m"}},
            "providers": {
                "p": {
                    "backend": "openai",
                    "base_url": "http://127.0.0.1:1",
                    "api_key_env": "PROBE_KEY",
                }
            },
        },
        isolation="none",
        transcript_dir=owner / "agent_transcripts",
        events_log=log,
        request=request,
    )
    assert agent.run_one(req).reason == "finish_session"
    return prov


@pytest.mark.parametrize("authoring", [False, True])
def test_a_restart_states_the_rendered_prompt_as_the_task(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, authoring: bool
) -> None:
    """A state's restart kept no task; an authoring request's kept only the typed words."""
    restarts: list[str] = []

    class Compacts(_Finishes):
        turns = 0

        @override
        def call(self, **kw: Any) -> ProviderResponse:
            messages = kw["messages"]
            if isinstance(messages[-1]["content"], str):
                return writers.response("Ready.")
            first = messages[0]["content"][0]["text"]
            if first.startswith("[harness context restart]"):
                restarts.append(first)
                return super().call(**kw)
            self.turns += 1
            assert self.turns <= 3
            call: dict[str, Any] = {
                "type": "tool_use",
                "id": str(self.turns),
                "name": "list_dir",
                "input": {},
            }
            return writers.response(
                "", tool_uses=(call,), stop_reason="tool_use", raw={"content": [call]}
            )

    owner = tmp_path / "machines" / "judge"
    owner.mkdir(parents=True)
    log = owner / "logs.jsonl" if authoring else owner / "states" / "0000-look" / "logs.jsonl"
    request = AgentRequest(
        prompt="Rendered request with machine output.",
        operator_task="Author a judge." if authoring else None,
        state_name="" if authoring else "look",
        timeout_s=60,
    )
    _run(
        tmp_path,
        monkeypatch,
        owner,
        log,
        request,
        Compacts(None),
        compaction={
            "keep_recent_turns": 0,
            "keep_recent_chars": 500,
            "drop_at_chars": 1000,
            "summarize_at_chars": 2000,
        },
    )
    (notice,) = restarts
    assert "TASK (the goal, verbatim):\nRendered request with machine output." in notice
    assert "Operator message:" not in notice


def _types(log: pathlib.Path) -> list[str]:
    return [json.loads(line)["type"] for line in log.read_text(encoding="utf-8").splitlines()]


def test_a_running_agent_state_takes_a_steer_and_its_log_closes_with_it(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An agent state's log opens with session.start and closes with session.stop.

    The harness was built without the state's log, so the log never opened: every surface
    refused a steer to the running state and hid its live turn.
    """
    instance = tmp_path / "machines" / "judge"
    instance.mkdir(parents=True)
    (instance / "machine.asm.toml").write_text(JUDGE, encoding="utf-8")
    machine_journal.MachineJournal(instance).begin(machine="judge", version=1)
    process.write_worker_pid(instance, os.getpid())  # the supervisor
    log = instance / "states" / "0000-look" / "logs.jsonl"
    request = AgentRequest(prompt="Judge the repository.", timeout_s=60.0, state_name="look")

    prov = _run(tmp_path, monkeypatch, instance, log, request)

    assert prov.steer_refusals == [""]
    types = _types(log)
    assert [t for t in types if t.startswith("session.")] == ["session.start", "session.stop"]
    assert types.index("session.start") < types.index("model.call")
    assert types.index("session.stop") > types.index("model.result")
    steer = machine_state.machine_verb_refusal(instance, "judge", "steer")
    assert steer == "machine 'judge' has no open agent state to steer"


def test_a_steer_written_while_no_front_end_claims_the_machine_reaches_the_state(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A steer the web wrote reached no state once its page had closed.

    At the next boundary the state found no front-end claim, dropped the steer on disk and
    logged it as empty.
    """
    instance = tmp_path / "machines" / "judge"
    instance.mkdir(parents=True)
    (instance / "machine.asm.toml").write_text(JUDGE, encoding="utf-8")
    machine_journal.MachineJournal(instance).begin(machine="judge", version=1)
    process.write_worker_pid(instance, os.getpid())  # the supervisor
    log = instance / "states" / "0000-look" / "logs.jsonl"
    request = AgentRequest(prompt="Judge the repository.", timeout_s=60.0, state_name="look")
    prov = _SteeredOnce(log.parent)

    _run(tmp_path, monkeypatch, instance, log, request, prov)

    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert [r["text"] for r in rows if r["type"] == "steer.injected"] == ["Read the tests first."]
    assert "Read the tests first." in prov.seen[1]


def test_the_authoring_agent_leaves_the_draft_log_to_machine_create(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An authoring attempt adds no session record to the draft log `machine create` opened.

    `machine create` writes the draft's start, attempt markers and stop around every attempt;
    an attempt's own start would reopen the draft between them and rewrite its worker pid.
    """
    draft = tmp_path / "sessions" / "machines" / "brisk-heron-7QK2LM"
    draft.mkdir(parents=True)
    log = draft / "logs.jsonl"
    start = journal.SessionStart(execution_id="draft", user_task="author it", mode="machine")
    process.emit_session_start(event_log.EventSink(log), draft, start)
    request = AgentRequest(
        prompt="Author a machine.", operator_task="author it", timeout_s=60.0, mode="run"
    )

    _run(tmp_path, monkeypatch, draft, log, request)

    types = _types(log)
    assert [t for t in types if t.startswith("session.")] == ["session.start"]
    assert "model.call" in types


def test_an_agent_state_reads_its_dollar_and_token_meters(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A state's harness held no budget tracker, so its budget line named no dollar or token."""

    class Records(_Finishes):
        requests: tuple[str, ...] = ()

        @override
        def call(self, **kw: Any) -> ProviderResponse:
            self.requests += (json.dumps(kw["messages"]),)
            return super().call(**kw)

    owner = tmp_path / "machines" / "judge"
    owner.mkdir(parents=True)
    log = owner / "states" / "0000-look" / "logs.jsonl"
    request = AgentRequest(prompt="Judge.", state_name="look", timeout_s=60, remaining_usd=2.0)
    prov = _run(tmp_path, monkeypatch, owner, log, request, Records(None))
    assert isinstance(prov, Records)
    line = "[harness budget] this execution: metered=$0/$2.00; fallback tokens=0/2000000"
    assert line in prov.requests[0]
