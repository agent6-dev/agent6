# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A machine agent state's output_schema as an in-execution finish contract.

The request carries the schema table, the execution's task states the contract, and the loop refuses
a non-conforming finish with the problems so the retry happens in-execution.
"""

from __future__ import annotations

import pathlib
from typing import Any
from unittest import mock

from agent6.app import machine_agent
from agent6.config import Config
from agent6.harness import _chain, _conversation, _finish_gates, _loop_state, loop
from agent6.machine import AgentRequest, spec
from tests.unit.turn_context import turn_context

_SCHEMAS = {
    "verdict": {
        "ok": spec.FieldSpec(type="bool"),
        "detail": spec.FieldSpec(type="finding", optional=True),
    },
    "finding": {"label": spec.FieldSpec(type="str", enum=("pass", "fail"))},
}


def _request(schema: str | None) -> AgentRequest:
    return AgentRequest(
        prompt="judge the tree",
        timeout_s=60.0,
        output_schema=schema,
        schemas=_SCHEMAS if schema else {},
    )


def test_the_task_states_the_contract_with_nested_records() -> None:
    task = machine_agent._task_with_contract(_request("verdict"))
    assert task.startswith("judge the tree")
    assert "matching schema 'verdict'" in task
    assert "verdict = {ok: bool; detail: finding (optional)}" in task
    assert "finding = {label: str one of [pass, fail]}" in task


def test_a_schemaless_request_leaves_the_task_alone() -> None:
    assert machine_agent._task_with_contract(_request(None)) == "judge the tree"
    assert machine_agent._finish_validator(_request(None)) is None


def _wf(validator: Any) -> loop.Harness:
    wf = loop.Harness(
        chain=_chain.RunChain(pathlib.Path("/tmp")),
        config=Config.model_validate({}),
        provider=mock.MagicMock(),
        dispatcher=mock.MagicMock(),
        logger=lambda _m: None,
        mode="run",
        finish_validator=validator,
    )
    return wf


def _finishing_turn(payload: dict[str, Any] | None) -> _loop_state.TurnState:
    turn = _loop_state.TurnState(iteration=3, resp=mock.MagicMock(), assistant=mock.MagicMock())
    turn.finish = _finish_gates.FinishCall("finish_session", "done", payload)
    return turn


def test_a_nonconforming_finish_is_refused_with_the_problems() -> None:
    validator = machine_agent._finish_validator(_request("verdict"))
    assert validator is not None
    wf = _wf(validator)
    state = _loop_state.LoopState(original_task="t", tool_calls=0)
    turn = _finishing_turn(None)
    ctx = wf._turn_context(state, iteration=3, execution_start=1)  # pyright: ignore[reportPrivateUsage]
    refusal = _finish_gates.finish_contract(turn, state, ctx)
    assert refusal is not None and refusal.event == "loop.finish_contract.refused"
    assert refusal.fields["iteration"] == 3 and refusal.fields["problems"]
    wf._turn_finish_gates(state, turn, ctx)  # pyright: ignore[reportPrivateUsage]
    assert turn.finish is None
    notices = [r.text for r in turn.tool_results if isinstance(r, _conversation.Notice)]
    assert any("finish_session refused" in n and "verdict" in n for n in notices)

    # The retry with a conforming payload stands.
    turn2 = _finishing_turn({"ok": True})
    assert _finish_gates.finish_contract(turn2, state, ctx) is None


def test_a_run_without_a_contract_is_never_gated() -> None:
    turn = _finishing_turn(None)
    assert (
        _finish_gates.finish_contract(
            turn, _loop_state.LoopState(original_task="t", tool_calls=0), turn_context()
        )
        is None
    )
