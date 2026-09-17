# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The tool-error ladder: a streak of identical tool errors draws a nudge,
an escalation, then a stop; a denial streak is named a refusal; a metric
run and the other modes are left to their own machinery."""

from __future__ import annotations

from unittest.mock import MagicMock

from agent6.workflows._conversation import AssistantTurn
from agent6.workflows._guards import Nudge, Stop, tool_error_ladder
from agent6.workflows._loop_state import LoopState, TurnState
from agent6.workflows._nudges import (
    TOOL_DENIED_NUDGE,
    TOOL_ERROR_ESCALATE_AFTER,
    TOOL_ERROR_ESCALATION,
    TOOL_ERROR_NUDGE,
    TOOL_ERROR_NUDGE_AFTER,
    TOOL_ERROR_STOP_AFTER,
)
from tests.unit.turn_context import turn_context


def _turn(iteration: int = 3) -> TurnState:
    return TurnState(iteration=iteration, resp=MagicMock(), assistant=AssistantTurn((), ()))


def _climb(state: LoopState, failures: int, *, denial: bool = False) -> list[Nudge | Stop]:
    """The ladder's answers over *failures* identical errors in one turn."""
    answers: list[Nudge | Stop] = []
    for _ in range(failures):
        state.spiral.note_error("read_file:bad path", denial=denial, content="{}")
        answer = tool_error_ladder(_turn(), state, turn_context())
        if answer is not None:
            answers.append(answer)
    return answers


def test_the_ladder_nudges_escalates_then_stops() -> None:
    state = LoopState(original_task="t", tool_calls=0)
    nudge, escalation, stop = _climb(state, TOOL_ERROR_STOP_AFTER)
    assert isinstance(nudge, Nudge) and nudge.text == TOOL_ERROR_NUDGE
    assert nudge.fields == {"iteration": 3, "streak": TOOL_ERROR_NUDGE_AFTER, "level": 1}
    assert isinstance(escalation, Nudge) and escalation.text == TOOL_ERROR_ESCALATION
    assert escalation.fields["streak"] == TOOL_ERROR_ESCALATE_AFTER
    assert escalation.fields["level"] == 2
    assert isinstance(stop, Stop) and stop.end().reason == "tool_error_stuck"
    assert stop.soft == "" and stop.declared == ""
    assert f"failed {TOOL_ERROR_STOP_AFTER} times" in stop.end().summary
    assert stop.log == f"LOOP: tool_error stop at iter 3 (streak {TOOL_ERROR_STOP_AFTER})"


def test_a_denial_streak_is_named_a_refusal() -> None:
    state = LoopState(original_task="t", tool_calls=0)
    nudge = _climb(state, TOOL_ERROR_NUDGE_AFTER, denial=True)[0]
    assert isinstance(nudge, Nudge) and nudge.text == TOOL_DENIED_NUDGE


def test_a_metric_run_and_the_other_modes_leave_the_ladder_alone() -> None:
    for ctx in (turn_context(metric=True), turn_context(mode="plan"), turn_context(mode="ask")):
        state = LoopState(original_task="t", tool_calls=0)
        for _ in range(TOOL_ERROR_STOP_AFTER):
            state.spiral.note_error("read_file:bad path", denial=False, content="{}")
            assert tool_error_ladder(_turn(), state, ctx) is None
