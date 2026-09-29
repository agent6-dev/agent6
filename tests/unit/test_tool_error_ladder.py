# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The tool-error ladder: a streak of identical tool errors draws a nudge, an escalation, a stop.

A denial streak is named a refusal; a metric run and the other modes keep their own machinery.
"""

from __future__ import annotations

from unittest import mock

from agent6.harness import _advice, _conversation, _guards, _loop_state, _nudges
from tests.unit.turn_context import turn_context


def _turn(iteration: int = 3) -> _loop_state.TurnState:
    return _loop_state.TurnState(
        iteration=iteration, resp=mock.MagicMock(), assistant=_conversation.AssistantTurn((), ())
    )


def _climb(
    state: _loop_state.LoopState, failures: int, *, denial: bool = False
) -> list[_advice.Nudge | _advice.Stop]:
    """The ladder's answers over *failures* identical errors in one turn."""
    answers: list[_advice.Nudge | _advice.Stop] = []
    for _ in range(failures):
        state.spiral.note_error("read_file:bad path", denial=denial, content="{}")
        answer = _guards.tool_error_ladder(_turn(), state, turn_context())
        if answer is not None:
            answers.append(answer)
    return answers


def test_the_ladder_nudges_escalates_then_stops() -> None:
    state = _loop_state.LoopState(original_task="t", tool_calls=0)
    nudge, escalation, stop = _climb(state, _nudges.TOOL_ERROR_STOP_AFTER)
    assert isinstance(nudge, _advice.Nudge) and nudge.text == _nudges.TOOL_ERROR_NUDGE
    assert nudge.fields == {"iteration": 3, "streak": _nudges.TOOL_ERROR_NUDGE_AFTER, "level": 1}
    assert (
        isinstance(escalation, _advice.Nudge) and escalation.text == _nudges.TOOL_ERROR_ESCALATION
    )
    assert escalation.fields["streak"] == _nudges.TOOL_ERROR_ESCALATE_AFTER
    assert escalation.fields["level"] == 2
    assert isinstance(stop, _advice.Stop) and stop.end().reason == "tool_error_stuck"
    assert stop.soft == "" and stop.declared == ""
    assert f"failed {_nudges.TOOL_ERROR_STOP_AFTER} times" in stop.end().summary
    assert stop.log == f"LOOP: tool_error stop at iter 3 (streak {_nudges.TOOL_ERROR_STOP_AFTER})"


def test_a_denial_streak_is_named_a_refusal() -> None:
    state = _loop_state.LoopState(original_task="t", tool_calls=0)
    nudge = _climb(state, _nudges.TOOL_ERROR_NUDGE_AFTER, denial=True)[0]
    assert isinstance(nudge, _advice.Nudge) and nudge.text == _nudges.TOOL_DENIED_NUDGE


def test_a_metric_run_and_the_other_modes_leave_the_ladder_alone() -> None:
    for ctx in (turn_context(metric=True), turn_context(mode="plan"), turn_context(mode="ask")):
        state = _loop_state.LoopState(original_task="t", tool_calls=0)
        for _ in range(_nudges.TOOL_ERROR_STOP_AFTER):
            state.spiral.note_error("read_file:bad path", denial=False, content="{}")
            assert _guards.tool_error_ladder(_turn(), state, ctx) is None
