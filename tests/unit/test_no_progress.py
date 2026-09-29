# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The no-progress ladder.

A plain run's streak of identical verify failures draws a nudge, an escalation, then a soft stop; a
metric run is left alone.
"""

from __future__ import annotations

from unittest import mock

from agent6.harness import _advice, _conversation, _guards, _loop_state, _nudges
from tests.unit.turn_context import turn_context


def _failed_turn(iteration: int) -> _loop_state.TurnState:
    turn = _loop_state.TurnState(
        iteration=iteration, resp=mock.MagicMock(), assistant=_conversation.AssistantTurn((), ())
    )
    turn.verify_just_failed = True
    return turn


def _climb(
    state: _loop_state.LoopState, failures: int, **ctx: object
) -> list[_advice.Nudge | _advice.Stop]:
    answers: list[_advice.Nudge | _advice.Stop] = []
    for i in range(1, failures + 1):
        state.verify.note_fail("same signature")
        answer = _guards.no_progress(_failed_turn(i), state, turn_context(**ctx))
        if answer is not None:
            answers.append(answer)
    return answers


def test_the_ladder_nudges_escalates_then_stops_softly() -> None:
    state = _loop_state.LoopState(original_task="t", tool_calls=0)
    nudge, escalation, stop = _climb(state, _nudges.NO_PROGRESS_STOP_AFTER)
    assert isinstance(nudge, _advice.Nudge) and nudge.text == _nudges.NO_PROGRESS_NUDGE
    assert nudge.fields == {"iteration": _nudges.NO_PROGRESS_NUDGE_AFTER, "streak": 4, "level": 1}
    assert (
        isinstance(escalation, _advice.Nudge) and escalation.text == _nudges.NO_PROGRESS_ESCALATION
    )
    assert escalation.fields["streak"] == _nudges.NO_PROGRESS_ESCALATE_AFTER
    assert isinstance(stop, _advice.Stop) and stop.end().reason == "no_progress"
    assert stop.soft == "no_progress" and stop.declared == ""
    assert f"through {_nudges.NO_PROGRESS_STOP_AFTER} consecutive runs" in stop.end().summary
    assert (
        stop.log == f"LOOP: no_progress stop at iter {_nudges.NO_PROGRESS_STOP_AFTER} (streak 10)"
    )


def test_a_turn_whose_verify_passed_or_a_metric_run_climbs_nothing() -> None:
    state = _loop_state.LoopState(original_task="t", tool_calls=0)
    state.verify.fail_streak = _nudges.NO_PROGRESS_STOP_AFTER
    quiet = _loop_state.TurnState(
        iteration=3, resp=mock.MagicMock(), assistant=_conversation.AssistantTurn((), ())
    )
    assert _guards.no_progress(quiet, state, turn_context()) is None
    for facts in ({"metric": True}, {"mode": "plan"}):
        fresh = _loop_state.LoopState(original_task="t", tool_calls=0)
        assert _climb(fresh, _nudges.NO_PROGRESS_STOP_AFTER, **facts) == []
