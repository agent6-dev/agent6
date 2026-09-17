# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The no-progress ladder: a plain run's streak of identical verify failures
draws a nudge, an escalation, then a soft stop; a metric run is left alone."""

from __future__ import annotations

from unittest.mock import MagicMock

from agent6.workflows._conversation import AssistantTurn
from agent6.workflows._guards import Nudge, Stop, no_progress
from agent6.workflows._loop_state import LoopState, TurnState
from agent6.workflows._nudges import (
    NO_PROGRESS_ESCALATE_AFTER,
    NO_PROGRESS_ESCALATION,
    NO_PROGRESS_NUDGE,
    NO_PROGRESS_NUDGE_AFTER,
    NO_PROGRESS_STOP_AFTER,
)
from tests.unit.turn_context import turn_context


def _failed_turn(iteration: int) -> TurnState:
    turn = TurnState(iteration=iteration, resp=MagicMock(), assistant=AssistantTurn((), ()))
    turn.verify_just_failed = True
    return turn


def _climb(state: LoopState, failures: int, **ctx: object) -> list[Nudge | Stop]:
    answers: list[Nudge | Stop] = []
    for i in range(1, failures + 1):
        state.verify.note_fail("same signature")
        answer = no_progress(_failed_turn(i), state, turn_context(**ctx))
        if answer is not None:
            answers.append(answer)
    return answers


def test_the_ladder_nudges_escalates_then_stops_softly() -> None:
    state = LoopState(original_task="t", tool_calls=0)
    nudge, escalation, stop = _climb(state, NO_PROGRESS_STOP_AFTER)
    assert isinstance(nudge, Nudge) and nudge.text == NO_PROGRESS_NUDGE
    assert nudge.fields == {"iteration": NO_PROGRESS_NUDGE_AFTER, "streak": 4, "level": 1}
    assert isinstance(escalation, Nudge) and escalation.text == NO_PROGRESS_ESCALATION
    assert escalation.fields["streak"] == NO_PROGRESS_ESCALATE_AFTER
    assert isinstance(stop, Stop) and stop.end().reason == "no_progress"
    assert stop.soft == "no_progress" and stop.declared == ""
    assert f"through {NO_PROGRESS_STOP_AFTER} consecutive runs" in stop.end().summary
    assert stop.log == f"LOOP: no_progress stop at iter {NO_PROGRESS_STOP_AFTER} (streak 10)"


def test_a_turn_whose_verify_passed_or_a_metric_run_climbs_nothing() -> None:
    state = LoopState(original_task="t", tool_calls=0)
    state.verify.fail_streak = NO_PROGRESS_STOP_AFTER
    quiet = TurnState(iteration=3, resp=MagicMock(), assistant=AssistantTurn((), ()))
    assert no_progress(quiet, state, turn_context()) is None
    for facts in ({"metric": True}, {"mode": "plan"}):
        fresh = LoopState(original_task="t", tool_calls=0)
        assert _climb(fresh, NO_PROGRESS_STOP_AFTER, **facts) == []
