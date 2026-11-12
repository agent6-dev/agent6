# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The no-progress ladder.

A plain run's streak of identical verify failures draws a nudge, an escalation, then a soft stop; a
metric run is left alone.
"""

from __future__ import annotations

from unittest import mock

from agent6.harness import advice, guards, loop_conversation, loop_state, nudges
from agent6.sessions import journal
from tests.unit.turn_context import turn_context


def _failed_turn(turn: int) -> loop_state.TurnState:
    turn_state = loop_state.TurnState(
        number=turn, resp=mock.MagicMock(), assistant=loop_conversation.AssistantMessage((), ())
    )
    turn_state.verify_just_failed = True
    return turn_state


def _climb(
    state: loop_state.LoopState, failures: int, **ctx: object
) -> list[advice.Nudge | advice.Stop]:
    answers: list[advice.Nudge | advice.Stop] = []
    for i in range(1, failures + 1):
        state.verify.note_fail("same signature")
        answer = guards.no_progress(_failed_turn(i), state, turn_context(**ctx))
        if answer is not None:
            answers.append(answer)
    return answers


def test_the_ladder_nudges_escalates_then_stops_softly() -> None:
    state = loop_state.LoopState(original_task="t", tool_calls=0)
    nudge, escalation, stop = _climb(state, nudges.NO_PROGRESS_STOP_AFTER)
    assert isinstance(nudge, advice.Nudge) and nudge.text == nudges.NO_PROGRESS_NUDGE
    assert nudge.event == journal.LoopNoProgressNudge(
        turn=nudges.NO_PROGRESS_NUDGE_AFTER, streak=4, level=1
    )
    assert isinstance(escalation, advice.Nudge) and escalation.text == nudges.NO_PROGRESS_ESCALATION
    assert isinstance(escalation.event, journal.LoopNoProgressNudge)
    assert escalation.event.streak == nudges.NO_PROGRESS_ESCALATE_AFTER
    assert isinstance(stop, advice.Stop) and stop.end().reason == "no_progress"
    assert stop.soft == "no_progress" and stop.declared == ""
    assert f"through {nudges.NO_PROGRESS_STOP_AFTER} consecutive runs" in stop.end().summary
    assert stop.log == f"LOOP: no_progress stop at turn {nudges.NO_PROGRESS_STOP_AFTER} (streak 10)"


def test_a_turn_whose_verify_passed_or_a_metric_run_climbs_nothing() -> None:
    state = loop_state.LoopState(original_task="t", tool_calls=0)
    state.verify.fail_streak = nudges.NO_PROGRESS_STOP_AFTER
    quiet = loop_state.TurnState(
        number=3, resp=mock.MagicMock(), assistant=loop_conversation.AssistantMessage((), ())
    )
    assert guards.no_progress(quiet, state, turn_context()) is None
    for facts in ({"metric": True}, {"mode": "plan"}):
        fresh = loop_state.LoopState(original_task="t", tool_calls=0)
        assert _climb(fresh, nudges.NO_PROGRESS_STOP_AFTER, **facts) == []
