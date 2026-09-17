# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The metric plateau advisor: a tie draws the notice while runway remains,
counts against the patience in the final budget slice, then stops; a
ceiling stops at once. Its end grounds on the tree."""

from __future__ import annotations

from unittest.mock import MagicMock

from agent6.workflows._advice import Nudge, Stop
from agent6.workflows._conversation import AssistantTurn
from agent6.workflows._guards import MetricGuard, metric_plateau
from agent6.workflows._loop_state import LoopState, TurnState
from agent6.workflows._metric import METRIC_PLATEAU_PATIENCE, MetricSample
from tests.unit.turn_context import turn_context


def _tie(iteration: int) -> TurnState:
    turn = TurnState(iteration=iteration, resp=MagicMock(), assistant=AssistantTurn((), ()))
    turn.metric_plateau_finish = "score plateaued at 10"
    return turn


def test_a_tie_with_runway_is_a_notice_that_costs_no_patience() -> None:
    state = LoopState(original_task="t", tool_calls=0)
    ctx = turn_context(metric=True, budget_remaining=lambda: 0.8)
    for i in range(1, 6):
        answer = metric_plateau(_tie(i), state, ctx)
        assert isinstance(answer, Nudge) and "80% of the budget remains" in answer.text
        assert answer.fields == {"iteration": i, "nudges_used": 0, "budget_remaining": 0.8}
    assert state.metric.plateau_nudges_used == 0


def test_final_slice_ties_spend_the_patience_then_stop() -> None:
    state = LoopState(original_task="t", tool_calls=0)
    ctx = turn_context(metric=True, open_subtasks=lambda: [("t1", "measure")])
    answers = [metric_plateau(_tie(i), state, ctx) for i in range(1, METRIC_PLATEAU_PATIENCE + 2)]
    notices, stop = answers[:-1], answers[-1]
    assert all(isinstance(a, Nudge) and "unknown" in a.text for a in notices)
    assert [a.fields["nudges_used"] for a in notices if isinstance(a, Nudge)] == [1, 2, 3]
    assert isinstance(stop, Stop) and stop.event == ""
    assert stop.soft == "metric_plateau" and stop.declared == "metric_plateau"
    assert stop.log == f"LOOP: metric_plateau at iter {METRIC_PLATEAU_PATIENCE + 1}"
    end = stop.end()
    assert end.reason == "metric_plateau" and end.verdict == "grounded" and end.completed
    assert end.summary == "score plateaued at 10 (1 open task(s): measure)"


def test_a_ceiling_stops_at_once_with_its_event() -> None:
    state = LoopState(original_task="t", tool_calls=0)
    state.metric = MetricGuard(
        history=[MetricSample(label="best", score=27, returncode=0, at_ceiling=True)]
    )
    stop = metric_plateau(_tie(4), state, turn_context(metric=True, budget_remaining=lambda: 0.9))
    assert isinstance(stop, Stop)
    assert stop.event == "loop.metric_ceiling.stop" and stop.fields == {"iteration": 4}
    assert stop.end().reason == "metric_plateau"


def test_a_turn_without_a_tie_is_quiet() -> None:
    turn = TurnState(iteration=1, resp=MagicMock(), assistant=AssistantTurn((), ()))
    state = LoopState(original_task="t", tool_calls=0)
    assert metric_plateau(turn, state, turn_context(metric=True)) is None
