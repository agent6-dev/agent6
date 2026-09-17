# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The finish gates that reach the panel, the metric budget and the
standing goal through the turn context: the panel's rejection revokes a
finish with no text of its own, an early finish on a metric run is refused
while runway remains, and a standing task re-enters instead of ending."""

from __future__ import annotations

from unittest.mock import MagicMock

from agent6.workflows._conversation import AssistantTurn
from agent6.workflows._finish_gates import FINISH_GATES, review_finish, standing_finish
from agent6.workflows._loop_state import LoopState, TurnState
from agent6.workflows._metric import (
    METRIC_EARLY_FINISH_PATIENCE,
    METRIC_FINISH_NUDGE,
    MetricGuard,
    MetricSample,
    early_finish_refusal,
    metric_early_finish,
)
from tests.unit.turn_context import turn_context


def _finishing(iteration: int = 4) -> TurnState:
    turn = TurnState(iteration=iteration, resp=MagicMock(), assistant=AssistantTurn((), ()))
    turn.finish_signal = "done"
    turn.finish_kind = "finish_session"
    return turn


def _state() -> LoopState:
    return LoopState(original_task="t", tool_calls=0)


def test_the_gates_run_in_precedence_order() -> None:
    assert [gate.__name__ for gate in FINISH_GATES] == [
        "finish_contract",
        "review_finish",
        "metric_early_finish",
        "open_tasks_finish",
        "verify_finish",
        "memory_finish",
        "standing_finish",
    ]


def test_the_panels_rejection_revokes_the_finish_without_a_text() -> None:
    seen: list[str] = []

    def rejecting(_turn: TurnState, ending: str) -> bool:
        seen.append(ending)
        return True

    refusal = review_finish(_finishing(), _state(), turn_context(end_reviewed=rejecting))
    assert refusal is not None and refusal.text == "" and refusal.event == ""
    assert seen == ["finish_session"]
    assert review_finish(_finishing(), _state(), turn_context()) is None


def test_an_early_finish_on_a_metric_run_is_refused_while_runway_remains() -> None:
    state = _state()
    runway = turn_context(metric=True, budget_remaining=lambda: 0.9)
    refusals = [metric_early_finish(_finishing(i), state, runway) for i in range(1, 5)]
    assert [r is not None for r in refusals] == [True, True, True, False]
    first = refusals[0]
    assert first is not None and first.text == METRIC_FINISH_NUDGE
    assert first.event == "loop.metric_early_finish.rejected"
    assert first.fields == {"iteration": 1, "nudges_used": 1, "budget_remaining": 0.9}
    assert first.log == "  metric early-finish rejected #1 at iter 1 (budget 90% left)"
    assert state.metric.finish_nudges_used == METRIC_EARLY_FINISH_PATIENCE

    silent = early_finish_refusal(_state(), runway, iteration=2, trigger="silent_finish")
    assert silent is not None and silent.fields["trigger"] == "silent_finish"
    assert "(silent)" in silent.log


def test_a_ceiling_the_final_slice_no_budget_signal_or_a_plain_run_lets_it_through() -> None:
    at_ceiling = _state()
    at_ceiling.metric = MetricGuard(
        history=[MetricSample(label="best", score=27, returncode=0, at_ceiling=True)]
    )
    runway = turn_context(metric=True, budget_remaining=lambda: 0.9)
    assert metric_early_finish(_finishing(), at_ceiling, runway) is None
    final_slice = turn_context(metric=True, budget_remaining=lambda: 0.2)
    assert metric_early_finish(_finishing(), _state(), final_slice) is None
    assert metric_early_finish(_finishing(), _state(), turn_context(metric=True)) is None
    assert (
        metric_early_finish(_finishing(), _state(), turn_context(budget_remaining=lambda: 0.9))
        is None
    )


def test_a_standing_task_re_enters_instead_of_ending() -> None:
    asked: list[tuple[str, int]] = []

    def absorb(reason: str, iteration: int) -> str:
        asked.append((reason, iteration))
        return "[harness] the standing task continues"

    refusal = standing_finish(_finishing(7), _state(), turn_context(standing_absorb=absorb))
    assert refusal is not None and refusal.text == "[harness] the standing task continues"
    assert asked == [("finish_session", 7)]
    assert standing_finish(_finishing(), _state(), turn_context()) is None
    plan = turn_context(mode="plan", standing_absorb=absorb)
    assert standing_finish(_finishing(), _state(), plan) is None and len(asked) == 1
