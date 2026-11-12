# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The finish gates reach the panel, the metric budget and the standing goal via the context.

The panel's rejection revokes a finish with no text of its own, an early finish on a metric run is
refused while runway remains, and a standing task re-enters instead of ending.
"""

from __future__ import annotations

from unittest import mock

from agent6.harness import finish_gates, loop_conversation, loop_state, metric_rules
from agent6.sessions import journal
from tests.unit.turn_context import turn_context


def _finishing(turn: int = 4) -> loop_state.TurnState:
    turn_state = loop_state.TurnState(
        number=turn, resp=mock.MagicMock(), assistant=loop_conversation.AssistantMessage((), ())
    )
    turn_state.finish = finish_gates.FinishCall("finish_session", "done")
    return turn_state


def _state() -> loop_state.LoopState:
    return loop_state.LoopState(original_task="t", tool_calls=0)


def test_the_gates_run_in_precedence_order() -> None:
    """One precedence for every end: the declared-end path has no copy of the gates."""
    assert [gate.__name__ for gate in finish_gates.FINISH_GATES] == [
        "finish_contract",
        "verify_finish",
        "review_finish",
        "metric_early_finish",
        "open_tasks_finish",
        "standing_finish",
    ]
    assert [gate.__name__ for gate in finish_gates.END_GATES] == [
        "verify_finish",
        "review_finish",
        "open_tasks_finish",
    ]
    assert [gate.__name__ for gate in finish_gates.SILENT_END_GATES] == [
        "verify_finish",
        "review_finish",
        "metric_early_finish",
        "open_tasks_finish",
    ]


def test_the_panels_rejection_revokes_the_finish_without_a_text() -> None:
    seen: list[str] = []

    def rejecting(_turn: loop_state.TurnState, ending: str) -> bool:
        seen.append(ending)
        return True

    refusal = finish_gates.review_finish(
        _finishing(), _state(), turn_context(end_rejected=rejecting)
    )
    assert refusal is not None and refusal.text == "" and refusal.event is None
    assert seen == ["finish_session"]
    assert finish_gates.review_finish(_finishing(), _state(), turn_context()) is None


def test_an_early_finish_on_a_metric_run_is_refused_while_runway_remains() -> None:
    state = _state()
    runway = turn_context(metric=True, budget_remaining=lambda: 0.9)
    refusals = [metric_rules.metric_early_finish(_finishing(i), state, runway) for i in range(1, 5)]
    assert [r is not None for r in refusals] == [True, True, True, False]
    first = refusals[0]
    assert first is not None and first.text == metric_rules.METRIC_FINISH_NUDGE
    assert first.event == journal.FinishRefusedMetric(turn=1, nudges_used=1, budget_remaining=0.9)
    assert first.log == "  metric early-finish rejected #1 at turn 1 (budget 90% left)"
    assert state.metric.finish_nudges_used == metric_rules.METRIC_EARLY_FINISH_PATIENCE

    quiet = _finishing(2)
    quiet.finish, quiet.ending = None, "silent_finish"
    silent = metric_rules.metric_early_finish(quiet, _state(), runway)
    assert silent is not None
    assert isinstance(silent.event, journal.FinishRefusedMetric)
    assert silent.event.trigger == "silent_finish"
    assert "(silent)" in silent.log


def test_a_ceiling_the_final_slice_no_budget_signal_or_a_plain_run_lets_it_through() -> None:
    at_ceiling = _state()
    at_ceiling.metric = metric_rules.MetricGuard(
        history=[metric_rules.MetricSample(label="best", score=27, returncode=0, at_ceiling=True)]
    )
    runway = turn_context(metric=True, budget_remaining=lambda: 0.9)
    assert metric_rules.metric_early_finish(_finishing(), at_ceiling, runway) is None
    final_slice = turn_context(metric=True, budget_remaining=lambda: 0.2)
    assert metric_rules.metric_early_finish(_finishing(), _state(), final_slice) is None
    assert (
        metric_rules.metric_early_finish(_finishing(), _state(), turn_context(metric=True)) is None
    )
    assert (
        metric_rules.metric_early_finish(
            _finishing(), _state(), turn_context(budget_remaining=lambda: 0.9)
        )
        is None
    )


def test_a_standing_task_re_enters_instead_of_ending() -> None:
    asked: list[tuple[str, int]] = []

    def absorb(reason: str, turn: int) -> str:
        asked.append((reason, turn))
        return "[harness] the standing task continues"

    refusal = finish_gates.standing_finish(
        _finishing(7), _state(), turn_context(standing_absorb=absorb)
    )
    assert refusal is not None and refusal.text == "[harness] the standing task continues"
    assert asked == [("finish_session", 7)]
    assert finish_gates.standing_finish(_finishing(), _state(), turn_context()) is None
    plan = turn_context(mode="plan", standing_absorb=absorb)
    assert finish_gates.standing_finish(_finishing(), _state(), plan) is None and len(asked) == 1
