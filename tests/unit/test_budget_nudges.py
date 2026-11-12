# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The before-call budget nudges.

A planner's one-shot finish directive on a low budget or too many turns, and a plain run's one-shot
wrap-up directive on a low budget, each once per run and only in its mode.
"""

from __future__ import annotations

from agent6.harness import guards, loop_state, nudges
from agent6.sessions import journal
from tests.unit.turn_context import turn_context


def _state() -> loop_state.LoopState:
    return loop_state.LoopState(original_task="t", tool_calls=0)


def test_the_planner_is_told_once_on_a_low_budget_or_too_many_turns() -> None:
    low = turn_context(mode="plan", budget_remaining=lambda: nudges.PLAN_BUDGET_NUDGE_BELOW)
    state = _state()
    nudge = guards.plan_budget_nudge(state, low)
    assert nudge is not None and nudge.text == nudges.PLAN_BUDGET_NUDGE
    assert nudge.event == journal.LoopPlanFinishNudge(
        turn=1, budget_remaining=nudges.PLAN_BUDGET_NUDGE_BELOW
    )
    assert nudge.log == "LOOP: plan finish-nudge at turn 1 (turns=False, low_budget=True)"
    assert guards.plan_budget_nudge(state, low) is None

    late = turn_context(mode="plan", turn=nudges.PLAN_NUDGE_AFTER_TURNS + 4, execution_start=5)
    nudge = guards.plan_budget_nudge(_state(), late)
    assert nudge is not None and "turns=True, low_budget=False" in nudge.log
    early = turn_context(mode="plan", turn=nudges.PLAN_NUDGE_AFTER_TURNS, execution_start=2)
    assert guards.plan_budget_nudge(_state(), early) is None
    assert guards.plan_budget_nudge(_state(), turn_context(budget_remaining=lambda: 0.1)) is None


def test_a_plain_run_is_told_once_on_a_low_budget_naming_its_gate() -> None:
    low = turn_context(budget_remaining=lambda: nudges.RUN_BUDGET_NUDGE_BELOW, turn=9)
    state = _state()
    nudge = guards.run_budget_nudge(state, low)
    assert nudge is not None and nudge.text == nudges.RUN_BUDGET_NUDGE_GATELESS
    assert nudge.event == journal.LoopRunBudgetNudge(
        turn=9, budget_remaining=nudges.RUN_BUDGET_NUDGE_BELOW
    )
    assert nudge.log == "LOOP: run budget-nudge at turn 9"
    assert guards.run_budget_nudge(state, low) is None

    gated = turn_context(budget_remaining=lambda: 0.1, gate_present=lambda: True)
    nudge = guards.run_budget_nudge(_state(), gated)
    assert nudge is not None and nudge.text == nudges.RUN_BUDGET_NUDGE
    assert guards.run_budget_nudge(_state(), turn_context(budget_remaining=lambda: 0.5)) is None
    assert guards.run_budget_nudge(_state(), turn_context()) is None
    metric = turn_context(budget_remaining=lambda: 0.1, metric=True)
    assert guards.run_budget_nudge(_state(), metric) is None
    plan = turn_context(budget_remaining=lambda: 0.1, mode="plan")
    assert guards.run_budget_nudge(_state(), plan) is None
