# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The before-call budget nudges: a planner's one-shot finish directive on a
low budget or too many turns, and a plain run's one-shot wrap-up directive on
a low budget, each once per run and only in its mode."""

from __future__ import annotations

from agent6.workflows._guards import plan_budget_nudge, run_budget_nudge
from agent6.workflows._loop_state import LoopState
from agent6.workflows._nudges import (
    PLAN_BUDGET_NUDGE,
    PLAN_BUDGET_NUDGE_BELOW,
    PLAN_NUDGE_AFTER_ITERS,
    RUN_BUDGET_NUDGE,
    RUN_BUDGET_NUDGE_BELOW,
    RUN_BUDGET_NUDGE_GATELESS,
)
from tests.unit.turn_context import turn_context


def _state() -> LoopState:
    return LoopState(original_task="t", tool_calls=0)


def test_the_planner_is_told_once_on_a_low_budget_or_too_many_turns() -> None:
    low = turn_context(mode="plan", budget_remaining=lambda: PLAN_BUDGET_NUDGE_BELOW)
    state = _state()
    nudge = plan_budget_nudge(state, low)
    assert nudge is not None and nudge.text == PLAN_BUDGET_NUDGE
    assert nudge.fields == {"iteration": 1, "budget_remaining": PLAN_BUDGET_NUDGE_BELOW}
    assert nudge.log == "LOOP: plan finish-nudge at iter 1 (turns=False, low_budget=True)"
    assert plan_budget_nudge(state, low) is None

    late = turn_context(mode="plan", iteration=PLAN_NUDGE_AFTER_ITERS + 4, leg_start=5)
    nudge = plan_budget_nudge(_state(), late)
    assert nudge is not None and "turns=True, low_budget=False" in nudge.log
    early = turn_context(mode="plan", iteration=PLAN_NUDGE_AFTER_ITERS, leg_start=2)
    assert plan_budget_nudge(_state(), early) is None
    assert plan_budget_nudge(_state(), turn_context(budget_remaining=lambda: 0.1)) is None


def test_a_plain_run_is_told_once_on_a_low_budget_naming_its_gate() -> None:
    low = turn_context(budget_remaining=lambda: RUN_BUDGET_NUDGE_BELOW, iteration=9)
    state = _state()
    nudge = run_budget_nudge(state, low)
    assert nudge is not None and nudge.text == RUN_BUDGET_NUDGE_GATELESS
    assert nudge.fields == {"iteration": 9, "budget_remaining": RUN_BUDGET_NUDGE_BELOW}
    assert nudge.log == "LOOP: run budget-nudge at iter 9"
    assert run_budget_nudge(state, low) is None

    gated = turn_context(budget_remaining=lambda: 0.1, gate_present=lambda: True)
    nudge = run_budget_nudge(_state(), gated)
    assert nudge is not None and nudge.text == RUN_BUDGET_NUDGE
    assert run_budget_nudge(_state(), turn_context(budget_remaining=lambda: 0.5)) is None
    assert run_budget_nudge(_state(), turn_context()) is None
    metric = turn_context(budget_remaining=lambda: 0.1, metric=True)
    assert run_budget_nudge(_state(), metric) is None
    plan = turn_context(budget_remaining=lambda: 0.1, mode="plan")
    assert run_budget_nudge(_state(), plan) is None
