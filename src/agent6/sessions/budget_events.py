# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Build cumulative budget events for provider calls, child settlement and finalization."""

from __future__ import annotations

from collections.abc import Callable

from agent6 import budget as agent6_budget
from agent6.sessions import journal


def budget_turn(budget: agent6_budget.BudgetTracker, turn: int) -> journal.BudgetTurn:
    """Return a `budget.turn` event: the tracker's cumulative tokens and cost at a turn."""
    snap = budget.snapshot()
    return journal.BudgetTurn(
        turn=turn,
        input_tokens=snap.input_total,
        output_tokens=snap.output_total,
        cache_read_tokens=snap.cache_read_total,
        cost_usd=round(snap.usd_total, 6),
    )


def budget_update(budget: agent6_budget.BudgetTracker) -> journal.BudgetUpdate:
    """Return a `budget.update` row without consuming its balance readings."""
    snap = budget.snapshot(for_journal=True)
    account = None
    latest = next(reversed(snap.plans.items()), None)  # the last reading any provider gave
    if latest is not None:
        provider, spend = latest
        plan = spend.usage
        account = journal.AccountReading(
            provider=provider,
            used_percent=plan.used_percent,
            resets_at=plan.resets_at,
            windows=[
                journal.AccountWindow(
                    name=w.name,
                    used_percent=w.used_percent,
                    window_minutes=w.window_minutes,
                    resets_at=w.resets_at,
                )
                for w in plan.windows
            ],
            credits_has=plan.has_credits,
            credits_unlimited=plan.credits_unlimited,
            credits_balance=plan.credits_balance,
        )
    return journal.BudgetUpdate(
        settled_children=tuple(sorted(snap.settled_children)),
        input_settled_children=snap.child_model_tokens.input_tokens,
        output_settled_children=snap.child_model_tokens.output_tokens,
        input_total=snap.input_total,
        output_total=snap.output_total,
        cache_read_total=snap.cache_read_total,
        cache_write_total=snap.cache_write_total,
        usd_total=snap.usd_total,
        usd_partial=snap.usd_partial,
        usd_own_calls=snap.usd_own_calls,
        usd_observed_credits=snap.usd_observed_credits,
        credit_readings=snap.credits.readings,
        credit_origins=snap.credits.origins,
        usd_cap=snap.max_usd,
        usd_running_children=snap.usd_running_children,
        usd_settled_children=snap.usd_settled_children,
        tokens_unmetered=snap.unmetered_tokens,
        tokens_running_children=snap.tokens_running_children,
        tokens_fallback_cap=snap.max_tokens_fallback,
        plan_observed_points=snap.plan_observed_points,
        plan_cap=snap.max_plan_points,
        account=account,
        # Unset without a search, so the row is the one written before searches were counted.
        **({"web_searches": snap.web_searches} if snap.web_searches else {}),
    )


def emit_budget_update(
    budget: agent6_budget.BudgetTracker, emit: Callable[[journal.BudgetUpdate], None]
) -> None:
    """Journal the meter and confirm its balance readings after the append succeeds."""
    with budget.journal_lock:
        row = budget_update(budget)
        emit(row)
        budget.credit_readings_published(row.credits)
