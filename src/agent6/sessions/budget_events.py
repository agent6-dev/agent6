# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Build cumulative budget events for provider calls, lane settlement and finalization."""

from __future__ import annotations

from typing import Any

from agent6 import budget as agent6_budget
from agent6.sessions import journal


def budget_turn(budget: agent6_budget.BudgetTracker, turn: int) -> journal.BudgetTurn:
    """Return a `budget.turn` event: the tracker's cumulative tokens and cost at a turn."""
    snap = budget.snapshot()
    cost, _ = budget.estimate_usd()
    return journal.BudgetTurn(
        turn=turn,
        input_tokens=snap.input_total,
        output_tokens=snap.output_total,
        cache_read_tokens=snap.cache_read_total,
        cost_usd=round(cost, 6),
    )


def budget_update_fields(budget: agent6_budget.BudgetTracker) -> dict[str, Any]:
    """Return a `budget.update` event's fields: the tracker's cumulative spend and plan reading."""
    snap = budget.snapshot()
    usd_total, usd_partial = budget.estimate_usd()
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
    return dict(
        settled_lanes=tuple(sorted(snap.settled_lanes)),
        input_total=snap.input_total,
        output_total=snap.output_total,
        cache_read_total=snap.cache_read_total,
        cache_write_total=snap.cache_write_total,
        usd_total=usd_total,
        usd_partial=usd_partial,
        usd_cap=snap.max_usd,
        usd_running_lanes=snap.usd_running_lanes,
        tokens_unmetered=snap.unmetered_tokens,
        tokens_running_lanes=snap.tokens_running_lanes,
        tokens_fallback_cap=snap.max_tokens_fallback,
        plan_observed_points=snap.plan_observed_points,
        plan_cap=snap.max_plan_points,
        account=account,
    )
