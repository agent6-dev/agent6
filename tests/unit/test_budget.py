# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for agent6.budget, the hard-stop token tracker."""

from __future__ import annotations

import json
from typing import Any
from unittest import mock

import pytest

from agent6 import budget, kinds


@pytest.fixture(autouse=True)
def price_cache(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory) -> None:
    """Inject prices via a real models-cache file (there is no static table)."""
    cache = tmp_path_factory.mktemp("price-cache")
    (cache / "agent6" / "models").mkdir(parents=True, exist_ok=True)
    (cache / "agent6" / "models" / "testprovider.json").write_text(
        json.dumps(
            {
                "models": [],
                "pricing": {
                    "claude-sonnet-4-5": [3.0, 15.0],
                    "claude-sonnet-4-20250514": [3.0, 15.0],
                    "free-or-unpriced": [0.0, 0.0],  # OpenRouter reports 0/0 for some routes
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache))


def _t(*, fallback: int = 100) -> budget.BudgetTracker:
    # Model "m" is unpriced in the fixture cache, so its tokens land in the fallback ledger.
    return budget.BudgetTracker(max_usd=-1, max_tokens_fallback=fallback, max_plan_points=-1)


def test_usd_ceiling_counts_cache_tokens_token_caps_would_miss() -> None:
    # Token caps huge and fresh input ~0, but cache_write alone costs over $1.
    t = budget.BudgetTracker(max_usd=1.0, max_tokens_fallback=-1, max_plan_points=-1)
    # sonnet-4 input $3/M, cache_write 1.25x = $3.75/M; 300k * 3.75/1e6 = $1.125 > $1.
    t.record(
        route=kinds.ModelRoute("claude-sonnet-4-20250514", "claude-sonnet-4-20250514"),
        input_tokens=10,
        output_tokens=10,
        cache_read_tokens=0,
        cache_write_tokens=300_000,
    )
    with pytest.raises(budget.BudgetExceededError) as exc:
        t.check()
    assert "USD budget" in str(exc.value)


def test_usd_ceiling_off_when_unlimited() -> None:
    # max_usd = -1 (unlimited): the same heavy-cache call trips nothing.
    t = budget.BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_plan_points=-1)
    t.record(
        route=kinds.ModelRoute("claude-sonnet-4-20250514", "claude-sonnet-4-20250514"),
        input_tokens=10,
        output_tokens=10,
        cache_read_tokens=0,
        cache_write_tokens=300_000,
    )
    t.check()  # no raise


def test_negative_usage_never_reduces_the_ledger() -> None:
    """Negative usage never reduces the ledger.

    A gateway's counts are third-party arithmetic; the one sink clamps signs, so spend only ever
    grows.
    """
    t = _t()
    t.record(
        route=kinds.ModelRoute("m", "m"),
        input_tokens=5,
        output_tokens=3,
        cache_read_tokens=1,
        cache_write_tokens=2,
    )
    before = t.snapshot()
    t.record(
        route=kinds.ModelRoute("m", "m"),
        input_tokens=-500,
        output_tokens=-500,
        cache_read_tokens=-500,
        cache_write_tokens=-500,
        cost_usd=-9.0,
    )
    snap = t.snapshot()
    assert snap.input_total == before.input_total
    assert snap.output_total == before.output_total
    assert snap.cache_read_total == before.cache_read_total
    assert snap.cache_write_total == before.cache_write_total
    assert t.estimate_usd()[0] >= 0.0


def test_record_accumulates() -> None:
    t = _t()
    t.record(
        route=kinds.ModelRoute("m", "m"),
        input_tokens=5,
        output_tokens=3,
        cache_read_tokens=1,
        cache_write_tokens=2,
    )
    t.record(
        route=kinds.ModelRoute("m", "m"),
        input_tokens=4,
        output_tokens=2,
        cache_read_tokens=0,
        cache_write_tokens=0,
    )
    snap = t.snapshot()
    assert snap.input_total == 9
    assert snap.output_total == 5
    assert snap.cache_read_total == 1
    assert snap.cache_write_total == 2
    assert snap.exhausted is False
    t.check()  # should not raise


def test_fallback_ceiling_hard_stop() -> None:
    # The unmetered ledger sums input and output; the call that reaches the cap exhausts it.
    t = _t(fallback=10)
    t.record(
        route=kinds.ModelRoute("m", "m"),
        input_tokens=7,
        output_tokens=3,
        cache_read_tokens=0,
        cache_write_tokens=0,
    )
    assert t.is_exhausted()
    with pytest.raises(budget.BudgetExceededError, match="fallback token budget"):
        t.check()


def test_per_model_tracking() -> None:
    t = _t(fallback=1000)
    t.record(
        route=kinds.ModelRoute("a", "a"),
        input_tokens=10,
        output_tokens=2,
        cache_read_tokens=0,
        cache_write_tokens=0,
    )
    t.record(
        route=kinds.ModelRoute("b", "b"),
        input_tokens=20,
        output_tokens=4,
        cache_read_tokens=0,
        cache_write_tokens=0,
    )
    t.record(
        route=kinds.ModelRoute("a", "a"),
        input_tokens=5,
        output_tokens=1,
        cache_read_tokens=0,
        cache_write_tokens=0,
    )
    pm = t.snapshot().per_model
    assert pm[kinds.ModelRoute("a", "a")].input_tokens == 15
    assert pm[kinds.ModelRoute("a", "a")].calls == 2
    assert pm[kinds.ModelRoute("b", "b")].input_tokens == 20
    assert pm[kinds.ModelRoute("b", "b")].calls == 1


def test_format_summary_renders_known_and_unknown_prices() -> None:
    # claude-sonnet-4-5 is priced by the fixture: 1000 in + 100 out = $0.003 + $0.0015 = $0.0045.
    t = _t(fallback=10000)
    t.record(
        route=kinds.ModelRoute("claude-sonnet-4-5", "claude-sonnet-4-5"),
        input_tokens=1000,
        output_tokens=100,
        cache_read_tokens=0,
        cache_write_tokens=0,
    )
    t.record(
        route=kinds.ModelRoute("totally-fake-model", "totally-fake-model"),
        input_tokens=500,
        output_tokens=50,
        cache_read_tokens=0,
        cache_write_tokens=0,
    )
    summary = t.format_summary()
    assert "claude-sonnet-4-5" in summary
    assert "$0.0045" in summary  # the PRICED path rendered a real figure
    assert "totally-fake-model" in summary
    assert "$? (unknown price)" in summary
    assert "TOTAL:" in summary


def test_format_summary_names_no_ceiling_before_any_call() -> None:
    """A run that made no model call names no dollar ceiling: nothing was metered against it.

    The line read `cost=$0.0000 of $10.00` on a subscription-plan model, where a run that
    made calls names no dollar ceiling.
    """
    t = budget.BudgetTracker(max_usd=10.0, max_tokens_fallback=-1, max_plan_points=-1)
    assert "  TOTAL: in=0 out=0 cost=$0.0000\n" in t.format_summary() + "\n"


def test_format_summary_marks_exhausted() -> None:
    """A standalone cost summary identifies a reached cap even when a call crosses it."""
    t = _t(fallback=5)
    t.record(
        route=kinds.ModelRoute("m", "m"),
        input_tokens=10,
        output_tokens=0,
        cache_read_tokens=0,
        cache_write_tokens=0,
    )
    assert "STATUS: fallback token budget exhausted" in t.format_summary()


def test_the_caps_have_one_home() -> None:
    """The caps are required arguments; BudgetConfig is the one place the defaults live."""
    import inspect
    import pathlib

    from agent6.config import BudgetConfig

    params = inspect.signature(budget.BudgetTracker).parameters
    for name in ("max_usd", "max_tokens_fallback", "max_plan_points"):
        assert params[name].default is inspect.Parameter.empty, name

    cfg = BudgetConfig()
    docs = pathlib.Path(__file__).resolve().parents[2] / "docs" / "config.md"
    row_usd, row_tokens = "", ""
    for line in docs.read_text(encoding="utf-8").splitlines():
        if line.startswith("| `max_usd`"):
            row_usd = line
        elif line.startswith("| `max_tokens_fallback`"):
            row_tokens = line
    assert f"`{cfg.max_usd}`" in row_usd, row_usd
    assert f"`{cfg.max_tokens_fallback}`" in row_tokens, row_tokens


def test_percent_meter_sawtooth_and_cap() -> None:
    """The percent meter: a sawtooth over window resets, and the cap trips at max_plan_points.

    The first reading is the baseline, rises are the run's consumption and a later reset time
    starts a new window counted from zero; plan calls never drain the fallback ledger and
    report an authoritative $0.
    """
    t = budget.BudgetTracker(max_usd=10.0, max_tokens_fallback=100, max_plan_points=10.0)

    def rec(pct: float, resets_at: float = 2e9) -> None:
        t.record(
            route=kinds.ModelRoute("gpt-5.6-sol", "gpt-5.6-sol"),
            input_tokens=1000,
            output_tokens=50,
            cache_read_tokens=0,
            cache_write_tokens=0,
            plan_usage=budget.PlanUsage.single(
                used_percent=pct, window_minutes=10080, resets_at=resets_at
            ),
        )

    reset = 2e9 + 7 * 86400
    rec(37.0)  # baseline
    rec(40.0)  # +3
    rec(2.0, reset)  # a new window: +2
    rec(5.0, reset)  # +3 -> consumed 8
    t.check()  # under the cap of 10
    snap = t.snapshot()
    assert snap.plan_observed_points == pytest.approx(8.0)
    assert snap.plan_latest is not None and snap.plan_latest.used_percent == 5.0
    assert snap.unmetered_tokens == 0  # percent-metered, never fallback
    usd, partial = t.estimate_usd()
    assert usd == 0.0 and partial is False  # authoritative $0, not unpriced
    rec(8.0, reset)  # +3 -> consumed 11 >= 10
    with pytest.raises(budget.BudgetExceededError, match="plan budget exhausted"):
        t.check()
    assert "plan usage (gpt-5.6-sol): 8% of the 7-day window" in t.format_summary()
    assert "(subscription)" in t.format_summary()


def test_a_plan_cap_names_the_same_window_as_its_usage_line() -> None:
    """A plan cap error named the primary window where usage named the 7-day window."""
    tracker = budget.BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_plan_points=1)
    tracker.record_plan_preflight("chatgpt", budget.PlanUsage.single(48, 10080, 2e9))
    tracker.record(
        route=kinds.ModelRoute("chatgpt", "m"),
        input_tokens=5,
        output_tokens=0,
        cache_read_tokens=0,
        cache_write_tokens=0,
        plan_usage=budget.PlanUsage.single(49, 10080, 2e9),
    )
    with pytest.raises(budget.BudgetExceededError, match="account at 49% on its 7-day window"):
        tracker.check()
    assert "49% of the 7-day window" in tracker.format_summary()


def test_percent_zero_refuses_plan_metered_calls() -> None:
    t = budget.BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_plan_points=0)
    t.record(
        route=kinds.ModelRoute("gpt-5.6-sol", "gpt-5.6-sol"),
        input_tokens=10,
        output_tokens=1,
        cache_read_tokens=0,
        cache_write_tokens=0,
        plan_usage=budget.PlanUsage.single(used_percent=1.0, window_minutes=300, resets_at=2e9),
    )
    with pytest.raises(budget.BudgetExceededError, match="percent budget is 0"):
        t.check()


def _plan_with_credits(used: float, *, unlimited: bool = False) -> budget.PlanUsage:
    return budget.PlanUsage.single(
        used_percent=used,
        window_minutes=10080,
        resets_at=2e9,
        has_credits=True,
        credits_unlimited=unlimited,
        credits_balance="$12.50",
    )


def _record_plan(t: budget.BudgetTracker, plan: budget.PlanUsage) -> None:
    t.record(
        route=kinds.ModelRoute("gpt-5.6-sol", "gpt-5.6-sol"),
        input_tokens=10,
        output_tokens=1,
        cache_read_tokens=0,
        cache_write_tokens=0,
        plan_usage=plan,
    )


def test_exhausted_window_with_credits_refuses_by_default() -> None:
    """An exhausted window with purchased credits refuses by default.

    Past the included window, chatgpt calls draw real money the $0-authoritative stance would hide;
    the refusal names [budget].allow_plan_overage.
    """
    t = budget.BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_plan_points=-1)
    _record_plan(t, _plan_with_credits(100.0))
    with pytest.raises(budget.BudgetExceededError, match="allow_plan_overage"):
        t.check()


def test_credits_spend_allowed_when_opted_in() -> None:
    t = budget.BudgetTracker(
        max_usd=-1, max_tokens_fallback=-1, max_plan_points=-1, allow_plan_overage=True
    )
    _record_plan(t, _plan_with_credits(100.0))
    t.check()


def test_zero_usd_refuses_paid_credits_even_when_opted_in() -> None:
    tracker = budget.BudgetTracker(
        max_usd=0, max_tokens_fallback=-1, max_plan_points=-1, allow_plan_overage=True
    )
    tracker.record_plan_preflight("chatgpt", _plan_with_credits(100.0))
    with pytest.raises(budget.BudgetExceededError, match="USD budget is 0"):
        tracker.check()


def test_credits_inside_the_included_window_do_not_refuse() -> None:
    t = budget.BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_plan_points=-1)
    _record_plan(t, _plan_with_credits(41.0))
    t.check()


@pytest.mark.parametrize("max_plan_points", [-1, 10])
def test_summary_labels_shared_plan_movement(max_plan_points: float) -> None:
    """Another consumer's window movement is not presented as this run's spend."""
    tracker = budget.BudgetTracker(
        max_usd=-1, max_tokens_fallback=-1, max_plan_points=max_plan_points
    )
    tracker.record_plan_preflight("claude", budget.PlanUsage.single(68, 10080, 0))
    # Another consumer uses seven points before this run's next reading.
    tracker.record_plan_preflight("claude", budget.PlanUsage.single(75, 10080, 0))

    cap = "" if max_plan_points == -1 else " of max_plan_points 10"
    assert (
        "plan usage (claude): 75% of the 7-day window"
        f" (window moved ~7 points{cap} during this run; resets in 0h)"
    ) in tracker.format_summary()
    assert "this run ~7 points" not in tracker.format_summary()


def test_unlimited_credits_do_not_refuse() -> None:
    t = budget.BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_plan_points=-1)
    _record_plan(t, _plan_with_credits(100.0, unlimited=True))
    t.check()


def test_fraction_remaining_counts_the_plan_percent_ledger() -> None:
    """fraction_remaining counts the plan ledger as used = plan_observed_points / max_plan_points.

    Without it a plan-metered run reads ~1.0 remaining until the hard stop and no near-budget
    behaviour engages.
    """
    t = budget.BudgetTracker(max_usd=-1.0, max_tokens_fallback=-1, max_plan_points=5.0)
    t.record(
        route=kinds.ModelRoute("gpt-5.6-sol", "gpt-5.6-sol"),
        input_tokens=10,
        output_tokens=10,
        cache_read_tokens=0,
        cache_write_tokens=0,
        plan_usage=budget.PlanUsage.single(used_percent=40.0, window_minutes=300, resets_at=0.0),
    )
    t.record(
        route=kinds.ModelRoute("gpt-5.6-sol", "gpt-5.6-sol"),
        input_tokens=10,
        output_tokens=10,
        cache_read_tokens=0,
        cache_write_tokens=0,
        plan_usage=budget.PlanUsage.single(used_percent=42.5, window_minutes=300, resets_at=0.0),
    )
    # The run consumed ~2.5 points of its 5-point cap.
    assert 0.4 <= t.fraction_remaining() <= 0.6


def test_preflight_reading_seeds_the_baseline_and_guards_credits() -> None:
    """A preflight reading seeds the baseline and guards credits before any call.

    A secondary window at 100 counts as exhausted.
    """
    t = budget.BudgetTracker(max_usd=1.0, max_tokens_fallback=100, max_plan_points=-1)
    t.record_plan_preflight(
        "m", budget.PlanUsage.single(used_percent=40.0, window_minutes=10080, resets_at=0)
    )
    t.record(
        route=kinds.ModelRoute("m", "m"),
        input_tokens=1,
        output_tokens=1,
        cache_read_tokens=0,
        cache_write_tokens=0,
        plan_usage=budget.PlanUsage.single(used_percent=42.0, window_minutes=10080, resets_at=0),
    )
    assert t.snapshot().plan_observed_points == 2.0
    guarded = budget.BudgetTracker(max_usd=1.0, max_tokens_fallback=100, max_plan_points=-1)
    guarded.record_plan_preflight(
        "m",
        budget.PlanUsage.single(
            used_percent=10.0,
            window_minutes=10080,
            resets_at=0,
            has_credits=True,
            secondary_used_percent=100.0,
        ),
    )
    with pytest.raises(budget.BudgetExceededError, match="purchased"):
        guarded.check()


def test_the_binding_window_meters_the_run_whatever_its_name() -> None:
    """The cap binds on the window that moved most, whatever its name.

    A per-model family the run burns counts while the primary window barely moves.
    """

    def reading(primary: float, spark: float) -> budget.PlanUsage:
        return budget.PlanUsage(
            windows=(
                budget.PlanWindow("primary", primary, 10080, 2e9),
                budget.PlanWindow("gpt-5-6-spark", spark, 300, 2e9),
            )
        )

    t = budget.BudgetTracker(max_usd=1.0, max_tokens_fallback=100, max_plan_points=5.0)
    _record_plan(t, reading(10.0, 40.0))  # baselines
    _record_plan(t, reading(10.5, 43.0))
    assert t.snapshot().plan_observed_points == 3.0  # spark moved 3, primary 0.5
    _record_plan(t, reading(11.0, 46.0))
    with pytest.raises(budget.BudgetExceededError, match="5-hour window"):
        t.check()
    # A reset on one window restarts that window's count from zero only.
    t2 = budget.BudgetTracker(max_usd=1.0, max_tokens_fallback=100, max_plan_points=50.0)
    _record_plan(t2, reading(10.0, 90.0))
    _record_plan(t2, reading(12.0, 1.0))
    assert t2.snapshot().plan_observed_points == 2.0


def test_percent_cap_names_the_window_whose_consumption_bound_it() -> None:
    """The max-plan-points error names the window that consumed the capped points."""

    def reading(primary: float, opus: float) -> budget.PlanUsage:
        return budget.PlanUsage(
            windows=(
                budget.PlanWindow("seven_day", primary, 10080, 2e9),
                budget.PlanWindow("seven_day_opus", opus, 10080, 2e9),
            )
        )

    tracker = budget.BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_plan_points=5)
    _record_plan(tracker, reading(90, 10))
    _record_plan(tracker, reading(91, 16))

    with pytest.raises(budget.BudgetExceededError, match=r"7-day window \(opus\)"):
        tracker.check()
    # The latest reading need not carry the window that bound; the backend adds and drops windows.
    _record_plan(
        tracker, budget.PlanUsage(windows=(budget.PlanWindow("seven_day", 92, 10080, 2e9),))
    )
    with pytest.raises(budget.BudgetExceededError, match=r"7-day window \(opus\)"):
        tracker.check()


def test_purchased_credit_spend_meters_against_max_usd() -> None:
    """With allow_plan_overage, the balance that left the account counts into the USD estimate.

    A balance that is not a number meters nothing.
    """
    t = budget.BudgetTracker(
        max_usd=1.0, max_tokens_fallback=100, max_plan_points=-1, allow_plan_overage=True
    )

    def reading(balance: str) -> budget.PlanUsage:
        return budget.PlanUsage.single(
            used_percent=100.0,
            window_minutes=10080,
            resets_at=2e9,
            has_credits=True,
            credits_balance=balance,
        )

    _record_plan(t, reading("$12.50"))
    _record_plan(t, reading("$11.90"))
    assert t.estimate_usd()[0] == pytest.approx(0.60)
    # The summary states the figure the ceiling enforces, not the sum of the per-model lines.
    assert "cost=$0.60" in t.format_summary()
    t.check()
    _record_plan(t, reading("$11.40"))
    with pytest.raises(budget.BudgetExceededError, match=r"USD budget exhausted: .*\$1.10"):
        t.check()
    opaque = budget.BudgetTracker(
        max_usd=1.0, max_tokens_fallback=100, max_plan_points=-1, allow_plan_overage=True
    )
    _record_plan(opaque, reading("lots"))
    _record_plan(opaque, reading("fewer"))
    assert opaque.estimate_usd()[0] == 0.0


@pytest.mark.parametrize("operation", ["record", "preflight", "settle"])
def test_plan_readings_enforce_combined_usd_spend(operation: str) -> None:
    """API dollars and purchased credits share the USD ceiling after every plan reading."""
    tracker = budget.BudgetTracker(
        max_usd=10, max_tokens_fallback=-1, max_plan_points=-1, allow_plan_overage=True
    )
    tracker.record(
        route=kinds.ModelRoute("api", "api"),
        input_tokens=1,
        output_tokens=1,
        cache_read_tokens=0,
        cache_write_tokens=0,
        cost_usd=8,
    )
    tracker.record_plan_preflight(
        "gpt-5.6-sol",
        budget.PlanUsage.single(100, 10080, 2e9, has_credits=True, credits_balance="$40"),
    )
    tracker.check()
    reading = budget.PlanUsage.single(100, 10080, 2e9, has_credits=True, credits_balance="$35")
    if operation == "record":
        _record_plan(tracker, reading)
    elif operation == "preflight":
        tracker.record_plan_preflight("gpt-5.6-sol", reading)
    else:
        tracker.register_plan_reader("gpt-5.6-sol", lambda: reading)
        assert tracker.settle()

    assert tracker.estimate_usd()[0] == 13
    assert "cost=$13.00" in tracker.format_summary()
    assert tracker.snapshot().exhausted
    assert tracker.is_exhausted()
    assert tracker.fraction_remaining() == 0
    with pytest.raises(budget.BudgetExceededError, match=r"USD budget exhausted: .*\$13.00"):
        tracker.check()


def test_settlement_enforces_plan_point_ceiling() -> None:
    """Settlement marks the tracker exhausted when the last reading crosses its plan cap."""
    tracker = budget.BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_plan_points=5)
    tracker.record_plan_preflight("m", _window(20, 2e9))
    tracker.register_plan_reader("m", lambda: _window(26, 2e9))
    tracker.check()

    assert tracker.settle()

    assert tracker.snapshot().plan_observed_points == 6
    assert tracker.snapshot().exhausted
    assert tracker.is_exhausted()
    with pytest.raises(budget.BudgetExceededError, match="plan budget exhausted"):
        tracker.check()


def test_credit_balances_are_tracked_per_provider_entry() -> None:
    """Balances from separate subscription accounts must never be compared."""
    tracker = budget.BudgetTracker(
        max_usd=-1, max_tokens_fallback=-1, max_plan_points=-1, allow_plan_overage=True
    )

    def record(route: kinds.ModelRoute, balance: str) -> None:
        tracker.record(
            route=route,
            input_tokens=1,
            output_tokens=1,
            cache_read_tokens=0,
            cache_write_tokens=0,
            plan_usage=budget.PlanUsage.single(
                100.0,
                10080,
                0.0,
                has_credits=True,
                credits_balance=balance,
            ),
        )

    record(kinds.ModelRoute("provider-a", "model-a"), "$10.00")
    record(kinds.ModelRoute("provider-b", "model-b"), "$100.00")
    record(kinds.ModelRoute("provider-a", "model-a"), "$9.00")
    assert tracker.estimate_usd()[0] == pytest.approx(1.0)


def test_credits_balance_units_convert_to_usd() -> None:
    """A bare credits balance converts at 25 credits per dollar; a "$"-prefixed one is dollars."""

    def usd(balance: str) -> float | None:
        return budget.PlanUsage.single(
            0.0, 10080, 0.0, has_credits=True, credits_balance=balance
        ).credits_usd

    assert usd("500") == 20.0
    assert usd("1,000") == 40.0
    assert usd("$12.50") == 12.50
    assert usd("") is None
    assert usd("n/a") is None
    assert usd("nan") is None
    assert usd("$inf") is None


def test_plan_usage_line_does_not_invent_a_window_length() -> None:
    """A window with no reported duration prints no invented length or backend key."""
    t = budget.BudgetTracker(max_usd=1.0, max_tokens_fallback=100, max_plan_points=-1)
    _record_plan(
        t,
        budget.PlanUsage(
            windows=(
                budget.PlanWindow("primary", 1.0, 10080, 2e9),
                budget.PlanWindow("secondary", 3.0, 0, 2e9),
            )
        ),
    )
    route, spend = next(iter(t.snapshot().plans.items()))
    line = budget.format_plan_usage(route, spend, -1)
    assert "3% of the window (window moved" in line
    assert "secondary" not in line
    assert "0-minute" not in line


def test_an_all_unpriced_run_does_not_claim_a_usd_ceiling() -> None:
    """An all-unpriced run's receipt claims no USD ceiling and spells the fallback cap.

    The cap reads as the preflight spells it, with the lower-bound `+` before the parenthetical.
    """
    t = budget.BudgetTracker(max_usd=10.0, max_tokens_fallback=2_000_000, max_plan_points=-1)
    t.record(
        route=kinds.ModelRoute("unpriced-xyz", "unpriced-xyz"),
        input_tokens=10,
        output_tokens=5,
        cache_read_tokens=0,
        cache_write_tokens=0,
        cost_usd=None,
    )

    total = next(ln for ln in t.format_summary().splitlines() if "TOTAL" in ln)

    assert "of $" not in total, total
    assert "cost~$0.0000+" in total
    assert "2,000,000 fallback tokens" in total


def test_each_subscription_plan_prints_its_own_named_line() -> None:
    """Each subscription plan prints its own named line and meters its windows apart.

    Two `primary` windows at different percents are two plans, not one plan that reset.
    """
    t = budget.BudgetTracker(max_usd=1.0, max_tokens_fallback=100, max_plan_points=-1)

    def rec(route: kinds.ModelRoute, pct: float) -> None:
        t.record(
            route=route,
            input_tokens=10,
            output_tokens=1,
            cache_read_tokens=0,
            cache_write_tokens=0,
            plan_usage=budget.PlanUsage.single(
                used_percent=pct, window_minutes=10080, resets_at=2e9
            ),
        )

    rec(kinds.ModelRoute("chatgpt", "gpt-5.6-sol"), 28.0)
    rec(kinds.ModelRoute("claude", "claude-opus-5"), 54.0)
    rec(
        kinds.ModelRoute("chatgpt", "gpt-5.6-sol"), 29.0
    )  # +1 on the chatgpt plan; the claude plan is untouched
    snap = t.snapshot()
    assert set(snap.plans) == {"chatgpt", "claude"}
    assert snap.plans["chatgpt"].observed_points == pytest.approx(1.0)
    assert snap.plans["claude"].observed_points == 0.0
    assert snap.plan_observed_points == pytest.approx(1.0)
    assert snap.plan_latest is not None and snap.plan_latest.used_percent == 29.0
    summary = t.format_summary()
    assert "plan usage (chatgpt): 29% of the 7-day window" in summary
    assert "plan usage (claude): 54% of the 7-day window" in summary


@pytest.mark.parametrize(
    ("name", "minutes", "label"),
    [
        ("seven_day_overage_included", 10080, "7-day window"),
        ("seven_day", 10080, "7-day window"),
        ("five_hour", 300, "5-hour window"),
        ("seven_day_opus", 10080, "7-day window"),
        ("primary", 10080, "7-day window"),
        ("gpt-5-6-spark", 300, "5-hour window"),
        ("unknown", 0, "window"),
        ("short", 30, "30-minute window"),
    ],
)
def test_unique_duration_plan_output_omits_backend_keys(
    name: str, minutes: int, label: str
) -> None:
    """A unique duration needs no key in the run receipt or fan-out reading."""
    window = budget.PlanWindow(name, 30, minutes, 0)
    usage = budget.PlanUsage(windows=(window,))
    tracker = budget.BudgetTracker(max_usd=1, max_tokens_fallback=100, max_plan_points=-1)
    _record_plan(tracker, usage)
    assert budget.format_plan_window(window, usage.windows) == label
    assert (
        f"30% of the {label} (window moved ~0 points during this run; resets in 0h)"
        in tracker.format_summary()
    )
    assert budget.format_plan_reading("claude", usage) == (
        f"plan usage (claude): 30% of the {label} (resets in 0h)"
    )
    assert tracker.snapshot().plan_latest == usage
    assert usage.binding.name == name


@pytest.mark.parametrize(
    ("keys", "labels"),
    [
        (("seven_day_opus", "seven_day_overage"), ("opus", "overage")),
        (("custom_alpha", "custom_beta"), ("alpha", "beta")),
        (("primary", "secondary"), ("primary", "secondary")),
    ],
)
def test_window_suffixes_come_from_shared_key_parts(
    keys: tuple[str, str], labels: tuple[str, str]
) -> None:
    """Equal-duration keys retain their distinguishing underscore-separated parts."""
    windows = tuple(budget.PlanWindow(key, 30, 10080, 0) for key in keys)
    for window, label in zip(windows, labels, strict=True):
        assert budget.format_plan_window(window, windows) == f"7-day window ({label})"


def test_colliding_plan_windows_have_distinguishing_suffixes() -> None:
    """A model-specific weekly window is distinguishable from the general weekly window."""
    for name, label in [("seven_day", "7-day window"), ("seven_day_opus", "7-day window (opus)")]:
        usage = budget.PlanUsage(
            windows=tuple(
                budget.PlanWindow(key, 30 if key == name else 10, 10080, 0)
                for key in ("seven_day", "seven_day_opus")
            )
        )
        tracker = budget.BudgetTracker(max_usd=1, max_tokens_fallback=100, max_plan_points=-1)
        _record_plan(tracker, usage)
        assert (
            f"30% of the {label} (window moved ~0 points during this run; resets in 0h)"
            in tracker.format_summary()
        )
        assert budget.format_plan_reading("claude", usage) == (
            f"plan usage (claude): 30% of the {label} (resets in 0h)"
        )


def _window(used: float, resets_at: float) -> budget.PlanUsage:
    return budget.PlanUsage.single(used_percent=used, window_minutes=10080, resets_at=resets_at)


def test_an_out_of_order_reading_counts_nothing() -> None:
    """Concurrent seats delivered readings out of order and each drop counted as a reset."""
    tracker = _t()
    for used in (20.0, 23.0, 21.0, 23.0):
        tracker.record_plan_preflight("m", _window(used, 2e9))
    assert tracker.snapshot().plan_observed_points == 3.0


def test_a_reset_time_later_by_more_than_a_minute_is_a_new_window() -> None:
    """A higher reading on a later window counted only its rise over the old window."""
    tracker = _t()
    tracker.record_plan_preflight("m", _window(10.0, 2e9))
    tracker.record_plan_preflight("m", _window(12.0, 2e9 + 7 * 86400))
    assert tracker.snapshot().plan_observed_points == 12.0


def test_a_reading_from_a_window_already_left_counts_nothing() -> None:
    """A delayed reading from the previous window counted as a rise and moved the reset back."""
    tracker = _t()
    later = 2e9 + 7 * 86400
    for used, at in ((90.0, 2e9), (2.0, later), (90.0, 2e9), (3.0, later)):
        tracker.record_plan_preflight("m", _window(used, at))
    assert tracker.snapshot().plan_observed_points == 3.0


def test_a_reset_time_within_a_minute_is_the_same_window() -> None:
    """A reported reset time that jittered by a second restarted the count."""
    tracker = _t()
    for used, at in ((93.0, 2e9), (93.0, 2e9 + 1), (94.0, 2e9 - 1), (93.0, 2e9), (94.0, 2e9)):
        tracker.record_plan_preflight("m", _window(used, at))
    assert tracker.snapshot().plan_observed_points == 1.0


@pytest.mark.parametrize("anchored_at", [2e9 + 150, 2e9 + 7200], ids=["within", "beyond"])
def test_an_unanchored_window_slides_with_the_clock_and_counts_nothing(anchored_at: float) -> None:
    """A window's reset time slides with the clock until its first use anchors it."""
    tracker = _t()
    tracker.record_plan_preflight("m", _window(0.0, 2e9))
    tracker.record_plan_preflight("m", _window(0.0, 2e9 + 120))
    tracker.record_plan_preflight("m", _window(1.0, anchored_at))
    assert tracker.snapshot().plan_observed_points == 1.0


def test_settle_reads_each_metered_plan_once_more() -> None:
    """A reading came from a header or the preflight, so a run's last calls never counted."""
    tracker = _t()
    metered = mock.Mock(side_effect=[_window(25.0, 2e9)])
    idle = mock.Mock(return_value=_window(99.0, 2e9))
    tracker.register_plan_reader("m", metered)
    tracker.register_plan_reader("idle", idle)
    tracker.record_plan_preflight("m", _window(20.0, 2e9))
    tracker.settle()
    tracker.settle()
    assert tracker.snapshot().plan_observed_points == 5.0
    assert metered.call_count == 1 and idle.call_count == 0


# ---------------------------------------------------------------------------
# Lane reservations: a parent splits what it has left and charges what its lanes used.
# ---------------------------------------------------------------------------


def _plan(pct: float, resets_at: float = 2e9, **rest: Any) -> budget.PlanUsage:
    return budget.PlanUsage.single(pct, 10080, resets_at, **rest)


def _usage(usd: float = 0.0, tokens: int = 0, *readings: tuple[str, budget.PlanUsage]) -> Any:
    return budget.UsageSettlement(usd=usd, usd_partial=False, tokens=tokens, plan_readings=readings)


def test_reserve_splits_what_is_available_and_a_spent_parent_refuses() -> None:
    """Three lanes share the six dollars left; a parent at its cap grants nothing."""
    t = budget.BudgetTracker(max_usd=10.0, max_tokens_fallback=-1, max_plan_points=-1)
    t.record(
        route=kinds.ModelRoute("testprovider", "claude-sonnet-4-5"),
        input_tokens=0,
        output_tokens=0,
        cache_read_tokens=0,
        cache_write_tokens=0,
        cost_usd=4.0,
    )
    assert t.available().usd == pytest.approx(6.0)
    grant = t.reserve("p1", ["a", "b", "c"], owner="exec-1")
    assert [share.usd for _, share in grant.shares] == pytest.approx([2.0, 2.0, 2.0])
    assert [share.tokens for _, share in grant.shares] == [-1, -1, -1]
    assert t.available().usd == pytest.approx(0.0)
    assert t.snapshot().reserved.usd == pytest.approx(6.0)
    assert t.fraction_remaining() == pytest.approx(0.0)
    # The breaker counts settled spend, not held shares: the parent's own call still runs.
    t.check()
    # The same identity is the same grant, never a second one.
    assert t.reserve("p1", ["a", "b", "c"], owner="exec-1") == grant
    spent = budget.BudgetTracker(max_usd=1.0, max_tokens_fallback=-1, max_plan_points=-1)
    spent.record(
        route=kinds.ModelRoute("testprovider", "claude-sonnet-4-5"),
        input_tokens=0,
        output_tokens=0,
        cache_read_tokens=0,
        cache_write_tokens=0,
        cost_usd=1.0,
    )
    with pytest.raises(budget.BudgetExceededError):
        spent.reserve("p2", ["a"], owner="exec-1")


def test_reserve_is_atomic_across_threads() -> None:
    """Simultaneous reservations never grant more than the parent had."""
    import threading

    t = budget.BudgetTracker(max_usd=8.0, max_tokens_fallback=-1, max_plan_points=-1)
    granted: list[float] = []
    refused: list[str] = []
    ready = threading.Barrier(8)

    def take(i: int) -> None:
        ready.wait()
        try:
            grant = t.reserve(f"p{i}", [f"p{i}-l1"], owner="exec-1")
        except budget.BudgetExceededError as exc:
            refused.append(str(exc))
        else:
            granted.extend(share.usd for _, share in grant.shares)

    threads = [threading.Thread(target=take, args=(i,)) for i in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert sum(granted) <= 8.0 + 1e-9
    assert len(granted) + len(refused) == 8
    assert t.available().usd == pytest.approx(8.0 - sum(granted))


def test_settle_charges_actual_usage_and_releases_the_rest() -> None:
    """Usage above a share is charged whole; usage below it frees the difference."""
    t = budget.BudgetTracker(max_usd=10.0, max_tokens_fallback=100, max_plan_points=-1)
    grant = t.reserve("p1", ["a", "b"], owner="exec-1")
    assert [share.tokens for _, share in grant.shares] == [50, 50]
    t.settle_lane("p1", "a", _usage(usd=7.0, tokens=10))  # above its $5 share
    assert t.estimate_usd()[0] == pytest.approx(7.0)
    assert t.snapshot().unmetered_tokens == 10
    assert t.available().usd == pytest.approx(0.0)  # $7 spent, $5 still held: nothing left
    t.settle_lane("p1", "b", _usage(usd=1.0, tokens=0))  # below: the rest is released
    assert t.estimate_usd()[0] == pytest.approx(8.0)
    assert t.available().usd == pytest.approx(2.0)
    assert t.snapshot().reserved.usd == 0.0
    # A duplicate settlement changes nothing.
    t.settle_lane("p1", "b", _usage(usd=1.0, tokens=0))
    assert t.estimate_usd()[0] == pytest.approx(8.0)
    t.check()
    # Actual usage over the cap trips the breaker for the parent's next call.
    over = budget.BudgetTracker(max_usd=10.0, max_tokens_fallback=-1, max_plan_points=-1)
    over.reserve("p1", ["a"], owner="exec-1")
    over.settle_lane("p1", "a", _usage(usd=12.0))
    with pytest.raises(budget.BudgetExceededError, match="USD budget exhausted"):
        over.check()


def test_unknown_usage_stays_held_until_it_is_known() -> None:
    t = budget.BudgetTracker(max_usd=10.0, max_tokens_fallback=-1, max_plan_points=-1)
    t.reserve("p1", ["a", "b"], owner="exec-1")
    t.settle_lane("p1", "a", None)  # the lane's record never came back
    assert t.snapshot().reserved.usd == pytest.approx(10.0)
    assert t.available().usd == pytest.approx(0.0)
    t.settle_lane("p1", "b", _usage(usd=1.0))
    assert t.snapshot().reserved.usd == pytest.approx(5.0)
    t.settle_lane("p1", "a", _usage(usd=2.0))  # known at last: released and charged
    assert t.snapshot().reserved.usd == 0.0
    assert t.estimate_usd()[0] == pytest.approx(3.0)
    assert "reserved for lanes" not in t.format_summary()


def test_the_summary_names_held_shares() -> None:
    t = budget.BudgetTracker(max_usd=10.0, max_tokens_fallback=-1, max_plan_points=-1)
    t.reserve("p1", ["a"], owner="exec-1")
    assert "reserved for lanes: $10.00" in t.format_summary()


def test_sibling_plan_readings_count_the_observed_union_once() -> None:
    """Two lanes that saw the same account movement charge it once; order and jitter add nothing."""
    t = budget.BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_plan_points=10.0)
    t.record_plan_preflight("chatgpt", _plan(40.0))
    t.reserve("p1", ["a", "b", "c", "d"], owner="exec-1")
    t.settle_lane("p1", "a", _usage(0.0, 0, ("chatgpt", _plan(42.0))))  # +2
    t.settle_lane("p1", "b", _usage(0.0, 0, ("chatgpt", _plan(42.0))))  # the same movement: +0
    t.settle_lane("p1", "c", _usage(0.0, 0, ("chatgpt", _plan(41.0))))  # arrived late, lower: +0
    assert t.snapshot().plan_observed_points == pytest.approx(2.0)
    # Reset-time jitter within a minute is the same window; a later reset is a new one.
    t.settle_lane("p1", "d", _usage(0.0, 0, ("chatgpt", _plan(43.0, resets_at=2e9 + 30))))  # +1
    assert t.snapshot().plan_observed_points == pytest.approx(3.0)
    t.reserve("p2", ["e", "f"], owner="exec-1")
    t.settle_lane(
        "p2", "e", _usage(0.0, 0, ("chatgpt", _plan(3.0, resets_at=2e9 + 7 * 86400)))
    )  # +3
    t.settle_lane("p2", "f", _usage(0.0, 0, ("claude", _plan(10.0))))  # another provider's baseline
    snap = t.snapshot()
    assert snap.plan_observed_points == pytest.approx(6.0)
    assert set(snap.plans) == {"chatgpt", "claude"}
    assert not t.is_exhausted()


def test_credit_balance_drops_seen_through_lanes_are_money_once() -> None:
    t = budget.BudgetTracker(max_usd=10.0, max_tokens_fallback=-1, max_plan_points=-1)
    t.record_plan_preflight("chatgpt", _plan(40.0, has_credits=True, credits_balance="$5.00"))
    t.reserve("p1", ["a", "b"], owner="exec-1")
    t.settle_lane(
        "p1",
        "a",
        _usage(0.0, 0, ("chatgpt", _plan(41.0, has_credits=True, credits_balance="$3.50"))),
    )
    t.settle_lane(
        "p1",
        "b",
        _usage(0.0, 0, ("chatgpt", _plan(41.0, has_credits=True, credits_balance="$3.50"))),
    )
    assert t.estimate_usd()[0] == pytest.approx(1.5)
