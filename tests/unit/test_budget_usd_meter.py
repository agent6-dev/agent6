# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Unit tests for `BudgetTracker.estimate_usd` (live cost meter)."""

from __future__ import annotations

import json

import pytest

from agent6 import budget


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
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache))


def test_estimate_usd_zero_when_no_calls() -> None:
    bt = budget.BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1)
    usd, partial = bt.estimate_usd()
    assert usd == 0.0
    assert partial is False


def test_estimate_usd_known_model() -> None:
    bt = budget.BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1)
    # sonnet-4-5 is $3 / Mtok in, $15 / Mtok out.
    bt.record(
        model="claude-sonnet-4-5",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_read_tokens=0,
        cache_creation_tokens=0,
    )
    usd, partial = bt.estimate_usd()
    assert usd == 18.0
    assert partial is False


def test_estimate_usd_unknown_model_flags_partial() -> None:
    bt = budget.BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1)
    bt.record(
        model="some-future-model-not-in-table",
        input_tokens=500_000,
        output_tokens=100_000,
        cache_read_tokens=0,
        cache_creation_tokens=0,
    )
    usd, partial = bt.estimate_usd()
    assert usd == 0.0  # unknown model contributes nothing
    assert partial is True


def test_estimate_usd_cache_read_priced_at_10_percent() -> None:
    bt = budget.BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1)
    # sonnet at $3/Mtok input -> $0.30/Mtok for cache_read.
    bt.record(
        model="claude-sonnet-4-5",
        input_tokens=0,
        output_tokens=0,
        cache_read_tokens=1_000_000,
        cache_creation_tokens=0,
    )
    usd, _ = bt.estimate_usd()
    assert abs(usd - 0.30) < 1e-9


def test_estimate_usd_cache_creation_priced_at_125_percent() -> None:
    """Anthropic bills 5-minute cache_creation at 1.25x the input rate.

    Sonnet at $3/Mtok input is $3.75/Mtok for cache_creation.
    """
    bt = budget.BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1)
    bt.record(
        model="claude-sonnet-4-5",
        input_tokens=0,
        output_tokens=0,
        cache_read_tokens=0,
        cache_creation_tokens=1_000_000,
    )
    usd, _ = bt.estimate_usd()
    assert abs(usd - 3.75) < 1e-9


def test_estimate_usd_fresh_input_excludes_cache_creation() -> None:
    """cache_creation tokens are priced as their own term, never summed into the input term."""
    bt = budget.BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1)
    bt.record(
        model="claude-sonnet-4-5",
        input_tokens=1_000_000,
        output_tokens=0,
        cache_read_tokens=0,
        cache_creation_tokens=1_000_000,
    )
    # 1M fresh @ $3 + 1M cache_creation @ $3.75 = $6.75 (NOT 2M @ $3 = $6).
    usd, _ = bt.estimate_usd()
    assert abs(usd - 6.75) < 1e-9


def test_estimate_usd_matches_format_summary_total() -> None:
    """The live meter and the end-of-run summary must agree on the total."""
    bt = budget.BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1)
    bt.record(
        model="claude-sonnet-4-5",
        input_tokens=12_345,
        output_tokens=6_789,
        cache_read_tokens=100,
        cache_creation_tokens=42,
    )
    bt.record(
        model="claude-haiku-4-5",
        input_tokens=500,
        output_tokens=200,
        cache_read_tokens=0,
        cache_creation_tokens=0,
    )
    usd, _ = bt.estimate_usd()
    summary = bt.format_summary()
    # format_summary prints the total through format_usd, marked `~`.
    assert f"cost~{budget.format_usd(usd)}" in summary


def test_reported_cost_overrides_table_estimate() -> None:
    """A reported `usage.cost` on every call to a model overrides the table estimate verbatim."""
    bt = budget.BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1)
    # A priced model whose provider also reports a different cost, so the reported value wins.
    bt.record(
        model="claude-sonnet-4-5",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        cost_usd=99.99,  # would be $18.00 by the table
    )
    usd, partial = bt.estimate_usd()
    assert usd == 99.99
    assert partial is False
    summary = bt.format_summary()
    assert "(reported)" in summary
    # Every call reported its cost: the total is exact, not an estimate.
    assert "cost=$99.99 of" in summary


def test_reported_cost_works_for_unknown_model() -> None:
    """A model not in the price table contributes its reported cost."""
    bt = budget.BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1)
    bt.record(
        model="future/unknown-model",
        input_tokens=10_000,
        output_tokens=2_000,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        cost_usd=0.1234,
    )
    usd, partial = bt.estimate_usd()
    assert usd == 0.1234
    assert partial is False


def test_mixed_reported_cost_adds_table_estimate_for_unreported_calls() -> None:
    """When only some calls carried `usage.cost`, the table prices only the unreported calls.

    A whole-model table fallback would present a lower estimate as exact.
    """
    bt = budget.BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1)
    bt.record(
        model="claude-sonnet-4-5",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        cost_usd=50.0,  # authoritative; this call's tokens must NOT be re-priced
    )
    bt.record(
        model="claude-sonnet-4-5",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_read_tokens=1_000_000,  # 1M @ $3*0.1/M = $0.30: cache banks per call too
        cache_creation_tokens=0,
        # no cost_usd
    )
    usd, partial = bt.estimate_usd()
    # $50.00 reported + table for call 2 only (1M @ $3 + 1M @ $15 + $0.30).
    assert usd == pytest.approx(68.30)
    # Every dollar is reported or table-priced; nothing is a known under-estimate.
    assert partial is False
    assert "(reported + estimated)" in bt.format_summary()


def test_mixed_reported_cost_counts_toward_usd_ceiling() -> None:
    """The enforced ceiling sees reported plus estimated spend."""
    bt = budget.BudgetTracker(max_usd=60.0, max_tokens_fallback=-1, max_percent=-1)
    bt.record(
        model="claude-sonnet-4-5",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        cost_usd=50.0,
    )
    bt.record(
        model="claude-sonnet-4-5",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_read_tokens=0,
        cache_creation_tokens=0,
    )
    assert bt.is_exhausted()


def test_fraction_remaining_tracks_usd_ceiling() -> None:
    """`fraction_remaining` counts the USD ceiling, not just the token caps.

    On a cache-heavy run the USD ceiling alone includes cache cost, so it is what hard-stops the run
    and what the wind-down nudges have to see.
    """
    # Token caps huge so only the $5 USD ceiling binds; a cache-heavy turn counts zero toward them.
    bt = budget.BudgetTracker(max_usd=5.0, max_tokens_fallback=-1, max_percent=-1)
    bt.record(
        model="claude-sonnet-4-5",
        input_tokens=100_000,  # $0.30 fresh input
        output_tokens=100_000,  # $1.50 output
        cache_read_tokens=10_000_000,  # 10M @ $3*0.1/M = $3.00
        cache_creation_tokens=0,
    )
    usd, _ = bt.estimate_usd()
    assert usd == pytest.approx(4.8)  # 0.30 + 1.50 + 3.00
    # Token axes are ~1% used but 96% of the USD budget is gone, so the figure is ~0.04.
    assert bt.fraction_remaining() == pytest.approx(1.0 - 4.8 / 5.0, abs=1e-6)


def test_fraction_remaining_unlimited_usd_never_depletes() -> None:
    """With max_usd = -1, metered spend depletes nothing; only a positive cap can."""
    bt = budget.BudgetTracker(max_usd=-1, max_tokens_fallback=1_000, max_percent=-1)
    bt.record(
        model="claude-sonnet-4-5",
        input_tokens=5_000_000,
        output_tokens=1_000_000,  # ~$30 metered, uncapped
        cache_read_tokens=0,
        cache_creation_tokens=0,
    )
    assert bt.fraction_remaining() == 1.0
    bt.record(
        model="local-unpriced",
        input_tokens=400,
        output_tokens=100,  # 500 of 1000 fallback tokens
        cache_read_tokens=0,
        cache_creation_tokens=0,
    )
    assert bt.fraction_remaining() == pytest.approx(0.5)


def test_partially_reported_unpriced_model_keeps_the_reported_spend() -> None:
    """A partially reported unpriced model keeps the reported spend, marked partial.

    Dropping it would read $0.00 and never trip the best-effort USD cap.
    """
    bt = budget.BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1)
    bt.record(
        model="future/unpriced-model",
        input_tokens=1_000,
        output_tokens=500,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        cost_usd=0.60,  # the provider reported this call
    )
    bt.record(
        model="future/unpriced-model",
        input_tokens=1_000,
        output_tokens=500,
        cache_read_tokens=0,
        cache_creation_tokens=0,  # this one's usage block carried no cost
    )
    usd, partial = bt.estimate_usd()
    assert usd == pytest.approx(0.60)  # not dropped to 0.0
    assert partial is True  # and flagged as an under-estimate


def test_a_sub_cent_cap_prints_at_the_spends_precision() -> None:
    """A sub-cent cap prints at the spend's precision: cap and spend share one formatter.

    Cents at $1 and above, four decimals below.
    """
    assert budget.format_usd(0.004) == "$0.0040"
    assert budget.format_usd(10.0) == "$10.00"
    bt = budget.BudgetTracker(max_usd=0.004, max_tokens_fallback=-1, max_percent=-1)
    bt.record(
        model="claude-haiku-4-5",
        input_tokens=5000,
        output_tokens=2000,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        cost_usd=0.0046,  # reported by the provider: this spend IS metered
    )
    summary = bt.format_summary()
    assert "of $0.0040" in summary  # the cap, at the spend's precision
    assert "cost=$0.0046" in summary
    assert ">= $0.0040" in summary  # and the exhaustion reason, not ">= $0.00"
