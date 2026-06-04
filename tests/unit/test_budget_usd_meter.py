# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Unit tests for `BudgetTracker.estimate_usd` (live cost meter)."""

from __future__ import annotations

from agent6.budget import BudgetTracker


def test_estimate_usd_zero_when_no_calls() -> None:
    bt = BudgetTracker(max_input_tokens=1_000_000, max_output_tokens=1_000_000)
    usd, partial = bt.estimate_usd()
    assert usd == 0.0
    assert partial is False


def test_estimate_usd_known_model() -> None:
    bt = BudgetTracker(max_input_tokens=1_000_000, max_output_tokens=1_000_000)
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
    bt = BudgetTracker(max_input_tokens=1_000_000, max_output_tokens=1_000_000)
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
    bt = BudgetTracker(max_input_tokens=10_000_000, max_output_tokens=10_000_000)
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
    """Anthropic bills 5-minute cache_creation at 1.25x the input
    rate (cache-write surcharge). Sonnet at $3/Mtok input -> $3.75/Mtok
    for cache_creation."""
    bt = BudgetTracker(max_input_tokens=10_000_000, max_output_tokens=10_000_000)
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
    """regression: prior to the fix, cache_creation_tokens were
    summed into the `input` term at full rate, double-counting the cache
    write. Verify the two are now priced via independent terms."""
    bt = BudgetTracker(max_input_tokens=10_000_000, max_output_tokens=10_000_000)
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
    bt = BudgetTracker(max_input_tokens=10_000_000, max_output_tokens=10_000_000)
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
    # format_summary prints `cost~$X.XXXX` for the total.
    assert f"cost~${usd:.4f}" in summary


def test_reported_cost_overrides_table_estimate() -> None:
    """When the provider returns ``usage.cost`` for every call to
    a model, the reported sum is used verbatim instead of the price-table
    estimate. This is what OpenRouter does."""
    bt = BudgetTracker(max_input_tokens=10_000_000, max_output_tokens=10_000_000)
    # A model that IS in the price table; provider also reports a cost
    # different from what the table would compute, to prove the reported
    # value wins.
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
    assert "(reported)" in bt.format_summary()


def test_reported_cost_works_for_unknown_model() -> None:
    """A model not in the price table contributes its reported cost
    instead of being silently dropped."""
    bt = BudgetTracker(max_input_tokens=10_000_000, max_output_tokens=10_000_000)
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


def test_partial_reported_cost_falls_back_to_table() -> None:
    """If only some calls to a model carried ``usage.cost`` the totals
    would be inconsistent — fall back to the table for the whole model."""
    bt = BudgetTracker(max_input_tokens=10_000_000, max_output_tokens=10_000_000)
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
        # no cost_usd
    )
    usd, _ = bt.estimate_usd()
    # Table says 2M @ $3 + 2M @ $15 = $36.
    assert usd == 36.0
