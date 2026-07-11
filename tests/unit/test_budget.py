# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for agent6.budget — hard-stop token tracker."""

from __future__ import annotations

import json

import pytest

from agent6.budget import BudgetExceeded, BudgetTracker


@pytest.fixture(autouse=True)
def price_cache(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory) -> None:
    """Inject prices via a real models-cache file (there is no static table)."""
    cache = tmp_path_factory.mktemp("price-cache")
    (cache / "models").mkdir()
    (cache / "models" / "testprovider.json").write_text(
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
    monkeypatch.setenv("AGENT6_CACHE_HOME", str(cache))


def _t(*, input_max: int = 100, output_max: int = 100) -> BudgetTracker:
    return BudgetTracker(max_input_tokens=input_max, max_output_tokens=output_max)


def test_usd_budget_to_tokens_zero_priced_model_returns_none_not_crash() -> None:
    """A 0/0-priced model (free, or transiently unpriced by OpenRouter) must
    return None like the no-price case, not raise ZeroDivisionError -- which
    crashed `agent6 run --max-usd` at config load."""
    from agent6.budget import usd_budget_to_tokens

    assert usd_budget_to_tokens(2.0, worker_model="free-or-unpriced") is None
    assert usd_budget_to_tokens(2.0, worker_model="not-in-cache-at-all") is None
    # A normally-priced model still converts.
    converted = usd_budget_to_tokens(2.0, worker_model="claude-sonnet-4-5")
    assert converted is not None and converted[0] > 0 and converted[1] > 0


def test_usd_ceiling_counts_cache_tokens_token_caps_would_miss() -> None:
    # Token caps huge (never fire) + fresh input ~0, but cache_creation alone
    # costs > $1: the USD ceiling must catch the overspend the token caps miss.
    t = BudgetTracker(max_input_tokens=10_000_000, max_output_tokens=10_000_000, max_usd=1.0)
    # sonnet-4 input $3/M; cache_creation surcharge 1.25x -> $3.75/M.
    # 300k * 3.75/1e6 = $1.125 > $1.
    t.record(
        model="claude-sonnet-4-20250514",
        input_tokens=10,
        output_tokens=10,
        cache_read_tokens=0,
        cache_creation_tokens=300_000,
    )
    with pytest.raises(BudgetExceeded) as exc:
        t.check()
    assert "USD budget" in str(exc.value)


def test_usd_ceiling_off_when_max_usd_zero() -> None:
    # max_usd defaults to 0 (disabled) -- the same heavy-cache call does not trip
    # any ceiling, so token-capped runs (e.g. benches) are unaffected.
    t = BudgetTracker(max_input_tokens=10_000_000, max_output_tokens=10_000_000)
    t.record(
        model="claude-sonnet-4-20250514",
        input_tokens=10,
        output_tokens=10,
        cache_read_tokens=0,
        cache_creation_tokens=300_000,
    )
    t.check()  # no raise


def test_record_accumulates() -> None:
    t = _t()
    t.record(
        model="m", input_tokens=5, output_tokens=3, cache_read_tokens=1, cache_creation_tokens=2
    )
    t.record(
        model="m", input_tokens=4, output_tokens=2, cache_read_tokens=0, cache_creation_tokens=0
    )
    snap = t.snapshot()
    assert snap.input_total == 9
    assert snap.output_total == 5
    assert snap.cache_read_total == 1
    assert snap.cache_creation_total == 2
    assert snap.exhausted is False
    t.check()  # should not raise


def test_input_ceiling_hard_stop() -> None:
    t = _t(input_max=10, output_max=1000)
    t.record(
        model="m", input_tokens=10, output_tokens=0, cache_read_tokens=0, cache_creation_tokens=0
    )
    assert t.is_exhausted()
    with pytest.raises(BudgetExceeded, match="input token budget"):
        t.check()


def test_output_ceiling_hard_stop() -> None:
    t = _t(input_max=1000, output_max=5)
    t.record(
        model="m", input_tokens=0, output_tokens=5, cache_read_tokens=0, cache_creation_tokens=0
    )
    assert t.is_exhausted()
    with pytest.raises(BudgetExceeded, match="output token budget"):
        t.check()


def test_per_model_tracking() -> None:
    t = _t(input_max=1000, output_max=1000)
    t.record(
        model="a", input_tokens=10, output_tokens=2, cache_read_tokens=0, cache_creation_tokens=0
    )
    t.record(
        model="b", input_tokens=20, output_tokens=4, cache_read_tokens=0, cache_creation_tokens=0
    )
    t.record(
        model="a", input_tokens=5, output_tokens=1, cache_read_tokens=0, cache_creation_tokens=0
    )
    pm = t.snapshot().per_model
    assert pm["a"].input_tokens == 15
    assert pm["a"].calls == 2
    assert pm["b"].input_tokens == 20
    assert pm["b"].calls == 1


def test_format_summary_renders_known_and_unknown_prices() -> None:
    t = _t(input_max=10000, output_max=10000)
    t.record(
        model="claude-opus-4-5-20250929",
        input_tokens=1000,
        output_tokens=100,
        cache_read_tokens=0,
        cache_creation_tokens=0,
    )
    t.record(
        model="totally-fake-model",
        input_tokens=500,
        output_tokens=50,
        cache_read_tokens=0,
        cache_creation_tokens=0,
    )
    summary = t.format_summary()
    assert "claude-opus-4-5-20250929" in summary
    assert "totally-fake-model" in summary
    assert "$? (unknown price)" in summary
    assert "TOTAL:" in summary


def test_format_summary_marks_exhausted() -> None:
    t = _t(input_max=5, output_max=1000)
    t.record(
        model="m", input_tokens=10, output_tokens=0, cache_read_tokens=0, cache_creation_tokens=0
    )
    assert "BUDGET EXCEEDED" in t.format_summary()
