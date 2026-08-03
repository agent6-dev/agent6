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


def _t(*, fallback: int = 100) -> BudgetTracker:
    # model "m" is unpriced in the fixture cache, so its tokens land in the
    # fallback ledger; max_usd stays unlimited to keep the tests single-ledger.
    return BudgetTracker(max_usd=-1, max_tokens_fallback=fallback)


def test_usd_ceiling_counts_cache_tokens_token_caps_would_miss() -> None:
    # Token caps huge (never fire) + fresh input ~0, but cache_creation alone
    # costs > $1: the USD ceiling must catch the overspend the token caps miss.
    t = BudgetTracker(max_usd=1.0, max_tokens_fallback=-1)
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


def test_usd_ceiling_off_when_unlimited() -> None:
    # max_usd = -1 (unlimited): the same heavy-cache call trips nothing.
    t = BudgetTracker(max_usd=-1, max_tokens_fallback=-1)
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


def test_fallback_ceiling_hard_stop() -> None:
    # The unmetered ledger sums input+output; the call that reaches the cap
    # exhausts it (exclusive ceiling, enforced on the next check).
    t = _t(fallback=10)
    t.record(
        model="m", input_tokens=7, output_tokens=3, cache_read_tokens=0, cache_creation_tokens=0
    )
    assert t.is_exhausted()
    with pytest.raises(BudgetExceeded, match="fallback token budget"):
        t.check()


def test_per_model_tracking() -> None:
    t = _t(fallback=1000)
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
    # claude-sonnet-4-5 IS priced by the fixture ($3/$15 per Mtok): the known
    # half must render a real dollar figure, not fall through to "$?" with the
    # priced path unexercised. 1000 in + 100 out = $0.003 + $0.0015 = $0.0045.
    t = _t(fallback=10000)
    t.record(
        model="claude-sonnet-4-5",
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
    assert "claude-sonnet-4-5" in summary
    assert "$0.0045" in summary  # the PRICED path rendered a real figure
    assert "totally-fake-model" in summary
    assert "$? (unknown price)" in summary
    assert "TOTAL:" in summary


def test_format_summary_marks_exhausted() -> None:
    t = _t(fallback=5)
    t.record(
        model="m", input_tokens=10, output_tokens=0, cache_read_tokens=0, cache_creation_tokens=0
    )
    assert "BUDGET EXCEEDED" in t.format_summary()
