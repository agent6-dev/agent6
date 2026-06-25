# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Validate the pure-Python statistics in bench/sweep/stats.py against known
reference values, so the benchmark's numbers are trustworthy."""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bench" / "sweep"))

import stats  # path-injected bench module  # pyright: ignore[reportMissingImports]


def test_wilson_ci_known_values() -> None:
    # p=1 edge: 95% Wilson for 10/10 is approx [0.7225, 1.0].
    lo, hi = stats.wilson_ci(10, 10)
    assert abs(lo - 0.7225) < 0.002
    assert abs(hi - 1.0) < 1e-9
    # symmetric: 5/10 is approx [0.2366, 0.7634].
    lo, hi = stats.wilson_ci(5, 10)
    assert abs(lo - 0.2366) < 0.002
    assert abs(hi - 0.7634) < 0.002
    # n=0 is the uninformative [0, 1].
    assert stats.wilson_ci(0, 0) == (0.0, 1.0)


def test_fisher_exact_known_values() -> None:
    # Classic 2x2 [[3,1],[1,3]] -> two-sided p = 0.4857 (lady-tasting-tea shape).
    assert abs(stats.fisher_exact_two_sided(3, 1, 1, 3) - 0.4857) < 0.001
    # Perfectly separated 10/0 vs 0/10 -> 2 / C(20,10) = 1.08e-5.
    assert stats.fisher_exact_two_sided(10, 0, 0, 10) < 1e-4
    # No association at all -> p ~ 1.
    assert stats.fisher_exact_two_sided(5, 5, 5, 5) > 0.99


def test_mann_whitney_fully_separated() -> None:
    r = stats.mann_whitney_u([1, 2, 3], [4, 5, 6])
    assert r.u == 0  # A entirely below B
    assert abs(r.cliffs_delta + 1.0) < 1e-9  # delta = -1
    assert 0.05 < r.p_two_sided < 0.12  # normal approx of the exact 0.10


def test_mann_whitney_identical_distributions() -> None:
    r = stats.mann_whitney_u([1, 2, 3, 4], [1, 2, 3, 4])
    assert abs(r.cliffs_delta) < 1e-9
    assert r.p_two_sided > 0.9


def test_bootstrap_median_ci_is_seeded_and_brackets_median() -> None:
    xs = [1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    a = stats.bootstrap_median_ci(xs)
    b = stats.bootstrap_median_ci(xs)
    assert a == b  # deterministic (seeded)
    assert a[0] <= statistics.median(xs) <= a[1]


def test_effective_cost_derives_for_unpriced_anthropic_only() -> None:
    # opus list price 5/25 per 1M; 1M in + 1M out -> $30.
    opus = {
        "model_slug": "claude-opus-4-8",
        "cost_usd": 0.0,
        "input_tokens": 1_000_000,
        "output_tokens": 1_000_000,
    }
    assert abs(stats.effective_cost(opus) - 30.0) < 1e-6
    # A measured (OpenRouter) cost is always preferred over derivation.
    kimi = {
        "model_slug": "moonshotai/kimi-k2.6-20260420",
        "cost_usd": 0.05,
        "input_tokens": 1,
        "output_tokens": 1,
    }
    assert stats.effective_cost(kimi) == 0.05
    # Unknown unpriced model stays at its measured (0) cost, not invented.
    unknown = {"model_slug": "x/y", "cost_usd": 0.0, "input_tokens": 9, "output_tokens": 9}
    assert stats.effective_cost(unknown) == 0.0


def test_quartiles_and_gmean() -> None:
    q1, med, q3 = stats.quartiles([1, 2, 3, 4, 5])
    assert med == 3
    assert q1 < med < q3
    assert abs(stats.gmean([1, 10, 100]) - 10.0) < 1e-9  # geometric mean of 1,10,100
