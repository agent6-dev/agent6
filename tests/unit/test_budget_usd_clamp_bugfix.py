# SPDX-License-Identifier: Apache-2.0
"""Regression test: a tiny USD budget must not floor a token ceiling to 0.

Before the fix, `usd_budget_to_tokens` used a bare `int()` which floored to 0
for an extreme-but-legal tiny budget, synthesizing an invalid 0 token ceiling
that then failed the BudgetConfig `gt=0` validator with a misleading error.
"""

from __future__ import annotations

import pytest

import agent6.budget as budget_mod
from agent6.budget import usd_budget_to_tokens


def test_tiny_usd_budget_clamps_to_at_least_one_token(monkeypatch: pytest.MonkeyPatch) -> None:
    # Stub a priced worker model ($3/M in, $15/M out) so the conversion runs.
    def _priced(_m: str) -> tuple[float, float]:
        return (3.0, 15.0)

    monkeypatch.setattr(budget_mod, "lookup_price", _priced)

    # 5e-05 USD floors the output side to 0 tokens under a bare int().
    result = usd_budget_to_tokens(0.00005, worker_model="stub/worker")
    assert result is not None
    max_in, max_out = result
    # Both ceilings must be >= 1 (gt=0 in BudgetConfig).
    assert max_in >= 1
    assert max_out >= 1


def test_no_price_still_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    def _unpriced(_m: str) -> tuple[float, float] | None:
        return None

    monkeypatch.setattr(budget_mod, "lookup_price", _unpriced)
    assert usd_budget_to_tokens(0.00005, worker_model="stub/worker") is None
