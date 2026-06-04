# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Unit tests for the USD-to-tokens budget converter (~budget.py)."""

from __future__ import annotations

import pytest

from agent6.budget import INPUT_TO_OUTPUT_RATIO_FOR_USD_BUDGET, usd_budget_to_tokens


def test_usd_converter_sonnet_known_model() -> None:
    """Sonnet 4.5 at $3/M input + $15/M output, $5 budget."""
    max_in, max_out = usd_budget_to_tokens(5.0, worker_model="claude-sonnet-4-5")
    # Ratio of 5 input:output value means input gets 5/6 of the budget,
    # output gets 1/6.
    expected_in_usd = 5.0 * 5 / 6  # ~$4.17
    expected_out_usd = 5.0 * 1 / 6  # ~$0.83
    assert abs(max_in - int(expected_in_usd * 1_000_000 / 3.0)) <= 1
    assert abs(max_out - int(expected_out_usd * 1_000_000 / 15.0)) <= 1


def test_usd_converter_unknown_model_uses_fallback() -> None:
    """Unknown OpenRouter / Kimi / etc model falls back to sonnet rates."""
    max_in_known, max_out_known = usd_budget_to_tokens(5.0, worker_model="claude-sonnet-4-5")
    max_in_unk, max_out_unk = usd_budget_to_tokens(5.0, worker_model="kimi/k2.5-preview")
    # Same fallback (sonnet) -> same numbers.
    assert max_in_unk == max_in_known
    assert max_out_unk == max_out_known


def test_usd_converter_custom_fallback_for_cheap_model() -> None:
    """Operator can override fallback rates for cheap third-party models."""
    # DeepSeek-class pricing: $0.27/M in, $1.10/M out (illustrative).
    max_in, _max_out = usd_budget_to_tokens(
        1.0,
        worker_model="deepseek/v3.2",
        fallback_input_per_mtok=0.27,
        fallback_output_per_mtok=1.10,
    )
    # Same $1 budget buys 11x more input tokens vs sonnet ($3/M).
    sonnet_in, _ = usd_budget_to_tokens(1.0, worker_model="claude-sonnet-4-5")
    assert max_in > 10 * sonnet_in


def test_usd_converter_rejects_zero_or_negative() -> None:
    with pytest.raises(ValueError):
        usd_budget_to_tokens(0.0, worker_model="claude-sonnet-4-5")
    with pytest.raises(ValueError):
        usd_budget_to_tokens(-1.0, worker_model="claude-sonnet-4-5")


def test_usd_converter_scale_linearity() -> None:
    """Doubling the budget doubles both token ceilings (within rounding)."""
    in_5, out_5 = usd_budget_to_tokens(5.0, worker_model="claude-sonnet-4-5")
    in_10, out_10 = usd_budget_to_tokens(10.0, worker_model="claude-sonnet-4-5")
    assert abs((in_10 / in_5) - 2.0) < 0.01
    assert abs((out_10 / out_5) - 2.0) < 0.01


def test_usd_converter_ratio_constant_value() -> None:
    """The hard-coded input:output ratio constant is what we documented
    (5.0). Trip wire so changing it requires updating the docstring."""
    assert INPUT_TO_OUTPUT_RATIO_FOR_USD_BUDGET == 5.0
