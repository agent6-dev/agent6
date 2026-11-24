# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for cache-fetched model pricing (agent6.models.pricing + models_cache)."""

from __future__ import annotations

import json
import pathlib

import pytest

from agent6 import config, kinds
from agent6.models import cache as models_cache  # pyright: ignore[reportPrivateUsage]
from agent6.models import pricing as models_pricing
from tests import price_env


def test_price_needs_the_listing_entry_binding(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    route = kinds.ModelRoute("testprovider", "a/model")
    rates = {route.model: [0.5, 2.5]}
    assert models_pricing.lookup_price(route, price_env.PROVIDERS) is None
    path = tmp_path / "agent6/models/testprovider.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"pricing": rates}), encoding="utf-8")
    assert models_pricing.lookup_price(route, price_env.PROVIDERS) is None
    price_env.write(
        rates, entry=config.OpenAIProviderEntry(backend="openai", base_url="http://localhost:2")
    )
    assert models_pricing.lookup_price(route, price_env.PROVIDERS) is None
    price_env.write(rates)
    assert models_pricing.lookup_price(route, price_env.PROVIDERS) == models_pricing.Quote(
        models_pricing.Price(0.5, 2.5), route
    )


def test_lookup_price_ignores_malformed_entries(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    price_env.write({"neg": [-1, 2], "short": [1], "infinite": [1.0, float("inf")]})
    for model in ("neg", "short", "infinite", "missing"):
        assert (
            models_pricing.lookup_price(
                kinds.ModelRoute("testprovider", model), price_env.PROVIDERS
            )
            is None
        )


def test_parse_pricing_openrouter_shape() -> None:
    # OpenRouter reports USD per TOKEN as strings; normalized to per-MTok.
    payload = {
        "data": [
            {
                "id": "moonshotai/kimi-k2.6",
                "pricing": {"prompt": "0.00000068", "completion": "0.00000341"},
            },
            {"id": "no-pricing-model"},
            {"id": "bad-pricing", "pricing": {"prompt": "free", "completion": "0"}},
        ]
    }
    got = models_cache._parse_pricing(payload)
    assert (got["moonshotai/kimi-k2.6"].input, got["moonshotai/kimi-k2.6"].output) == (
        pytest.approx(0.68),
        pytest.approx(3.41),
    )
    assert got["moonshotai/kimi-k2.6"].cache_read is None
    assert "no-pricing-model" not in got
    assert "bad-pricing" not in got


@pytest.mark.parametrize(
    ("name", "model", "listed"),
    [
        ("anthropic", "claude-haiku-4-5-20251001", "anthropic/claude-haiku-4.5"),
        ("anthropic", "claude-opus-4-8", "anthropic/claude-opus-4.8"),
        ("openai", "gpt-x", "openai/gpt-x"),
        ("responses", "gpt-x", "openai/gpt-x"),
        ("gateway", "m", "vendor/m"),
    ],
)
def test_reference_quote_names_the_matched_listing(
    name: str, model: str, listed: str, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    providers = config.Config.model_validate(
        {
            "providers": {
                "anthropic": {"backend": "anthropic"},
                "openai": {"backend": "openai"},
                "responses": {"backend": "responses", "base_url": "https://api.openai.com/v1"},
                "openrouter": {"backend": "openai", "base_url": "https://prices.example"},
                "gateway": {
                    "backend": "openai",
                    "base_url": "https://gateway.example",
                    "price_from": "openrouter",
                },
            }
        }
    ).providers
    route = kinds.ModelRoute(name, model)
    rates = {listed: [1.0, 5.0]}
    price_env.write(rates, name="openrouter", entry=providers["openrouter"])

    def quote() -> models_pricing.Quote | None:
        return models_pricing.lookup_price(route, providers)

    assert quote() == models_pricing.Quote(
        models_pricing.Price(1.0, 5.0), kinds.ModelRoute("openrouter", listed)
    )
    price_env.write({**rates, model: [2.0, 3.0]}, name="openrouter", entry=providers["openrouter"])
    assert quote() == models_pricing.Quote(
        models_pricing.Price(2.0, 3.0), kinds.ModelRoute("openrouter", model)
    )


@pytest.mark.parametrize("model", ["x", "claude-x"])
def test_gateway_reference_does_not_choose_between_vendors(
    model: str, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    providers = {
        **price_env.PROVIDERS,
        "local": price_env.ENTRY.model_copy(update={"price_from": "testprovider"}),
    }
    price_env.write({f"one/{model}": [0, 1], f"anthropic/{model}": [1, 2]})
    assert models_pricing.lookup_price(kinds.ModelRoute("local", model), providers) is None


def test_vendor_prefixed_id_precedes_a_normalized_claude_alias(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    model = "claude-haiku-4-5-20251001"
    providers = {
        **price_env.PROVIDERS,
        "direct": config.AnthropicProviderEntry(backend="anthropic", price_from="testprovider"),
    }
    price_env.write({f"anthropic/{model}": [1, 2], "anthropic/claude-haiku-4.5": [3, 4]})
    quote = models_pricing.lookup_price(kinds.ModelRoute("direct", model), providers)
    assert quote == models_pricing.Quote(
        models_pricing.Price(1, 2), kinds.ModelRoute("testprovider", f"anthropic/{model}")
    )


def test_a_listing_that_publishes_cache_rates_prices_them(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """A listing that publishes cache rates prices them.

    OpenRouter publishes each model's own `input_cache_read` and `input_cache_write` (OpenAI's
    cached input is 0.5x and carries no write premium); Anthropic's multipliers (0.1x read, 1.25x
    write) apply only where a listing publishes none.
    """
    from agent6 import budget as agent6_budget

    got = models_cache._parse_pricing(
        {
            "data": [
                {
                    "id": "openai/gpt-x",
                    "pricing": {
                        "prompt": "0.000002",
                        "completion": "0.000008",
                        "input_cache_read": "0.000001",
                        "input_cache_write": "0.000002",
                    },
                }
            ]
        }
    )
    price = got["openai/gpt-x"]
    assert (price.cache_read, price.cache_write) == (pytest.approx(1.0), pytest.approx(2.0))
    assert price.as_list() == pytest.approx([2.0, 8.0, 1.0, 2.0])

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    price_env.write({"openai/gpt-x": [2.0, 8.0, 1.0, 2.0], "listed-only/plain": [2.0, 8.0]})
    listed = agent6_budget.BudgetTracker(
        providers=price_env.PROVIDERS, max_usd=10.0, max_tokens_fallback=-1, max_plan_points=-1
    )
    listed.record(
        route=kinds.ModelRoute("testprovider", "openai/gpt-x"),
        input_tokens=0,
        output_tokens=0,
        cache_read_tokens=1_000_000,
        cache_write_tokens=0,
    )
    assert listed.estimate_usd()[0] == pytest.approx(1.0)  # the listed 0.5x, not 0.1x
    assumed = agent6_budget.BudgetTracker(
        providers=price_env.PROVIDERS, max_usd=10.0, max_tokens_fallback=-1, max_plan_points=-1
    )
    assumed.record(
        route=kinds.ModelRoute("testprovider", "listed-only/plain"),
        input_tokens=0,
        output_tokens=0,
        cache_read_tokens=1_000_000,
        cache_write_tokens=0,
    )
    assert assumed.estimate_usd()[0] == pytest.approx(0.2)  # Anthropic's 0.1x of $2
    assert "cache rates assumed" in assumed.format_summary()
    assert "cache rates assumed" not in listed.format_summary()
