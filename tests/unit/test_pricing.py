# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for cache-fetched model pricing (agent6.models.pricing + models_cache)."""

from __future__ import annotations

import json
import pathlib

import pytest

from agent6 import kinds
from agent6.config import model as config_model
from agent6.config import provider_config
from agent6.models import cache as models_cache
from agent6.models import listing_store
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
        rates,
        entry=provider_config.OpenAIProviderEntry(backend="openai", base_url="http://localhost:2"),
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
    providers = config_model.Config.model_validate(
        {
            "providers": {
                "anthropic": {"backend": "anthropic"},
                "openai": {"backend": "openai"},
                "responses": {"backend": "responses", "base_url": "https://api.openai.com/v1"},
                "gateway": {
                    "backend": "openai",
                    "base_url": "https://gateway.example",
                    "price_list_fallback": provider_config.DEFAULT_PRICE_LIST,
                },
            }
        }
    ).providers
    route = kinds.ModelRoute(name, model)
    rates = {listed: [1.0, 5.0]}
    tier = "default" if name in ("openai", "responses") else None
    price_env.write_list(rates, url=provider_config.DEFAULT_PRICE_LIST, tier=tier)

    def quote() -> models_pricing.Quote | None:
        return models_pricing.lookup_price(route, providers)

    assert quote() == models_pricing.Quote(
        models_pricing.Price(1.0, 5.0), kinds.ModelRoute("openrouter.ai", listed)
    )
    price_env.write_list(
        {**rates, model: [2.0, 3.0]}, url=provider_config.DEFAULT_PRICE_LIST, tier=tier
    )
    assert quote() == models_pricing.Quote(
        models_pricing.Price(2.0, 3.0), kinds.ModelRoute("openrouter.ai", model)
    )


@pytest.mark.parametrize("model", ["x", "claude-x"])
def test_gateway_reference_does_not_choose_between_vendors(
    model: str, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    providers = {
        **price_env.PROVIDERS,
        "local": price_env.ENTRY.model_copy(update={"price_list_fallback": price_env.PRICE_LIST}),
    }
    price_env.write_list({f"one/{model}": [0, 1], f"anthropic/{model}": [1, 2]})
    assert models_pricing.lookup_price(kinds.ModelRoute("local", model), providers) is None


@pytest.mark.parametrize(
    ("provider", "model", "unpriced", "rated"),
    [
        ("gateway", "x", "a/x", "b/x"),
        ("gateway", "x", "x", "b/x"),
        ("direct", "claude-x-4-5", "anthropic/claude-x-4-5", "anthropic/claude-x-4.5"),
    ],
)
def test_unpriced_listing_ids_still_decide_the_match(
    provider: str,
    model: str,
    unpriced: str,
    rated: str,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing rate must not make another listing id the price source."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    providers = {
        **price_env.PROVIDERS,
        "gateway": price_env.ENTRY.model_copy(update={"price_list_fallback": price_env.PRICE_LIST}),
        "direct": provider_config.AnthropicProviderEntry(
            backend="anthropic", price_list_fallback=price_env.PRICE_LIST
        ),
    }
    name, listing = models_pricing.price_list_source(price_env.PRICE_LIST)
    models_cache._write_cache(
        listing_store.cache_path(name),
        models_cache.Listing([unpriced, rated], {rated: models_pricing.Price(1, 2)}, {}, {}),
        listing,
    )
    assert models_pricing.lookup_price(kinds.ModelRoute(provider, model), providers) is None


def test_vendor_prefixed_id_precedes_a_normalized_claude_alias(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    model = "claude-haiku-4-5-20251001"
    providers = {
        **price_env.PROVIDERS,
        "direct": provider_config.AnthropicProviderEntry(
            backend="anthropic", price_list_fallback=price_env.PRICE_LIST
        ),
    }
    price_env.write_list({f"anthropic/{model}": [1, 2], "anthropic/claude-haiku-4.5": [3, 4]})
    quote = models_pricing.lookup_price(kinds.ModelRoute("direct", model), providers)
    assert quote == models_pricing.Quote(
        models_pricing.Price(1, 2),
        kinds.ModelRoute(price_env.PRICE_LIST_NAME, f"anthropic/{model}"),
    )


_VERTEX = {
    "deployment": "vertex",
    "base_url": "https://us-east5-aiplatform.googleapis.com/v1/projects/p/locations/us-east5"
    "/publishers/anthropic/models",
    "token_command": ["true"],
}


@pytest.mark.parametrize("settings", [{"base_url": "https://proxy.example"}, _VERTEX])
def test_an_explicit_price_list_prices_claude_behind_any_anthropic_entry(
    settings: dict[str, object], tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Claude on Vertex or behind a proxy stayed unpriced with a fallback price list set."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    providers = config_model.Config.model_validate(
        {
            "providers": {
                "named": {
                    "backend": "anthropic",
                    **settings,
                    "price_list_fallback": price_env.PRICE_LIST,
                },
                "unset": {"backend": "anthropic", **settings},
            }
        }
    ).providers
    listed = "anthropic/claude-sonnet-4.5"
    price_env.write_list({listed: [3.0, 15.0], "z-ai/glm-x": [1.0, 2.0]})
    for model, source in (
        ("claude-sonnet-4-5", listed),
        ("claude-sonnet-4-5-20250929", listed),
        ("glm-x", "z-ai/glm-x"),
    ):
        quote = models_pricing.lookup_price(kinds.ModelRoute("named", model), providers)
        assert quote is not None and quote.source == kinds.ModelRoute(
            price_env.PRICE_LIST_NAME, source
        )
        assert models_pricing.lookup_price(kinds.ModelRoute("unset", model), providers) is None


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


def test_only_a_vendors_own_api_defaults_to_the_public_price_list() -> None:
    """A gateway or a local server takes no fallback list unless the operator names one.

    A local model that shares a hosted model's name would otherwise be charged list prices.
    """
    default = provider_config.DEFAULT_PRICE_LIST
    assert (
        provider_config.AnthropicProviderEntry(backend="anthropic").price_list_fallback == default
    )
    assert provider_config.OpenAIProviderEntry(backend="openai").price_list_fallback == default
    local = provider_config.OpenAIProviderEntry(
        backend="openai", base_url="http://localhost:11434/v1"
    )
    assert local.price_list_fallback == ""
    gateway = provider_config.OpenAIProviderEntry(
        backend="openai", base_url="https://gateway.example/v1"
    )
    assert gateway.price_list_fallback == ""
    off = provider_config.AnthropicProviderEntry(backend="anthropic", price_list_fallback="")
    assert off.price_list_fallback == ""


@pytest.mark.parametrize(
    "address", ["https://prices.example/v1", "prices.example/models", "ftp://prices.example/models"]
)
def test_a_price_list_address_must_name_a_models_listing(address: str) -> None:
    with pytest.raises(ValueError, match=r"price_list_fallback|base_url|http"):
        provider_config.OpenAIProviderEntry(
            backend="openai", base_url="https://gateway.example/v1", price_list_fallback=address
        )


def test_a_providers_own_listing_prices_before_its_fallback_list(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fallback list prices only a model the provider's own listing does not."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    entry = price_env.ENTRY.model_copy(update={"price_list_fallback": price_env.PRICE_LIST})
    providers = {"gateway": entry}
    price_env.write({"m": [1.0, 2.0], "unpriced": [0.0, 0.0]}, name="gateway", entry=entry)
    price_env.write_list({"vendor/m": [3.0, 4.0], "vendor/other": [5.0, 6.0]})
    own = models_pricing.lookup_price(kinds.ModelRoute("gateway", "m"), providers)
    assert own == models_pricing.Quote(
        models_pricing.Price(1.0, 2.0), kinds.ModelRoute("gateway", "m")
    )
    listed = models_pricing.lookup_price(kinds.ModelRoute("gateway", "other"), providers)
    assert listed == models_pricing.Quote(
        models_pricing.Price(5.0, 6.0), kinds.ModelRoute(price_env.PRICE_LIST_NAME, "vendor/other")
    )
