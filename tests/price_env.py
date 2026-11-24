"""Bound model listings for budget and provider tests."""

from __future__ import annotations

import json

from agent6.config import provider_config
from agent6.models import listing_store, pricing

ENTRY = provider_config.OpenAIProviderEntry(backend="openai", base_url="http://localhost:1")
PROVIDERS = {"testprovider": ENTRY}
# A fallback price list address; quotes from it cite its host.
PRICE_LIST = "https://prices.example/v1/models"
PRICE_LIST_NAME = "prices.example"


def write(
    rates: dict[str, list[float]],
    *,
    name: str = "testprovider",
    entry: provider_config.ProviderEntry = ENTRY,
    tier: str | None = None,
) -> None:
    """Write one price table under the test's XDG cache home."""
    path = listing_store.cache_path(name)
    assert path is not None
    path.parent.mkdir(parents=True, exist_ok=True)
    table = {"tier_pricing": {tier: rates}} if tier is not None else {"pricing": rates}
    path.write_text(
        json.dumps({"binding": listing_store.binding(entry), "models": list(rates), **table}),
        encoding="utf-8",
    )


def write_list(
    rates: dict[str, list[float]], *, url: str = PRICE_LIST, tier: str | None = None
) -> None:
    """Write one fallback price list's table under the test's XDG cache home."""
    name, entry = pricing.price_list_source(url)
    write(rates, name=name, entry=entry, tier=tier)
