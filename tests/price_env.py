"""Bound model listings for budget and provider tests."""

from __future__ import annotations

import json

from agent6 import config
from agent6.models import listing_store

ENTRY = config.OpenAIProviderEntry(backend="openai", base_url="http://localhost:1")
PROVIDERS = {"testprovider": ENTRY}


def write(
    rates: dict[str, list[float]],
    *,
    name: str = "testprovider",
    entry: config.ProviderEntry = ENTRY,
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
