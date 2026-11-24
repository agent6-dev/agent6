# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Resolve token rates from the one configured listing that prices a route."""

from __future__ import annotations

import dataclasses
import math
import re
from collections.abc import Mapping

from agent6 import config, kinds
from agent6.models import listing_store

_CLAUDE_DATE_SUFFIX_RE = re.compile(r"-20\d{6}$")
_CLAUDE_TRAILING_VERSION_RE = re.compile(r"-(\d+)-(\d+)$")


def _openrouter_alias(model: str) -> str | None:
    """Return the OpenRouter listing id for a bare `claude-*` model id.

    Drops a `-YYYYMMDD` suffix, then dots a trailing `-N-M` version. An id the rules do
    not cover (`claude-3-5-sonnet`) yields a candidate that misses the price map.

    Args:
        model: The model id.

    Returns:
        The candidate id, or None for a namespaced or non-Claude id.
    """
    if "/" in model or not model.startswith("claude-"):
        return None
    base = _CLAUDE_DATE_SUFFIX_RE.sub("", model)
    base = _CLAUDE_TRAILING_VERSION_RE.sub(r"-\1.\2", base)
    return f"anthropic/{base}"


@dataclasses.dataclass(frozen=True, slots=True)
class Price:
    """A model's listed rates in USD per 1M tokens.

    Attributes:
        input: The fresh input rate.
        output: The output rate.
        cache_read: The cache read rate, or None when the listing carries none (the cost
            arithmetic then applies Anthropic's multipliers to the input rate).
        cache_write: The cache write rate, or None likewise.
    """

    input: float
    output: float
    cache_read: float | None = None
    cache_write: float | None = None

    def estimate(self, fresh: int, output: int, read: int, write: int) -> float:
        """Return the quoted cost of one call, using cache ratios only for absent cache rates."""
        read_rate = self.cache_read if self.cache_read is not None else self.input * 0.1
        write_rate = self.cache_write if self.cache_write is not None else self.input * 1.25
        return (
            fresh * self.input + output * self.output + read * read_rate + write * write_rate
        ) / 1e6

    def as_list(self) -> list[float]:
        """Return the cache file's row: `[input, output]`, plus both cache rates when known."""
        if self.cache_read is None or self.cache_write is None:
            return [self.input, self.output]
        return [self.input, self.output, self.cache_read, self.cache_write]


@dataclasses.dataclass(frozen=True, slots=True)
class Quote:
    """Token rates and the provider/model listing that published them."""

    rates: Price
    source: kinds.ModelRoute


def _reference_id(model: str, entry: config.PricedProviderEntry, models: set[str]) -> str | None:
    """Return the reference listing's id for a model it does not list by the route's id.

    A direct endpoint names its vendor; a gateway takes the one listed id that ends in the
    model. An `anthropic` entry with neither takes the catalog spelling of a Claude id.

    Args:
        model: The route's bare model id.
        entry: The provider entry that receives the call.
        models: Every id the reference lists, including those without rates.

    Returns:
        The candidate id, or None when no id or more than one id matches.
    """
    vendor = entry.direct_vendor
    literal = (
        [f"{vendor}/{model}"] if vendor else [k for k in models if k.partition("/")[2] == model]
    )
    listed = [candidate for candidate in literal if candidate in models]
    if listed:
        return listed[0] if len(listed) == 1 else None
    return _openrouter_alias(model) if entry.backend == "anthropic" else None


def lookup_price(
    route: kinds.ModelRoute,
    providers: Mapping[str, config.ProviderEntry],
    *,
    served_tier: str | None = None,
) -> Quote | None:
    """Read a bound price without touching the network.

    The standard tier (`default` on the wire) reads the listing's unqualified rates when the
    listing has no row for that tier; `flex` and `priority` read only their own rows.

    Args:
        route: The provider and model that receive the call.
        providers: The current provider entries.
        served_tier: The observed wire tier, or None to use the request tier.

    Returns:
        The exact quote, a vendor-prefixed reference quote, or None when unpriced.
    """
    entry = providers.get(route.provider)
    if not isinstance(entry, config.PricedProviderEntry):
        return None
    source = entry.price_from or route.provider
    listing = providers.get(source)
    if listing is None:
        return None
    data = listing_store.read(listing_store.cache_path(source))
    if data.get("binding") != listing_store.binding(listing):
        return None
    tier = served_tier
    if tier is None and isinstance(
        entry, (config.OpenAIProviderEntry, config.ResponsesProviderEntry)
    ):
        tier = entry.wire_tier
    models = set(listing_store.model_ids(data) or ())
    model = route.model
    if model not in models and source != route.provider and "/" not in model:
        model = _reference_id(model, entry, models)
    if model is None or model not in models:
        return None
    tiers = data.get("tier_pricing")
    listed = tiers.get(tier) if tier is not None and isinstance(tiers, dict) else None
    row = listed.get(model) if isinstance(listed, dict) else None
    if row is None and tier in (None, "default"):
        prices = data.get("pricing")
        row = prices.get(model) if isinstance(prices, dict) else None
    if (
        not isinstance(row, list)
        or len(row) not in (2, 4)
        or not all(
            isinstance(x, (int, float)) and not isinstance(x, bool) and x >= 0 and math.isfinite(x)
            for x in row
        )
    ):
        return None
    return Quote(Price(*row), kinds.ModelRoute(source, model))
