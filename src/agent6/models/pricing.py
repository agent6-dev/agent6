# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Cache-only model price lookups, in USD per 1M tokens.

There is no static price table: a price came from a provider's models endpoint (cached by
`agent6.models.cache` under `$XDG_CACHE_HOME/agent6/models/<provider>.json`) or it is
unknown, and reports then render "$?". OpenRouter publishes pricing; Anthropic's models API
does not, so a bare `claude-*` id reads its OpenRouter listing when one is cached
(`claude-haiku-4-5-20251001` -> `anthropic/claude-haiku-4.5`). The module imports only
the stdlib and `agent6.paths`, so `agent6.budget` can use it; reads never touch the network.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import json
import pathlib
import re

from agent6 import paths

__all__ = ["lookup_price"]

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


def _models_cache_dir() -> pathlib.Path | None:
    """Return the models cache directory, or None when no cache dir resolves."""
    with contextlib.suppress(OSError, RuntimeError):
        return paths.cache_dir() / "models"
    return None


def _cache_state() -> tuple[tuple[str, float], ...]:
    """Return (name, mtime) per cache file, the memoization key for the parsed map.

    A fetch that lands mid-process (the preflight refresh) bumps an mtime and invalidates
    the memo.
    """
    root = _models_cache_dir()
    if root is None or not root.is_dir():
        return ()
    out: list[tuple[str, float]] = []
    with contextlib.suppress(OSError):
        for path in sorted(root.glob("*.json")):
            with contextlib.suppress(OSError):
                out.append((path.name, path.stat().st_mtime))
    return tuple(out)


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

    def as_list(self) -> list[float]:
        """Return the cache file's row: `[input, output]`, plus both cache rates when known."""
        if self.cache_read is None or self.cache_write is None:
            return [self.input, self.output]
        return [self.input, self.output, self.cache_read, self.cache_write]


@functools.lru_cache(maxsize=4)
def _load_pricing(
    state: tuple[tuple[str, float], ...],
) -> dict[str, dict[str, Price]]:
    """Return the pricing map of every provider cache file, keyed by provider name.

    Args:
        state: The (name, mtime) pairs from `_cache_state`.

    Returns:
        Per provider, the model to price map; never raises.
    """
    out: dict[str, dict[str, Price]] = {}
    root = _models_cache_dir()
    if root is None:
        return out
    for name, _mtime in state:
        path = root / name
        with contextlib.suppress(OSError, ValueError, TypeError):
            data = json.loads(path.read_text(encoding="utf-8"))
            pricing = data.get("pricing") if isinstance(data, dict) else None
            if not isinstance(pricing, dict):
                continue
            table = out.setdefault(path.stem, {})
            for model, row in pricing.items():
                if (
                    isinstance(model, str)
                    and isinstance(row, list)
                    and len(row) in (2, 4)
                    and all(isinstance(x, (int, float)) and x >= 0 for x in row)
                ):
                    rates = [float(x) for x in row]
                    table[model] = Price(*rates[:2], *rates[2:])
    return out


def _price_in(table: dict[str, Price], model: str) -> Price | None:
    """Return the table's price for the model, by its id or its OpenRouter alias."""
    hit = table.get(model)
    if hit is not None:
        return hit
    alias = _openrouter_alias(model)
    return table.get(alias) if alias is not None else None


def lookup_price(model: str, provider: str = "") -> Price | None:
    """Return the listed price of a model.

    Two providers can list one id at different prices, and the route's own listing bills,
    so a model the named provider does not list is unpriced.

    Args:
        model: The model id.
        provider: The config entry the call went through; "" or a provider with no cached
            listing lets the first listing that has the id answer, by file name.

    Returns:
        The price, or None when unknown.
    """
    tables = _load_pricing(_cache_state())
    if provider and provider in tables:
        return _price_in(tables[provider], model)
    for name in sorted(tables):
        hit = _price_in(tables[name], model)
        if hit is not None:
            return hit
    return None
