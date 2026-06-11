# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Live, cached provider model listings for shell completion + interactive prompts.

Model catalogs change constantly (new OpenRouter routes, new Claude/GPT
snapshots), so agent6 never ships a curated static list that would go stale.
Instead it queries each provider's list endpoint on demand and caches the
result under ``$XDG_CACHE_HOME/agent6/models/<provider>.json`` for a short
TTL — long enough that tab-completion does not hammer the network on every
keystroke, short enough that a freshly-released model shows up within minutes
without the operator hunting for a cache to clear.

This runs in the operator's own shell process (completion / interactive
`agent6 model`), never inside a run sandbox, so a direct HTTP call is fine.
Everything here is best-effort: :func:`list_models` NEVER raises — on a cache
miss + network failure it falls back to the stale cache, then to an empty
list, so completion degrades to free-text rather than breaking the shell.
"""

from __future__ import annotations

import contextlib
import json
import time
from pathlib import Path

import httpx

from agent6.config import AnthropicProviderEntry, ProviderEntry
from agent6.paths import cache_dir

__all__ = ["list_models"]

_ANTHROPIC_MODELS_URL = "https://api.anthropic.com/v1/models"
_ANTHROPIC_VERSION = "2023-06-01"
_CACHE_TTL_S = 600  # 10 minutes
_FETCH_TIMEOUT_S = 1.5  # keep tab-completion snappy


def _cache_path(provider_name: str) -> Path | None:
    """Cache file for *provider_name*, or None when the name is not a safe
    single path component. Provider names are config table keys; guard against
    ``/`` or ``..`` so a crafted name can't write the cache outside cache_dir().
    """
    if provider_name in ("", ".", "..") or provider_name != Path(provider_name).name:
        return None
    return cache_dir() / "models" / f"{provider_name}.json"


def _read_cache(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    models = data.get("models") if isinstance(data, dict) else None
    if isinstance(models, list) and all(isinstance(m, str) for m in models):
        return models
    return None


def _write_cache(
    path: Path | None, models: list[str], pricing: dict[str, tuple[float, float]]
) -> None:
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        body: dict[str, object] = {"models": models}
        if pricing:
            # Consumed by agent6.pricing.lookup_price (USD per 1M tokens,
            # [input, output]). Only providers that publish pricing on their
            # models endpoint (OpenRouter does, Anthropic does not) get this
            # key; there is deliberately no static fallback anywhere.
            body["pricing"] = {m: [p[0], p[1]] for m, p in pricing.items()}
        path.write_text(json.dumps(body), encoding="utf-8")
    except OSError:
        pass  # cache is throwaway; a write failure must not break completion


def _parse_models(payload: object) -> list[str]:
    """Extract model ids from an OpenAI-/Anthropic-style ``{"data": [...]}`` body."""
    data = payload.get("data") if isinstance(payload, dict) else None
    out: list[str] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                mid = item.get("id")
                if isinstance(mid, str) and mid:
                    out.append(mid)
    return out


def _parse_pricing(payload: object) -> dict[str, tuple[float, float]]:
    """Extract per-model pricing from an OpenRouter-style ``{"data": [...]}`` body.

    OpenRouter reports ``pricing.prompt``/``pricing.completion`` as USD per
    TOKEN strings; normalize to USD per 1M tokens. Models without a usable
    pair are simply absent (unknown beats wrong)."""
    data = payload.get("data") if isinstance(payload, dict) else None
    out: dict[str, tuple[float, float]] = {}
    if not isinstance(data, list):
        return out
    for item in data:
        if not isinstance(item, dict):
            continue
        mid = item.get("id")
        pricing = item.get("pricing")
        if not (isinstance(mid, str) and mid and isinstance(pricing, dict)):
            continue
        try:
            in_mtok = float(pricing.get("prompt", "")) * 1_000_000
            out_mtok = float(pricing.get("completion", "")) * 1_000_000
        except (TypeError, ValueError):
            continue
        if in_mtok >= 0 and out_mtok >= 0:
            out[mid] = (in_mtok, out_mtok)
    return out


def _fetch(
    entry: ProviderEntry, api_key: str | None, timeout_s: float
) -> tuple[list[str], dict[str, tuple[float, float]]]:
    if isinstance(entry, AnthropicProviderEntry):
        url = _ANTHROPIC_MODELS_URL
        headers = {"anthropic-version": _ANTHROPIC_VERSION}
        if api_key:
            headers["x-api-key"] = api_key
    else:
        url = entry.base_url.rstrip("/") + "/models"
        headers = dict(entry.extra_headers)
        if api_key:
            headers["authorization"] = f"Bearer {api_key}"
    resp = httpx.get(url, headers=headers, timeout=timeout_s)
    resp.raise_for_status()
    payload = resp.json()
    return _parse_models(payload), _parse_pricing(payload)


def list_models(
    provider_name: str,
    entry: ProviderEntry,
    api_key: str | None,
    *,
    ttl_s: int = _CACHE_TTL_S,
    timeout_s: float = _FETCH_TIMEOUT_S,
) -> list[str]:
    """Best-effort list of model ids offered by *entry*. Never raises.

    Returns a fresh cache when one exists within *ttl_s*; otherwise fetches
    live, rewrites the cache, and returns it. On any failure (no key, network
    error, bad payload) falls back to a stale cache, then an empty list.
    """
    path = _cache_path(provider_name)
    cached = _read_cache(path)
    age = float("inf")
    if path is not None:
        with contextlib.suppress(OSError):
            age = time.time() - path.stat().st_mtime
    if cached is not None and age < ttl_s:
        return cached
    try:
        models, pricing = _fetch(entry, api_key, timeout_s)
    except (httpx.HTTPError, ValueError, OSError):
        return cached or []
    if models:
        _write_cache(path, models, pricing)
        return models
    return cached or []
