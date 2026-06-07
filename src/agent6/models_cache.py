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


def _cache_path(provider_name: str) -> Path:
    return cache_dir() / "models" / f"{provider_name}.json"


def _read_cache(path: Path) -> list[str] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    models = data.get("models") if isinstance(data, dict) else None
    if isinstance(models, list) and all(isinstance(m, str) for m in models):
        return models
    return None


def _write_cache(path: Path, models: list[str]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"models": models}), encoding="utf-8")
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


def _fetch(entry: ProviderEntry, api_key: str | None, timeout_s: float) -> list[str]:
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
    return _parse_models(resp.json())


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
    try:
        age = time.time() - path.stat().st_mtime
    except OSError:
        age = float("inf")
    if cached is not None and age < ttl_s:
        return cached
    try:
        models = _fetch(entry, api_key, timeout_s)
    except (httpx.HTTPError, ValueError, OSError):
        return cached or []
    if models:
        _write_cache(path, models)
        return models
    return cached or []
