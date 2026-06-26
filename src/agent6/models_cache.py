# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Live, cached provider model listings for shell completion + interactive prompts.

Model catalogs change constantly (new OpenRouter routes, new Claude/GPT
snapshots), so agent6 never ships a curated static list that would go stale.
Instead it queries each provider's list endpoint on demand and caches the
result under ``$XDG_CACHE_HOME/agent6/models/<provider>.json`` for a short
TTL, long enough that tab-completion does not hammer the network on every
keystroke, short enough that a freshly-released model shows up within minutes
without the operator hunting for a cache to clear.

This runs in the operator's own shell process (completion / interactive
`agent6 model`), never inside a run sandbox, so a direct HTTP call is fine.
Everything here is best-effort: :func:`list_models` NEVER raises, on a cache
miss + network failure it falls back to the stale cache, then to an empty
list, so completion degrades to free-text rather than breaking the shell.
"""

from __future__ import annotations

import contextlib
import json
import re
import time
from pathlib import Path

import httpx

from agent6.config import AnthropicProviderEntry, Config, ProviderEntry
from agent6.paths import cache_dir
from agent6.providers.wire import auth_header

__all__ = ["compaction_thresholds", "context_window", "list_models", "resolved_adaptive_values"]

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
    path: Path | None,
    models: list[str],
    pricing: dict[str, tuple[float, float]],
    context: dict[str, int],
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
        if context:
            # Per-model context window in tokens, consumed by ``context_window``
            # to size adaptive compaction. Same story as pricing: only providers
            # that publish ``context_length`` (OpenRouter does) populate it.
            body["context"] = dict(context)
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


def _parse_context(payload: object) -> dict[str, int]:
    """Extract per-model context window (tokens) from a ``{"data": [...]}`` body.

    OpenRouter reports ``context_length`` per model; Anthropic's listing does
    not, so those models simply fall back to the bundled table. Models without
    a usable positive integer are absent (unknown beats wrong)."""
    data = payload.get("data") if isinstance(payload, dict) else None
    out: dict[str, int] = {}
    if not isinstance(data, list):
        return out
    for item in data:
        if not isinstance(item, dict):
            continue
        mid = item.get("id")
        ctx = item.get("context_length")
        if isinstance(mid, str) and mid and isinstance(ctx, int) and ctx > 0:
            out[mid] = ctx
    return out


def _fetch(
    entry: ProviderEntry, api_key: str | None, timeout_s: float
) -> tuple[list[str], dict[str, tuple[float, float]], dict[str, int]]:
    url = entry.base_url.rstrip("/") + "/models"
    headers = dict(entry.extra_headers)
    # Anthropic's direct /models needs the version header; Vertex/Azure have no
    # uniform /models endpoint, so listing there is best-effort (the caller
    # swallows the failure). Auth uses the same style the call path uses.
    if isinstance(entry, AnthropicProviderEntry) and entry.deployment == "direct":
        headers["anthropic-version"] = _ANTHROPIC_VERSION
    authed = auth_header(entry.auth_style, api_key or "")
    if authed is not None:
        headers[authed[0]] = authed[1]
    resp = httpx.get(url, headers=headers, timeout=timeout_s)
    resp.raise_for_status()
    payload = resp.json()
    return _parse_models(payload), _parse_pricing(payload), _parse_context(payload)


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
        models, pricing, context = _fetch(entry, api_key, timeout_s)
    except (httpx.HTTPError, ValueError, OSError):
        return cached or []
    if models:
        _write_cache(path, models, pricing, context)
        return models
    return cached or []


def cached_models(provider_name: str) -> list[str]:
    """Model ids from the on-disk cache only (no network). ``[]`` if nothing has
    been cached for *provider_name* yet. For instant typeahead suggestions; pair
    with :func:`list_models` (in a worker) to refresh from the live listing."""
    return _read_cache(_cache_path(provider_name)) or []


# --- context window + adaptive compaction sizing --------------------------

# Curated context windows (TOKENS) for models we have tested or that are
# popular, used to size adaptive compaction without a network round-trip and to
# cover providers whose listing omits the window (Anthropic's /models does not
# report it). The live cache (``context_length`` from the provider listing)
# covers everything else; this table just guarantees good behaviour offline and
# on the first run before any listing has been fetched. Keep ids canonical (no
# date/`:tag` suffix -- ``_normalize_model_id`` strips those before matching).
BUNDLED_CONTEXT_WINDOWS: dict[str, int] = {
    # Anthropic (standard window; the 1M-context beta is opt-in, so pin it
    # explicitly in [workflow] if you enable it).
    "claude-opus-4-8": 200_000,
    "claude-sonnet-4-6": 200_000,
    "claude-sonnet-4-5": 200_000,
    "claude-haiku-4-5": 200_000,
    "claude-3-5-sonnet": 200_000,
    "claude-3-5-haiku": 200_000,
    # OpenRouter open-weights we bench against (cross-checked against the live
    # listing; the cache supersedes these once fetched).
    "moonshotai/kimi-k2.6": 262_144,
    "moonshotai/kimi-k2": 131_072,
    "qwen/qwen3-coder": 1_048_576,
    "qwen/qwen3-coder-30b-a3b-instruct": 160_000,
    "z-ai/glm-4.6": 202_752,
    "z-ai/glm-5.2": 1_048_576,
    "deepseek/deepseek-v3.2-exp": 163_840,
}

# Adaptive sizing. tokens ~= chars/4 (matches the loop's ``context_chars``
# approximation). Tier-1 elides old tool_results once they pass ~45% of the
# window; tier-2 summarises+restarts at ~80%, leaving headroom for the next
# turn's output and the summary call itself.
_CHARS_PER_TOKEN = 4
_DROP_FRACTION = 0.45
_SUMMARISE_FRACTION = 0.80
# Used when the window is unknown: the historical fixed defaults, so behaviour
# is unchanged for unsizable models. Mirrors workflows._compaction
# DROP_BLOCKS_AT_CHARS / SUMMARISE_AT_CHARS.
_FALLBACK_DROP_CHARS = 256_000
_FALLBACK_SUMMARISE_CHARS = 768_000


def _normalize_model_id(model_id: str) -> str:
    """Strip a trailing ``-YYYYMMDD`` snapshot date or ``:tag`` so dated/tagged
    ids (``claude-haiku-4-5-20251001``, ``qwen/qwen3-coder:free``) match the
    canonical bundled key."""
    base = model_id.split(":", 1)[0]
    return re.sub(r"-\d{8}$", "", base)


def _bundled_context_window(model_id: str) -> int | None:
    if model_id in BUNDLED_CONTEXT_WINDOWS:
        return BUNDLED_CONTEXT_WINDOWS[model_id]
    return BUNDLED_CONTEXT_WINDOWS.get(_normalize_model_id(model_id))


def _cached_context_window(provider_name: str, model_id: str) -> int | None:
    """Read ``context_length`` for *model_id* from the provider's model cache,
    if a listing has been fetched. Best-effort: returns None on any miss."""
    path = _cache_path(provider_name)
    if path is None:
        return None
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    ctx = body.get("context") if isinstance(body, dict) else None
    if not isinstance(ctx, dict):
        return None
    for key in (model_id, _normalize_model_id(model_id)):
        val = ctx.get(key)
        if isinstance(val, int) and val > 0:
            return val
    return None


def context_window(provider_name: str, model_id: str) -> int | None:
    """Best-effort context window (tokens) for a configured model. Never raises.

    Bundled table (curated; tested models + Anthropic) first, then the live
    model cache (``context_length`` from the provider listing, populated by
    completion / ``agent6 model``), then None. Reads only -- never triggers a
    network fetch -- so it is safe and fast on the run path.
    """
    return _bundled_context_window(model_id) or _cached_context_window(provider_name, model_id)


def compaction_thresholds(
    provider_name: str,
    model_id: str,
    *,
    drop_override: int | None,
    summarise_override: int | None,
) -> tuple[int, int]:
    """Effective ``(compact_drop_at_chars, compact_summarise_at_chars)``.

    Explicit config wins (both set, by construction -- the config validator
    requires both-or-neither). Otherwise size from the model's context window
    (tier-1 ~45%, tier-2 ~80%); if the window is unknown, the historical fixed
    defaults. Never raises.
    """
    if drop_override is not None and summarise_override is not None:
        return drop_override, summarise_override
    ctx = context_window(provider_name, model_id)
    if ctx is None or ctx <= 0:
        return _FALLBACK_DROP_CHARS, _FALLBACK_SUMMARISE_CHARS
    drop = int(ctx * _CHARS_PER_TOKEN * _DROP_FRACTION)
    summarise = int(ctx * _CHARS_PER_TOKEN * _SUMMARISE_FRACTION)
    return drop, summarise


def resolved_adaptive_values(cfg: Config) -> dict[str, object]:
    """Config settings whose effective value is resolved at runtime, so a UI
    (`config show`, the TUI/web config page) can display the real number rather
    than the unset/adaptive placeholder. Currently the adaptive compaction
    thresholds, sized from the worker model's context window; empty when no
    worker model is configured."""
    rm = cfg.models.resolve("worker")
    if rm is None:
        return {}
    drop, summarise = compaction_thresholds(
        rm.provider,
        rm.model,
        drop_override=cfg.workflow.compact_drop_at_chars,
        summarise_override=cfg.workflow.compact_summarise_at_chars,
    )
    return {
        "workflow.compact_drop_at_chars": drop,
        "workflow.compact_summarise_at_chars": summarise,
    }
