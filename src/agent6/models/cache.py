# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Cached provider model listings for completion and the interactive prompts.

Each provider's list endpoint is queried on demand and cached under
`$XDG_CACHE_HOME/agent6/models/<provider>.json` for a short TTL. The fetch runs in the
operator's own process, never in a jail. `list_models` never raises: on a miss plus a
network failure it falls back to the stale cache, then to an empty list.
"""

from __future__ import annotations

import contextlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx2

from agent6.config import (
    AnthropicProviderEntry,
    ChatGPTProviderEntry,
    ClaudeCodeProviderEntry,
    OpenAIProviderEntry,
    ProviderEntry,
)
from agent6.models.pricing import Price
from agent6.paths import cache_dir
from agent6.providers.types import ProviderError
from agent6.providers.wire import auth_header
from agent6.secrets import load_oauth_tokens

__all__ = ["cached_context_window", "list_models"]

_ANTHROPIC_VERSION = "2023-06-01"
_CACHE_TTL_S = 600
_FETCH_TIMEOUT_S = 1.5  # tab completion waits on this


def _cache_path(provider_name: str) -> Path | None:
    """Return the provider's cache file, or None when the name is not one path component.

    Provider names are config table keys; a `/` or `..` would write outside the cache dir.
    """
    if provider_name in ("", ".", "..") or provider_name != Path(provider_name).name:
        return None
    return cache_dir() / "models" / f"{provider_name}.json"


def _read_cache(path: Path | None) -> list[str] | None:
    """Return the cached model ids, or None when the file is missing or malformed."""
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
    pricing: dict[str, Price],
    context: dict[str, int],
) -> None:
    """Write the listing, plus the pricing and context keys where the provider publishes them."""
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        body: dict[str, object] = {"models": models}
        if pricing:
            body["pricing"] = {m: p.as_list() for m, p in pricing.items()}
        if context:
            body["context"] = dict(context)
        path.write_text(json.dumps(body), encoding="utf-8")
    except OSError:
        pass  # the cache is throwaway; a write failure must not break completion


def _parse_models(payload: object) -> list[str]:
    """Return the model ids of an OpenAI- or Anthropic-style `{"data": [...]}` body."""
    data = payload.get("data") if isinstance(payload, dict) else None
    out: list[str] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                mid = item.get("id")
                if isinstance(mid, str) and mid:
                    out.append(mid)
    return out


def _per_mtok(pricing: dict[str, Any], key: str) -> float | None:
    """Return one OpenRouter per-token rate as USD per 1M tokens.

    Args:
        pricing: The listing's pricing block.
        key: The rate's key.

    Returns:
        The rate, or None when absent, boolean (float(True) is $1) or unparseable.
    """
    raw = pricing.get(key)
    if raw is None or isinstance(raw, bool):
        return None
    try:
        value = float(raw) * 1_000_000
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


def _parse_pricing(payload: object) -> dict[str, Price]:
    """Return per-model prices from an OpenRouter-style `{"data": [...]}` body.

    A model without a usable prompt and completion pair is absent (unknown beats wrong); the
    cache rates ride along only when both parse.
    """
    data = payload.get("data") if isinstance(payload, dict) else None
    out: dict[str, Price] = {}
    if not isinstance(data, list):
        return out
    for item in data:
        if not isinstance(item, dict):
            continue
        mid = item.get("id")
        pricing = item.get("pricing")
        if not (isinstance(mid, str) and mid and isinstance(pricing, dict)):
            continue
        in_mtok, out_mtok = _per_mtok(pricing, "prompt"), _per_mtok(pricing, "completion")
        if in_mtok is None or out_mtok is None:
            continue
        read, write = (
            _per_mtok(pricing, "input_cache_read"),
            _per_mtok(pricing, "input_cache_write"),
        )
        if read is None or write is None:
            out[mid] = Price(in_mtok, out_mtok)
        else:
            out[mid] = Price(in_mtok, out_mtok, read, write)
    return out


def _parse_context(payload: object) -> dict[str, int]:
    """Return per-model context windows in tokens from a `{"data": [...]}` body.

    A model without a positive integer `context_length` is absent (unknown beats wrong).
    """
    data = payload.get("data") if isinstance(payload, dict) else None
    out: dict[str, int] = {}
    if not isinstance(data, list):
        return out
    for item in data:
        if not isinstance(item, dict):
            continue
        mid = item.get("id")
        ctx = item.get("context_length")
        # bool subclasses int: JSON `true` would cache a 1-token window and collapse compaction.
        if (
            isinstance(mid, str)
            and mid
            and isinstance(ctx, int)
            and not isinstance(ctx, bool)
            and ctx > 0
        ):
            out[mid] = ctx
    return out


def _models_endpoint(
    entry: AnthropicProviderEntry | OpenAIProviderEntry, api_key: str | None
) -> tuple[str, dict[str, str]]:
    """Return the (url, headers) of the entry's `/models` listing, auth included.

    The cache fetch and the `connect` key probe share it, so both authenticate as the
    call path does.
    """
    url = entry.base_url.rstrip("/") + "/models"
    headers = dict(entry.extra_headers)
    # Vertex and Azure have no uniform /models endpoint; the caller swallows that failure.
    if isinstance(entry, AnthropicProviderEntry) and entry.deployment == "direct":
        headers["anthropic-version"] = _ANTHROPIC_VERSION
    authed = auth_header(entry.auth_style, api_key or "")
    if authed is not None:
        headers[authed[0]] = authed[1]
    return url, headers


# The backend hides models whose minimal_client_version is above this; the pin names the wire
# feature set agent6 implements and is raised deliberately, never claimed past.
_CHATGPT_CLIENT_VERSION = "1.0.0"


def _chatgpt_models_endpoint(
    provider_name: str, entry: ChatGPTProviderEntry
) -> tuple[str, dict[str, str]]:
    """Return the (url, headers) of the subscription backend's listing.

    An expired access token fails the fetch and the caller falls back to the cache; runs
    refresh tokens, listings do not.

    Raises:
        ProviderError: No sign-in is stored for the provider.
    """
    tokens = load_oauth_tokens(provider_name)
    if tokens is None:
        raise ProviderError(
            f"no ChatGPT sign-in stored for {provider_name!r}; run `agent6 connect {provider_name}`"
        )
    url = f"{entry.base_url.rstrip('/')}/models?client_version={_CHATGPT_CLIENT_VERSION}"
    headers = dict(entry.extra_headers)
    headers["authorization"] = f"Bearer {tokens.access_token}"
    if tokens.account_id:
        headers["chatgpt-account-id"] = tokens.account_id
    headers["originator"] = "agent6"
    return url, headers


def _chatgpt_listing(payload: object) -> tuple[list[str], dict[str, int]]:
    """Return the (ids, context windows) of a ChatGPT `{"models": [...]}` body.

    Hidden entries are left out; a typed hidden slug still works, the backend validates.
    """
    models = payload.get("models") if isinstance(payload, dict) else None
    ids: list[str] = []
    context: dict[str, int] = {}
    if not isinstance(models, list):
        return ids, context
    for item in models:
        if not isinstance(item, dict):
            continue
        slug = item.get("slug")
        if not isinstance(slug, str) or not slug or item.get("visibility") == "hide":
            continue
        ids.append(slug)
        ctx = item.get("context_window")
        if isinstance(ctx, int) and not isinstance(ctx, bool) and ctx > 0:
            context[slug] = ctx
    return ids, context


def _fetch(
    provider_name: str, entry: ProviderEntry, api_key: str | None, timeout_s: float
) -> tuple[list[str], dict[str, Price], dict[str, int]]:
    """Fetch the entry's listing; raises on any failure.

    Returns:
        The ids, the pricing and the context windows.
    """
    if isinstance(entry, ClaudeCodeProviderEntry):
        return [], {}, {}  # no endpoint: the binary resolves model names itself
    if isinstance(entry, ChatGPTProviderEntry):
        url, headers = _chatgpt_models_endpoint(provider_name, entry)
        resp = httpx2.get(url, headers=headers, timeout=timeout_s)
        resp.raise_for_status()
        ids, context = _chatgpt_listing(resp.json())
        return ids, {}, context  # a subscription prices nothing
    url, headers = _models_endpoint(entry, api_key)
    resp = httpx2.get(url, headers=headers, timeout=timeout_s)
    resp.raise_for_status()
    payload = resp.json()
    return _parse_models(payload), _parse_pricing(payload), _parse_context(payload)


@dataclass(frozen=True, slots=True)
class KeyProbeResult:
    """The outcome of a `connect` key probe.

    Attributes:
        ok: The key is usable, or the probe cannot tell.
        status: What the probe found.
        detail: One line for the operator.
    """

    ok: bool
    status: Literal["ok", "auth_failed", "unreachable", "unsupported"]
    detail: str


def probe_provider_key(
    entry: ProviderEntry, api_key: str, *, timeout_s: float = 10.0
) -> KeyProbeResult:
    """Check whether a key authenticates against the entry's `/models`, by a read-only GET.

    A 401 or 403 is a reliable bad key; a 2xx proves validity only where `/models` is auth
    gated, so OpenRouter's public listing is probed at `/key` instead, and another provider
    with a public `/models` would report a false `ok`.

    Args:
        entry: The provider entry.
        api_key: The key to check.
        timeout_s: The request timeout.

    Returns:
        The probe's outcome; `unsupported` where the deployment has no `/models` listing.
    """
    if (
        isinstance(entry, (ChatGPTProviderEntry, ClaudeCodeProviderEntry))
        or entry.deployment != "direct"
    ):
        if isinstance(entry, ChatGPTProviderEntry):
            detail = "ChatGPT signs in via OAuth, not a key"
        elif isinstance(entry, ClaudeCodeProviderEntry):
            detail = "Claude Code signs in with its own login, not a key"
        else:
            detail = "no /models listing for this deployment"
        return KeyProbeResult(ok=True, status="unsupported", detail=detail)
    try:
        url, headers = _models_endpoint(entry, api_key)
    except ProviderError as exc:
        # A credential auth_header refuses (a control char, non-ASCII) is an unusable key.
        return KeyProbeResult(ok=False, status="auth_failed", detail=str(exc)[:200])
    # OpenRouter's /models is public; probe its auth-gated /key, matched on the parsed host.
    host = (urlsplit(entry.base_url).hostname or "").lower()
    if host == "openrouter.ai" or host.endswith(".openrouter.ai"):
        url = entry.base_url.rstrip("/") + "/key"
    try:
        resp = httpx2.get(url, headers=headers, timeout=timeout_s)
    except (httpx2.HTTPError, OSError) as exc:
        return KeyProbeResult(ok=False, status="unreachable", detail=str(exc)[:200])
    if resp.status_code in (401, 403):
        return KeyProbeResult(ok=False, status="auth_failed", detail=f"HTTP {resp.status_code}")
    if resp.status_code >= 400:
        return KeyProbeResult(ok=False, status="unreachable", detail=f"HTTP {resp.status_code}")
    try:
        n = len(_parse_models(resp.json()))
        detail = f"provider returned {n} models" if n else "provider accepted the key"
    except (ValueError, json.JSONDecodeError):
        detail = "provider accepted the key"
    return KeyProbeResult(ok=True, status="ok", detail=detail)


def list_models(
    provider_name: str,
    entry: ProviderEntry,
    api_key: str | None,
    *,
    ttl_s: int = _CACHE_TTL_S,
    timeout_s: float = _FETCH_TIMEOUT_S,
) -> list[str]:
    """Return the model ids the entry offers; never raises.

    Args:
        provider_name: The provider's config name.
        entry: The provider entry.
        api_key: The key, or None for a keyless attempt.
        ttl_s: How old a cache may be before a live fetch.
        timeout_s: The fetch timeout.

    Returns:
        A fresh cache, else the live listing (cached), else the stale cache, else [].
    """
    path = _cache_path(provider_name)
    cached = _read_cache(path)
    age = float("inf")
    if path is not None:
        with contextlib.suppress(OSError):
            age = time.time() - path.stat().st_mtime
    if cached is not None and age < ttl_s:
        return cached
    return fetch_models_live(provider_name, entry, api_key, timeout_s=timeout_s) or cached or []


def fetch_models_live(
    provider_name: str,
    entry: ProviderEntry,
    api_key: str | None,
    *,
    timeout_s: float = _FETCH_TIMEOUT_S,
) -> list[str] | None:
    """Fetch the entry's live listing now, TTL ignored; never raises.

    `models.validate` refuses a model only on a listing this returned.

    Args:
        provider_name: The provider's config name.
        entry: The provider entry.
        api_key: The key, or None for a keyless attempt.
        timeout_s: The fetch timeout.

    Returns:
        The ids, with the cache rewritten, or None on any failure or an empty listing.
    """
    try:
        models, pricing, context = _fetch(provider_name, entry, api_key, timeout_s)
    except (httpx2.HTTPError, ValueError, OSError, ProviderError):
        return None  # ProviderError: a malformed credential, or no ChatGPT sign-in
    if not models:
        return None
    _write_cache(_cache_path(provider_name), models, pricing, context)
    return models


# The price source for bare `claude-*` ids: the public OpenRouter listing, fetched keyless.
# Security: a fixed host, a keyless GET, nothing from the response is executed.
_PRICING_CATALOG_BASE_URL = "https://openrouter.ai/api/v1"


def refresh_pricing_catalog(*, ttl_s: int = _CACHE_TTL_S) -> None:
    """Refresh the OpenRouter catalog cache keyless, under the TTL.

    Bare `claude-*` ids are priced through this catalog, which a config with only
    `[providers.anthropic]` would otherwise never fetch.

    Args:
        ttl_s: How old the cache may be before a fetch.
    """
    entry = OpenAIProviderEntry(api_format="openai", base_url=_PRICING_CATALOG_BASE_URL)
    list_models("openrouter", entry, None, ttl_s=ttl_s)


def cached_models(provider_name: str) -> list[str]:
    """Return the provider's cached model ids without touching the network, or []."""
    return _read_cache(_cache_path(provider_name)) or []


def cached_context_window(provider_name: str, keys: tuple[str, ...]) -> int | None:
    """Return the cached context window of the first key that has one.

    Args:
        provider_name: The provider's config name.
        keys: The raw and normalized model ids, as `models.registry` passes them.

    Returns:
        The window in tokens, or None on any miss.
    """
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
    for key in keys:
        val = ctx.get(key)
        if isinstance(val, int) and not isinstance(val, bool) and val > 0:
            return val
    return None
