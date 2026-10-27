# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Local credential state and free live checks for `agent6 connect --status`."""

from __future__ import annotations

import math
import os
import pathlib
import time
from typing import Any
from urllib import parse

import httpx2

from agent6 import secret_store
from agent6.config import (
    ChatGPTProviderEntry,
    ClaudeCodeProviderEntry,
    ConfigError,
    ProviderEntry,
    layer,
)
from agent6.models import cache
from agent6.providers import claude_code, oauth, token_command, types
from agent6.ui.cli import _common

_LIVE_TIMEOUT_S = 7.0


def _token_life(remaining: float) -> str:
    """Return the remaining life in the largest two units, or expired."""
    if remaining <= 0:
        return "expired"
    hours, minutes = divmod(math.ceil(remaining / 60), 60)
    days, hours = divmod(hours, 24)
    if days:
        return f"expires in {days} d {hours} h"
    if hours:
        return f"expires in {hours} h {minutes} min"
    return f"expires in {minutes} min"


def _credential_status(name: str, entry: ProviderEntry, secrets: dict[str, Any]) -> tuple[str, str]:
    """Return the credential source and local state without using the credential."""
    if isinstance(entry, ClaudeCodeProviderEntry) or entry.auth_style == "none":
        return "none", "ok"
    if isinstance(entry, ChatGPTProviderEntry) or entry.oauth is not None:
        source = (
            "ChatGPT sign-in"
            if isinstance(entry, ChatGPTProviderEntry)
            else parse.urlsplit(oauth.profile_for(entry).issuer).netloc
        )
        try:
            grant = oauth.load_bound_grant(name, entry, secrets=secrets)
        except oauth.BoundGrantError as exc:
            return source, exc.cause
        remaining = None if grant.expires_at is None else grant.expires_at - time.time()
        expired = remaining is not None and remaining <= 0
        state = "expired" if expired and not grant.refresh_token else "ok"
        if grant.account_id:
            state += f" {grant.account_id}"
        if remaining is not None and (not expired or grant.refresh_token):
            state += f" {_token_life(remaining)}"
        return source, state
    if entry.token_command:
        return "token command", "ok"
    key = secret_store.resolve_api_key(name, entry.api_key_env, secrets=secrets)
    source = entry.api_key_env or "secrets.toml"
    if key and not (entry.api_key_env and os.environ.get(entry.api_key_env, "").strip()):
        source = "secrets.toml"
    return source, "ok" if key else "key unset"


def _live_status(name: str, entry: ProviderEntry, secrets: dict[str, Any], state: str) -> str:
    """Return a free check's result without refreshing, exchanging or spending."""
    if state.split(" ", 1)[0] != "ok":
        return f"skipped: {state}"
    try:
        if isinstance(entry, ClaudeCodeProviderEntry):
            refusal = claude_code.login_status(entry.binary, timeout_s=_LIVE_TIMEOUT_S)
            live = f"refused: {' '.join(refusal.split())[:200]}" if refusal else "verified"
        elif entry.deployment != "direct":
            live = "skipped: no /models listing for this deployment"
        elif isinstance(entry, ChatGPTProviderEntry) or entry.oauth or entry.auth_style == "none":
            cache._fetch(name, entry, None, _LIVE_TIMEOUT_S, refresh=False)
            live = "verified"
        else:
            key = (
                token_command.CommandToken(
                    entry.token_command, run_timeout_s=_LIVE_TIMEOUT_S
                ).token()
                if entry.token_command
                else secret_store.resolve_api_key(name, entry.api_key_env, secrets=secrets)
            )
            result = cache.probe_provider_key(entry, key or "", timeout_s=_LIVE_TIMEOUT_S)
            label = {
                "ok": "verified",
                "auth_failed": "refused",
                "unreachable": "unreachable",
                "unsupported": "skipped",
            }[result.status]
            if result.detail.startswith("HTTP "):
                label = "refused"
            detail = " ".join(result.detail.split())[:200] or "request failed"
            live = label if result.status == "ok" else f"{label}: {detail}"
    except httpx2.HTTPStatusError as exc:
        live = f"refused: HTTP {exc.response.status_code}"
    except (httpx2.HTTPError, OSError, ValueError) as exc:
        live = f"unreachable: {' '.join(str(exc).split())[:200] or type(exc).__name__}"
    except (types.ProviderError, secret_store.SecretsError) as exc:
        live = f"refused: {' '.join(str(exc).split())[:200]}"
    return live


def show(provider: str, config_paths: tuple[pathlib.Path, ...]) -> int:
    """Print configured providers, local state and live checks, always returning zero.

    Args:
        provider: A single configured name; empty lists all entries in config order.
        config_paths: Explicit config overlays.

    Returns:
        Zero, including when a credential or config cannot be read.
    """
    try:
        entries = layer.load_effective(pathlib.Path.cwd(), config_paths).config.providers
        secrets = secret_store.load_secrets()
    except (ConfigError, secret_store.SecretsError) as exc:
        _common.error(str(exc))
    else:
        rows = [("name", "api_format", "authentication", "state", "live")]
        for name, entry in entries.items():
            if not provider or name == provider:
                source, state = _credential_status(name, entry, secrets)
                live = _live_status(name, entry, secrets, state)
                rows.append((name, entry.api_format, source, state, live))
        if provider and provider not in entries:
            _common.warn(f"provider {provider!r} is not configured.")
        widths = [max(len(row[i]) for row in rows) for i in range(4)]
        for name, api_format, source, state, live in rows:
            print(
                f"{name:<{widths[0]}}  {api_format:<{widths[1]}}  {source:<{widths[2]}}"
                f"  {state:<{widths[3]}}  {live}"
            )
    return 0
