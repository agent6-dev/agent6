# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Local credential listings for `agent6 connect --status`."""

from __future__ import annotations

import math
import os
import pathlib
import time
from typing import Any
from urllib import parse

from agent6 import secret_store
from agent6.config import (
    ChatGPTProviderEntry,
    ClaudeCodeProviderEntry,
    ConfigError,
    ProviderEntry,
    layer,
)
from agent6.providers import oauth
from agent6.ui.cli import _common


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
        if expired and grant.refresh_token:
            state += " expired"
        elif remaining is not None and not expired:
            state += f" expires in {math.ceil(remaining / 60)} min"
        return source, state
    if entry.token_command:
        return "token command", "ok"
    key = secret_store.resolve_api_key(name, entry.api_key_env, secrets=secrets)
    source = entry.api_key_env or "secrets.toml"
    if key and not (entry.api_key_env and os.environ.get(entry.api_key_env, "").strip()):
        source = "secrets.toml"
    return source, "ok" if key else "key unset"


def show(provider: str, config_paths: tuple[pathlib.Path, ...]) -> int:
    """Print configured providers and local credential state, always returning zero.

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
        rows = [("name", "api_format", "authentication", "state")]
        for name, entry in entries.items():
            if not provider or name == provider:
                source, state = _credential_status(name, entry, secrets)
                rows.append((name, entry.api_format, source, state))
        if provider and provider not in entries:
            _common.warn(f"provider {provider!r} is not configured.")
        widths = [max(len(row[i]) for row in rows) for i in range(3)]
        for name, api_format, source, state in rows:
            print(f"{name:<{widths[0]}}  {api_format:<{widths[1]}}  {source:<{widths[2]}}  {state}")
    print("local check; a provider's acceptance is verified on first use")
    return 0
