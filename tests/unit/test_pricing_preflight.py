# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The preflight fetches the OpenRouter pricing catalog for bare claude-* ids.

Prices for direct-Anthropic ids live only in that catalog (pricing's alias);
a config with just [providers.anthropic] refreshed nothing that carries
prices, so the $ cap ran unpriced on a cold cache."""

from __future__ import annotations

import pytest

from agent6.app import _setup
from agent6.config import Config


def _cfg(model: str, provider_block: dict[str, object]) -> Config:
    return Config.model_validate(
        {
            "providers": provider_block,
            "models": {"worker": {"provider": next(iter(provider_block)), "model": model}},
        }
    )


def test_claude_only_config_refreshes_the_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[bool] = []

    def _key(*_a: object, **_k: object) -> str:
        return "sk-test"

    def _models(*_a: object, **_k: object) -> list[str]:
        return []

    monkeypatch.setattr(_setup, "refresh_pricing_catalog", lambda: called.append(True))
    monkeypatch.setattr(_setup, "load_secrets", dict)
    monkeypatch.setattr(_setup, "resolve_api_key", _key)
    monkeypatch.setattr(_setup, "list_models", _models)
    cfg = _cfg(
        "claude-opus-5",
        {"anthropic": {"api_format": "anthropic", "api_key_env": "X_KEY"}},
    )
    assert _setup.check_provider_keys(cfg) is None
    assert called, "bare claude-* with no openrouter provider must refresh the catalog"


def test_openrouter_config_does_not_double_refresh(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[bool] = []

    def _key(*_a: object, **_k: object) -> str:
        return "sk-test"

    def _models(*_a: object, **_k: object) -> list[str]:
        return []

    monkeypatch.setattr(_setup, "refresh_pricing_catalog", lambda: called.append(True))
    monkeypatch.setattr(_setup, "load_secrets", dict)
    monkeypatch.setattr(_setup, "resolve_api_key", _key)
    monkeypatch.setattr(_setup, "list_models", _models)
    # A BARE claude-* id through openrouter: the one shape where the guard
    # decides (a non-claude model skips the refresh before the guard is read).
    cfg = _cfg(
        "claude-opus-5",
        {
            "openrouter": {
                "api_format": "openai",
                "api_key_env": "X_KEY",
                "base_url": "https://openrouter.ai/api/v1",
            }
        },
    )
    assert _setup.check_provider_keys(cfg) is None
    assert not called, "a configured openrouter provider already refreshes with a key"
