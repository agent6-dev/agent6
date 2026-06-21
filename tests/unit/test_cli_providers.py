# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for agent6.cli.providers provider construction (config -> provider)."""

from __future__ import annotations

from unittest.mock import MagicMock

from agent6.cli.providers import (
    _build_role_provider,  # pyright: ignore[reportPrivateUsage]
)
from agent6.config import Config, ModelsConfig, OpenAIProviderEntry, RoleModel
from agent6.providers import OpenAIProvider


def test_build_role_provider_forwards_extra_body_and_headers() -> None:
    # The config -> provider pass-through is a one-liner; pin it so dropping
    # `extra_body=...` (or extra_headers) can't silently stop reaching the wire
    # — that would make `provider` routing / caching config a no-op with no
    # failing test.
    cfg = Config(
        providers={
            "openrouter": OpenAIProviderEntry(
                api_format="openai",
                base_url="https://openrouter.ai/api/v1",
                extra_headers={"X-Title": "agent6"},
                extra_body={"provider": {"sort": "throughput"}},
            )
        },
        models=ModelsConfig(worker=RoleModel(provider="openrouter", model="kimi")),
    )
    prov = _build_role_provider(  # pyright: ignore[reportPrivateUsage]
        cfg, "worker", transcript_sink=MagicMock(), budget=MagicMock()
    )
    assert isinstance(prov, OpenAIProvider)
    assert prov.extra_body == {"provider": {"sort": "throughput"}}
    assert ("X-Title", "agent6") in prov.extra_headers
