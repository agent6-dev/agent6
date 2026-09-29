# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for agent6.app.providers provider construction (config -> provider)."""

from __future__ import annotations

from unittest import mock

from agent6.app import providers
from agent6.config import Config, ModelsConfig, OpenAIProviderEntry, RoleModel
from agent6.providers import OpenAIProvider


def test_build_role_provider_forwards_extra_body_and_headers() -> None:
    # Dropping `extra_body` or `extra_headers` from the pass-through would make the config a no-op.
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
    prov = providers.build_role_provider(
        cfg, "worker", transcript_sink=mock.MagicMock(), budget=mock.MagicMock()
    )
    assert isinstance(prov, OpenAIProvider)
    assert prov.extra_body == {"provider": {"sort": "throughput"}}
    assert ("X-Title", "agent6") in prov.extra_headers


def test_reviewer_family_builders_stamp_their_own_seats() -> None:
    """The reviser, summariser and a bare-persona seat stamp their own seats.

    They share the reviewer route; a transcript still tells which actor made a call.
    """
    from agent6.config import PromptConfig, ReviewConfig

    cfg = Config(
        providers={"o": OpenAIProviderEntry(api_format="openai", base_url="https://x/v1")},
        models=ModelsConfig(
            worker=RoleModel(provider="o", model="m"),
            reviewer=RoleModel(provider="o", model="m"),
        ),
        review=ReviewConfig(trigger="on_verify_fail", seats=("security",)),
        prompt=PromptConfig(revise_prompt="auto"),
    )

    sink = mock.MagicMock()
    providers.build_prompt_reviser_provider(
        cfg, transcript_sink=sink, budget=mock.MagicMock(), events=mock.MagicMock()
    )
    assert sink.for_seat.call_args == mock.call("prompt_reviser")

    sink = mock.MagicMock()
    providers.reviewer_seat_provider(
        cfg, "summariser", transcript_sink=sink, budget=mock.MagicMock(), events=mock.MagicMock()
    )
    assert sink.for_seat.call_args == mock.call("summariser")

    sink = mock.MagicMock()
    providers.build_review_seats(cfg, transcript_sink=sink, budget=mock.MagicMock(), n=1)
    assert sink.for_seat.call_args == mock.call("review:security")

    # The role builders keep stamping the role itself.
    sink = mock.MagicMock()
    providers.build_role_provider(cfg, "worker", transcript_sink=sink, budget=mock.MagicMock())
    assert sink.for_seat.call_args == mock.call("worker")
