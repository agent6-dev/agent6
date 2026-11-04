# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for agent6.app.role_providers provider construction (config -> provider)."""

from __future__ import annotations

import dataclasses
import pathlib
import time
from unittest import mock

import pytest

from agent6.app import role_providers
from agent6.config import (
    ChatGPTProviderEntry,
    Config,
    ModelsConfig,
    OpenAIProviderEntry,
    RoleModel,
    secret_store,
)
from agent6.providers import OpenAIProvider, chatgpt, oauth, oauth_protocol, provider_types


@pytest.mark.parametrize("state", ["old_format", "rebound"])
def test_chatgpt_binding_refuses_before_provider_construction(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, state: str
) -> None:
    """Inference and usage cannot acquire a provider from an unbound sign-in."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    entry = ChatGPTProviderEntry(api_format="chatgpt")
    if state == "old_format":
        secret_store.save_secret(
            "work",
            "",
            extra={
                "oauth_access_token": "AT",
                "oauth_refresh_token": "RT",
                "oauth_expires_at": str(time.time() + 3600),
                "oauth_account_id": "acct",
            },
        )
    else:
        grant = oauth.stored_grant_for(
            oauth_protocol.TokenGrant("AT", "RT", 3600), entry, now=time.time()
        )
        secret_store.save_oauth_grant(
            "work", dataclasses.replace(grant, binding="other", account_id="acct")
        )
    cfg = Config(
        providers={"work": entry}, models=ModelsConfig(worker=RoleModel(provider="work", model="m"))
    )
    message = (
        "No ChatGPT sign-in stored for provider 'work'"
        if state == "old_format"
        else "other provider settings"
    )
    with pytest.raises(provider_types.ProviderError, match=message) as refused:
        role_providers.build_role_provider(
            cfg, "worker", transcript_sink=mock.MagicMock(), budget=mock.MagicMock()
        )
    assert "agent6 connect work" in str(refused.value)


def test_chatgpt_construction_reads_account_through_shared_holder(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The adapter gets its account header from the bound holder, not stored metadata."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    entry = ChatGPTProviderEntry(api_format="chatgpt")
    grant = oauth.stored_grant_for(
        oauth_protocol.TokenGrant("AT", "RT", 3600), entry, now=time.time()
    )
    secret_store.save_oauth_grant("work", dataclasses.replace(grant, account_id="stored"))

    def account(self: oauth.OAuthCredential) -> str:
        return "from-holder"

    monkeypatch.setattr(oauth.OAuthCredential, "account_id", account)
    cfg = Config(
        providers={"work": entry}, models=ModelsConfig(worker=RoleModel(provider="work", model="m"))
    )
    provider = role_providers.build_role_provider(
        cfg, "worker", transcript_sink=mock.MagicMock(), budget=mock.MagicMock()
    )
    assert isinstance(provider, chatgpt.ChatGPTProvider)
    assert isinstance(provider.credential, oauth.OAuthCredential)
    assert provider.account_id == "from-holder"


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
    prov = role_providers.build_role_provider(
        cfg, "worker", transcript_sink=mock.MagicMock(), budget=mock.MagicMock()
    )
    assert isinstance(prov, OpenAIProvider)
    assert prov.extra_body == {"provider": {"sort": "throughput"}}
    assert ("x-title", "agent6") in prov.extra_headers.literal


def test_reviewer_family_builders_stamp_their_own_seats() -> None:
    """The reviser, summariser and a bare-focus seat stamp their own seats.

    They share the reviewer route; a transcript still tells which actor made a call.
    """
    from agent6.config import PromptConfig, ReviewConfig, ReviewerConfig

    cfg = Config(
        providers={"o": OpenAIProviderEntry(api_format="openai", base_url="https://x/v1")},
        models=ModelsConfig(
            worker=RoleModel(provider="o", model="m"),
            reviewer=RoleModel(provider="o", model="m"),
        ),
        review=ReviewConfig(
            trigger="on_verify_fail", reviewers=(ReviewerConfig(focus="security"),)
        ),
        prompt=PromptConfig(revise_prompt="auto"),
    )

    sink = mock.MagicMock()
    role_providers.build_prompt_reviser_provider(
        cfg, transcript_sink=sink, budget=mock.MagicMock(), events=mock.MagicMock()
    )
    assert sink.for_seat.call_args == mock.call("prompt_reviser")

    sink = mock.MagicMock()
    role_providers.reviewer_seat_provider(
        cfg, "summariser", transcript_sink=sink, budget=mock.MagicMock(), events=mock.MagicMock()
    )
    assert sink.for_seat.call_args == mock.call("summariser")

    sink = mock.MagicMock()
    role_providers.build_review_seats(cfg, transcript_sink=sink, budget=mock.MagicMock(), n=1)
    assert sink.for_seat.call_args == mock.call("review:security")

    # Each instrumented seat carries its roster index, so concurrent seats' calls pair up.
    unpinned = cfg.model_copy(update={"review": ReviewConfig(trigger="on_verify_fail")})
    seats = role_providers.build_review_seats(
        unpinned, transcript_sink=sink, budget=mock.MagicMock(), n=2, events=mock.MagicMock()
    )
    assert [getattr(s.provider, "seat_index", None) for s in seats] == [0, 1]

    # The role builders keep stamping the role itself.
    sink = mock.MagicMock()
    role_providers.build_role_provider(cfg, "worker", transcript_sink=sink, budget=mock.MagicMock())
    assert sink.for_seat.call_args == mock.call("worker")
