# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the freeform code-review sub-agent."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from agent6.agents.code_review import CodeReviewError, code_review
from agent6.providers import ProviderError, ProviderResponse, ToolDefinition


@dataclass
class _FakeProvider:
    """Captures the last call and returns a canned response."""

    response_text: str = "LGTM"
    raise_error: bool = False
    last_system: str = ""
    last_user: str = ""
    last_max_tokens: int = 0

    def call(
        self,
        *,
        system: str,
        messages: list[dict[str, object]],
        tools: list[ToolDefinition] | None = None,
        max_tokens: int = 1024,
    ) -> ProviderResponse:
        if self.raise_error:
            raise ProviderError("boom")
        self.last_system = system
        self.last_user = str(messages[0]["content"])
        self.last_max_tokens = max_tokens
        return ProviderResponse(
            text=self.response_text,
            tool_uses=(),
            stop_reason="end_turn",
            input_tokens=10,
            output_tokens=5,
            cache_read_tokens=0,
            cache_creation_tokens=0,
        )


def test_code_review_passes_diff_and_context() -> None:
    provider = _FakeProvider(response_text="LGTM with nits\n- [nit] foo")
    out = code_review(
        provider,  # type: ignore[arg-type]
        diff="diff --git a/x b/x\n+pass\n",
        agents_md="# project rules",
        recent_log="abc fix bug",
        extra_context="PR title: fix bug",
    )
    assert out.startswith("LGTM")
    assert "diff --git" in provider.last_user
    assert "project rules" in provider.last_user
    assert "abc fix bug" in provider.last_user
    assert "PR title" in provider.last_user
    assert "senior code reviewer" in provider.last_system


def test_code_review_truncates_huge_diff() -> None:
    provider = _FakeProvider()
    huge = "x" * 200_000
    code_review(provider, diff=huge)  # type: ignore[arg-type]
    # Diff is truncated to 60k chars in the prompt; user content must be smaller
    # than the raw input.
    assert len(provider.last_user) < len(huge)


def test_code_review_rejects_empty_response() -> None:
    provider = _FakeProvider(response_text="   ")
    with pytest.raises(CodeReviewError, match="empty"):
        code_review(provider, diff="diff")  # type: ignore[arg-type]


def test_code_review_wraps_provider_error() -> None:
    provider = _FakeProvider(raise_error=True)
    with pytest.raises(CodeReviewError, match="provider call failed"):
        code_review(provider, diff="diff")  # type: ignore[arg-type]
