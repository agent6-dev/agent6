# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Provider package.

Both `AnthropicProvider` (Anthropic Messages) and `OpenAIProvider` (any
OpenAI Chat Completions-compatible endpoint: OpenAI, OpenRouter, Ollama,
vLLM, llama.cpp) satisfy the `Provider` Protocol and can serve ANY
sub-agent role. Role-to-provider routing lives in `[models.<role>]` in
`agent6.toml`; the providers themselves are interchangeable from the
sub-agents' point of view.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from agent6.providers.anthropic import (
    AnthropicProvider,
    ProviderError,
    ProviderResponse,
    ToolDefinition,
    TranscriptSink,
)
from agent6.providers.openai import OpenAIProvider


@runtime_checkable
class Provider(Protocol):
    """Vendor-agnostic surface used by every sub-agent.

    `AnthropicProvider` and `OpenAIProvider` both satisfy this. No
    sub-agent currently exercises tool use through the provider — tools
    are dispatched in Python via `ToolDispatcher` — so the `tools`
    parameter exists for forward-compatibility only.
    """

    def call(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition] | None = ...,
        max_tokens: int = ...,
    ) -> ProviderResponse: ...


__all__ = [
    "AnthropicProvider",
    "OpenAIProvider",
    "Provider",
    "ProviderError",
    "ProviderResponse",
    "ToolDefinition",
    "TranscriptSink",
]
