# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Model providers behind one `Provider` protocol.

`AnthropicProvider` (Anthropic Messages), `OpenAIProvider` (any Chat Completions
endpoint: OpenAI, OpenRouter, Ollama, vLLM, llama.cpp), `ChatGPTProvider` (the
ChatGPT subscription's Codex backend) and `ClaudeCodeProvider` (the installed
Claude Code binary) are interchangeable; `[models.<role>]` routes each role.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from agent6.providers._claude_code_wire import CLAUDE_CODE_PERSIST_BYTES, result_text
from agent6.providers.anthropic import AnthropicProvider
from agent6.providers.chatgpt import ChatGPTProvider
from agent6.providers.chatgpt_oauth import ChatGPTCredential
from agent6.providers.claude_code import ClaudeCodeProvider
from agent6.providers.openai import OpenAIProvider
from agent6.providers.token_command import CommandToken
from agent6.providers.types import (
    ProviderAborted,
    ProviderError,
    ProviderInterrupted,
    ProviderResponse,
    RoleTranscriptSink,
    ToolDefinition,
    TranscriptRecorder,
    TranscriptSink,
    output_cap_truncated,
)


@runtime_checkable
class Provider(Protocol):
    """The vendor-agnostic surface every role calls.

    Tools are declared every turn and executed Python-side by the dispatcher. The
    delta callbacks opt into streaming: with either set a provider may stream text
    or reasoning deltas as they arrive; with both None it uses the non-streaming path.
    """

    def call(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition] | None = ...,
        max_tokens: int = ...,
        temperature: float | None = ...,
        reasoning_effort: str | None = ...,
        text_delta_callback: Callable[[str], None] | None = ...,
        thinking_delta_callback: Callable[[str], None] | None = ...,
        should_abort: Callable[[], bool] | None = ...,
        should_interrupt: Callable[[], bool] | None = ...,
    ) -> ProviderResponse:
        """Make one model call.

        Args:
            system: The system prompt.
            messages: The conversation in the provider's message shape.
            tools: The tools the model may call; None declares none.
            max_tokens: The output cap.
            temperature: The sampling temperature; None takes the provider's default.
            reasoning_effort: The reasoning level; None takes the provider's default.
            text_delta_callback: Receives visible text as it streams.
            thinking_delta_callback: Receives reasoning text as it streams.
            should_abort: Polled during the call; True abandons it with `ProviderAborted`.
            should_interrupt: Polled during the call; True ends it early with
                `ProviderInterrupted`.

        Returns:
            The parsed response.
        """
        ...


def call_for_text(provider: Provider, *, system: str, user: str, max_tokens: int) -> str | None:
    """Make one text-only call for best-effort drafting.

    The broad except is the point: the caller holds a deterministic fallback, and
    a drafting failure never surfaces as a run error.

    Args:
        provider: The provider to call.
        system: The system prompt.
        user: The one user message.
        max_tokens: The output cap.

    Returns:
        The stripped reply, or None on any failure or an empty reply.
    """
    try:
        resp = provider.call(
            system=system,
            messages=[{"role": "user", "content": user}],
            tools=None,
            max_tokens=max_tokens,
        )
    except Exception:
        return None
    return (resp.text or "").strip() or None


__all__ = [
    "CLAUDE_CODE_PERSIST_BYTES",
    "AnthropicProvider",
    "ChatGPTCredential",
    "ChatGPTProvider",
    "ClaudeCodeProvider",
    "CommandToken",
    "OpenAIProvider",
    "Provider",
    "ProviderAborted",
    "ProviderError",
    "ProviderInterrupted",
    "ProviderResponse",
    "RoleTranscriptSink",
    "ToolDefinition",
    "TranscriptRecorder",
    "TranscriptSink",
    "call_for_text",
    "output_cap_truncated",
    "result_text",
]
