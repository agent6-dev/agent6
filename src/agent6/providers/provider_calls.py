# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The provider call protocol and its text-only drafting helper."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from agent6.providers import provider_types, request_headers


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
        tools: list[provider_types.ToolDefinition] | None = ...,
        max_tokens: int = ...,
        temperature: float | None = ...,
        reasoning_effort: str | None = ...,
        text_delta_callback: Callable[[str], None] | None = ...,
        thinking_delta_callback: Callable[[str], None] | None = ...,
        should_abort: Callable[[], bool] | None = ...,
        should_interrupt: Callable[[], bool] | None = ...,
        compaction: provider_types.CompactionRequest | None = ...,
        context: request_headers.RequestContext | None = ...,
    ) -> provider_types.ProviderResponse:
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
            compaction: The history operation on a supporting endpoint.
            context: The application's context for the call: the conversation id, the turn
                state and the initiator; None outside a conversation.

        Returns:
            The parsed response.
        """
        ...


def call_for_text(provider: Provider, *, system: str, user: str, max_tokens: int) -> str | None:
    """Make one text-only call for best-effort drafting.

    The caller has a deterministic fallback for a drafting failure.

    Args:
        provider: The provider to call.
        system: The system prompt.
        user: The one user message.
        max_tokens: The output cap.

    Returns:
        The stripped reply, or None for an empty reply.

    Raises:
        provider_types.ProviderError: The call failed. The message gives the reason.
    """
    try:
        resp = provider.call(
            system=system,
            messages=[{"role": "user", "content": user}],
            tools=None,
            max_tokens=max_tokens,
        )
    except provider_types.ProviderError:
        raise
    except Exception as exc:
        raise provider_types.ProviderError(f"{type(exc).__name__}: {exc}") from exc
    return (resp.text or "").strip() or None
