# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The worker's call names its initiator: the operator after the task, a steer or an answer."""

from __future__ import annotations

from typing import Any

from agent6.harness import loop_conversation, provider_call
from agent6.providers import ProviderError, ProviderResponse, provider_types, request_headers


def _ok() -> ProviderResponse:
    return ProviderResponse(
        text="ok",
        tool_uses=(),
        stop_reason="end_turn",
        input_tokens=1,
        output_tokens=1,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        raw={},
    )


def _caller(provider: Any, *, retry_count: int = 0) -> provider_call.ProviderCaller:
    return provider_call.ProviderCaller(
        provider=provider,
        retry_count=retry_count,
        retry_delay_s=0.0,
        retry_max_delay_s=0.0,
        temperature=None,
        should_abort=lambda: False,
        should_interrupt=lambda: False,
        log=lambda _m: None,
        emit=lambda *_a, **_k: None,
    )


def test_the_initiator_follows_the_latest_notice_origin() -> None:
    conv = loop_conversation.Conversation()
    assert conv.initiator == "agent"
    conv.notice("TASK: do it", origin="operator")
    assert conv.initiator == "user"
    conv.notice("[harness verify] the gate's baseline is green")  # a preamble keeps the turn
    assert conv.initiator == "user"
    conv.assistant([{"type": "text", "text": "working"}])
    assert conv.initiator == "agent"
    conv.notice("[no-progress] nudge")
    assert conv.initiator == "agent"
    conv.notice("OPERATOR STEERING: faster", origin="operator")
    assert conv.initiator == "user"
    # The wire carries text only: a snapshot read back is the harness's.
    reloaded = loop_conversation.Conversation.from_wire(conv.to_wire())
    assert reloaded.initiator == "agent"


def test_the_caller_passes_the_initiator_as_a_partial_context() -> None:
    seen: list[Any] = []

    class _Provider:
        def call(self, **kw: Any) -> ProviderResponse:
            seen.append(kw["context"])
            return _ok()

    caller = _caller(_Provider())
    caller.call("s", [], [], 10, initiator="user")
    caller.call("s", [], [], 10)
    assert [c.initiator for c in seen] == ["user", "agent"]
    assert all(
        isinstance(c, request_headers.RequestContext) and c.conversation_id == "" for c in seen
    )


def test_a_retried_call_keeps_its_logical_call_id() -> None:
    """A transient error's retry carries the call id of the attempt it repeats."""
    seen: list[str] = []

    class _Flaky:
        def call(self, **kw: Any) -> ProviderResponse:
            # As the instrumented provider does: mint an id only when the caller sent none.
            call_context = kw["context"].call_context or provider_types.CallContext()
            seen.append(call_context.model_call_id)
            if len(seen) == 1:
                raise ProviderError("busy", status_code=503)
            return _ok()

    _caller(_Flaky(), retry_count=1).call("s", [], [], 10)
    assert len(seen) == 2
    assert len(set(seen)) == 1, seen
