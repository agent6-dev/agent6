# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The worker's call names its initiator: the operator after the task, a steer or an answer."""

from __future__ import annotations

from typing import Any

from agent6.harness import _conversation, _provider_call
from agent6.providers import ProviderResponse, _headers


def test_the_initiator_follows_the_latest_notice_origin() -> None:
    conv = _conversation.Conversation()
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
    reloaded = _conversation.Conversation.from_wire(conv.to_wire())
    assert reloaded.initiator == "agent"


def test_the_caller_passes_the_initiator_as_a_partial_context() -> None:
    seen: list[Any] = []

    class _Provider:
        def call(self, **kw: Any) -> ProviderResponse:
            seen.append(kw["context"])
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

    caller = _provider_call.ProviderCaller(
        provider=_Provider(),  # type: ignore[arg-type]
        retry_count=0,
        retry_delay_s=0.0,
        retry_max_delay_s=0.0,
        temperature=None,
        should_abort=lambda: False,
        should_interrupt=lambda: False,
        log=lambda _m: None,
        emit=lambda *_a, **_k: None,
    )
    caller.call("s", [], [], 10, initiator="user")
    caller.call("s", [], [], 10)
    assert [c.initiator for c in seen] == ["user", "agent"]
    assert all(isinstance(c, _headers.RequestContext) and c.session_id == "" for c in seen)
