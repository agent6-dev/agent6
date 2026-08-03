# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The cli ``InstrumentedProvider`` wrapper must forward every
provider.call kwarg to the inner provider. A missing passthrough is
invisible to unit tests that call providers directly but crashes every
real run (regression: ``reasoning_effort`` was added to the providers
and the loop but not the wrapper, so the perf bench died with
``TypeError: ... got an unexpected keyword argument 'reasoning_effort'``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from agent6.app.providers import InstrumentedProvider
from agent6.budget import BudgetTracker
from agent6.providers import ProviderResponse


def _resp() -> ProviderResponse:
    return ProviderResponse(
        text="ok",
        tool_uses=(),
        stop_reason="end_turn",
        input_tokens=1,
        output_tokens=1,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        raw={"content": [{"type": "text", "text": "ok"}]},
    )


def _wrap(inner: MagicMock) -> InstrumentedProvider:
    return InstrumentedProvider(
        inner=inner,
        role="worker",
        model="moonshotai/kimi-k2.6",
        provider_name="openai",
        events=MagicMock(),
        budget=BudgetTracker(max_usd=-1, max_tokens_fallback=-1),
    )


def test_instrumented_provider_forwards_reasoning_effort() -> None:
    inner = MagicMock()
    inner.call.return_value = _resp()
    wrapper = _wrap(inner)

    wrapper.call(
        system="s",
        messages=[{"role": "user", "content": "hi"}],
        reasoning_effort="off",
    )

    kwargs: dict[str, Any] = inner.call.call_args.kwargs
    assert kwargs["reasoning_effort"] == "off"


def test_instrumented_provider_forwards_should_abort() -> None:
    inner = MagicMock()
    inner.call.return_value = _resp()
    wrapper = _wrap(inner)

    def _abort() -> bool:
        return True

    wrapper.call(system="s", messages=[{"role": "user", "content": "hi"}], should_abort=_abort)
    assert inner.call.call_args.kwargs["should_abort"] is _abort


def test_instrumented_provider_defaults_reasoning_effort_to_none() -> None:
    inner = MagicMock()
    inner.call.return_value = _resp()
    wrapper = _wrap(inner)

    wrapper.call(system="s", messages=[{"role": "user", "content": "hi"}])

    assert inner.call.call_args.kwargs["reasoning_effort"] is None


def test_the_journal_records_what_the_assistant_said(tmp_path: Path) -> None:
    """The contract three readers depend on: `read_session`, `/btw`, and the
    transcript fold all reconstruct the conversation from this event.

    The prose used to reach the journal only as `role.text_delta`, emitted only
    when streaming is on, so a headless run recorded none of it -- and each
    reader had a hand-written fixture inventing this field, so all three were
    green against a shape the engine never emitted. Pinned at the EMITTER: a
    fixture can drift, this cannot.
    """
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from agent6.app.providers import InstrumentedProvider
    from agent6.budget import BudgetTracker
    from agent6.events import EventSink

    events = EventSink(tmp_path / "logs.jsonl")
    inner = MagicMock()
    inner.call.return_value = SimpleNamespace(
        text="the answer",
        tool_uses=(),
        stop_reason="end_turn",
        input_tokens=1,
        output_tokens=1,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        raw={},
    )
    InstrumentedProvider(
        inner=inner,
        role="worker",
        model="m",
        provider_name="p",
        events=events,
        budget=BudgetTracker(max_usd=-1, max_tokens_fallback=-1),
    ).call(system="s", messages=[], tools=[], max_tokens=8)

    settled = [
        json.loads(line)
        for line in (tmp_path / "logs.jsonl").read_text(encoding="utf-8").splitlines()
        if json.loads(line)["type"] == "role.result"
    ]
    assert [e["text"] for e in settled] == ["the answer"]
