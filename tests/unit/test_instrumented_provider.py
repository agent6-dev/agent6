# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`InstrumentedProvider` forwards every provider.call kwarg to the inner provider.

A missing passthrough is invisible to unit tests that call providers directly and crashes every real
run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

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
        budget=BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1),
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
    """The assistant text event is pinned at the emitter: three readers rebuild the conversation.

    `read_session`, `/btw` and the transcript fold; a fixture can drift, the emitter cannot.
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
        refused={},
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
        budget=BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1),
    ).call(system="s", messages=[], tools=[], max_tokens=8)

    settled = [
        json.loads(line)
        for line in (tmp_path / "logs.jsonl").read_text(encoding="utf-8").splitlines()
        if json.loads(line)["type"] == "role.result"
    ]
    assert [e["text"] for e in settled] == ["the answer"]


def test_a_failed_call_still_reports_what_it_spent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A cut stream is billed, and `budget.update` is the only path spend takes to a surface.

    The live meters, `sessions list` and the machine spend ledger all read that event.
    """
    from agent6.events import EventSink
    from agent6.providers import ProviderError

    # The USD assertion needs a table price; the suite isolates the price cache, so seed one.
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    (tmp_path / "agent6" / "models").mkdir(parents=True, exist_ok=True)
    pricing = {"anthropic/claude-haiku-4.5": [1.0, 5.0]}
    (tmp_path / "agent6" / "models" / "anthropic.json").write_text(
        json.dumps({"models": list(pricing), "pricing": pricing}), encoding="utf-8"
    )
    events = EventSink(tmp_path / "logs.jsonl")
    budget = BudgetTracker(max_usd=10.0, max_tokens_fallback=2_000_000, max_percent=-1)

    def _cut_stream(**_: object) -> ProviderResponse:
        budget.record(
            model="anthropic/claude-haiku-4.5",
            input_tokens=50_000,
            output_tokens=120,
            cache_read_tokens=7_000,
            cache_creation_tokens=1_100,
        )
        raise ProviderError("stream cut before completion")

    inner = MagicMock()
    inner.call.side_effect = _cut_stream
    wrapper = InstrumentedProvider(
        inner=inner,
        role="worker",
        model="anthropic/claude-haiku-4.5",
        provider_name="anthropic",
        events=events,
        budget=budget,
    )

    with pytest.raises(ProviderError):
        wrapper.call(system="s", messages=[])

    updates = [
        json.loads(line)
        for line in (tmp_path / "logs.jsonl").read_text(encoding="utf-8").splitlines()
        if json.loads(line)["type"] == "budget.update"
    ]
    assert [(e["input_total"], e["output_total"]) for e in updates] == [(50_000, 120)]
    # The cached side rides the same event: the scan and every surface read it there.
    assert (updates[0]["cache_read_total"], updates[0]["cache_creation_total"]) == (7_000, 1_100)
    assert updates[0]["usd_total"] > 0.0


def test_a_provider_error_is_stamped_with_the_provider_name() -> None:
    """The credential hint names the failing provider's config key, not a `<name>` placeholder."""
    from agent6.providers import ProviderError

    inner = MagicMock()
    inner.call.side_effect = ProviderError("401 nope", status_code=401)
    with pytest.raises(ProviderError) as info:
        _wrap(inner).call(system="s", messages=[])
    assert info.value.provider == "openai"
