# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A stream that dies after the provider reported usage still cost money.

`budget.record` ran only when a stream completed, so every early exit -- a
retryable mid-stream error, the idle watchdog, an operator steer or stop --
spent money `max_usd` never saw. Each retry re-sends the whole input and is
billed again, so a run with any flakiness had no ceiling at all: the operator
set a number for the task and could pass it without being told.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest import mock

import httpx2
import pytest

from agent6.budget import BudgetTracker
from agent6.providers import AnthropicProvider, OpenAIProvider
from tests.unit.test_anthropic_streaming import FakeStreamResponse


class _CutAfter(FakeStreamResponse):
    """Serves lines until `cut_at`, then dies like a dropped connection."""

    cut_at: int = 0

    def iter_lines(self) -> Any:
        for i, line in enumerate(self._lines):
            if i >= self.cut_at:
                raise httpx2.ReadError("connection dropped mid-stream")
            yield line


def _cut(lines: list[str], at: int) -> _CutAfter:
    resp = _CutAfter(status_code=200, lines=lines)
    resp.cut_at = at
    return resp


def _sse(event: str, data: dict[str, Any]) -> list[str]:
    return [f"event: {event}", f"data: {json.dumps(data)}", ""]


def test_anthropic_records_what_a_cut_stream_already_cost(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The USD assertion needs a table price; the suite isolates the model-price
    # cache, so seed one (the suite never reads the developer's real cache).
    monkeypatch.setenv("AGENT6_CACHE_HOME", str(tmp_path))
    (tmp_path / "models").mkdir()
    pricing = {"claude-sonnet-4-5": [3.0, 15.0]}
    (tmp_path / "models" / "anthropic.json").write_text(
        json.dumps({"models": list(pricing), "pricing": pricing}), encoding="utf-8"
    )
    lines = _sse(
        "message_start",
        {
            "message": {
                "usage": {
                    "input_tokens": 50_000,
                    "output_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                }
            }
        },
    )
    lines += _sse(
        "content_block_start", {"index": 0, "content_block": {"type": "text", "text": ""}}
    )
    lines += _sse(
        "content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": "partial"}}
    )

    budget = BudgetTracker(max_usd=10.0, max_tokens_fallback=-1)
    provider = AnthropicProvider(api_key="k", model="claude-sonnet-4-5", budget=budget)
    with (
        mock.patch("agent6.providers._stream.http_stream", return_value=_cut(lines, at=9)),
        pytest.raises(Exception),
    ):
        provider.call(
            messages=[{"role": "user", "content": "hi"}],
            system="",
            tools=[],
            # The callback is what selects the STREAMING path.
            text_delta_callback=lambda _t: None,
        )

    snap = budget.snapshot()
    assert snap.input_total == 50_000, "the provider billed this input; the cap must see it"
    spent, _ = budget.estimate_usd()
    assert spent > 0


def test_openai_records_a_cut_stream_and_keeps_the_cached_split() -> None:
    """Through parse_response, so the cached-vs-fresh mapping has one owner."""

    def chunk(obj: dict[str, Any]) -> list[str]:
        return [f"data: {json.dumps(obj)}", ""]

    lines = chunk({"choices": [{"index": 0, "delta": {"role": "assistant", "content": "hi"}}]})
    lines += chunk(
        {
            "usage": {
                "prompt_tokens": 40_000,
                "completion_tokens": 120,
                "prompt_tokens_details": {"cached_tokens": 10_000},
            },
            "choices": [],
        }
    )
    lines += chunk({"choices": [{"index": 0, "delta": {"content": " more"}}]})

    budget = BudgetTracker(max_usd=10.0, max_tokens_fallback=-1)
    provider = OpenAIProvider(api_key="k", model="gpt-4o", budget=budget)
    with (
        mock.patch("agent6.providers._stream.http_stream", return_value=_cut(lines, at=6)),
        pytest.raises(Exception),
    ):
        provider.call(
            messages=[{"role": "user", "content": "hi"}],
            system="",
            tools=[],
            # The callback is what selects the STREAMING path.
            text_delta_callback=lambda _t: None,
        )

    snap = budget.snapshot()
    assert snap.input_total == 30_000  # fresh input, cached counted separately
    assert snap.cache_read_total == 10_000
    assert snap.output_total == 120


def test_a_stream_that_reported_nothing_records_nothing() -> None:
    """An unknown amount is not a licence to invent one: a stream cut before
    any usage arrived must leave the ledger untouched."""
    lines = _sse("content_block_start", {"index": 0, "content_block": {"type": "text", "text": ""}})

    budget = BudgetTracker(max_usd=10.0, max_tokens_fallback=-1)
    provider = AnthropicProvider(api_key="k", model="claude-sonnet-4-5", budget=budget)
    with (
        mock.patch("agent6.providers._stream.http_stream", return_value=_cut(lines, at=1)),
        pytest.raises(Exception),
    ):
        provider.call(
            messages=[{"role": "user", "content": "hi"}],
            system="",
            tools=[],
            # The callback is what selects the STREAMING path.
            text_delta_callback=lambda _t: None,
        )

    snap = budget.snapshot()
    assert snap.input_total == 0
    assert snap.output_total == 0
