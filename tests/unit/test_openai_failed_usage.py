from __future__ import annotations

import contextlib
import json
import pathlib
import threading
from collections.abc import Iterator
from typing import Any, Literal
from unittest import mock

import httpx2
import pytest

from agent6 import budget
from agent6.app import role_providers
from agent6.providers import openai, provider_types, stream_lifecycle
from agent6.sessions import event_log
from agent6.viewmodel import telemetry, telemetry_records

Failure = Literal[
    "http_error", "provider_error", "malformed_delta", "abort", "truncated", "failed_completion"
]


def _chunk(data: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(data)}\n\n".encode()


class _Stream(httpx2.SyncByteStream):
    """Deliver usage before a failure; an abort waits for the real watchdog to close it."""

    def __init__(self, usage: dict[str, Any] | None, failure: Failure | None) -> None:
        self.usage = usage
        self.failure: Failure | None = failure
        self.abort = threading.Event()
        self.closed = threading.Event()

    def __iter__(self) -> Iterator[bytes]:
        if self.usage is not None:
            yield _chunk({"choices": [], "usage": self.usage})
        match self.failure:
            case "http_error":
                raise httpx2.ReadError("connection dropped")
            case "provider_error":
                yield _chunk({"error": {"code": 502, "message": "upstream failed"}})
            case "malformed_delta":
                yield _chunk({"choices": [{"delta": {"content": {"bad": "shape"}}}]})
            case "abort":
                self.abort.set()
                assert self.closed.wait(5), "the watchdog did not close the aborted stream"
                raise httpx2.ReadError("stream closed")
            case None:
                yield _chunk({"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]})
                yield b"data: [DONE]\n\n"
            case "truncated":
                return
            case "failed_completion":
                yield _chunk({"choices": [{"delta": {}, "finish_reason": "error"}]})
                yield b"data: [DONE]\n\n"

    def close(self) -> None:
        self.closed.set()


@pytest.mark.parametrize(
    "failure",
    ["http_error", "provider_error", "malformed_delta", "abort", "truncated", "failed_completion"],
)
@pytest.mark.parametrize("reported_cost", [None, 0.25])
def test_failed_stream_usage_reaches_the_journal_and_telemetry(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: Failure,
    reported_cost: float | None,
) -> None:
    """A failed attempt keeps its own reported tokens, cost and route without billing twice."""
    monkeypatch.setattr(stream_lifecycle, "STREAM_WATCHDOG_TICK_S", 0.01)
    usage: dict[str, Any] = {
        "prompt_tokens": 125,
        "completion_tokens": 10,
        "prompt_tokens_details": {"cached_tokens": 25},
    }
    if reported_cost is not None:
        usage["cost"] = reported_cost
    stream = _Stream(usage, failure)
    tracker = budget.BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_plan_points=-1)
    events = event_log.EventSink(tmp_path / "logs.jsonl")
    wrapper = role_providers.InstrumentedProvider(
        inner=openai.OpenAIProvider(api_key="test", model="test-model", budget=tracker),
        role="worker",
        model="test-model",
        provider_name="test-provider",
        events=events,
        budget=tracker,
        stream_text=True,
    )
    response = httpx2.Response(200, stream=stream)
    with (
        mock.patch("httpx2.stream", return_value=contextlib.closing(response)),
        pytest.raises(provider_types.ProviderError) as caught,
    ):
        wrapper.call(system="s", messages=[], should_abort=stream.abort.is_set)
    if failure == "abort":
        assert isinstance(caught.value, provider_types.ProviderAborted)
    elif failure == "provider_error":
        assert caught.value.status_code == 502
    elif failure == "malformed_delta":
        assert "content delta was not a string" in str(caught.value)
    assert caught.value.provider == "test-provider"

    snap = tracker.snapshot()
    assert (
        snap.input_total,
        snap.output_total,
        snap.cache_read_total,
        snap.cache_creation_total,
    ) == (100, 10, 25, 0)
    assert snap.per_model["test-model"].calls == 1
    assert snap.per_model["test-model"].reported_cost_usd == (reported_cost or 0.0)

    rows = [json.loads(line) for line in events.path.read_text().splitlines()]
    (call,) = [row for row in rows if row["type"] == "role.call"]
    (result,) = [row for row in rows if row["type"] == "role.result"]
    assert result["ok"] is False
    assert result["model_call_id"] == call["model_call_id"]
    assert {
        key: result.get(key) for key in ("tokens_in", "tokens_out", "cache_read", "cache_creation")
    } == {"tokens_in": 100, "tokens_out": 10, "cache_read": 25, "cache_creation": 0}
    if reported_cost is None:
        assert "cost_usd" not in result
    else:
        assert result["cost_usd"] == reported_cost

    fold = telemetry.TelemetryFold("failed-stream")
    fold.feed({"type": "session.start", "execution_id": "e"})
    spans = [span for row in rows for span in fold.feed(row).spans]
    (model_span,) = [span for span in spans if span.name == telemetry_records.SPAN_MODEL_CALL]
    assert model_span.status == "error"
    assert dict(model_span.attributes)["agent6.tokens_in"] == 100
    counters = fold.metrics().counters
    labels = (("model", "test-model"), ("provider", "test-provider"))
    for kind, total in (("input", 100), ("output", 10), ("cache_read", 25)):
        assert counters[("agent6_tokens_total", (*labels, ("type", kind)))] == total
    assert counters[("agent6_model_calls_total", (labels[0], ("outcome", "error"), labels[1]))] == 1
    if reported_cost is not None:
        assert counters[("agent6_cost_usd_total", ())] == reported_cost


@pytest.mark.parametrize("reported_cost", [None, 0.25])
def test_successful_stream_reports_the_same_usage_fields(
    tmp_path: pathlib.Path, reported_cost: float | None
) -> None:
    """A successful result includes a reported cost without inventing one when absent."""
    usage: dict[str, Any] = {"prompt_tokens": 100, "completion_tokens": 10}
    if reported_cost is not None:
        usage["cost"] = reported_cost
    tracker = budget.BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_plan_points=-1)
    events = event_log.EventSink(tmp_path / "logs.jsonl")
    wrapper = role_providers.InstrumentedProvider(
        inner=openai.OpenAIProvider(api_key="test", model="test-model", budget=tracker),
        role="worker",
        model="test-model",
        provider_name="test-provider",
        events=events,
        budget=tracker,
        stream_text=True,
    )
    response = httpx2.Response(200, stream=_Stream(usage, None))
    with mock.patch("httpx2.stream", return_value=contextlib.closing(response)):
        wrapper.call(system="s", messages=[])
    rows = [json.loads(line) for line in events.path.read_text().splitlines()]
    (result,) = [row for row in rows if row["type"] == "role.result"]
    assert result["ok"] is True
    assert (
        result["tokens_in"],
        result["tokens_out"],
        result["cache_read"],
        result["cache_creation"],
    ) == (100, 10, 0, 0)
    if reported_cost is None:
        assert "cost_usd" not in result
    else:
        assert result["cost_usd"] == reported_cost


@pytest.mark.parametrize(
    "failure", ["http_error", "provider_error", "malformed_delta", "abort", "truncated"]
)
def test_failure_before_usage_does_not_invent_tokens(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, failure: Failure
) -> None:
    """A failure without a usage event leaves both the journal and the budget unmetered."""
    monkeypatch.setattr(stream_lifecycle, "STREAM_WATCHDOG_TICK_S", 0.01)
    stream = _Stream(None, failure)
    tracker = budget.BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_plan_points=-1)
    events = event_log.EventSink(tmp_path / "logs.jsonl")
    wrapper = role_providers.InstrumentedProvider(
        inner=openai.OpenAIProvider(api_key="test", model="test-model", budget=tracker),
        role="worker",
        model="test-model",
        provider_name="test-provider",
        events=events,
        budget=tracker,
        stream_text=True,
    )
    response = httpx2.Response(200, stream=stream)
    with (
        mock.patch("httpx2.stream", return_value=contextlib.closing(response)),
        pytest.raises(provider_types.ProviderError),
    ):
        wrapper.call(system="s", messages=[], should_abort=stream.abort.is_set)
    rows = [json.loads(line) for line in events.path.read_text().splitlines()]
    (result,) = [row for row in rows if row["type"] == "role.result"]
    assert (
        not {"tokens_in", "tokens_out", "cache_read", "cache_creation", "cost_usd"} & result.keys()
    )
    assert tracker.snapshot().per_model == {}
