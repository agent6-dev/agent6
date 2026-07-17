# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Regression tests for provider bug fixes.

Covers five independently-reported bugs:

* Bug #1 - non-JSON 2xx body in the NON-streaming path is converted to a
  retryable ``ProviderError`` instead of leaking a ``json.JSONDecodeError``
  (both OpenAI and Anthropic).
* Bug #2 - the Anthropic SSE streaming path has an idle watchdog: a wedged
  upstream that only emits ``ping`` heartbeats is closed and surfaced as a
  retryable ``ProviderError`` (mirrors the existing OpenAI watchdog).
* Bug #3 - OpenAI-direct o-series/reasoning models receive
  ``max_completion_tokens`` (not ``max_tokens``) and no explicit
  ``temperature``; other hosts are unchanged.
* Bug #4 - native tool_calls that arrive with no id get a synthesised
  distinct id so tool_use/tool_result pairing stays one-to-one.
* Bug #5 - budgeted calls fail closed when the upstream omits token usage
  accounting instead of recording a zero-token turn.
"""

from __future__ import annotations

import json
import threading
from typing import Any
from unittest import mock

import httpx2
import pytest

from agent6.budget import BudgetTracker
from agent6.providers import _stream as stream_mod
from agent6.providers._openai_parse import parse_response as _parse_response
from agent6.providers.anthropic import AnthropicProvider, ProviderError
from agent6.providers.openai import OpenAIProvider


# --------------------------------------------------------------------------
# Bug #1: non-JSON 2xx body -> retryable ProviderError (non-streaming)
# --------------------------------------------------------------------------
class _FakeJSONResponse:
    def __init__(self, *, status_code: int, text: str) -> None:
        self.status_code = status_code
        self.headers: dict[str, str] = {}
        self.text = text

    def json(self) -> Any:
        return json.loads(self.text)  # raises on non-JSON


def test_openai_non_json_200_is_provider_error() -> None:
    provider = OpenAIProvider(api_key="sk-test", model="gpt-4o-mini")
    resp = _FakeJSONResponse(status_code=200, text="<html>502 Bad Gateway</html>")
    with (
        mock.patch("agent6.providers._transport.http_post", return_value=resp),
        pytest.raises(ProviderError) as ei,
    ):
        provider.call(system="sys", messages=[{"role": "user", "content": "x"}])
    # Retryable: status_code stays unset (None), not a non-retryable 4xx.
    assert ei.value.status_code is None
    assert "non-JSON" in str(ei.value)


def test_anthropic_non_json_200_is_provider_error() -> None:
    provider = AnthropicProvider(api_key="sk-test", model="claude-3-5-sonnet")
    resp = _FakeJSONResponse(status_code=200, text="<html>502 Bad Gateway</html>")
    with (
        mock.patch("agent6.providers._transport.http_post", return_value=resp),
        pytest.raises(ProviderError) as ei,
    ):
        provider.call(system="sys", messages=[{"role": "user", "content": "x"}])
    assert ei.value.status_code is None
    assert "non-JSON" in str(ei.value)


def test_openai_budgeted_response_requires_usage_tokens() -> None:
    budget = BudgetTracker(max_input_tokens=1, max_output_tokens=1)
    provider = OpenAIProvider(api_key="sk-test", model="gpt-4o-mini", budget=budget)
    resp = _FakeJSONResponse(
        status_code=200,
        text=json.dumps(
            {
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {},
            }
        ),
    )
    with (
        mock.patch("agent6.providers._transport.http_post", return_value=resp),
        pytest.raises(ProviderError) as ei,
    ):
        provider.call(system="sys", messages=[{"role": "user", "content": "x"}])
    assert ei.value.status_code == 422
    assert budget.snapshot().per_model == {}


def test_anthropic_budgeted_response_requires_usage_tokens() -> None:
    budget = BudgetTracker(max_input_tokens=1, max_output_tokens=1)
    provider = AnthropicProvider(api_key="sk-test", model="claude-3-5-sonnet", budget=budget)
    resp = _FakeJSONResponse(
        status_code=200,
        text=json.dumps(
            {
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {},
            }
        ),
    )
    with (
        mock.patch("agent6.providers._transport.http_post", return_value=resp),
        pytest.raises(ProviderError) as ei,
    ):
        provider.call(system="sys", messages=[{"role": "user", "content": "x"}])
    assert ei.value.status_code == 422
    assert budget.snapshot().per_model == {}


def test_openai_budgeted_response_rejects_zero_token_usage() -> None:
    # Presence is not enough: a gateway with usage tracking off returns 0/0, and
    # every turn would record zero so the budget never trips. Fail closed.
    budget = BudgetTracker(max_input_tokens=1, max_output_tokens=1)
    provider = OpenAIProvider(api_key="sk-test", model="gpt-4o-mini", budget=budget)
    resp = _FakeJSONResponse(
        status_code=200,
        text=json.dumps(
            {
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            }
        ),
    )
    with (
        mock.patch("agent6.providers._transport.http_post", return_value=resp),
        pytest.raises(ProviderError) as ei,
    ):
        provider.call(system="sys", messages=[{"role": "user", "content": "x"}])
    assert ei.value.status_code == 422
    assert budget.snapshot().per_model == {}


def test_anthropic_budgeted_response_rejects_zero_token_usage() -> None:
    budget = BudgetTracker(max_input_tokens=1, max_output_tokens=1)
    provider = AnthropicProvider(api_key="sk-test", model="claude-3-5-sonnet", budget=budget)
    resp = _FakeJSONResponse(
        status_code=200,
        text=json.dumps(
            {
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 0, "output_tokens": 0},
            }
        ),
    )
    with (
        mock.patch("agent6.providers._transport.http_post", return_value=resp),
        pytest.raises(ProviderError) as ei,
    ):
        provider.call(system="sys", messages=[{"role": "user", "content": "x"}])
    assert ei.value.status_code == 422
    assert budget.snapshot().per_model == {}


def test_anthropic_budgeted_response_accepts_fully_cached_turn() -> None:
    # A fully-cached turn legitimately reports input_tokens: 0 with a positive
    # cache_read count; the metering check must NOT false-reject it.
    budget = BudgetTracker(max_input_tokens=1_000_000, max_output_tokens=1_000_000)
    provider = AnthropicProvider(api_key="sk-test", model="claude-3-5-sonnet", budget=budget)
    resp = _FakeJSONResponse(
        status_code=200,
        text=json.dumps(
            {
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": 5,
                    "cache_read_input_tokens": 120,
                    "cache_creation_input_tokens": 0,
                },
            }
        ),
    )
    with mock.patch("agent6.providers._transport.http_post", return_value=resp):
        provider.call(system="sys", messages=[{"role": "user", "content": "x"}])
    assert budget.snapshot().per_model != {}


# --------------------------------------------------------------------------
# Bug #2: Anthropic SSE idle watchdog
# --------------------------------------------------------------------------
class _PingOnlyStreamResponse:
    """A stream that only ever emits ``ping`` heartbeats.

    ``iter_lines`` blocks (via an event) until the watchdog calls
    ``close()``, at which point it raises ``httpx2.ReadError`` exactly as
    httpx2 would when the underlying socket is closed mid-read.
    """

    def __init__(self) -> None:
        self.status_code = 200
        self.headers: dict[str, str] = {}
        self._closed = threading.Event()

    def __enter__(self) -> _PingOnlyStreamResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def close(self) -> None:
        self._closed.set()

    def iter_lines(self):  # type: ignore[no-untyped-def]
        # Emit a couple of ping heartbeats, then block until closed.
        yield "event: ping"
        yield 'data: {"type": "ping"}'
        yield ""
        yield "event: ping"
        yield 'data: {"type": "ping"}'
        yield ""
        # Now park as if waiting for real data. The watchdog must fire.
        if not self._closed.wait(timeout=10.0):
            raise AssertionError("watchdog never closed the response")
        raise httpx2.ReadError("connection closed by watchdog")


def test_anthropic_streaming_idle_watchdog_fires(monkeypatch: pytest.MonkeyPatch) -> None:
    # Ping-only, no data event ever = the prefill-wedge case, so the FIRST-data
    # timeout governs. Make it fire fast so the test runs in well under a second.
    monkeypatch.setattr(stream_mod, "STREAM_FIRST_DATA_TIMEOUT_S", 0.05)
    monkeypatch.setattr(stream_mod, "STREAM_WATCHDOG_TICK_S", 0.01)

    provider = AnthropicProvider(api_key="sk-test", model="claude-3-5-sonnet")

    def fake_stream(method: str, url: str, **kwargs: Any) -> _PingOnlyStreamResponse:
        return _PingOnlyStreamResponse()

    with (
        mock.patch("httpx2.stream", side_effect=fake_stream),
        pytest.raises(ProviderError) as ei,
    ):
        provider.call(
            system="sys",
            messages=[{"role": "user", "content": "x"}],
            text_delta_callback=lambda _s: None,
        )
    # Surfaced as a retryable (status_code None) idle error, not a generic
    # ReadError leaking out of the loop.
    assert ei.value.status_code is None
    assert "idle" in str(ei.value).lower()


# --------------------------------------------------------------------------
# Bug #3: OpenAI-direct o-series uses max_completion_tokens, drops temperature
# --------------------------------------------------------------------------
def _capture_body(provider: OpenAIProvider) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    class _Resp:
        status_code = 200
        text = "{}"

        def json(self) -> Any:
            return {
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }

    def fake_post(url: str, **kwargs: Any) -> _Resp:
        captured.update(json.loads(kwargs["content"]))
        return _Resp()

    with mock.patch("agent6.providers._transport.http_post", side_effect=fake_post):
        provider.call(
            system="sys",
            messages=[{"role": "user", "content": "x"}],
            temperature=0.2,
        )
    return captured


def test_openai_direct_oseries_uses_max_completion_tokens() -> None:
    provider = OpenAIProvider(
        api_key="sk-test",
        model="o3-mini",
        base_url="https://api.openai.com/v1",
        deployment="direct",
    )
    body = _capture_body(provider)
    assert "max_tokens" not in body
    assert "max_completion_tokens" in body
    # Temperature is dropped for o-series direct.
    assert "temperature" not in body


def test_openrouter_oseries_still_uses_max_tokens() -> None:
    # Same reasoning model, but routed via OpenRouter: must keep max_tokens
    # (OpenRouter normalises it) and forward temperature.
    provider = OpenAIProvider(
        api_key="sk-test",
        model="o3-mini",
        base_url="https://openrouter.ai/api/v1",
        deployment="direct",
    )
    body = _capture_body(provider)
    assert "max_tokens" in body
    assert "max_completion_tokens" not in body
    assert body.get("temperature") == 0.2


def test_openai_direct_nonreasoning_keeps_max_tokens() -> None:
    provider = OpenAIProvider(
        api_key="sk-test",
        model="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
        deployment="direct",
    )
    body = _capture_body(provider)
    assert "max_tokens" in body
    assert "max_completion_tokens" not in body
    assert body.get("temperature") == 0.2


# --------------------------------------------------------------------------
# Bug #4: synthesise distinct ids for native tool_calls missing an id
# --------------------------------------------------------------------------
def test_parse_response_synthesises_distinct_tool_call_ids() -> None:
    data = {
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "list_dir", "arguments": "{}"}},
                        {"function": {"name": "read_file", "arguments": "{}"}},
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }
    parsed = _parse_response(data)
    ids = [tu["id"] for tu in parsed.tool_uses]
    assert all(i for i in ids), "every tool_use must have a non-empty id"
    assert len(set(ids)) == len(ids), "ids must be distinct"


def test_parse_response_preserves_provided_tool_call_ids() -> None:
    data = {
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {"id": "call_real", "function": {"name": "list_dir", "arguments": "{}"}},
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }
    parsed = _parse_response(data)
    assert parsed.tool_uses[0]["id"] == "call_real"


# --------------------------------------------------------------------------
# claude-opus-4-8 rejects `temperature` (400) -> drop it and retry, then latch
# --------------------------------------------------------------------------
def test_anthropic_temperature_400_retries_without_temperature_then_latches() -> None:
    provider = AnthropicProvider(api_key="sk-test", model="claude-opus-4-8")
    err400 = _FakeJSONResponse(
        status_code=400,
        text=json.dumps(
            {
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": "`temperature` is deprecated for this model.",
                },
            }
        ),
    )
    ok200 = _FakeJSONResponse(
        status_code=200,
        text=json.dumps(
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-4-8",
                "content": [{"type": "text", "text": "hello"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }
        ),
    )

    bodies: list[dict[str, Any]] = []

    def first_call(*_a: object, **k: object) -> _FakeJSONResponse:
        bodies.append(json.loads(k["content"]))  # type: ignore[arg-type]
        return err400 if len(bodies) == 1 else ok200

    with mock.patch("agent6.providers._transport.http_post", side_effect=first_call):
        resp = provider.call(
            system="sys", messages=[{"role": "user", "content": "x"}], temperature=0.0
        )
    # First request carried temperature; the retry dropped it and succeeded.
    assert "temperature" in bodies[0]
    assert "temperature" not in bodies[1]
    assert resp.text == "hello"

    # The flag latched: a later call omits temperature from the very first request
    # (no wasted 400 + full-context resend every iteration).
    bodies2: list[dict[str, Any]] = []

    def second_call(*_a: object, **k: object) -> _FakeJSONResponse:
        bodies2.append(json.loads(k["content"]))  # type: ignore[arg-type]
        return ok200

    with mock.patch("agent6.providers._transport.http_post", side_effect=second_call):
        provider.call(system="sys", messages=[{"role": "user", "content": "y"}], temperature=0.0)
    assert "temperature" not in bodies2[0]


# --------------------------------------------------------------------------
# Connection errors name the dialled URL + api format (a bare "HTTP error
# calling OpenAI" pointed users at the wrong party for local endpoints).
# --------------------------------------------------------------------------
def test_openai_connection_error_names_url_and_format() -> None:
    provider = OpenAIProvider(api_key="", model="llama3", base_url="http://localhost:11434/v1")
    with (
        mock.patch(
            "agent6.providers._transport.http_post",
            side_effect=httpx2.HTTPError("[Errno 111] Connection refused"),
        ),
        pytest.raises(ProviderError) as ei,
    ):
        provider.call(system="sys", messages=[{"role": "user", "content": "x"}])
    msg = str(ei.value)
    assert "http://localhost:11434/v1/chat/completions" in msg
    assert "openai format" in msg
    assert "Connection refused" in msg


def test_anthropic_connection_error_names_url_and_format() -> None:
    provider = AnthropicProvider(api_key="sk-test", model="claude-3-5-sonnet")
    with (
        mock.patch(
            "agent6.providers._transport.http_post",
            side_effect=httpx2.HTTPError("[Errno 111] Connection refused"),
        ),
        pytest.raises(ProviderError) as ei,
    ):
        provider.call(system="sys", messages=[{"role": "user", "content": "x"}])
    msg = str(ei.value)
    assert "api.anthropic.com" in msg
    assert "anthropic format" in msg
