# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for OpenAIProvider SSE streaming.

Mirrors ``test_anthropic_streaming.py`` for the OpenAI Chat Completions
SSE shape. Validates that when a ``text_delta_callback`` is supplied
the provider:

* takes the streaming code path (``stream: true`` +
  ``stream_options.include_usage`` in the body);
* fans text deltas to the callback as they arrive;
* reassembles tool_call arguments spanning multiple chunks (indexed
  by the chunk-level ``index`` field, not the call id);
* reads ``usage`` from the terminal ``choices: []`` chunk;
* ignores SSE comment heartbeats (``:OPENROUTER PROCESSING``) and
  the ``data: [DONE]`` sentinel;
* synthesises a non-streaming-shape response so ``_parse_response``
  yields the same ProviderResponse it would for a normal call;
* surfaces HTTP errors as ``ProviderError``;
* never calls ``httpx2.stream`` when the callback is None.
"""

from __future__ import annotations

import json
from typing import Any
from unittest import mock

import httpx2
import pytest

from agent6.providers import ProviderError
from agent6.providers.openai import OpenAIProvider


class _FakeStreamResponse:
    """Mimics the subset of httpx2 streaming Response we use."""

    def __init__(self, *, status_code: int, lines: list[str], error_body: str = "") -> None:
        self.status_code = status_code
        self._lines = lines
        self._error_body = error_body

    def __enter__(self) -> _FakeStreamResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def iter_lines(self) -> list[str]:
        return self._lines

    def read(self) -> bytes:
        return self._error_body.encode("utf-8")


def _chunk(data: dict[str, Any]) -> list[str]:
    return [f"data: {json.dumps(data)}", ""]


def _text_stream() -> list[str]:
    out: list[str] = []
    # Leading heartbeat (OpenRouter style).
    out += [":OPENROUTER PROCESSING", ""]
    out += _chunk(
        {
            "id": "c1",
            "choices": [
                {"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}
            ],
        }
    )
    out += _chunk({"choices": [{"index": 0, "delta": {"content": "hello"}, "finish_reason": None}]})
    out += [":OPENROUTER PROCESSING", ""]
    out += _chunk(
        {"choices": [{"index": 0, "delta": {"content": " world"}, "finish_reason": None}]}
    )
    out += _chunk({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
    out += _chunk(
        {
            "choices": [],
            "usage": {
                "prompt_tokens": 42,
                "completion_tokens": 9,
                "prompt_tokens_details": {"cached_tokens": 7},
            },
        }
    )
    out += ["data: [DONE]", ""]
    return out


def _tool_stream() -> list[str]:
    """Tool call whose arguments arrive in 3 chunks; id+name on first."""
    out: list[str] = []
    out += _chunk(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_xyz",
                                "type": "function",
                                "function": {"name": "list_dir", "arguments": ""},
                            }
                        ],
                    },
                    "finish_reason": None,
                }
            ]
        }
    )
    out += _chunk(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"pa'}}]},
                    "finish_reason": None,
                }
            ]
        }
    )
    out += _chunk(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'th": '}}]},
                    "finish_reason": None,
                }
            ]
        }
    )
    out += _chunk(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"."}'}}]},
                    "finish_reason": "tool_calls",
                }
            ]
        }
    )
    out += _chunk({"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 4}})
    out += ["data: [DONE]", ""]
    return out


def test_streaming_calls_back_on_each_text_delta() -> None:
    provider = OpenAIProvider(api_key="sk-test", model="kimi")
    captured_bodies: list[dict[str, Any]] = []

    def fake_stream(method: str, url: str, **kwargs: Any) -> _FakeStreamResponse:
        assert method == "POST"
        captured_bodies.append(json.loads(kwargs["content"]))
        return _FakeStreamResponse(status_code=200, lines=_text_stream())

    pieces: list[str] = []
    with mock.patch("httpx2.stream", side_effect=fake_stream):
        resp = provider.call(
            system="sys",
            messages=[{"role": "user", "content": "x"}],
            text_delta_callback=pieces.append,
        )

    assert captured_bodies[0]["stream"] is True
    assert captured_bodies[0]["stream_options"] == {"include_usage": True}
    assert pieces == ["hello", " world"]
    assert resp.text == "hello world"
    assert resp.stop_reason == "stop"
    # input_tokens is fresh (non-cached) only; prompt_tokens=42 with
    # cached_tokens=7 means 35 fresh + 7 cached.
    assert resp.input_tokens == 35
    assert resp.output_tokens == 9
    assert resp.cache_read_tokens == 7


def test_streaming_reassembles_tool_call_arguments_across_chunks() -> None:
    provider = OpenAIProvider(api_key="sk-test", model="kimi")

    def fake_stream(method: str, url: str, **kwargs: Any) -> _FakeStreamResponse:
        return _FakeStreamResponse(status_code=200, lines=_tool_stream())

    with mock.patch("httpx2.stream", side_effect=fake_stream):
        resp = provider.call(
            system="sys",
            messages=[{"role": "user", "content": "x"}],
            text_delta_callback=lambda _p: None,
        )

    assert resp.text == ""
    assert resp.stop_reason == "tool_calls"
    assert len(resp.tool_uses) == 1
    tu = resp.tool_uses[0]
    assert tu["id"] == "call_xyz"
    assert tu["name"] == "list_dir"
    assert tu["input"] == {"path": "."}
    assert resp.input_tokens == 10
    assert resp.output_tokens == 4


def test_streaming_swallows_callback_exception() -> None:
    provider = OpenAIProvider(api_key="sk-test", model="kimi")

    def boom(_p: str) -> None:
        raise RuntimeError("ui exploded")

    def fake_stream(method: str, url: str, **kwargs: Any) -> _FakeStreamResponse:
        return _FakeStreamResponse(status_code=200, lines=_text_stream())

    with mock.patch("httpx2.stream", side_effect=fake_stream):
        resp = provider.call(
            system="sys",
            messages=[{"role": "user", "content": "x"}],
            text_delta_callback=boom,
        )

    assert resp.text == "hello world"


def test_streaming_http_error_raises_provider_error() -> None:
    provider = OpenAIProvider(api_key="sk-test", model="kimi")

    def fake_stream(method: str, url: str, **kwargs: Any) -> _FakeStreamResponse:
        return _FakeStreamResponse(status_code=429, lines=[], error_body='{"error": "rate limit"}')

    with (
        mock.patch("httpx2.stream", side_effect=fake_stream),
        pytest.raises(ProviderError, match="429"),
    ):
        provider.call(
            system="sys",
            messages=[{"role": "user", "content": "x"}],
            text_delta_callback=lambda _p: None,
        )


def test_streaming_httpx_transport_error_raises_provider_error() -> None:
    provider = OpenAIProvider(api_key="sk-test", model="kimi")

    def boom(method: str, url: str, **kwargs: Any) -> _FakeStreamResponse:
        raise httpx2.ReadTimeout("timed out")

    with (
        mock.patch("httpx2.stream", side_effect=boom),
        pytest.raises(ProviderError, match="HTTP error streaming OpenAI"),
    ):
        provider.call(
            system="sys",
            messages=[{"role": "user", "content": "x"}],
            text_delta_callback=lambda _p: None,
        )


def test_no_callback_does_not_stream() -> None:
    provider = OpenAIProvider(api_key="sk-test", model="kimi")

    def fake_post(*_a: Any, **kw: Any) -> httpx2.Response:
        return httpx2.Response(
            status_code=200,
            request=httpx2.Request("POST", "https://api.openai.com/v1/chat/completions"),
            content=json.dumps(
                {
                    "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                }
            ).encode("utf-8"),
            headers={"content-type": "application/json"},
        )

    with (
        mock.patch("httpx2.post", side_effect=fake_post) as post_mock,
        mock.patch("httpx2.stream") as stream_mock,
    ):
        resp = provider.call(
            system="sys",
            messages=[{"role": "user", "content": "x"}],
        )

    assert resp.text == "ok"
    assert post_mock.called
    assert not stream_mock.called


def test_streaming_idle_watchdog_kills_heartbeat_only_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stream that emits only ``:`` heartbeats must trip
    the idle watchdog. Without this guard, OpenRouter sessions where
    the upstream model wedges can pin the harness for 800+ seconds
    while heartbeats keep httpx2's read-timeout reset indefinitely.
    """
    import threading
    import time

    from agent6.providers import openai as openai_mod

    monkeypatch.setattr(openai_mod, "_STREAM_IDLE_TIMEOUT_S", 0.3)
    monkeypatch.setattr(openai_mod, "_STREAM_WATCHDOG_TICK_S", 0.05)

    provider = OpenAIProvider(api_key="sk-test", model="kimi")

    class _BlockingHeartbeatResponse:
        def __init__(self) -> None:
            self.status_code = 200
            self._closed = threading.Event()

        def __enter__(self) -> _BlockingHeartbeatResponse:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def close(self) -> None:
            self._closed.set()

        def read(self) -> bytes:
            return b""

        def iter_lines(self):
            # Emit heartbeats every 50ms until the watchdog closes us.
            while not self._closed.is_set():
                yield ":OPENROUTER PROCESSING"
                yield ""
                # Block briefly; on close(), raise to mimic httpx2.
                if self._closed.wait(0.05):
                    raise httpx2.ReadError("connection closed by watchdog")
            raise httpx2.ReadError("connection closed by watchdog")

    def fake_stream(method: str, url: str, **kwargs: Any) -> _BlockingHeartbeatResponse:
        return _BlockingHeartbeatResponse()

    started = time.monotonic()
    with (
        mock.patch("httpx2.stream", side_effect=fake_stream),
        pytest.raises(ProviderError, match="idle for"),
    ):
        provider.call(
            system="sys",
            messages=[{"role": "user", "content": "x"}],
            text_delta_callback=lambda _p: None,
        )
    elapsed = time.monotonic() - started
    # Should fire well under 2s with a 0.3s threshold + 0.05s tick.
    assert elapsed < 2.0, f"watchdog took {elapsed:.2f}s"


def test_streaming_idle_watchdog_does_not_fire_when_data_flows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real SSE ``data:`` lines must reset the idle clock so
    a healthy long stream does not get killed.
    """
    from agent6.providers import openai as openai_mod

    monkeypatch.setattr(openai_mod, "_STREAM_IDLE_TIMEOUT_S", 0.5)
    monkeypatch.setattr(openai_mod, "_STREAM_WATCHDOG_TICK_S", 0.05)

    provider = OpenAIProvider(api_key="sk-test", model="kimi")

    def fake_stream(method: str, url: str, **kwargs: Any) -> _FakeStreamResponse:
        return _FakeStreamResponse(status_code=200, lines=_text_stream())

    with mock.patch("httpx2.stream", side_effect=fake_stream):
        resp = provider.call(
            system="sys",
            messages=[{"role": "user", "content": "x"}],
            text_delta_callback=lambda _p: None,
        )

    assert resp.text == "hello world"
    assert resp.stop_reason == "stop"
