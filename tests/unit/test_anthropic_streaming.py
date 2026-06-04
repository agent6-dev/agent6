# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for AnthropicProvider SSE streaming.

Validate that when a ``text_delta_callback`` is supplied, the provider:

* takes the streaming code path (sets ``stream: true`` in the body);
* fans text deltas to the callback as they arrive;
* reassembles a ProviderResponse with shape identical to the
  non-streaming path (text + tool_uses + usage + stop_reason);
* records a transcript identical in shape to non-streaming;
* handles tool_use blocks whose ``input_json_delta`` chunks span
  multiple SSE events.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from agent6.providers import AnthropicProvider, ProviderError, TranscriptSink


class _FakeStreamResponse:
    """Mimics the subset of httpx streaming Response we use."""

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


def _sse(events: list[tuple[str, dict[str, Any]]]) -> list[str]:
    """Turn (event_type, data) pairs into the raw line list httpx
    .iter_lines() would yield. SSE frames are separated by a blank line."""
    out: list[str] = []
    for et, data in events:
        out.append(f"event: {et}")
        out.append(f"data: {json.dumps(data)}")
        out.append("")
    return out


def _basic_text_stream() -> list[str]:
    return _sse(
        [
            (
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_1",
                        "role": "assistant",
                        "content": [],
                        "usage": {
                            "input_tokens": 42,
                            "cache_read_input_tokens": 7,
                            "cache_creation_input_tokens": 0,
                        },
                    },
                },
            ),
            (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "hello"},
                },
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": " world"},
                },
            ),
            (
                "content_block_stop",
                {"type": "content_block_stop", "index": 0},
            ),
            (
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {"output_tokens": 9},
                },
            ),
            ("message_stop", {"type": "message_stop"}),
        ]
    )


def _tool_use_stream() -> list[str]:
    """tool_use whose input arrives in 3 input_json_delta chunks."""
    return _sse(
        [
            (
                "message_start",
                {
                    "message": {
                        "usage": {
                            "input_tokens": 10,
                            "cache_read_input_tokens": 0,
                            "cache_creation_input_tokens": 0,
                        }
                    }
                },
            ),
            (
                "content_block_start",
                {
                    "index": 0,
                    "content_block": {
                        "type": "tool_use",
                        "id": "tu_xyz",
                        "name": "list_dir",
                        "input": {},
                    },
                },
            ),
            (
                "content_block_delta",
                {
                    "index": 0,
                    "delta": {"type": "input_json_delta", "partial_json": '{"pa'},
                },
            ),
            (
                "content_block_delta",
                {
                    "index": 0,
                    "delta": {"type": "input_json_delta", "partial_json": 'th": '},
                },
            ),
            (
                "content_block_delta",
                {
                    "index": 0,
                    "delta": {"type": "input_json_delta", "partial_json": '"."}'},
                },
            ),
            ("content_block_stop", {"index": 0}),
            (
                "message_delta",
                {"delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 4}},
            ),
            ("message_stop", {}),
        ]
    )


def test_streaming_calls_back_on_each_text_delta(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sink = TranscriptSink(tmp_path / "transcripts")
    provider = AnthropicProvider(
        api_key="sk-test", model="claude-test", prompt_caching=False, transcript_sink=sink
    )

    captured_bodies: list[dict[str, Any]] = []

    def fake_stream(method: str, url: str, **kwargs: Any) -> _FakeStreamResponse:
        assert method == "POST"
        body = json.loads(kwargs["content"])
        captured_bodies.append(body)
        return _FakeStreamResponse(status_code=200, lines=_basic_text_stream())

    monkeypatch.setattr(httpx, "stream", fake_stream)

    pieces: list[str] = []
    resp = provider.call(
        system="sys",
        messages=[{"role": "user", "content": "x"}],
        text_delta_callback=pieces.append,
    )

    assert captured_bodies[0]["stream"] is True
    assert pieces == ["hello", " world"]
    assert resp.text == "hello world"
    assert resp.stop_reason == "end_turn"
    assert resp.input_tokens == 42
    assert resp.output_tokens == 9
    assert resp.cache_read_tokens == 7
    # raw must be shaped like a non-streaming response so downstream
    # assistant-block reconstruction in Workflow keeps working.
    assert resp.raw["content"] == [{"type": "text", "text": "hello world"}]
    assert resp.raw["stop_reason"] == "end_turn"

    # Transcript records the synthesised response, not raw SSE.
    files = list((tmp_path / "transcripts").glob("*.json"))
    assert len(files) == 1
    doc = json.loads(files[0].read_text(encoding="utf-8"))
    assert doc["response"]["status"] == 200
    assert doc["response"]["body"]["content"] == [{"type": "text", "text": "hello world"}]


def test_streaming_reassembles_tool_use_input_across_deltas(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sink = TranscriptSink(tmp_path / "transcripts")
    provider = AnthropicProvider(
        api_key="sk-test", model="claude-test", prompt_caching=False, transcript_sink=sink
    )

    def fake_stream(method: str, url: str, **kwargs: Any) -> _FakeStreamResponse:
        return _FakeStreamResponse(status_code=200, lines=_tool_use_stream())

    monkeypatch.setattr(httpx, "stream", fake_stream)

    pieces: list[str] = []
    resp = provider.call(
        system="sys",
        messages=[{"role": "user", "content": "x"}],
        text_delta_callback=pieces.append,
    )

    assert pieces == []  # tool_use has no text_delta callbacks
    assert resp.text == ""
    assert len(resp.tool_uses) == 1
    tu = resp.tool_uses[0]
    assert tu["name"] == "list_dir"
    assert tu["id"] == "tu_xyz"
    assert tu["input"] == {"path": "."}
    assert resp.stop_reason == "tool_use"


def test_streaming_callback_exception_does_not_break_stream(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A buggy TUI callback must NOT take the run down. The callback
    surface is cosmetic; the loop must complete and return the full
    response either way."""
    sink = TranscriptSink(tmp_path / "transcripts")
    provider = AnthropicProvider(
        api_key="sk-test", model="claude-test", prompt_caching=False, transcript_sink=sink
    )

    def fake_stream(method: str, url: str, **kwargs: Any) -> _FakeStreamResponse:
        return _FakeStreamResponse(status_code=200, lines=_basic_text_stream())

    monkeypatch.setattr(httpx, "stream", fake_stream)

    def boom(_piece: str) -> None:
        raise RuntimeError("renderer exploded")

    resp = provider.call(
        system="sys",
        messages=[{"role": "user", "content": "x"}],
        text_delta_callback=boom,
    )

    assert resp.text == "hello world"
    assert resp.stop_reason == "end_turn"


def test_streaming_propagates_http_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    sink = TranscriptSink(tmp_path / "transcripts")
    provider = AnthropicProvider(
        api_key="sk-test", model="claude-test", prompt_caching=False, transcript_sink=sink
    )

    def fake_stream(method: str, url: str, **kwargs: Any) -> _FakeStreamResponse:
        return _FakeStreamResponse(
            status_code=429,
            lines=[],
            error_body='{"error":{"type":"rate_limit"}}',
        )

    monkeypatch.setattr(httpx, "stream", fake_stream)

    with pytest.raises(ProviderError):
        provider.call(
            system="sys",
            messages=[{"role": "user", "content": "x"}],
            text_delta_callback=lambda _p: None,
        )


def test_non_streaming_path_unchanged_when_callback_is_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The default behaviour must NOT call httpx.stream. Bench runs
    rely on the audited non-streaming code path."""
    sink = TranscriptSink(tmp_path / "transcripts")
    provider = AnthropicProvider(
        api_key="sk-test", model="claude-test", prompt_caching=False, transcript_sink=sink
    )

    stream_called = False

    def fake_stream(*_a: Any, **_kw: Any) -> _FakeStreamResponse:
        nonlocal stream_called
        stream_called = True
        return _FakeStreamResponse(status_code=200, lines=[])

    class _R:
        status_code = 200
        text = ""

        def json(self) -> dict[str, Any]:
            return {
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }

    def fake_post(*_a: Any, **_kw: Any) -> _R:
        return _R()

    monkeypatch.setattr(httpx, "stream", fake_stream)
    monkeypatch.setattr(httpx, "post", fake_post)

    resp = provider.call(system="sys", messages=[{"role": "user", "content": "x"}])
    assert resp.text == "ok"
    assert stream_called is False
