# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for Anthropic extended-thinking wiring.

Validate that the provider's ``thinking`` level (off/low/medium/high):

* enables the ``thinking`` block in the request body with the mapped
  ``budget_tokens`` and lifts ``max_tokens`` above it;
* drops ``temperature`` (Anthropic rejects temperature with thinking);
* leaves the body unchanged when ``thinking`` is off/unset;
* preserves streamed ``thinking`` blocks (with signature) in the
  reconstructed assistant content so they round-trip on the next turn.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx2
import pytest

from agent6.providers import AnthropicProvider, TranscriptSink
from agent6.providers.anthropic import (
    _THINKING_BUDGET_TOKENS,  # pyright: ignore[reportPrivateUsage]
)


class _FakeResponse:
    def __init__(self, *, status_code: int, payload: dict[str, Any]) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = ""

    def json(self) -> dict[str, Any]:
        return self._payload


def _ok_payload() -> dict[str, Any]:
    return {
        "content": [{"type": "text", "text": "hi"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


def _capture_body(monkeypatch: pytest.MonkeyPatch, bodies: list[dict[str, Any]]) -> None:
    def fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        bodies.append(json.loads(kwargs["content"]))
        return _FakeResponse(status_code=200, payload=_ok_payload())

    monkeypatch.setattr(httpx2, "post", fake_post)


@pytest.mark.parametrize("level", ["low", "medium", "high"])
def test_thinking_enables_budget_and_drops_temperature(
    monkeypatch: pytest.MonkeyPatch, level: str
) -> None:
    bodies: list[dict[str, Any]] = []
    _capture_body(monkeypatch, bodies)
    provider = AnthropicProvider(
        api_key="sk-test", model="claude-test", prompt_caching=False, thinking=level
    )
    provider.call(
        system="sys",
        messages=[{"role": "user", "content": "x"}],
        temperature=0.0,
        max_tokens=8192,
    )
    body = bodies[0]
    budget = _THINKING_BUDGET_TOKENS[level]
    assert body["thinking"] == {"type": "enabled", "budget_tokens": budget}
    # Temperature must not be sent while thinking is enabled.
    assert "temperature" not in body
    # max_tokens must exceed the thinking budget so the model can answer.
    assert body["max_tokens"] > budget


def test_thinking_off_is_a_plain_call(monkeypatch: pytest.MonkeyPatch) -> None:
    bodies: list[dict[str, Any]] = []
    _capture_body(monkeypatch, bodies)
    provider = AnthropicProvider(
        api_key="sk-test", model="claude-test", prompt_caching=False, thinking="off"
    )
    provider.call(
        system="sys",
        messages=[{"role": "user", "content": "x"}],
        temperature=0.3,
        max_tokens=8192,
    )
    body = bodies[0]
    assert "thinking" not in body
    assert body["temperature"] == 0.3
    assert body["max_tokens"] == 8192


def test_thinking_unset_is_a_plain_call(monkeypatch: pytest.MonkeyPatch) -> None:
    bodies: list[dict[str, Any]] = []
    _capture_body(monkeypatch, bodies)
    provider = AnthropicProvider(api_key="sk-test", model="claude-test", prompt_caching=False)
    provider.call(
        system="sys",
        messages=[{"role": "user", "content": "x"}],
        temperature=0.0,
        max_tokens=8192,
    )
    assert "thinking" not in bodies[0]


class _FakeStreamResponse:
    def __init__(self, *, status_code: int, lines: list[str]) -> None:
        self.status_code = status_code
        self._lines = lines

    def __enter__(self) -> _FakeStreamResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def iter_lines(self) -> list[str]:
        return self._lines

    def read(self) -> bytes:
        return b""


def _sse(events: list[tuple[str, dict[str, Any]]]) -> list[str]:
    out: list[str] = []
    for et, data in events:
        out.append(f"event: {et}")
        out.append(f"data: {json.dumps(data)}")
        out.append("")
    return out


def _thinking_then_tool_stream() -> list[str]:
    """A thinking block (with signature) followed by a tool_use block."""
    return _sse(
        [
            (
                "message_start",
                {
                    "message": {
                        "usage": {
                            "input_tokens": 5,
                            "cache_read_input_tokens": 0,
                            "cache_creation_input_tokens": 0,
                        }
                    }
                },
            ),
            (
                "content_block_start",
                {"index": 0, "content_block": {"type": "thinking", "thinking": ""}},
            ),
            (
                "content_block_delta",
                {"index": 0, "delta": {"type": "thinking_delta", "thinking": "let me "}},
            ),
            (
                "content_block_delta",
                {"index": 0, "delta": {"type": "thinking_delta", "thinking": "think"}},
            ),
            (
                "content_block_delta",
                {"index": 0, "delta": {"type": "signature_delta", "signature": "sig-abc"}},
            ),
            ("content_block_stop", {"index": 0}),
            (
                "content_block_start",
                {
                    "index": 1,
                    "content_block": {
                        "type": "tool_use",
                        "id": "tu_1",
                        "name": "list_dir",
                        "input": {},
                    },
                },
            ),
            (
                "content_block_delta",
                {
                    "index": 1,
                    "delta": {"type": "input_json_delta", "partial_json": '{"path": "."}'},
                },
            ),
            ("content_block_stop", {"index": 1}),
            (
                "message_delta",
                {"delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 7}},
            ),
            ("message_stop", {}),
        ]
    )


def test_streaming_preserves_thinking_blocks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sink = TranscriptSink(tmp_path / "transcripts")
    provider = AnthropicProvider(
        api_key="sk-test",
        model="claude-test",
        prompt_caching=False,
        transcript_sink=sink,
        thinking="high",
    )

    def fake_stream(method: str, url: str, **kwargs: Any) -> _FakeStreamResponse:
        return _FakeStreamResponse(status_code=200, lines=_thinking_then_tool_stream())

    monkeypatch.setattr(httpx2, "stream", fake_stream)

    resp = provider.call(
        system="sys",
        messages=[{"role": "user", "content": "x"}],
        text_delta_callback=lambda _s: None,
    )

    # The thinking block must survive (with its signature) and precede the
    # tool_use block, exactly as Anthropic requires when echoing the
    # assistant turn back on the next request.
    blocks = resp.raw["content"]
    assert blocks[0] == {
        "type": "thinking",
        "thinking": "let me think",
        "signature": "sig-abc",
    }
    assert blocks[1]["type"] == "tool_use"
    assert blocks[1]["input"] == {"path": "."}
    assert resp.stop_reason == "tool_use"
