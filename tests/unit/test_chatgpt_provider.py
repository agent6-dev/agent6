# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for ChatGPTProvider (Responses request build, SSE parse, auth)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from agent6.budget import BudgetTracker
from agent6.providers import ProviderError
from agent6.providers.chatgpt import (
    ChatGPTProvider,
    responses_input,
    tools_to_responses,
)
from agent6.providers.chatgpt_oauth import ChatGPTCredential
from agent6.providers.types import ToolDefinition
from agent6.secrets import OAuthTokens, save_oauth_tokens


@pytest.fixture
def signed_in(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> ChatGPTCredential:
    """A gcfg-backed credential holding an unexpired sign-in."""
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(tmp_path / "g"))
    save_oauth_tokens("chatgpt", OAuthTokens("AT0", "RT1", time.time() + 3600, "acct-1"))
    return ChatGPTCredential("chatgpt", issuer="https://auth.example", client_id="app_X")


def _provider(credential: ChatGPTCredential, **kwargs: Any) -> ChatGPTProvider:
    return ChatGPTProvider(
        model="gpt-5-codex",
        credential=credential,
        account_id="acct-1",
        base_url="https://chatgpt.com/backend-api/codex",
        **kwargs,
    )


class _FakeStreamResponse:
    def __init__(self, *, status_code: int, lines: list[str], error_body: str = "") -> None:
        self.status_code = status_code
        self._lines = lines
        self._error_body = error_body
        self.headers: dict[str, str] = {}

    def __enter__(self) -> _FakeStreamResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def iter_lines(self) -> list[str]:
        return self._lines

    def read(self) -> bytes:
        return self._error_body.encode("utf-8")


def _evt(data: dict[str, Any]) -> list[str]:
    return [f"event: {data.get('type', '')}", f"data: {json.dumps(data)}", ""]


_USAGE = {
    "input_tokens": 42,
    "input_tokens_details": {"cached_tokens": 7},
    "output_tokens": 9,
    "total_tokens": 51,
}


def _serve(lines: list[str]):
    """A stream stub serving *lines* regardless of the request."""

    def stream(method: str, url: str, **kwargs: Any) -> _FakeStreamResponse:
        del method, url, kwargs
        return _FakeStreamResponse(status_code=200, lines=lines)

    return stream


def _happy_stream() -> list[str]:
    out: list[str] = []
    out += _evt({"type": "response.created", "response": {"id": "resp_1"}})
    out += _evt({"type": "response.output_text.delta", "delta": "hel"})
    out += _evt({"type": "response.output_text.delta", "delta": "lo"})
    out += _evt(
        {
            "type": "response.output_item.done",
            "item": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hello"}],
            },
        }
    )
    out += _evt(
        {
            "type": "response.completed",
            "response": {"id": "resp_1", "status": "completed", "usage": _USAGE},
        }
    )
    return out


def test_request_body_and_headers_speak_the_codex_dialect(
    signed_in: ChatGPTCredential,
) -> None:
    provider = _provider(signed_in, reasoning_effort="medium")
    captured: dict[str, Any] = {}

    def fake_stream(method: str, url: str, **kwargs: Any) -> _FakeStreamResponse:
        captured["url"] = url
        captured["headers"] = kwargs["headers"]
        captured["body"] = json.loads(kwargs["content"])
        return _FakeStreamResponse(status_code=200, lines=_happy_stream())

    tools = [ToolDefinition(name="read_file", description="d", input_schema={"type": "object"})]
    history: list[dict[str, Any]] = [
        {"role": "user", "content": "task"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "private"},
                {"type": "text", "text": "on it"},
                {"type": "tool_use", "id": "call_1", "name": "read_file", "input": {"p": "."}},
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_1",
                    "content": [{"type": "text", "text": "ok"}],
                }
            ],
        },
    ]
    with mock.patch("httpx2.stream", side_effect=fake_stream):
        resp = provider.call(system="SYS", messages=history, tools=tools, temperature=0.7)

    assert captured["url"] == "https://chatgpt.com/backend-api/codex/responses"
    body = captured["body"]
    assert body["model"] == "gpt-5-codex" and body["instructions"] == "SYS"
    assert body["store"] is False and body["stream"] is True
    assert body["prompt_cache_key"] == provider.session_id
    assert len(provider.session_id) <= 64
    assert body["reasoning"] == {"effort": "medium", "summary": "auto"}
    assert "max_output_tokens" not in body and "temperature" not in body
    assert body["tools"] == [
        {
            "type": "function",
            "name": "read_file",
            "description": "d",
            "parameters": {"type": "object"},
            "strict": False,
        }
    ]
    assert body["input"] == [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "task"}]},
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "on it"}],
        },
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "read_file",
            "arguments": '{"p": "."}',
        },
        {"type": "function_call_output", "call_id": "call_1", "output": "ok"},
    ]
    headers = captured["headers"]
    assert headers["authorization"] == "Bearer AT0"
    assert headers["chatgpt-account-id"] == "acct-1"
    assert headers["originator"] == "agent6"
    assert headers["openai-beta"] == "responses=experimental"
    assert headers["accept"] == "text/event-stream"
    assert headers["session-id"] == provider.session_id
    assert resp.text == "hello"


def test_stream_deltas_feed_callbacks_and_usage_normalises(
    signed_in: ChatGPTCredential,
) -> None:
    provider = _provider(signed_in)
    pieces: list[str] = []
    with mock.patch(
        "httpx2.stream",
        side_effect=_serve(_happy_stream()),
    ):
        resp = provider.call(
            system="s",
            messages=[{"role": "user", "content": "x"}],
            text_delta_callback=pieces.append,
        )
    assert pieces == ["hel", "lo"]
    assert resp.text == "hello" and resp.stop_reason == "end_turn"
    assert (resp.input_tokens, resp.cache_read_tokens, resp.output_tokens) == (35, 7, 9)
    assert resp.cost_usd == 0.0


def test_tool_call_and_reasoning_items_parse(signed_in: ChatGPTCredential) -> None:
    lines: list[str] = []
    lines += _evt(
        {
            "type": "response.output_item.done",
            "item": {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "thought"}],
            },
        }
    )
    lines += _evt(
        {
            "type": "response.output_item.done",
            "item": {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_9",
                "name": "run",
                "arguments": '{"cmd": "ls"}',
            },
        }
    )
    lines += _evt(
        {
            "type": "response.completed",
            "response": {"id": "r", "status": "completed", "usage": _USAGE},
        }
    )
    provider = _provider(signed_in)
    with mock.patch(
        "httpx2.stream",
        side_effect=_serve(lines),
    ):
        resp = provider.call(system="s", messages=[{"role": "user", "content": "x"}])
    assert resp.tool_uses == ({"id": "call_9", "name": "run", "input": {"cmd": "ls"}},)
    assert resp.raw["content"][0] == {"type": "thinking", "thinking": "thought"}


def test_incomplete_max_output_tokens_maps_to_max_tokens(
    signed_in: ChatGPTCredential,
) -> None:
    lines = _evt(
        {
            "type": "response.incomplete",
            "response": {
                "id": "r",
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
                "usage": _USAGE,
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "partial"}],
                    }
                ],
            },
        }
    )
    provider = _provider(signed_in)
    with mock.patch(
        "httpx2.stream",
        side_effect=_serve(lines),
    ):
        resp = provider.call(system="s", messages=[{"role": "user", "content": "x"}])
    assert resp.stop_reason == "max_tokens" and resp.text == "partial"


def test_failed_event_raises_with_usage_limit_status(signed_in: ChatGPTCredential) -> None:
    lines = _evt(
        {
            "type": "response.failed",
            "response": {
                "error": {
                    "code": "usage_limit_reached",
                    "message": "limit hit",
                    "plan_type": "plus",
                }
            },
        }
    )
    provider = _provider(signed_in)
    with (
        mock.patch(
            "httpx2.stream",
            side_effect=_serve(lines),
        ),
        pytest.raises(ProviderError) as exc,
    ):
        provider.call(system="s", messages=[{"role": "user", "content": "x"}])
    assert exc.value.status_code == 429 and "plus plan" in str(exc.value)


def test_cut_stream_is_retryable_not_a_completed_turn(signed_in: ChatGPTCredential) -> None:
    lines = _evt({"type": "response.output_text.delta", "delta": "half"})
    provider = _provider(signed_in)
    with (
        mock.patch(
            "httpx2.stream",
            side_effect=_serve(lines),
        ),
        pytest.raises(ProviderError, match="ended without"),
    ):
        provider.call(system="s", messages=[{"role": "user", "content": "x"}])


def test_401_refreshes_the_credential_once_and_retries(
    signed_in: ChatGPTCredential, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 401 from the backend invalidates the cached token, refreshes via the
    token endpoint, and re-sends with the fresh bearer."""

    def fake_refresh(url: str, data: dict[str, str], timeout_s: float) -> Any:
        class R:
            status_code = 200
            text = ""

            @staticmethod
            def json() -> dict[str, Any]:
                return {"access_token": "AT1", "refresh_token": "RT2", "expires_in": 3600}

        return R()

    monkeypatch.setattr("agent6.providers.chatgpt_oauth._post_form", fake_refresh)
    seen_auth: list[str] = []

    def fake_stream(method: str, url: str, **kwargs: Any) -> _FakeStreamResponse:
        seen_auth.append(kwargs["headers"]["authorization"])
        if len(seen_auth) == 1:
            return _FakeStreamResponse(status_code=401, lines=[], error_body="expired")
        return _FakeStreamResponse(status_code=200, lines=_happy_stream())

    provider = _provider(signed_in)
    with mock.patch("httpx2.stream", side_effect=fake_stream):
        resp = provider.call(system="s", messages=[{"role": "user", "content": "x"}])
    assert seen_auth == ["Bearer AT0", "Bearer AT1"]
    assert resp.text == "hello"


def test_budgeted_call_requires_usage(signed_in: ChatGPTCredential) -> None:
    lines = _evt({"type": "response.completed", "response": {"id": "r", "status": "completed"}})
    provider = _provider(signed_in, budget=BudgetTracker(max_usd=-1, max_tokens_fallback=1_000_000))
    with (
        mock.patch(
            "httpx2.stream",
            side_effect=_serve(lines),
        ),
        pytest.raises(ProviderError, match="usage"),
    ):
        provider.call(system="s", messages=[{"role": "user", "content": "x"}])


def test_responses_input_flattens_odd_content() -> None:
    items = responses_input(
        [
            {"role": "system", "content": "mapped to user"},
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "c", "content": "plain"}],
            },
            {"role": "assistant", "content": ""},
        ]
    )
    assert items == [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "mapped to user"}],
        },
        {"type": "function_call_output", "call_id": "c", "output": "plain"},
    ]
    assert tools_to_responses([])[0:0] == []


def test_responses_input_drops_blank_name_calls_and_their_results() -> None:
    """A blank-name tool_use (another provider's malformed call, carried in a
    resumed history) is skipped together with its paired tool_result, so the
    replayed conversation never holds an output with no matching call."""
    items = responses_input(
        [
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "bad", "name": " ", "input": {}}],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "bad", "content": "x"}],
            },
        ]
    )
    assert items == []
