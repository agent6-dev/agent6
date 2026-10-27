# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`api_format = "responses"`: the generic Responses adapter over a configured endpoint."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any
from unittest import mock

import httpx2
import pytest

from agent6 import budget as agent6_budget
from agent6.config import Config, ResponsesProviderEntry
from agent6.models import registry
from agent6.providers import ProviderError, ToolDefinition, _responses, responses, wire

_BASE = "https://models.example/v1"
_USAGE = {
    "input_tokens": 42,
    "input_tokens_details": {"cached_tokens": 7},
    "output_tokens": 9,
    "total_tokens": 51,
}


def _provider(**kwargs: Any) -> responses.ResponsesProvider:
    return responses.ResponsesProvider(api_key="sk-test", model="r-1", base_url=_BASE, **kwargs)


def _json_response(body: dict[str, Any], status: int = 200) -> httpx2.Response:
    return httpx2.Response(
        status_code=status,
        request=httpx2.Request("POST", f"{_BASE}/responses"),
        content=json.dumps(body).encode("utf-8"),
        headers={"content-type": "application/json"},
    )


def _message(text: str) -> dict[str, Any]:
    return {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text}],
    }


class _FakeStreamResponse:
    def __init__(self, *, lines: list[str]) -> None:
        self.status_code = 200
        self._lines = lines
        self.headers: dict[str, str] = {}

    def __enter__(self) -> _FakeStreamResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def iter_lines(self) -> list[str]:
        return self._lines

    def read(self) -> bytes:
        return b""

    def iter_bytes(self) -> Iterator[bytes]:
        yield b""


def _evt(data: dict[str, Any]) -> list[str]:
    return [f"event: {data.get('type', '')}", f"data: {json.dumps(data)}", ""]


def test_the_entry_needs_a_base_url_and_the_direct_deployment() -> None:
    entry = Config.model_validate(
        {"providers": {"work": {"api_format": "responses", "base_url": _BASE}}}
    ).providers["work"]
    assert isinstance(entry, ResponsesProviderEntry)
    assert (entry.auth_style, entry.deployment) == ("bearer", "direct")
    with pytest.raises(Exception, match="base_url is required"):
        Config.model_validate({"providers": {"work": {"api_format": "responses"}}})
    with pytest.raises(Exception, match="direct"):
        Config.model_validate(
            {
                "providers": {
                    "work": {"api_format": "responses", "base_url": _BASE, "deployment": "azure"}
                }
            }
        )


def test_the_url_appends_responses_and_the_effort_passes_through() -> None:
    url, in_body = wire.request_url(
        api_format="responses", deployment="direct", base_url=_BASE, model="r-1", streaming=True
    )
    assert (url, in_body) == (f"{_BASE}/responses", True)
    cfg = Config.model_validate(
        {
            "providers": {"work": {"api_format": "responses", "base_url": _BASE}},
            "models": {"worker": {"provider": "work", "model": "r-1", "effort": "high"}},
        }
    )
    assert registry.role_effort(cfg, "worker") == "high"


def test_a_json_call_sends_the_contract_and_parses_the_response_object() -> None:
    captured: dict[str, Any] = {}

    def fake_post(*_a: Any, **kw: Any) -> httpx2.Response:
        captured["headers"] = kw["headers"]
        captured["body"] = json.loads(kw["content"])
        return _json_response(
            {
                "id": "resp_1",
                "status": "completed",
                "output": [
                    {"type": "reasoning", "id": "rs_1", "encrypted_content": "X", "summary": []},
                    _message("hello"),
                    {
                        "type": "function_call",
                        "call_id": "c1",
                        "name": "read_file",
                        "arguments": '{"path": "x"}',
                    },
                ],
                "usage": _USAGE,
            }
        )

    provider = _provider(
        reasoning_effort="off",
        extra_body={"store": True, "previous_response_id": "resp_0", "metadata": {"k": "v"}},
    )
    tools = [ToolDefinition(name="read_file", description="d", input_schema={"type": "object"})]
    with mock.patch("agent6.providers._transport.http_post", side_effect=fake_post):
        resp = provider.call(
            system="be brief",
            messages=[{"role": "user", "content": "go"}],
            tools=tools,
            max_tokens=777,
            temperature=0.2,
        )
    body = captured["body"]
    assert captured["headers"]["authorization"] == "Bearer sk-test"
    assert body["instructions"] == "be brief" and body["max_output_tokens"] == 777
    assert body["store"] is False and "previous_response_id" not in body
    assert body["metadata"] == {"k": "v"} and body["temperature"] == 0.2
    assert body["reasoning"] == {"effort": "none", "summary": "auto"}
    assert body["include"] == ["reasoning.encrypted_content"] and "stream" not in body
    assert body["tools"][0]["name"] == "read_file" and body["input"][0]["type"] == "message"
    assert resp.text == "hello" and resp.stop_reason == "tool_use"
    assert resp.tool_uses == ({"id": "c1", "name": "read_file", "input": {"path": "x"}},)
    assert (resp.input_tokens, resp.cache_read_tokens, resp.output_tokens) == (35, 7, 9)
    route = _responses.replay_route("responses", _BASE, "r-1")
    assert resp.raw["content"][0]["reasoning_replay"]["route"] == route
    # The next call replays the reasoning item on this route and keeps the pairing.
    items = _responses.responses_input(
        [
            {"role": "assistant", "content": resp.raw["content"]},
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "x"}],
            },
        ],
        route=route,
    )
    assert [i["type"] for i in items] == [
        "reasoning",
        "message",
        "function_call",
        "function_call_output",
    ]


def test_an_incomplete_response_reads_as_the_output_cap() -> None:
    def fake_post(*_a: Any, **kw: Any) -> httpx2.Response:
        return _json_response(
            {
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
                "output": [_message("partial")],
                "usage": _USAGE,
            }
        )

    with mock.patch("agent6.providers._transport.http_post", side_effect=fake_post):
        resp = _provider().call(system="s", messages=[{"role": "user", "content": "go"}])
    assert resp.stop_reason == "max_tokens" and resp.text == "partial"


def test_a_failed_response_object_raises_its_error() -> None:
    def fake_post(*_a: Any, **kw: Any) -> httpx2.Response:
        return _json_response(
            {"status": "failed", "error": {"code": "server_error", "message": "boom"}}
        )

    with (
        mock.patch("agent6.providers._transport.http_post", side_effect=fake_post),
        pytest.raises(ProviderError, match="server_error: boom"),
    ):
        _provider().call(system="s", messages=[{"role": "user", "content": "go"}])


def test_a_budgeted_call_without_usage_is_refused_and_a_priced_one_is_booked() -> None:
    budget = agent6_budget.BudgetTracker(max_usd=10.0, max_tokens_fallback=-1, max_plan_points=-1)

    def unmetered(*_a: Any, **kw: Any) -> httpx2.Response:
        return _json_response({"status": "completed", "output": [_message("x")], "usage": {}})

    with (
        mock.patch("agent6.providers._transport.http_post", side_effect=unmetered),
        pytest.raises(ProviderError, match=r"usage\.input_tokens"),
    ):
        _provider(budget=budget).call(system="s", messages=[{"role": "user", "content": "go"}])

    def metered(*_a: Any, **kw: Any) -> httpx2.Response:
        return _json_response({"status": "completed", "output": [_message("x")], "usage": _USAGE})

    with mock.patch("agent6.providers._transport.http_post", side_effect=metered):
        _provider(budget=budget).call(system="s", messages=[{"role": "user", "content": "go"}])
    snap = budget.snapshot()
    assert (snap.input_total, snap.cache_read_total, snap.output_total) == (35, 7, 9)


def test_a_temperature_refusal_drops_it_for_the_run() -> None:
    bodies: list[dict[str, Any]] = []

    def fake_post(*_a: Any, **kw: Any) -> httpx2.Response:
        body = json.loads(kw["content"])
        bodies.append(body)
        if "temperature" in body:
            return _json_response(
                {"error": {"message": "Unsupported parameter: 'temperature'"}}, status=400
            )
        return _json_response({"status": "completed", "output": [_message("ok")], "usage": _USAGE})

    provider = _provider()
    with mock.patch("agent6.providers._transport.http_post", side_effect=fake_post):
        first = provider.call(
            system="s", messages=[{"role": "user", "content": "go"}], temperature=0.5
        )
        second = provider.call(
            system="s", messages=[{"role": "user", "content": "go"}], temperature=0.5
        )
    assert first.text == "ok" and second.text == "ok"
    assert [("temperature" in b) for b in bodies] == [True, False, False]


def test_a_stream_feeds_the_callbacks_and_settles_on_the_terminal_response() -> None:
    lines: list[str] = []
    lines += _evt({"type": "response.created", "response": {"id": "resp_1"}})
    lines += _evt({"type": "response.output_text.delta", "delta": "hel"})
    lines += _evt({"type": "response.reasoning_summary_text.delta", "delta": "think"})
    lines += _evt({"type": "response.output_text.delta", "delta": "lo"})
    lines += _evt({"type": "response.output_item.done", "item": _message("hello")})
    lines += _evt(
        {
            "type": "response.completed",
            "response": {"id": "resp_1", "status": "completed", "usage": _USAGE},
        }
    )
    seen: dict[str, Any] = {}

    def stream(method: str, url: str, **kwargs: Any) -> _FakeStreamResponse:
        seen["url"] = url
        seen["headers"] = kwargs["headers"]
        seen["body"] = json.loads(kwargs["content"])
        return _FakeStreamResponse(lines=lines)

    text: list[str] = []
    thinking: list[str] = []
    budget = agent6_budget.BudgetTracker(max_usd=10.0, max_tokens_fallback=-1, max_plan_points=-1)
    with mock.patch("agent6.providers._stream.http_stream", side_effect=stream):
        resp = _provider(budget=budget).call(
            system="s",
            messages=[{"role": "user", "content": "go"}],
            text_delta_callback=text.append,
            thinking_delta_callback=thinking.append,
        )
    assert seen["url"] == f"{_BASE}/responses" and seen["body"]["stream"] is True
    assert seen["headers"]["accept"] == "text/event-stream"
    assert (text, thinking) == (["hel", "lo"], ["think"])
    assert resp.text == "hello" and resp.stop_reason == "end_turn"
    assert budget.snapshot().output_total == 9


def test_a_cut_stream_is_truncated_and_books_what_it_cost() -> None:
    lines: list[str] = []
    lines += _evt({"type": "response.output_text.delta", "delta": "hel"})

    def stream(method: str, url: str, **kwargs: Any) -> _FakeStreamResponse:
        return _FakeStreamResponse(lines=lines)

    with (
        mock.patch("agent6.providers._stream.http_stream", side_effect=stream),
        pytest.raises(ProviderError, match="cut off"),
    ):
        _provider().call(
            system="s",
            messages=[{"role": "user", "content": "go"}],
            text_delta_callback=lambda _s: None,
        )


def test_a_failed_stream_books_the_usage_it_reports() -> None:
    lines: list[str] = []
    lines += _evt({"type": "response.output_text.delta", "delta": "hel"})
    lines += _evt(
        {
            "type": "response.failed",
            "response": {"status": "failed", "error": {"code": "server_error"}, "usage": _USAGE},
        }
    )

    def stream(method: str, url: str, **kwargs: Any) -> _FakeStreamResponse:
        return _FakeStreamResponse(lines=lines)

    budget = agent6_budget.BudgetTracker(max_usd=10.0, max_tokens_fallback=-1, max_plan_points=-1)
    with (
        mock.patch("agent6.providers._stream.http_stream", side_effect=stream),
        pytest.raises(ProviderError, match="server_error"),
    ):
        _provider(budget=budget).call(
            system="s", messages=[{"role": "user", "content": "go"}], text_delta_callback=print
        )
    snap = budget.snapshot()
    assert (snap.input_total, snap.cache_read_total, snap.output_total) == (35, 7, 9)
