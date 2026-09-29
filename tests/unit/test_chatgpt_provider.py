# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for ChatGPTProvider (Responses request build, SSE parse, auth)."""

from __future__ import annotations

import json
import pathlib
import time
from collections.abc import Iterator
from typing import Any
from unittest import mock

import pytest

from agent6 import budget as agent6_budget
from agent6 import secrets
from agent6.providers import ProviderError, chatgpt, chatgpt_oauth, types


@pytest.fixture
def signed_in(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> chatgpt_oauth.ChatGPTCredential:
    """A gcfg-backed credential holding an unexpired sign-in."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "g"))
    secrets.save_oauth_tokens(
        "chatgpt", secrets.OAuthTokens("AT0", "RT1", time.time() + 3600, "acct-1")
    )
    return chatgpt_oauth.ChatGPTCredential(
        "chatgpt", issuer="https://auth.example", client_id="app_X"
    )


def _provider(
    credential: chatgpt_oauth.ChatGPTCredential, **kwargs: Any
) -> chatgpt.ChatGPTProvider:
    return chatgpt.ChatGPTProvider(
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

    def iter_bytes(self) -> Iterator[bytes]:
        yield self._error_body.encode("utf-8")


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
    signed_in: chatgpt_oauth.ChatGPTCredential,
) -> None:
    provider = _provider(signed_in, reasoning_effort="medium")
    captured: dict[str, Any] = {}

    def fake_stream(method: str, url: str, **kwargs: Any) -> _FakeStreamResponse:
        captured["url"] = url
        captured["headers"] = kwargs["headers"]
        captured["body"] = json.loads(kwargs["content"])
        return _FakeStreamResponse(status_code=200, lines=_happy_stream())

    tools = [
        types.ToolDefinition(name="read_file", description="d", input_schema={"type": "object"})
    ]
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
    assert body["include"] == ["reasoning.encrypted_content"]
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
    signed_in: chatgpt_oauth.ChatGPTCredential,
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


def test_tool_call_and_reasoning_items_parse(signed_in: chatgpt_oauth.ChatGPTCredential) -> None:
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
    thinking = resp.raw["content"][0]
    assert thinking["type"] == "thinking" and thinking["thinking"] == "thought"
    assert thinking["chatgpt_reasoning"]["type"] == "reasoning"


def test_lenient_empty_tool_arguments_are_not_replaced_by_a_raw_sentinel() -> None:
    item = {
        "type": "function_call",
        "call_id": "call_1",
        "name": "list_tasks",
        "arguments": "{} </invoke>",
    }

    response = chatgpt.parse_output_items([item], usage={}, stop_reason="end_turn")

    assert response.tool_uses[0]["input"] == {}


def test_incomplete_max_output_tokens_maps_to_max_tokens(
    signed_in: chatgpt_oauth.ChatGPTCredential,
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


def test_response_done_uses_an_incomplete_response_status(
    signed_in: chatgpt_oauth.ChatGPTCredential,
) -> None:
    message = {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "partial"}],
    }
    lines = _evt(
        {
            "type": "response.done",
            "response": {
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
                "usage": _USAGE,
                "output": [message],
            },
        }
    )
    provider = _provider(signed_in)
    with mock.patch("httpx2.stream", side_effect=_serve(lines)):
        response = provider.call(system="s", messages=[{"role": "user", "content": "x"}])
    assert response.stop_reason == "max_tokens" and response.text == "partial"


def test_response_done_raises_and_bills_a_failed_response(
    signed_in: chatgpt_oauth.ChatGPTCredential,
) -> None:
    lines = _evt(
        {
            "type": "response.done",
            "response": {
                "status": "failed",
                "error": {"code": "server_error", "message": "failed after generation"},
                "usage": _USAGE,
            },
        }
    )
    budget = agent6_budget.BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1)
    provider = _provider(signed_in, budget=budget)
    no_preflight, _ = _usage_get({}, status=503)
    with (
        mock.patch("httpx2.get", side_effect=no_preflight),
        mock.patch("httpx2.stream", side_effect=_serve(lines)),
        pytest.raises(ProviderError, match="failed after generation"),
    ):
        provider.call(system="s", messages=[{"role": "user", "content": "x"}])
    snapshot = budget.snapshot()
    assert (snapshot.input_total, snapshot.cache_read_total, snapshot.output_total) == (35, 7, 9)


def test_response_done_raises_and_bills_a_cancelled_response(
    signed_in: chatgpt_oauth.ChatGPTCredential,
) -> None:
    lines = _evt({"type": "response.done", "response": {"status": "cancelled", "usage": _USAGE}})
    budget = agent6_budget.BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1)
    provider = _provider(signed_in, budget=budget)
    no_preflight, _ = _usage_get({}, status=503)
    with (
        mock.patch("httpx2.get", side_effect=no_preflight),
        mock.patch("httpx2.stream", side_effect=_serve(lines)),
        pytest.raises(ProviderError, match="cancelled"),
    ):
        provider.call(system="s", messages=[{"role": "user", "content": "x"}])
    snapshot = budget.snapshot()
    assert (snapshot.input_total, snapshot.cache_read_total, snapshot.output_total) == (35, 7, 9)


def test_failed_event_raises_with_usage_limit_status(
    signed_in: chatgpt_oauth.ChatGPTCredential,
) -> None:
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


def test_failed_event_usage_is_recorded(signed_in: chatgpt_oauth.ChatGPTCredential) -> None:
    lines = _evt(
        {
            "type": "response.failed",
            "response": {
                "error": {"code": "server_error", "message": "failed after generation"},
                "usage": _USAGE,
            },
        }
    )
    budget = agent6_budget.BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1)
    provider = _provider(signed_in, budget=budget)
    no_preflight, _ = _usage_get({}, status=503)

    with (
        mock.patch("httpx2.get", side_effect=no_preflight),
        mock.patch("httpx2.stream", side_effect=_serve(lines)),
        pytest.raises(ProviderError, match="failed after generation"),
    ):
        provider.call(system="s", messages=[{"role": "user", "content": "x"}])

    snapshot = budget.snapshot()
    assert (snapshot.input_total, snapshot.cache_read_total, snapshot.output_total) == (35, 7, 9)


def test_cut_stream_is_retryable_not_a_completed_turn(
    signed_in: chatgpt_oauth.ChatGPTCredential,
) -> None:
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
    signed_in: chatgpt_oauth.ChatGPTCredential, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 401 refreshes the credential once and re-sends with the fresh bearer."""

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


def test_budgeted_call_requires_usage(signed_in: chatgpt_oauth.ChatGPTCredential) -> None:
    lines = _evt({"type": "response.completed", "response": {"id": "r", "status": "completed"}})
    provider = _provider(
        signed_in,
        budget=agent6_budget.BudgetTracker(
            max_usd=-1, max_tokens_fallback=1_000_000, max_percent=-1
        ),
    )
    with (
        mock.patch(
            "httpx2.stream",
            side_effect=_serve(lines),
        ),
        pytest.raises(ProviderError, match="usage"),
    ):
        provider.call(system="s", messages=[{"role": "user", "content": "x"}])


def test_malformed_usage_count_is_a_provider_error(
    signed_in: chatgpt_oauth.ChatGPTCredential,
) -> None:
    usage = {**_USAGE, "output_tokens": "nine"}
    lines = _evt(
        {
            "type": "response.completed",
            "response": {"id": "r", "status": "completed", "usage": usage},
        }
    )
    provider = _provider(signed_in)

    with (
        mock.patch("httpx2.stream", side_effect=_serve(lines)),
        pytest.raises(ProviderError, match=r"usage\.output_tokens"),
    ):
        provider.call(system="s", messages=[{"role": "user", "content": "x"}])


@pytest.mark.parametrize("value", [-1, 1.5, True, {}, [], ""])
def test_usage_counts_reject_non_integer_values(value: object) -> None:
    with pytest.raises(ProviderError, match=r"usage\.output_tokens"):
        chatgpt.parse_output_items(
            [], usage={"input_tokens": 1, "output_tokens": value}, stop_reason="end_turn"
        )


def test_usage_counts_accept_integer_strings() -> None:
    response = chatgpt.parse_output_items(
        [],
        usage={
            "input_tokens": "10",
            "input_tokens_details": {"cached_tokens": "3"},
            "output_tokens": "2",
        },
        stop_reason="end_turn",
    )
    assert (response.input_tokens, response.cache_read_tokens, response.output_tokens) == (7, 3, 2)


def test_responses_input_flattens_odd_content() -> None:
    items = chatgpt.responses_input(
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
    assert chatgpt.tools_to_responses([])[0:0] == []


def test_responses_input_keeps_multiple_notices_separated() -> None:
    """Several harness notices in one turn stay separate text blocks."""
    items = chatgpt.responses_input(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "[harness verify] the gate is red"},
                    {"type": "text", "text": "[no-progress] you have made no progress"},
                ],
            }
        ]
    )
    assert items == [
        {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": (
                        "[harness verify] the gate is red\n\n"
                        "[no-progress] you have made no progress"
                    ),
                }
            ],
        }
    ]


def test_responses_input_drops_blank_name_calls_and_their_results() -> None:
    """A blank-name tool_use is dropped together with its paired tool_result."""
    items = chatgpt.responses_input(
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


def test_plan_usage_headers_feed_the_percent_budget(
    signed_in: chatgpt_oauth.ChatGPTCredential,
) -> None:
    """The x-codex primary-window headers feed the percent budget as plan-metered spend."""
    provider = _provider(
        signed_in,
        budget=agent6_budget.BudgetTracker(max_usd=10.0, max_tokens_fallback=100, max_percent=-1),
    )

    def stream(method: str, url: str, **kwargs: Any) -> _FakeStreamResponse:
        resp = _FakeStreamResponse(status_code=200, lines=_happy_stream())
        resp.headers = {
            "x-codex-primary-used-percent": "37",
            "x-codex-primary-window-minutes": "10080",
            "x-codex-primary-reset-at": "2000000000",
        }
        return resp

    with mock.patch("httpx2.stream", side_effect=stream):
        provider.call(system="s", messages=[{"role": "user", "content": "x"}])
    assert provider.budget is not None
    snap = provider.budget.snapshot()
    assert snap.plan_latest is not None
    assert snap.plan_latest.used_percent == 37.0
    assert snap.plan_latest.window_minutes == 10080
    assert snap.unmetered_tokens == 0
    assert "plan usage (gpt-5-codex): 37% of the 7-day window" in provider.budget.format_summary()


def test_http_error_plan_headers_reach_the_budget(
    signed_in: chatgpt_oauth.ChatGPTCredential,
) -> None:
    budget = agent6_budget.BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1)
    provider = _provider(signed_in, budget=budget)
    no_preflight, _ = _usage_get({}, status=503)

    def stream(method: str, url: str, **kwargs: Any) -> _FakeStreamResponse:
        del method, url, kwargs
        response = _FakeStreamResponse(status_code=429, lines=[], error_body="usage limit")
        response.headers = {
            "x-codex-primary-used-percent": "100",
            "x-codex-primary-window-minutes": "10080",
            "x-codex-primary-reset-at": "2000000000",
        }
        return response

    with (
        mock.patch("httpx2.get", side_effect=no_preflight),
        mock.patch("httpx2.stream", side_effect=stream),
        pytest.raises(ProviderError) as error,
    ):
        provider.call(system="s", messages=[{"role": "user", "content": "x"}])

    assert error.value.status_code == 429
    plan = budget.snapshot().plan_latest
    assert plan is not None and plan.used_percent == 100.0


def test_completed_stream_without_message_item_keeps_delta_text(
    signed_in: chatgpt_oauth.ChatGPTCredential,
) -> None:
    """A completed stream with no final message item still yields the streamed text.

    The turn's other blocks stay in history behind it.
    """
    lines: list[str] = []
    lines += _evt({"type": "response.output_text.delta", "delta": "half"})
    lines += _evt({"type": "response.output_text.delta", "delta": " answer"})
    lines += _evt(
        {
            "type": "response.output_item.done",
            "item": {"type": "function_call", "call_id": "c1", "name": "run", "arguments": "{}"},
        }
    )
    lines += _evt(
        {
            "type": "response.completed",
            "response": {"id": "r", "status": "completed", "usage": _USAGE},
        }
    )
    provider = _provider(signed_in)
    with mock.patch("httpx2.stream", side_effect=_serve(lines)):
        resp = provider.call(system="s", messages=[{"role": "user", "content": "x"}])
    assert resp.text == "half answer"
    assert [b["type"] for b in resp.raw["content"]] == ["text", "tool_use"]


def test_terminal_output_supplies_items_missing_from_done_events(
    signed_in: chatgpt_oauth.ChatGPTCredential,
) -> None:
    message = {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "I'll inspect it."}],
    }
    call = {
        "type": "function_call",
        "call_id": "c1",
        "name": "read_file",
        "arguments": '{"path": "x.py"}',
    }
    lines = _evt({"type": "response.output_item.done", "item": message})
    lines += _evt(
        {
            "type": "response.completed",
            "response": {"status": "completed", "usage": _USAGE, "output": [message, call]},
        }
    )

    provider = _provider(signed_in)
    with mock.patch("httpx2.stream", side_effect=_serve(lines)):
        response = provider.call(system="s", messages=[{"role": "user", "content": "x"}])

    assert response.text == "I'll inspect it."
    assert response.tool_uses == ({"id": "c1", "name": "read_file", "input": {"path": "x.py"}},)


def test_tool_calling_completed_turn_reports_tool_use(
    signed_in: chatgpt_oauth.ChatGPTCredential,
) -> None:
    """A completed response holding function_call items says stop_reason tool_use.

    That is what arms the loop's empty-tool-call contradiction detector for this wire.
    """
    lines: list[str] = []
    lines += _evt(
        {
            "type": "response.output_item.done",
            "item": {"type": "function_call", "call_id": "c1", "name": "run", "arguments": "{}"},
        }
    )
    lines += _evt(
        {
            "type": "response.completed",
            "response": {"id": "r", "status": "completed", "usage": _USAGE},
        }
    )
    provider = _provider(signed_in)
    with mock.patch("httpx2.stream", side_effect=_serve(lines)):
        resp = provider.call(system="s", messages=[{"role": "user", "content": "x"}])
    assert resp.stop_reason == "tool_use" and len(resp.tool_uses) == 1


def test_plan_usage_parses_the_credits_family() -> None:
    """The x-codex credits headers feed the paid-credit guard: has, unlimited and balance."""
    plan = chatgpt._plan_usage_of(
        {
            "x-codex-primary-used-percent": "100",
            "x-codex-primary-window-minutes": "10080",
            "x-codex-primary-reset-at": "2000000000",
            "x-codex-credits-has-credits": "true",
            "x-codex-credits-unlimited": "false",
            "x-codex-credits-balance": " $12.50 ",
        }
    )
    assert plan is not None
    assert plan.has_credits is True
    assert plan.credits_unlimited is False
    assert plan.credits_balance == "$12.50"
    bare = chatgpt._plan_usage_of({"x-codex-primary-used-percent": "40"})
    assert bare is not None and bare.has_credits is False


def test_reasoning_items_are_captured_and_replayed_in_order() -> None:
    """Reasoning items are kept opaque in wire position and replayed before their function_call.

    With store=false the encrypted item is the model's own chain-of-thought state.
    """
    reasoning = {
        "type": "reasoning",
        "id": "rs_1",
        "encrypted_content": "OPAQUE",
        "summary": [{"type": "summary_text", "text": "plan"}],
    }
    call = {
        "type": "function_call",
        "call_id": "c1",
        "name": "read_file",
        "arguments": '{"path": "x"}',
    }
    got = chatgpt.parse_output_items([reasoning, call], usage={}, stop_reason="end_turn")
    blocks = got.raw["content"]
    assert [b["type"] for b in blocks] == ["thinking", "tool_use"]
    assert blocks[0]["thinking"] == "plan" and blocks[0]["chatgpt_reasoning"] == reasoning

    items = chatgpt.responses_input(
        [
            {"role": "assistant", "content": blocks},
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "hi"}],
            },
        ]
    )
    assert [i["type"] for i in items] == ["reasoning", "function_call", "function_call_output"]
    assert items[0] == reasoning


def test_interleaved_items_persist_and_replay_in_wire_order() -> None:
    """Interleaved items persist as one block per output item and replay in wire order.

    The commentary message never hoists ahead of the reasoning that produced it; a display-only
    thinking block replays nothing.
    """
    r1 = {"type": "reasoning", "id": "rs_1", "encrypted_content": "A", "summary": []}
    note = {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "Looking at x."}],
    }
    r2 = {
        "type": "reasoning",
        "id": "rs_2",
        "encrypted_content": "B",
        "summary": [{"type": "summary_text", "text": "then read"}],
    }
    call = {"type": "function_call", "call_id": "c1", "name": "read_file", "arguments": "{}"}
    got = chatgpt.parse_output_items([r1, note, r2, call], usage={}, stop_reason="end_turn")
    blocks = got.raw["content"]
    assert [b["type"] for b in blocks] == ["thinking", "text", "thinking", "tool_use"]
    assert got.text == "Looking at x." and blocks[0]["thinking"] == ""
    assert blocks[2]["thinking"] == "then read"

    items = chatgpt.responses_input(
        [
            {
                "role": "assistant",
                "content": [{"type": "thinking", "thinking": "foreign"}, *blocks],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "body"}],
            },
        ]
    )
    assert [i.get("id") or i["type"] for i in items] == [
        "rs_1",
        "message",
        "rs_2",
        "function_call",
        "function_call_output",
    ]

    # A final answer keeps its reasoning ahead of the message too.
    final = chatgpt.parse_output_items([r1, note], usage={}, stop_reason="end_turn")
    items = chatgpt.responses_input([{"role": "assistant", "content": final.raw["content"]}])
    assert [i.get("id") or i["type"] for i in items] == ["rs_1", "message"]


def test_orphaned_reasoning_is_dropped_with_its_call() -> None:
    """A reasoning item whose paired call is dropped is dropped with it.

    An orphan 400s the request.
    """
    item = {"type": "reasoning", "id": "rs_1"}
    blocks = [
        {"type": "thinking", "thinking": "", "chatgpt_reasoning": item},
        {"type": "tool_use", "id": "c1", "name": "", "input": {}},
    ]
    items = chatgpt.responses_input([{"role": "assistant", "content": blocks}])
    assert items == []


def test_reasoning_without_a_following_output_item_is_not_replayed() -> None:
    item = {"type": "reasoning", "id": "rs_orphan", "encrypted_content": "OPAQUE"}

    items = chatgpt.responses_input(
        [
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "", "chatgpt_reasoning": item},
                    {"type": "text", "text": ""},
                ],
            }
        ]
    )

    assert items == []


def test_responses_input_drops_idless_tool_pairs() -> None:
    items = chatgpt.responses_input(
        [
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "name": "read_file", "input": {}}],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "content": "orphaned"}],
            },
        ]
    )
    assert items == []


_USAGE_BODY: dict[str, Any] = {
    "plan_type": "pro",
    "rate_limit": {
        "allowed": True,
        "limit_reached": False,
        "primary_window": {
            "used_percent": 38,
            "limit_window_seconds": 604800,
            "reset_after_seconds": 435664,
            "reset_at": 1787867609,
        },
        "secondary_window": None,
    },
    "credits": {"has_credits": False, "unlimited": False, "balance": "0"},
}


def test_usage_body_parses_both_windows_and_the_credit_family() -> None:
    plan = chatgpt.plan_usage_from_usage_body(_USAGE_BODY)
    assert plan is not None
    assert plan.used_percent == 38.0 and plan.window_minutes == 10080
    assert plan.resets_at == 1787867609.0 and [w.name for w in plan.windows] == ["primary"]
    assert plan.has_credits is False and plan.window_exhausted is False
    body = json.loads(json.dumps(_USAGE_BODY))
    body["rate_limit"]["limit_reached"] = True
    body["rate_limit"]["secondary_window"] = {"used_percent": 100}
    body["rate_limit"]["gpt-5.6-spark_window"] = {"used_percent": 55, "limit_window_seconds": 18000}
    body["credits"] = {"has_credits": True, "unlimited": False, "balance": "$12.50"}
    plan = chatgpt.plan_usage_from_usage_body(body)
    assert plan is not None
    assert [(w.name, w.used_percent) for w in plan.windows] == [
        ("primary", 38.0),
        ("secondary", 100.0),
        ("gpt-5.6-spark", 55.0),
    ]
    # The binding window is the one closest to its cap, whatever its name.
    assert plan.binding.name == "secondary" and plan.used_percent == 100.0
    assert plan.limit_reached and plan.has_credits and plan.credits_balance == "$12.50"
    assert plan.credits_usd == 12.5
    assert plan.window_exhausted
    assert chatgpt.plan_usage_from_usage_body({"credits": {}}) is None


def test_usage_body_does_not_treat_string_false_as_unlimited_credits() -> None:
    plan = chatgpt.plan_usage_from_usage_body(
        {
            "rate_limit": {"primary_window": {"used_percent": 100}},
            "credits": {"has_credits": True, "unlimited": "false", "balance": "500"},
        }
    )
    assert plan is not None and plan.credits_unlimited is False
    budget = agent6_budget.BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1)
    budget.record_plan_preflight("chatgpt", plan)
    with pytest.raises(agent6_budget.BudgetExceededError, match="purchased"):
        budget.check()


def test_secondary_window_header_rides_into_the_reading() -> None:
    plan = chatgpt._plan_usage_of(
        {"x-codex-primary-used-percent": "12", "x-codex-secondary-used-percent": "100"}
    )
    assert plan is not None and plan.binding.name == "secondary" and plan.used_percent == 100.0
    assert plan.window_exhausted


def test_every_used_percent_header_family_is_a_window() -> None:
    """Every used-percent header family is a window, parsed unnamed and binding when tightest.

    Primary stays first; a family with no primary is no reading.
    """
    plan = chatgpt._plan_usage_of(
        {
            "X-Codex-Primary-Used-Percent": "12",
            "x-codex-primary-window-minutes": "10080",
            "x-codex-primary-reset-at": "1787867609",
            "x-codex-gpt-5-6-spark-used-percent": "97.5",
            "x-codex-gpt-5-6-spark-window-minutes": "300",
            "x-codex-gpt-5-6-spark-reset-after-seconds": "600",
            "x-codex-credits-balance": "500",
        }
    )
    assert plan is not None
    assert [w.name for w in plan.windows] == ["primary", "gpt-5-6-spark"]
    assert plan.binding.name == "gpt-5-6-spark" and plan.used_percent == 97.5
    assert plan.window_minutes == 300 and 0 < plan.resets_at - time.time() <= 600
    # 500 credits at 25 per dollar (1,000-credit packs at $40): the header carries credits.
    assert plan.credits_usd == 20.0 and not plan.window_exhausted
    assert chatgpt._plan_usage_of({"x-codex-gpt-5-6-spark-used-percent": "97.5"}) is None


def test_nonfinite_plan_metadata_is_ignored() -> None:
    from_headers = chatgpt._plan_usage_of(
        {
            "x-codex-primary-used-percent": "10",
            "x-codex-primary-window-minutes": "inf",
            "x-codex-primary-reset-at": "nan",
        }
    )
    assert from_headers is not None
    assert from_headers.window_minutes == 0
    assert from_headers.resets_at > 0

    body = {
        "rate_limit": {
            "primary_window": {
                "used_percent": 10,
                "limit_window_seconds": "inf",
                "reset_at": "nan",
            }
        }
    }
    from_body = chatgpt.plan_usage_from_usage_body(body)
    assert from_body is not None
    assert from_body.window_minutes == 0
    assert from_body.resets_at > 0
    assert chatgpt._plan_usage_of({"x-codex-primary-used-percent": "nan"}) is None
    assert chatgpt._plan_usage_of({"x-codex-primary-used-percent": "-1"}) is None
    assert (
        chatgpt.plan_usage_from_usage_body(
            {"rate_limit": {"primary_window": {"used_percent": "inf"}}}
        )
        is None
    )
    assert (
        chatgpt.plan_usage_from_usage_body({"rate_limit": {"primary_window": {"used_percent": -1}}})
        is None
    )


def _usage_get(body: dict[str, Any], status: int = 200):
    class _Resp:
        status_code = status

        def json(self) -> Any:
            return body

    calls: list[str] = []

    def get(url: str, **kwargs: Any) -> _Resp:
        calls.append(url)
        return _Resp()

    return get, calls


def test_preflight_refuses_a_credit_spending_run_before_its_first_call(
    signed_in: chatgpt_oauth.ChatGPTCredential,
) -> None:
    """The preflight refuses a credit-spending run before its first call.

    An exhausted window with purchased credits and `allow_paid_credits = false` refuses with no
    request sent, and the reading seeds the percent ledger; with the knob on, the call proceeds.
    """
    body = json.loads(json.dumps(_USAGE_BODY))
    body["rate_limit"]["limit_reached"] = True
    body["credits"] = {"has_credits": True, "unlimited": False, "balance": "$5.00"}
    get, urls = _usage_get(body)
    streamed: list[str] = []

    def stream(method: str, url: str, **kwargs: Any) -> _FakeStreamResponse:
        streamed.append(url)
        return _FakeStreamResponse(status_code=200, lines=_happy_stream())

    provider = _provider(
        signed_in,
        budget=agent6_budget.BudgetTracker(max_usd=10.0, max_tokens_fallback=100, max_percent=-1),
    )
    with (
        mock.patch("httpx2.get", side_effect=get),
        mock.patch("httpx2.stream", side_effect=stream),
        pytest.raises(agent6_budget.BudgetExceededError, match="purchased"),
    ):
        provider.call(system="s", messages=[{"role": "user", "content": "x"}])
    assert urls == ["https://chatgpt.com/backend-api/codex/usage"] and streamed == []
    assert provider.budget is not None and provider.budget.snapshot().plan_latest is not None

    allowed = _provider(
        signed_in,
        budget=agent6_budget.BudgetTracker(
            max_usd=10.0, max_tokens_fallback=100, max_percent=-1, allow_paid_credits=True
        ),
    )
    with mock.patch("httpx2.get", side_effect=get), mock.patch("httpx2.stream", side_effect=stream):
        allowed.call(system="s", messages=[{"role": "user", "content": "x"}])
        allowed.call(system="s", messages=[{"role": "user", "content": "y"}])
    assert len(streamed) == 2 and len(urls) == 2  # one preflight per provider


def test_preflight_failure_never_blocks(signed_in: chatgpt_oauth.ChatGPTCredential) -> None:
    """A preflight transport error or non-200 reads as no reading, and the call proceeds."""
    get, _urls = _usage_get({}, status=503)
    provider = _provider(
        signed_in,
        budget=agent6_budget.BudgetTracker(max_usd=10.0, max_tokens_fallback=100, max_percent=-1),
    )
    with (
        mock.patch("httpx2.get", side_effect=get),
        mock.patch("httpx2.stream", side_effect=_serve(_happy_stream())),
    ):
        resp = provider.call(system="s", messages=[{"role": "user", "content": "x"}])
    assert resp.text == "hello"
    with (
        mock.patch("httpx2.get", side_effect=OSError("down")),
        mock.patch("httpx2.stream", side_effect=_serve(_happy_stream())),
    ):
        fresh = _provider(
            signed_in,
            budget=agent6_budget.BudgetTracker(
                max_usd=10.0, max_tokens_fallback=100, max_percent=-1
            ),
        )
        resp = fresh.call(system="s", messages=[{"role": "user", "content": "x"}])
    assert resp.text == "hello"


class _FlakyCredential(chatgpt_oauth.ChatGPTCredential):
    """The signed-in credential, raising once from `token()`."""

    def __init__(self) -> None:
        super().__init__("chatgpt", issuer="https://auth.example", client_id="app_X")
        self.faults = 1

    def token(self) -> str:
        if self.faults:
            self.faults -= 1
            raise ProviderError("refresh failed")
        return super().token()


def test_a_credential_fault_in_the_preflight_is_no_reading(
    signed_in: chatgpt_oauth.ChatGPTCredential,
) -> None:
    """A credential fault in the preflight is no reading.

    The call's own token path still runs.
    """
    get, _urls = _usage_get({}, status=200)
    provider = _provider(
        _FlakyCredential(),
        budget=agent6_budget.BudgetTracker(max_usd=10.0, max_tokens_fallback=100, max_percent=-1),
    )
    with (
        mock.patch("httpx2.get", side_effect=get),
        mock.patch("httpx2.stream", side_effect=_serve(_happy_stream())),
    ):
        resp = provider.call(system="s", messages=[{"role": "user", "content": "x"}])
    assert resp.text == "hello"


def test_a_completed_round_the_guard_refuses_still_books_its_plan_window(
    signed_in: chatgpt_oauth.ChatGPTCredential,
) -> None:
    """A completed round the guard refuses still books the plan window its headers reported.

    A `response.completed` with no `usage` body trips the no-input-tokens refusal.
    """
    lines = _evt({"type": "response.output_text.delta", "delta": "hello"})
    lines += _evt({"type": "response.completed", "response": {"output": [], "status": "completed"}})

    def stream(method: str, url: str, **kwargs: Any) -> _FakeStreamResponse:
        del method, url, kwargs
        resp = _FakeStreamResponse(status_code=200, lines=lines)
        resp.headers = {
            "x-codex-primary-used-percent": "37",
            "x-codex-primary-window-minutes": "10080",
            "x-codex-primary-reset-at": "2000000000",
        }
        return resp

    budget = agent6_budget.BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1)
    provider = _provider(signed_in, budget=budget)
    with (
        mock.patch("httpx2.stream", side_effect=stream),
        pytest.raises(types.ProviderError, match="no usage input tokens"),
    ):
        provider.call(system="s", messages=[{"role": "user", "content": "x"}])
    snap = budget.snapshot()
    assert "gpt-5-codex" in snap.per_model, "the refused round is on the ledger"
    assert snap.plan_latest is not None and snap.plan_latest.used_percent == 37.0


def test_two_message_items_in_a_response_stay_separated() -> None:
    """Two message items in a response stay separated in the settled text."""
    resp = chatgpt.parse_output_items(
        [
            {"type": "message", "content": [{"type": "output_text", "text": "First message."}]},
            {"type": "reasoning", "summary": [{"text": "thinking"}]},
            {"type": "message", "content": [{"type": "output_text", "text": "Second message."}]},
        ],
        usage={"input_tokens": 1, "output_tokens": 1},
        stop_reason="end_turn",
    )
    assert resp.text == "First message.\n\nSecond message."
