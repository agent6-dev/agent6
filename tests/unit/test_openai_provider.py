# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for `agent6.providers.openai.OpenAIProvider`."""

from __future__ import annotations

import json
from typing import Any
from unittest import mock

import httpx
import pytest

from agent6.providers import ProviderError, ToolDefinition
from agent6.providers.openai import OpenAIProvider


def _fake_response(body: dict[str, Any], status: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status,
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        content=json.dumps(body).encode("utf-8"),
        headers={"content-type": "application/json"},
    )


def test_call_translates_messages_and_parses_usage() -> None:
    provider = OpenAIProvider(api_key="sk-test", model="gpt-x")
    captured: dict[str, Any] = {}

    def fake_post(*_a: Any, **kw: Any) -> httpx.Response:
        captured["headers"] = kw["headers"]
        captured["body"] = json.loads(kw["content"])
        return _fake_response(
            {
                "choices": [
                    {
                        "message": {"content": "hello"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 25,
                    "prompt_tokens_details": {"cached_tokens": 40},
                },
            }
        )

    with mock.patch("httpx.post", side_effect=fake_post):
        resp = provider.call(
            system="you are a reviewer",
            messages=[{"role": "user", "content": "judge this"}],
        )

    assert captured["headers"]["authorization"] == "Bearer sk-test"
    assert captured["body"]["model"] == "gpt-x"
    assert captured["body"]["messages"][0] == {"role": "system", "content": "you are a reviewer"}
    assert captured["body"]["messages"][1] == {"role": "user", "content": "judge this"}
    assert resp.text == "hello"
    assert resp.stop_reason == "stop"
    assert resp.input_tokens == 100
    assert resp.output_tokens == 25
    assert resp.cache_read_tokens == 40


def test_call_flattens_anthropic_block_content() -> None:
    provider = OpenAIProvider(api_key="sk", model="gpt-x")
    captured: dict[str, Any] = {}

    def fake_post(*_a: Any, **kw: Any) -> httpx.Response:
        captured["body"] = json.loads(kw["content"])
        return _fake_response({"choices": [{"message": {"content": "ok"}}], "usage": {}})

    msg_content = [
        {"type": "text", "text": "hello "},
        {"type": "text", "text": "world"},
    ]
    with mock.patch("httpx.post", side_effect=fake_post):
        provider.call(system="s", messages=[{"role": "user", "content": msg_content}])

    assert captured["body"]["messages"][1] == {"role": "user", "content": "hello world"}


def test_call_refuses_tools() -> None:
    provider = OpenAIProvider(api_key="sk", model="gpt-x")
    tool = ToolDefinition(name="t", description="d", input_schema={"type": "object"})
    with pytest.raises(ProviderError, match="does not support tool use"):
        provider.call(system="s", messages=[], tools=[tool])


def test_call_raises_provider_error_on_http_status() -> None:
    provider = OpenAIProvider(api_key="sk", model="gpt-x")
    with (
        mock.patch("httpx.post", return_value=_fake_response({"error": "no"}, status=500)),
        pytest.raises(ProviderError, match="OpenAI API error 500"),
    ):
        provider.call(system="s", messages=[{"role": "user", "content": "x"}])


def test_from_env_missing_env_var_yields_no_auth_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty/unset env var is allowed (Ollama-style local endpoint)."""
    monkeypatch.delenv("MY_OAI_KEY", raising=False)
    provider = OpenAIProvider.from_env(model="gpt-x", env_var="MY_OAI_KEY")
    assert provider.api_key == ""

    captured: dict[str, Any] = {}

    def fake_post(_url: str, *_a: Any, **kw: Any) -> httpx.Response:
        captured["headers"] = kw["headers"]
        return _fake_response(
            {
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }
        )

    with mock.patch("httpx.post", side_effect=fake_post):
        provider.call(system="s", messages=[{"role": "user", "content": "x"}])

    assert "authorization" not in {k.lower() for k in captured["headers"]}


def test_from_env_none_env_var_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    """env_var=None is a valid Ollama/llama.cpp shape."""
    provider = OpenAIProvider.from_env(model="gpt-x", env_var=None)
    assert provider.api_key == ""


def test_base_url_override_and_extra_headers() -> None:
    """OpenRouter-style usage: custom endpoint + required identifying headers."""
    provider = OpenAIProvider(
        api_key="or-test",
        model="meta-llama/llama-3.3-70b-instruct",
        base_url="https://openrouter.ai/api/v1",
        extra_headers=(("HTTP-Referer", "https://example.com/r"), ("X-Title", "agent6")),
    )
    captured: dict[str, Any] = {}

    def fake_post(url: str, *_a: Any, **kw: Any) -> httpx.Response:
        captured["url"] = url
        captured["headers"] = kw["headers"]
        return _fake_response({"choices": [{"message": {"content": "k"}}], "usage": {}})

    with mock.patch("httpx.post", side_effect=fake_post):
        provider.call(system="s", messages=[{"role": "user", "content": "x"}])

    assert captured["url"] == "https://openrouter.ai/api/v1/chat/completions"
    # extra_headers are lowercased into the request:
    assert captured["headers"]["http-referer"] == "https://example.com/r"
    assert captured["headers"]["x-title"] == "agent6"
    # default auth still present
    assert captured["headers"]["authorization"] == "Bearer or-test"


def test_from_env_threads_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OR_KEY", "k")
    p = OpenAIProvider.from_env(
        model="m",
        env_var="OR_KEY",
        base_url="http://localhost:11434/v1",
        extra_headers={"X-Title": "t"},
    )
    assert p.base_url == "http://localhost:11434/v1"
    assert p.endpoint == "http://localhost:11434/v1/chat/completions"
    assert dict(p.extra_headers) == {"X-Title": "t"}
