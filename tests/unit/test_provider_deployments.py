# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Wire-level checks for the api_format x deployment x auth provider model:
URL shape, model placement (body vs URL path), protocol-version placement, and
auth-header style, asserted by capturing the mocked HTTP request."""

from __future__ import annotations

import json
from typing import Any
from unittest import mock

import httpx2

from agent6.providers.anthropic import AnthropicProvider
from agent6.providers.openai import OpenAIProvider

_VERTEX_ANTHROPIC = (
    "https://us-east5-aiplatform.googleapis.com/v1/projects/p/locations/us-east5"
    "/publishers/anthropic/models"
)


def _capture(resp: httpx2.Response) -> tuple[dict[str, Any], Any]:
    captured: dict[str, Any] = {}

    def fake_post(*args: Any, **kw: Any) -> httpx2.Response:
        captured["url"] = args[0] if args else kw.get("url")
        captured["headers"] = kw["headers"]
        captured["body"] = json.loads(kw["content"])
        return resp

    return captured, fake_post


def _anthropic_ok() -> httpx2.Response:
    return httpx2.Response(
        200,
        request=httpx2.Request("POST", "https://x"),
        json={
            "content": [{"type": "text", "text": "hi"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 2},
        },
    )


def _openai_ok() -> httpx2.Response:
    return httpx2.Response(
        200,
        request=httpx2.Request("POST", "https://x"),
        json={
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        },
    )


def test_anthropic_direct_wire() -> None:
    p = AnthropicProvider(api_key="sk-ant", model="claude-x")  # deployment/auth default
    captured, fake = _capture(_anthropic_ok())
    with mock.patch("httpx2.post", side_effect=fake):
        p.call(system="s", messages=[{"role": "user", "content": "q"}])
    assert captured["url"] == "https://api.anthropic.com/v1/messages"
    assert captured["headers"]["x-api-key"] == "sk-ant"
    assert captured["headers"]["anthropic-version"] == "2023-06-01"
    assert "authorization" not in captured["headers"]
    assert captured["body"]["model"] == "claude-x"
    assert "anthropic_version" not in captured["body"]


def test_anthropic_vertex_wire() -> None:
    # model in URL (:rawPredict), version in body, bearer auth, no model in body.
    p = AnthropicProvider(
        api_key="ya29.tok",
        model="claude-opus-4-8",
        base_url=_VERTEX_ANTHROPIC,
        deployment="vertex",
        auth_style="bearer",
    )
    captured, fake = _capture(_anthropic_ok())
    with mock.patch("httpx2.post", side_effect=fake):
        p.call(system="s", messages=[{"role": "user", "content": "q"}])
    assert captured["url"] == f"{_VERTEX_ANTHROPIC}/claude-opus-4-8:rawPredict"
    assert captured["headers"]["authorization"] == "Bearer ya29.tok"
    assert "anthropic-version" not in captured["headers"]
    assert captured["body"]["anthropic_version"] == "vertex-2023-10-16"
    assert "model" not in captured["body"]


def test_openai_direct_wire() -> None:
    p = OpenAIProvider(api_key="sk-oai", model="gpt-x")  # bearer default
    captured, fake = _capture(_openai_ok())
    with mock.patch("httpx2.post", side_effect=fake):
        p.call(system="s", messages=[{"role": "user", "content": "q"}])
    assert captured["url"] == "https://api.openai.com/v1/chat/completions"
    assert captured["headers"]["authorization"] == "Bearer sk-oai"
    assert captured["body"]["model"] == "gpt-x"


def test_openai_azure_wire() -> None:
    # deployment name in URL, api-version query param, api-key header, no model in body.
    p = OpenAIProvider(
        api_key="azkey",
        model="my-deployment",
        base_url="https://res.openai.azure.com",
        deployment="azure",
        auth_style="api_key_header",
        extra_query={"api-version": "2024-06-01"},
    )
    captured, fake = _capture(_openai_ok())
    with mock.patch("httpx2.post", side_effect=fake):
        p.call(system="s", messages=[{"role": "user", "content": "q"}])
    assert captured["url"] == (
        "https://res.openai.azure.com/openai/deployments/my-deployment"
        "/chat/completions?api-version=2024-06-01"
    )
    assert captured["headers"]["api-key"] == "azkey"
    assert "authorization" not in captured["headers"]
    assert "model" not in captured["body"]


def test_openai_none_auth_sends_no_auth_header() -> None:
    p = OpenAIProvider(
        api_key="", model="local", base_url="http://localhost:1234/v1", auth_style="none"
    )
    captured, fake = _capture(_openai_ok())
    with mock.patch("httpx2.post", side_effect=fake):
        p.call(system="s", messages=[{"role": "user", "content": "q"}])
    assert "authorization" not in captured["headers"]
    assert "api-key" not in captured["headers"]


def test_egress_allowlist_derives_from_anthropic_base_url() -> None:
    # Security: the egress allow-list must follow a provider's effective base_url
    # host (now configurable for anthropic too), not a hardcoded api.anthropic.com.
    from agent6.cli.egress import _provider_endpoints  # pyright: ignore[reportPrivateUsage]
    from agent6.config import AnthropicProviderEntry

    entry = AnthropicProviderEntry(
        api_format="anthropic", deployment="vertex", base_url=_VERTEX_ANTHROPIC, auth_style="bearer"
    )
    cfg = mock.Mock()
    cfg.providers = {"v": entry}
    hosts = {e.host for e in _provider_endpoints(cfg)}
    assert hosts == {"us-east5-aiplatform.googleapis.com"}
    assert "api.anthropic.com" not in hosts


def test_api_key_header_is_redacted_in_transcripts() -> None:
    from agent6.providers.anthropic import _redact_headers  # pyright: ignore[reportPrivateUsage]

    redacted = _redact_headers({"api-key": "secret", "x-api-key": "s2", "content-type": "json"})
    assert redacted["api-key"] == "<REDACTED>"
    assert redacted["x-api-key"] == "<REDACTED>"
    assert redacted["content-type"] == "json"
