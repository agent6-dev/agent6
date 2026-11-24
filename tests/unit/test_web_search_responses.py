# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The Responses codec's hosted web search, shared by the generic and the ChatGPT adapters.

The item shapes are the provider's documented ones; no live call verified them.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import time
from typing import Any

import httpx2
import pytest

from agent6.config import ChatGPTProviderEntry, secret_store
from agent6.providers import (
    ToolDefinition,
    anthropic,
    chatgpt,
    oauth,
    oauth_protocol,
    provider_types,
    responses,
    responses_codec,
)

_TOOLS = [ToolDefinition(name="read_file", description="d", input_schema={"type": "object"})]
_USAGE = {"input_tokens": 42, "output_tokens": 9, "total_tokens": 51}

# A reply that reasons, searches once, cites a result and calls one of agent6's own tools.
_REASONING = {"type": "reasoning", "id": "rs_1", "encrypted_content": "X", "summary": []}
_SEARCH_ITEM = {
    "type": "web_search_call",
    "id": "ws_1",
    "status": "completed",
    "action": {"type": "search", "query": "httpx2 release notes"},
}
_MESSAGE = {
    "type": "message",
    "role": "assistant",
    "content": [
        {
            "type": "output_text",
            "text": "The release adds X.",
            "annotations": [
                {
                    "type": "url_citation",
                    "url": "https://example.com/a",
                    "title": "A",
                    "start_index": 4,
                    "end_index": 11,
                },
                {"type": "url_citation", "url": "https://example.com/b", "title": "B"},
                {"type": "url_citation", "url": "https://example.com/a", "title": "A"},
            ],
        }
    ],
}
_CALL = {"type": "function_call", "call_id": "c1", "name": "read_file", "arguments": "{}"}
_OUTPUT: list[dict[str, Any]] = [_REASONING, _SEARCH_ITEM, _MESSAGE, _CALL]
_SEARCH = provider_types.WebSearch(
    id="ws_1",
    query="httpx2 release notes",
    status="completed",
    urls=("https://example.com/a", "https://example.com/b"),
)


class _FakeStream:
    def __init__(self, lines: list[str]) -> None:
        self.status_code = 200
        self.headers: dict[str, str] = {}
        self._lines = lines

    def __enter__(self) -> _FakeStream:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def iter_lines(self) -> list[str]:
        return self._lines


def _evt(data: dict[str, Any]) -> list[str]:
    return [f"event: {data['type']}", f"data: {json.dumps(data)}", ""]


def _stream_lines() -> list[str]:
    lines: list[str] = []
    for item in _OUTPUT:
        lines += _evt({"type": "response.output_item.done", "item": item})
    lines += _evt(
        {"type": "response.completed", "response": {"status": "completed", "usage": _USAGE}}
    )
    return lines


@pytest.fixture
def chatgpt_credential(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> oauth.OAuthCredential:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "g"))
    entry = ChatGPTProviderEntry(backend="chatgpt")
    grant = oauth.stored_grant_for(
        oauth_protocol.TokenGrant("AT0", "RT1", 3600), entry, now=time.time()
    )
    secret_store.save_oauth_grant("chatgpt", dataclasses.replace(grant, account_id="acct-1"))
    return oauth.OAuthCredential("chatgpt", entry)


def _providers(credential: oauth.OAuthCredential) -> dict[str, Any]:
    return {
        "responses": responses.ResponsesProvider(
            api_key="sk-test", model="r-1", base_url="https://api.openai.com/v1"
        ),
        "chatgpt": chatgpt.ChatGPTProvider(
            model="gpt-5-codex",
            credential=credential,
            account_id="acct-1",
            base_url="https://chatgpt.com/backend-api/codex",
        ),
    }


def _ignore(_piece: str) -> None:
    """Take a streamed piece, which opts the call into streaming."""


def _call(
    monkeypatch: pytest.MonkeyPatch,
    provider: Any,
    *,
    streamed: bool,
    messages: list[dict[str, Any]] | None = None,
    **options: Any,
) -> tuple[provider_types.ProviderResponse, dict[str, Any]]:
    """Make one call against the recorded reply and return it with the request body."""
    bodies: list[dict[str, Any]] = []

    def fake_post(url: str, **kwargs: Any) -> httpx2.Response:
        bodies.append(json.loads(kwargs["content"]))
        return httpx2.Response(
            status_code=200,
            request=httpx2.Request("POST", url),
            content=json.dumps({"status": "completed", "output": _OUTPUT, "usage": _USAGE}),
            headers={"content-type": "application/json"},
        )

    def fake_stream(method: str, url: str, **kwargs: Any) -> _FakeStream:
        bodies.append(json.loads(kwargs["content"]))
        return _FakeStream(_stream_lines())

    monkeypatch.setattr("agent6.providers.transport.http_post", fake_post)
    monkeypatch.setattr(httpx2, "stream", fake_stream)
    response = provider.call(
        system="sys",
        messages=messages or [{"role": "user", "content": "what changed?"}],
        tools=_TOOLS,
        text_delta_callback=_ignore if streamed else None,
        **options,
    )
    return response, bodies[-1]


@pytest.mark.parametrize("backend", ["responses", "chatgpt"])
def test_the_request_is_unchanged_until_a_call_offers_search(
    monkeypatch: pytest.MonkeyPatch, chatgpt_credential: oauth.OAuthCredential, backend: str
) -> None:
    """A call that does not offer search sends the bytes it sent before the key existed."""
    provider = _providers(chatgpt_credential)[backend]
    _, plain = _call(monkeypatch, provider, streamed=False)
    _, off = _call(monkeypatch, provider, streamed=False, web_search=False)
    _, on = _call(monkeypatch, provider, streamed=False, web_search=True)
    assert json.dumps(off) == json.dumps(plain)
    assert [tool["type"] for tool in plain["tools"]] == ["function"]
    assert on["tools"][-1] == {"type": "web_search"}
    assert json.dumps({**on, "tools": on["tools"][:-1]}) == json.dumps(plain)


@pytest.mark.parametrize(
    ("backend", "streamed"), [("responses", False), ("responses", True), ("chatgpt", True)]
)
def test_a_reply_with_a_search_parses_and_replays_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    chatgpt_credential: oauth.OAuthCredential,
    backend: str,
    streamed: bool,
) -> None:
    """A search call reaches history whole and returns in place on its own route.

    The canonical content dropped an unknown item before, so the reasoning item ahead of a
    search replayed without the item it led to.
    """
    provider = _providers(chatgpt_credential)[backend]
    response, _ = _call(monkeypatch, provider, streamed=streamed, web_search=True)
    assert response.web_searches == (_SEARCH,)
    assert response.web_searches_billed == 1
    assert response.text == "The release adds X."
    assert [use["name"] for use in response.tool_uses] == ["read_file"]
    assert [block["type"] for block in response.raw["content"]] == [
        "thinking",
        "web_search_call",
        "text",
        "tool_use",
    ]

    history = [
        {"role": "user", "content": "what changed?"},
        {"role": "assistant", "content": response.raw["content"]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "x"}]},
    ]
    _, replay = _call(monkeypatch, provider, streamed=streamed, messages=history, web_search=True)
    sent = replay["input"]
    assert [item["type"] for item in sent] == [
        "message",
        "reasoning",
        "web_search_call",
        "message",
        "function_call",
        "function_call_output",
    ]
    assert json.dumps(sent[2]) == json.dumps(_SEARCH_ITEM)


def test_a_search_call_stays_off_every_other_route(
    monkeypatch: pytest.MonkeyPatch, chatgpt_credential: oauth.OAuthCredential
) -> None:
    """Another Responses route and the Messages wire both drop the block and keep the text."""
    providers = _providers(chatgpt_credential)
    response, _ = _call(monkeypatch, providers["responses"], streamed=False, web_search=True)
    history = [
        {"role": "user", "content": "what changed?"},
        {"role": "assistant", "content": response.raw["content"]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "x"}]},
    ]
    _, other = _call(monkeypatch, providers["chatgpt"], streamed=True, messages=history)
    assert [item["type"] for item in other["input"]] == [
        "message",
        "message",
        "function_call",
        "function_call_output",
    ]
    shaped = anthropic.shape_anthropic_messages(history)
    assert [block["type"] for block in shaped[1]["content"]] == ["text", "tool_use"]


def test_each_distinct_search_id_counts_once() -> None:
    """Two calls bill twice, a repeated item once; citations join the search before them."""
    second = {**_SEARCH_ITEM, "id": "ws_2", "action": {"type": "search", "query": "second"}}
    searches = responses_codec.web_searches([_SEARCH_ITEM, _SEARCH_ITEM, second, _MESSAGE])
    assert [(s.id, s.query, s.urls) for s in searches] == [
        ("ws_1", "httpx2 release notes", ()),
        ("ws_2", "second", _SEARCH.urls),
    ]
