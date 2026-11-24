# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The Messages provider's hosted web search: the request entry and the reply's blocks.

The reply shapes are the provider's documented ones; no live call verified them.
"""

from __future__ import annotations

import copy
import json
from typing import Any

import httpx2
import pytest

from agent6.providers import AnthropicProvider, ToolDefinition, anthropic, provider_types

_TOOLS = [
    ToolDefinition(name="read_file", description="Read a file.", input_schema={"type": "object"}),
    ToolDefinition(name="list_dir", description="List a dir.", input_schema={"type": "object"}),
]

# A reply that searches once, cites a result and calls one of agent6's own tools.
_CONTENT: list[dict[str, Any]] = [
    {"type": "text", "text": "I will look that up."},
    {
        "type": "server_tool_use",
        "id": "srvtoolu_01",
        "name": "web_search",
        "input": {"query": "httpx2 release notes"},
    },
    {
        "type": "web_search_tool_result",
        "tool_use_id": "srvtoolu_01",
        "content": [
            {
                "type": "web_search_result",
                "url": "https://example.com/a",
                "title": "A",
                "encrypted_content": "enc-a",
                "page_age": "1 day ago",
            },
            {
                "type": "web_search_result",
                "url": "https://example.com/b",
                "title": "B",
                "encrypted_content": "enc-b",
                "page_age": None,
            },
        ],
    },
    {
        "type": "text",
        "text": "The release adds X.",
        "citations": [
            {
                "type": "web_search_result_location",
                "url": "https://example.com/a",
                "title": "A",
                "encrypted_index": "idx-a",
                "cited_text": "adds X",
            }
        ],
    },
    {"type": "tool_use", "id": "toolu_01", "name": "read_file", "input": {"path": "a.py"}},
]
_USAGE = {"input_tokens": 30, "output_tokens": 12, "server_tool_use": {"web_search_requests": 1}}
_SEARCH = provider_types.WebSearch(
    id="srvtoolu_01",
    query="httpx2 release notes",
    status="completed",
    urls=("https://example.com/a", "https://example.com/b"),
)


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.status_code = 200
        self.headers: dict[str, str] = {}
        self._payload = payload
        self.text = ""

    def json(self) -> dict[str, Any]:
        return self._payload


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


def _sse(events: list[tuple[str, dict[str, Any]]]) -> list[str]:
    out: list[str] = []
    for event, data in events:
        out.extend((f"event: {event}", f"data: {json.dumps(data)}", ""))
    return out


def _stream_of(content: list[dict[str, Any]]) -> list[str]:
    """Return the reply as the provider streams it: inputs, text and citations in deltas."""
    events: list[tuple[str, dict[str, Any]]] = [
        (
            "message_start",
            {
                "message": {
                    "usage": {
                        "input_tokens": _USAGE["input_tokens"],
                        "server_tool_use": {"web_search_requests": 0},
                    }
                }
            },
        )
    ]
    for index, block in enumerate(content):
        start = copy.deepcopy(block)
        deltas: list[dict[str, Any]] = []
        if block["type"] in ("tool_use", "server_tool_use"):
            start["input"] = {}
            raw = json.dumps(block["input"])
            deltas = [
                {"type": "input_json_delta", "partial_json": raw[:7]},
                {"type": "input_json_delta", "partial_json": raw[7:]},
            ]
        elif block["type"] == "text":
            start = {"type": "text", "text": ""}
            deltas = [
                {"type": "citations_delta", "citation": citation}
                for citation in block.get("citations", ())
            ]
            deltas.append({"type": "text_delta", "text": block["text"]})
        events.append(("content_block_start", {"index": index, "content_block": start}))
        events.extend(("content_block_delta", {"index": index, "delta": d}) for d in deltas)
        events.append(("content_block_stop", {"index": index}))
    events.append(
        (
            "message_delta",
            {
                "delta": {"stop_reason": "tool_use"},
                "usage": {
                    "output_tokens": _USAGE["output_tokens"],
                    "server_tool_use": _USAGE["server_tool_use"],
                },
            },
        )
    )
    events.append(("message_stop", {}))
    return _sse(events)


def _call(
    monkeypatch: pytest.MonkeyPatch,
    *,
    streamed: bool,
    content: list[dict[str, Any]] | None = None,
    messages: list[dict[str, Any]] | None = None,
    **options: Any,
) -> tuple[provider_types.ProviderResponse, dict[str, Any]]:
    """Make one call against the recorded reply and return it with the request body."""
    reply = _CONTENT if content is None else content
    bodies: list[dict[str, Any]] = []

    def fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        bodies.append(json.loads(kwargs["content"]))
        return _FakeResponse(
            {"content": copy.deepcopy(reply), "stop_reason": "tool_use", "usage": _USAGE}
        )

    def fake_stream(method: str, url: str, **kwargs: Any) -> _FakeStream:
        bodies.append(json.loads(kwargs["content"]))
        return _FakeStream(_stream_of(reply))

    monkeypatch.setattr("agent6.providers.transport.http_post", fake_post)
    monkeypatch.setattr(httpx2, "stream", fake_stream)
    provider = AnthropicProvider(api_key="sk-test", model="claude-test")
    response = provider.call(
        system="sys",
        messages=messages or [{"role": "user", "content": "what changed?"}],
        tools=_TOOLS,
        text_delta_callback=(lambda _piece: None) if streamed else None,
        **options,
    )
    return response, bodies[-1]


def test_the_request_is_unchanged_until_a_call_offers_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A call that does not offer search sends the bytes it sent before the key existed."""
    _, plain = _call(monkeypatch, streamed=False)
    _, off = _call(monkeypatch, streamed=False, web_search=False)
    _, on = _call(monkeypatch, streamed=False, web_search=True)
    assert json.dumps(off) == json.dumps(plain)
    assert [tool.get("name") for tool in plain["tools"]] == ["read_file", "list_dir"]
    assert all("type" not in tool for tool in plain["tools"])
    # The hosted entry follows the cache-marked tool, so the marked prefix is the same.
    assert on["tools"][-1] == anthropic.WEB_SEARCH_TOOL
    assert on["tools"][-1]["max_uses"] == 1
    assert json.dumps({**on, "tools": on["tools"][:-1]}) == json.dumps(plain)


@pytest.mark.parametrize("streamed", [False, True])
def test_a_reply_with_a_search_parses_and_replays_unchanged(
    monkeypatch: pytest.MonkeyPatch, streamed: bool
) -> None:
    """A search's blocks reach history as received, so the next request is accepted.

    A streamed `server_tool_use` builds its input from deltas, and a cited text block keeps
    its citations.
    """
    response, _ = _call(monkeypatch, streamed=streamed, web_search=True)
    assert response.web_searches == (_SEARCH,)
    assert response.web_searches_billed == 1
    assert response.text == "I will look that up.\n\nThe release adds X."
    assert [use["name"] for use in response.tool_uses] == ["read_file"]
    assert json.dumps(response.raw["content"]) == json.dumps(_CONTENT)

    history = [
        {"role": "user", "content": "what changed?"},
        {"role": "assistant", "content": response.raw["content"]},
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_01", "content": "ok"}],
        },
    ]
    _, replay = _call(monkeypatch, streamed=streamed, messages=history, web_search=True)
    assert json.dumps(replay["messages"][1]["content"]) == json.dumps(_CONTENT)


@pytest.mark.parametrize("streamed", [False, True])
def test_a_failed_search_reads_as_its_error_code(
    monkeypatch: pytest.MonkeyPatch, streamed: bool
) -> None:
    """An error result is an object, not a result list: it names no URL and bills nothing."""
    content = copy.deepcopy(_CONTENT[:3])
    content[2]["content"] = {
        "type": "web_search_tool_result_error",
        "error_code": "max_uses_exceeded",
    }
    monkeypatch.setitem(_USAGE, "server_tool_use", {"web_search_requests": 0})
    response, _ = _call(monkeypatch, streamed=streamed, content=content, web_search=True)
    assert response.web_searches == (
        provider_types.WebSearch(
            id="srvtoolu_01", query="httpx2 release notes", status="max_uses_exceeded"
        ),
    )
    assert response.web_searches_billed == 0
    assert json.dumps(response.raw["content"]) == json.dumps(content)
