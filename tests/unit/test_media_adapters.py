from __future__ import annotations

import base64
import contextlib
import json
from typing import Any

import httpx2
import pytest

from agent6 import kinds, media
from agent6.harness import _conversation, _provider_call
from agent6.providers import (
    Provider,
    _headers,
    _openai_messages,
    _responses,
    _stream,
    _transport,
    anthropic,
    chatgpt,
    openai,
    responses,
)


class _Credential:
    def token(self) -> str:
        return "test-token"

    def invalidate(self, status: int, rejected_bearer: str) -> bool:
        return False


def _provider(api: str) -> Provider:
    if api == "anthropic":
        return anthropic.AnthropicProvider(api_key="key", model="m", prompt_caching=False)
    if api == "openai":
        return openai.OpenAIProvider(api_key="key", model="m")
    if api == "responses":
        return responses.ResponsesProvider(api_key="key", model="m", base_url="https://test/v1")
    return chatgpt.ChatGPTProvider(
        model="m", credential=_Credential(), account_id="test", base_url="https://test/codex"
    )


def _conversation_with_media(mime: str) -> _conversation.Conversation:
    conv = _conversation.Conversation()
    conv.notice("inspect", origin="operator")
    conv.assistant(
        [
            {"type": "tool_use", "id": "a", "name": "read_file", "input": {"path": "dir/page"}},
            {"type": "tool_use", "id": "b", "name": "read_file", "input": {"path": "note.txt"}},
            {"type": "tool_use", "id": "c", "name": "read_file", "input": {"path": "other"}},
        ]
    )
    assistant = conv.messages[-1]
    assert isinstance(assistant, _conversation.AssistantMessage)
    conv.results(
        [
            _conversation.ToolResultItem(
                call.id,
                "result " + call.id,
                call,
                kinds.MediaAttachment(b"pixels" * 20, mime, str(call.input["path"]))
                if mime and call.id != "b"
                else None,
            )
            for call in assistant.tool_uses
        ]
    )
    return conv


def _response(api: str) -> dict[str, Any]:
    if api == "anthropic":
        return {"content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn"}
    if api == "openai":
        return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
    return {
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "ok"}],
            }
        ],
        "usage": {"input_tokens": 10, "output_tokens": 1},
    }


@pytest.mark.parametrize("api", ["anthropic", "openai", "responses", "chatgpt"])
@pytest.mark.parametrize("mime", ["", "image/png", "application/pdf"])
def test_media_wire_bodies_and_attribution(
    monkeypatch: pytest.MonkeyPatch, api: str, mime: str
) -> None:
    """Attachments follow all parallel replies without changing the initiating role."""
    conv = _conversation_with_media(mime)
    # One document per request; the second media result stays text-only for this case.
    if mime == "application/pdf":
        conv.set_result_content(2, 2, "result c", drop_attachment=True)
    canonical = conv.to_wire()
    captured: list[dict[str, Any]] = []
    facts: list[_headers.RequestFacts] = []

    def post(*args: Any, **kwargs: Any) -> httpx2.Response:
        captured.append(json.loads(kwargs["content"]))
        return httpx2.Response(200, json=_response(api))

    def stream(*args: Any, **kwargs: Any) -> contextlib.AbstractContextManager[httpx2.Response]:
        captured.append(json.loads(kwargs["content"]))
        event = {"type": "response.completed", "response": _response(api)}
        return contextlib.nullcontext(
            httpx2.Response(200, text="data: " + json.dumps(event) + "\n\n")
        )

    monkeypatch.setattr(_transport, "http_post", post)
    monkeypatch.setattr(_stream, "http_stream", stream)
    _provider(api).call(
        system="s",
        messages=canonical,
        context=_headers.RequestContext("sid", False, False, conv.initiator, observe=facts.append),
    )
    body = captured[0]
    assert facts[0].has_images is (mime == "image/png")
    assert facts[0].context is not None and facts[0].context.initiator == "agent"
    assert canonical == conv.to_wire()
    if api == "anthropic":
        assert body["messages"] == canonical
        return
    encoded = base64.b64encode(b"pixels" * 20).decode()
    parts: list[dict[str, Any]] = []
    for call_id, path in [("a", "dir/page"), ("c", "other")]:
        if not mime or (mime == "application/pdf" and call_id == "c"):
            continue
        is_responses = api != "openai"
        kind = "document" if mime == "application/pdf" else "image"
        parts.append(
            {
                "type": "input_text" if is_responses else "text",
                "text": f"{kind} from tool call {call_id}: {path}",
            }
        )
        url = f"data:{mime};base64,{encoded}"
        if kind == "image":
            parts.append(
                {"type": "input_image", "image_url": url}
                if is_responses
                else {"type": "image_url", "image_url": {"url": url}}
            )
        else:
            file = {"filename": "page", "file_data": url}
            parts.append(
                {"type": "input_file", **file} if is_responses else {"type": "file", "file": file}
            )
    if api == "openai":
        expected: list[dict[str, Any]] = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "inspect"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": b["id"],
                        "type": "function",
                        "function": {"name": "read_file", "arguments": json.dumps(b["input"])},
                    }
                    for b in canonical[1]["content"]
                ],
            },
            *[
                {"role": "tool", "tool_call_id": call_id, "content": "result " + call_id}
                for call_id in "abc"
            ],
        ]
        if parts:
            expected.append({"role": "user", "content": parts})
        assert body["messages"] == expected
    else:
        expected = [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "inspect"}],
            },
            *[
                {
                    "type": "function_call",
                    "call_id": b["id"],
                    "name": "read_file",
                    "arguments": json.dumps(b["input"]),
                }
                for b in canonical[1]["content"]
            ],
            *[
                {"type": "function_call_output", "call_id": call_id, "output": "result " + call_id}
                for call_id in "abc"
            ],
        ]
        if parts:
            expected.append({"type": "message", "role": "user", "content": parts})
        assert body["input"] == expected


@pytest.mark.parametrize("api", ["anthropic", "openai", "responses", "chatgpt"])
@pytest.mark.parametrize(
    "mime, code", [("image/png", "image_input_not_supported"), ("application/pdf", "invalid_file")]
)
def test_adapter_media_rejection_retries_text_only(
    monkeypatch: pytest.MonkeyPatch, api: str, mime: str, code: str
) -> None:
    """A real adapter error body removes media once and keeps every tool reply."""
    conv = _conversation_with_media(mime)
    captured: list[dict[str, Any]] = []
    facts: list[_headers.RequestFacts] = []
    error = {"error": {"code": code, "message": "rejected"}}

    def reply(kwargs: dict[str, Any], *, streaming: bool) -> httpx2.Response:
        captured.append(json.loads(kwargs["content"]))
        if len(captured) == 1:
            return httpx2.Response(400, json=error)
        if streaming:
            event = {"type": "response.completed", "response": _response(api)}
            return httpx2.Response(200, text="data: " + json.dumps(event) + "\n\n")
        return httpx2.Response(200, json=_response(api))

    def post(*args: Any, **kwargs: Any) -> httpx2.Response:
        return reply(kwargs, streaming=False)

    def stream(*args: Any, **kwargs: Any) -> contextlib.AbstractContextManager[httpx2.Response]:
        return contextlib.nullcontext(reply(kwargs, streaming=True))

    monkeypatch.setattr(_transport, "http_post", post)
    monkeypatch.setattr(_stream, "http_stream", stream)
    provider = _provider(api)
    recovery = _provider_call.MediaRecovery(media.InputCapabilities(kinds.ModelRoute(api, "m")))
    result = _provider_call.call_with_media_recovery(
        lambda messages: provider.call(
            system="s",
            messages=messages,
            context=_headers.RequestContext(
                "sid", False, False, conv.initiator, observe=facts.append
            ),
        ),
        conv,
        recovery,
    )
    assert result.text == "ok" and len(captured) == 2
    assert [f.has_images for f in facts] == [mime == "image/png", False]
    assert media.input_modalities(captured[0])
    assert not media.input_modalities(captured[1])
    assert "Cannot deliver" in json.dumps(captured[1])
    assert "dir/page" in json.dumps(captured[1])


@pytest.mark.parametrize("dialect", ["openai", "responses"])
@pytest.mark.parametrize("tool_input", ["bad arguments", [1], 1])
def test_media_path_lookup_preserves_malformed_text_only_calls(
    dialect: str, tool_input: Any
) -> None:
    """Media labels cannot make a text-only call's non-object arguments crash replay."""
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "a", "name": "read_file", "input": tool_input},
            ],
        }
    ]
    if dialect == "openai":
        outgoing = _openai_messages.anthropic_to_openai_messages("s", messages)
        arguments = outgoing[-1]["tool_calls"][0]["function"]["arguments"]
    else:
        outgoing = _responses.responses_input(messages, route="r")
        arguments = outgoing[-1]["arguments"]
    assert json.loads(arguments) == tool_input
