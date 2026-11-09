from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import pathlib
import threading
from collections.abc import Iterator
from typing import Any
from unittest import mock

import httpx2
import pytest

from agent6 import portable
from agent6.providers import openai, provider_types, request_headers, stream_lifecycle
from agent6.sessions import objects

_SECRET = "sk-transcript-secret-123456789"
_USAGE = {
    "prompt_tokens": 125,
    "completion_tokens": 10,
    "prompt_tokens_details": {"cached_tokens": 25},
    "cost": 0.0,
}


def _chunk(data: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(data)}\n\n".encode()


class _AttemptStream(httpx2.SyncByteStream):
    def __init__(self, ending: str) -> None:
        self.ending = ending
        self.stop = threading.Event()
        self.closed = threading.Event()

    def __iter__(self) -> Iterator[bytes]:
        yield _chunk(
            {
                "id": "response-id",
                "model": "served-model",
                "choices": [{"delta": {"content": "partial answer"}}],
            }
        )
        yield _chunk({"choices": [], "usage": _USAGE})
        if self.ending in ("aborted", "interrupted", "idle"):
            self.stop.set()
            assert self.closed.wait(5), "watchdog did not close the stream"
            raise httpx2.ReadError("stream closed")
        if self.ending == "cancelled":
            raise asyncio.CancelledError("task cancelled")
        if self.ending == "keyboard":
            raise KeyboardInterrupt("operator cancelled")
        if self.ending == "network":
            raise httpx2.ReadError("connection dropped")
        if self.ending == "malformed":
            yield _chunk({"choices": [{"delta": {"content": {"bad": "shape"}}}]})
            return
        if self.ending == "truncated":
            return
        if self.ending == "final_parse":
            yield _chunk(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": index,
                                        "id": "duplicate",
                                        "function": {"name": "list_dir", "arguments": "{}"},
                                    }
                                    for index in range(2)
                                ]
                            }
                        }
                    ]
                }
            )
        yield _chunk({"choices": [{"delta": {}, "finish_reason": "stop"}]})
        yield b"data: [DONE]\n\n"

    def close(self) -> None:
        self.closed.set()


@pytest.mark.parametrize(
    "ending, outcome",
    [
        ("completed", "completed"),
        ("aborted", "aborted"),
        ("interrupted", "interrupted"),
        ("malformed", "failed"),
        ("final_parse", "failed"),
        ("http_error", "failed"),
        ("network", "failed"),
        ("truncated", "failed"),
        ("idle", "failed"),
        ("cancelled", "aborted"),
        ("keyboard", "aborted"),
    ],
)
def test_stream_attempt_keeps_one_record_with_its_outcome(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, ending: str, outcome: str
) -> None:
    """Every stream exit finalizes its pre-send record with partial evidence and observed usage."""
    monkeypatch.setattr(stream_lifecycle, "STREAM_WATCHDOG_TICK_S", 0.01)
    monkeypatch.setattr(stream_lifecycle, "STREAM_IDLE_TIMEOUT_S", 0.05 if ending == "idle" else 45)
    writes: list[tuple[pathlib.Path, dict[str, Any]]] = []
    real_write = portable.atomic_write

    def write(path: pathlib.Path, data: str | bytes) -> None:
        if path.parent.name != "objects":
            writes.append((path, json.loads(data)))
        real_write(path, data)

    monkeypatch.setattr(portable, "atomic_write", write)
    stream = _AttemptStream(ending)
    response = (
        httpx2.Response(429, text="rate limited", headers={"retry-after": "7"})
        if ending == "http_error"
        else httpx2.Response(200, stream=stream, headers={"x-request-id": "upstream-id"})
    )
    provider = openai.OpenAIProvider(
        api_key=_SECRET,
        model="m",
        transcript_sink=provider_types.TranscriptSink(
            tmp_path, objects_dir=tmp_path / "objects"
        ).for_seat("worker"),
    )
    error: BaseException | None = None
    with mock.patch("httpx2.stream", return_value=contextlib.closing(response)):
        try:
            provider.call(
                system="s",
                messages=[{"role": "user", "content": "hi"}],
                text_delta_callback=lambda _: None,
                should_abort=stream.stop.is_set if ending == "aborted" else None,
                should_interrupt=stream.stop.is_set if ending == "interrupted" else None,
                context=request_headers.RequestContext.initiated_by(
                    "user", call_context=provider_types.CallContext(model_call_id="logical-call")
                ),
            )
        except (provider_types.ProviderError, asyncio.CancelledError, KeyboardInterrupt) as exc:
            error = exc
    assert (error is None) == (ending == "completed")
    if ending == "aborted":
        assert isinstance(error, provider_types.ProviderAborted)
    if ending == "interrupted":
        assert isinstance(error, provider_types.ProviderInterrupted)
    if ending == "final_parse":
        assert isinstance(error, provider_types.ProviderError)
        assert "duplicate tool_call.id" in str(error)
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    doc = json.loads(files[0].read_text())
    assert doc["outcome"] == outcome
    assert doc["model_call_id"] == "logical-call"
    assert doc["request_id"]
    assert doc["seat"] == "worker"
    assert len(writes) == 2
    assert writes[0][0] == writes[1][0] == files[0]
    assert writes[0][1]["outcome"] == "incomplete"
    assert writes[0][1]["request"] == doc["request"]
    assert (writes[0][1]["ts"], writes[0][1]["seq"]) == (doc["ts"], doc["seq"])
    assert _SECRET not in files[0].read_text()
    if ending == "http_error":
        assert doc["response"]["status"] == 429
        assert doc["response"]["body"] == "rate limited"
        assert doc["response"]["headers"]["retry-after"] == "7"
        assert doc["usage"] is None
    else:
        assert doc["response"]["status"] == 200
        assert doc["response"]["headers"]["x-request-id"] == "upstream-id"
        assert "events" not in doc["response"]
        assert doc["usage"] == {
            "input_tokens": 100,
            "output_tokens": 10,
            "cache_read_tokens": 25,
            "cache_creation_tokens": 0,
            "cost_usd": 0.0,
        }
    if ending != "completed":
        assert doc["error"]


def test_redacted_attempt_is_on_disk_before_the_http_send(tmp_path: pathlib.Path) -> None:
    """The pre-send record uses credential and media redaction before any network call."""
    encoded = base64.b64encode(b"private image pixels" * 8).decode()
    pre_send: list[dict[str, Any]] = []

    def send(*args: Any, **kwargs: Any) -> Any:
        for path in tmp_path.glob("*.json"):
            raw = json.loads(path.read_text())
            assert "$object" in raw["request"]["body"]["messages"][0]
            pre_send.append(objects.materialize(raw, tmp_path / "objects", record_path=path))
        assert len(pre_send) == 1, "request was sent without a transcript"
        raise httpx2.ConnectError("no route")

    provider = openai.OpenAIProvider(
        api_key=_SECRET,
        model="m",
        transcript_sink=provider_types.TranscriptSink(tmp_path, objects_dir=tmp_path / "objects"),
    )
    with mock.patch("httpx2.stream", side_effect=send), pytest.raises(provider_types.ProviderError):
        provider.call(
            system=_SECRET,
            messages=[
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "image-call",
                            "name": "read_file",
                            "input": {"path": "image.png"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "image-call",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": encoded,
                                    },
                                }
                            ],
                        }
                    ],
                },
            ],
            text_delta_callback=lambda _: None,
        )
    assert pre_send[0]["outcome"] == "incomplete"
    assert pre_send[0]["request"]["body"]["stream"] is True
    assert pre_send[0]["request"]["headers"]["authorization"] == "<REDACTED>"
    assert pre_send[0]["usage"] is None
    text = json.dumps(pre_send[0])
    assert _SECRET not in text and encoded not in text
    assert "payload_omitted" in text
    (path,) = tmp_path.glob("*.json")
    final = objects.materialize(json.loads(path.read_text()), tmp_path / "objects")
    assert final["request"] == pre_send[0]["request"]
    assert final["outcome"] == "failed"


@pytest.mark.parametrize("header", ["authorization", "set-cookie"])
def test_response_credentials_are_redacted_from_all_evidence(
    tmp_path: pathlib.Path, header: str
) -> None:
    """Retained response headers cannot expose credentials through headers or echoed deltas."""
    secret = "response-only-credential-123456789"
    value = f"Bearer {secret}" if header == "authorization" else f"session={secret}; HttpOnly"
    response = httpx2.Response(
        200,
        headers={header: value},
        content=_chunk(
            {"choices": [{"delta": {"content": value + " " + _SECRET}, "finish_reason": "stop"}]}
        )
        + b"data: [DONE]\n\n",
    )
    provider = openai.OpenAIProvider(
        api_key=_SECRET,
        model="m",
        transcript_sink=provider_types.TranscriptSink(tmp_path, objects_dir=tmp_path / "objects"),
    )
    with mock.patch("httpx2.stream", return_value=contextlib.closing(response)):
        provider.call(system="s", messages=[], text_delta_callback=lambda _: None)
    (path,) = tmp_path.glob("*.json")
    text = path.read_text()
    assert secret not in text and _SECRET not in text
    assert json.loads(text)["response"]["headers"][header] == "<REDACTED>"


@pytest.mark.parametrize("status", [200, 429, 302])
def test_nonstreaming_attempt_is_written_before_send_and_finalized(
    tmp_path: pathlib.Path, status: int
) -> None:
    """The JSON path uses the same pre-send and final-outcome contract as SSE."""
    pre_send: list[dict[str, Any]] = []

    def send(*args: Any, **kwargs: Any) -> httpx2.Response:
        pre_send.extend(json.loads(path.read_text()) for path in tmp_path.glob("*.json"))
        return httpx2.Response(
            status,
            json={
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": _USAGE,
            },
        )

    provider = openai.OpenAIProvider(
        api_key=_SECRET,
        model="m",
        transcript_sink=provider_types.TranscriptSink(tmp_path, objects_dir=tmp_path / "objects"),
    )
    with mock.patch("agent6.providers.transport.http_post", side_effect=send):
        if status == 200:
            provider.call(system="s", messages=[])
        else:
            with pytest.raises(provider_types.ProviderError):
                provider.call(system="s", messages=[])
    assert len(pre_send) == 1, "request was sent without a transcript"
    assert pre_send[0]["outcome"] == "incomplete"
    (path,) = tmp_path.glob("*.json")
    assert json.loads(path.read_text())["outcome"] == ("completed" if status == 200 else "failed")


def _streamed_text(doc: dict[str, Any]) -> str:
    """Join the content deltas a reader could reassemble from retained wire events."""
    parts: list[str] = []
    for event in doc["response"].get("events", []):
        try:
            data = json.loads(event["data"])
        except ValueError:
            continue
        parts.extend(c.get("delta", {}).get("content") or "" for c in data.get("choices", []))
    return "".join(parts)


def test_a_credential_split_across_deltas_never_reaches_a_failed_record(
    tmp_path: pathlib.Path,
) -> None:
    """A secret echoed over two deltas matches no whole-string scrub, so no raw events are kept."""
    half = len(_SECRET) // 2
    response = httpx2.Response(
        200,
        content=_chunk({"choices": [{"delta": {"content": _SECRET[:half]}}]})
        + _chunk({"choices": [{"delta": {"content": _SECRET[half:]}}]}),
    )
    provider = openai.OpenAIProvider(
        api_key=_SECRET,
        model="m",
        transcript_sink=provider_types.TranscriptSink(tmp_path, objects_dir=tmp_path / "objects"),
    )
    with (
        mock.patch("httpx2.stream", return_value=contextlib.closing(response)),
        pytest.raises(provider_types.ProviderError, match="prematurely"),
    ):
        provider.call(system="s", messages=[], text_delta_callback=lambda _: None)
    (path,) = tmp_path.glob("*.json")
    doc = json.loads(path.read_text())
    assert doc["outcome"] == "failed"
    assert _SECRET not in path.read_text()
    assert _SECRET not in _streamed_text(doc)
    assert "events" not in doc["response"]
