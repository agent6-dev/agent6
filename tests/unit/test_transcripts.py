# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the Anthropic provider transcript writer.

The critical security property: the literal `x-api-key` value must never
land on disk. The http_post seam is stubbed so no network call is made.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import pathlib
from typing import Any

import httpx2
import pytest

from agent6 import media, portable
from agent6.providers import (
    AnthropicProvider,
    ProviderError,
    TranscriptSink,
    types,  # pyright: ignore[reportPrivateUsage]
)


class _FakeResponse:
    def __init__(self, *, status_code: int, payload: dict[str, Any] | None = None, text: str = ""):
        self.status_code = status_code
        self.headers: dict[str, str] = {}
        self._payload = payload
        self.text = text

    def json(self) -> dict[str, Any]:
        assert self._payload is not None
        return self._payload


def _scan_for_secret(transcripts_dir: pathlib.Path, secret: str) -> list[pathlib.Path]:
    matches: list[pathlib.Path] = []
    for p in transcripts_dir.rglob("*"):
        if not p.is_file():
            continue
        if secret in p.read_text(encoding="utf-8"):
            matches.append(p)
    return matches


def test_transcript_redacts_api_key_on_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    sink = TranscriptSink(tmp_path / "transcripts")
    api_key = "sk-ant-supersecret-do-not-leak"
    provider = AnthropicProvider(
        api_key=api_key, model="claude-test", prompt_caching=False, transcript_sink=sink
    )

    def fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        return _FakeResponse(
            status_code=200,
            payload={
                "content": [{"type": "text", "text": "hi"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    monkeypatch.setattr("agent6.providers._transport.http_post", fake_post)
    resp = provider.call(system="sys", messages=[{"role": "user", "content": "x"}])
    assert resp.text == "hi"
    leaks = _scan_for_secret(tmp_path / "transcripts", api_key)
    assert leaks == [], f"API key leaked to transcripts: {leaks}"
    files = list((tmp_path / "transcripts").glob("*.json"))
    assert len(files) == 1
    doc = json.loads(files[0].read_text(encoding="utf-8"))
    assert doc["request"]["headers"]["x-api-key"] == "<REDACTED>"
    assert doc["response"]["status"] == 200


def test_transcript_redacts_api_key_on_http_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    sink = TranscriptSink(tmp_path / "transcripts")
    api_key = "sk-ant-secret-error-path"
    provider = AnthropicProvider(
        api_key=api_key, model="claude-test", prompt_caching=False, transcript_sink=sink
    )

    def fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        return _FakeResponse(status_code=429, text="rate limited")

    monkeypatch.setattr("agent6.providers._transport.http_post", fake_post)
    with pytest.raises(ProviderError):
        provider.call(system="sys", messages=[{"role": "user", "content": "x"}])
    leaks = _scan_for_secret(tmp_path / "transcripts", api_key)
    assert leaks == []


def test_transcript_redacts_api_key_on_network_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    sink = TranscriptSink(tmp_path / "transcripts")
    api_key = "sk-ant-secret-net-error"
    provider = AnthropicProvider(
        api_key=api_key, model="claude-test", prompt_caching=False, transcript_sink=sink
    )

    def fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        raise httpx2.ConnectError("no route")

    monkeypatch.setattr("agent6.providers._transport.http_post", fake_post)
    with pytest.raises(ProviderError):
        provider.call(system="sys", messages=[{"role": "user", "content": "x"}])
    leaks = _scan_for_secret(tmp_path / "transcripts", api_key)
    assert leaks == []


def test_a_response_body_echoing_the_credential_is_scrubbed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """A 401 body echoing the key carries the marker, in the transcript and the error text."""
    sink = TranscriptSink(tmp_path / "transcripts")
    api_key = "sk-ant-echoed-back-by-a-gateway"
    provider = AnthropicProvider(
        api_key=api_key, model="claude-test", prompt_caching=False, transcript_sink=sink
    )

    def fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        return _FakeResponse(status_code=401, text=f'{{"error": "bad key {api_key}"}}')

    monkeypatch.setattr("agent6.providers._transport.http_post", fake_post)
    with pytest.raises(ProviderError) as exc:
        provider.call(system="sys", messages=[{"role": "user", "content": "x"}])
    assert api_key not in str(exc.value)
    assert "<REDACTED>" in str(exc.value)
    assert _scan_for_secret(tmp_path / "transcripts", api_key) == []
    files = list((tmp_path / "transcripts").glob("*.json"))
    assert len(files) == 1 and "<REDACTED>" in files[0].read_text(encoding="utf-8")


def test_record_scrubs_credential_values_from_the_bodies(tmp_path: pathlib.Path) -> None:
    """A body string equal to a credential in the auth headers is scrubbed at serialization."""
    sink = TranscriptSink(tmp_path / "t")
    path = sink.record(
        request_headers={"authorization": "Bearer sk-tok-123456789"},
        request_body={"echo": "sk-tok-123456789"},
        response_status=200,
        response_body={"msg": "key sk-tok-123456789 rejected"},
    )
    text = path.read_text(encoding="utf-8")
    assert "sk-tok-123456789" not in text
    assert text.count("<REDACTED>") >= 3


def test_redact_headers_unit() -> None:
    out = types._redact_headers(  # pyright: ignore[reportPrivateUsage]
        {
            "x-api-key": "secret",
            "Authorization": "Bearer t",
            "chatgpt-account-id": "acc-uuid",
            "Other": "keep",
        }
    )
    assert out["x-api-key"] == "<REDACTED>"
    assert out["Authorization"] == "<REDACTED>"
    assert out["chatgpt-account-id"] == "<REDACTED>"
    assert out["Other"] == "keep"


def test_seq_continues_across_resume_executions(tmp_path: pathlib.Path) -> None:
    """Seq is per run: a new sink continues from the highest seq present.

    Restarting at 1 produced duplicate seqs that interleaved the executions, a scrambled
    conversation with a false 'context summarised' marker.
    """
    d = tmp_path / "transcripts"
    leg1 = TranscriptSink(d)
    for _ in range(2):
        leg1.record(request_headers={}, request_body={}, response_status=200, response_body={})
    leg2 = TranscriptSink(d)  # the resume's fresh sink over the same dir
    p = leg2.record(request_headers={}, request_body={}, response_status=200, response_body={})
    assert json.loads(p.read_text(encoding="utf-8"))["seq"] == 3
    from agent6.viewmodel import transcript_render

    assert [t["seq"] for t in transcript_render.load_transcripts(d)] == [1, 2, 3]


def test_transcript_record_publishes_via_atomic_write(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The writer publishes through atomic_write, never a predictable temp name a symlink can plant.

    Spying the primitive is the regression: the old write_text path never called it.
    """
    calls: list[pathlib.Path] = []
    real = portable.atomic_write

    def spy(path: pathlib.Path, data: str | bytes) -> None:
        calls.append(path)
        real(path, data)

    monkeypatch.setattr(portable, "atomic_write", spy)
    sink = TranscriptSink(tmp_path)
    path = sink.record(
        url="https://x",
        request_headers={"x-api-key": "sk-secret"},
        request_body={"m": 1},
        response_status=200,
        response_body={"ok": True},
    )
    assert calls == [path]  # published atomically, not via a predictable temp
    body = path.read_text(encoding="utf-8")
    assert json.loads(body)["seq"] == 1
    assert "sk-secret" not in body  # redaction intact
    assert not list(tmp_path.glob("*.json.tmp"))


@pytest.mark.parametrize("shape", ["native", "mcp", "chat", "responses"])
def test_transcripts_omit_images_and_error_echoes(tmp_path: pathlib.Path, shape: str) -> None:
    """Native, MCP and data URL images leave only metadata, including echoed payloads."""
    data = b"private image pixels" * 4
    encoded = base64.b64encode(data).decode()
    url = "data:image/png;base64," + encoded
    blocks = {
        "native": {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": encoded},
        },
        "mcp": {"type": "image", "mimeType": "image/png", "data": encoded},
        "chat": {"type": "image_url", "image_url": {"url": url}},
        "responses": {"type": "input_image", "image_url": url},
    }
    body = {
        "stdin": [
            {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps({"attachment": {"source_path": "page.png"}}),
                    },
                    blocks[shape],
                ]
            }
        ]
    }
    original = json.dumps(body)
    path = TranscriptSink(tmp_path).record(
        request_headers={},
        request_body=body,
        response_status=400,
        response_body=f"rejected {encoded}",
    )
    text = path.read_text()
    assert encoded not in text
    assert hashlib.sha256(data).hexdigest() in text
    assert '"payload_omitted": true' in text
    assert "page.png" in text
    assert json.dumps(body) == original


@pytest.mark.parametrize("streaming", ["off", "status", "frame"])
@pytest.mark.parametrize("copies", [1, 10000])
def test_image_error_echo_is_scrubbed_before_excerpting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, streaming: str, copies: int
) -> None:
    """A long base64 echo cannot escape through the transport's short error excerpts."""
    encoded = base64.b64encode(b"private pixels" * 8 * copies).decode()
    provider = AnthropicProvider(api_key="key", model="m", transcript_sink=TranscriptSink(tmp_path))

    def post(url: str, **kwargs: Any) -> _FakeResponse:
        return _FakeResponse(status_code=400, text="bad image: " + encoded)

    def stream(*args: Any, **kwargs: Any) -> contextlib.AbstractContextManager[httpx2.Response]:
        if streaming == "frame":
            error = {
                "type": "error",
                "error": {"type": "invalid_request_error", "message": "bad image: " + encoded},
            }
            return contextlib.nullcontext(
                httpx2.Response(200, text="event: error\ndata: " + json.dumps(error) + "\n\n")
            )
        return contextlib.nullcontext(httpx2.Response(400, text="bad image: " + encoded))

    monkeypatch.setattr("agent6.providers._transport.http_post", post)
    monkeypatch.setattr("agent6.providers._stream.http_stream", stream)
    with pytest.raises(ProviderError) as error:
        provider.call(
            system="s",
            text_delta_callback=(lambda _: None) if streaming != "off" else None,
            messages=[
                {
                    "role": "user",
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
        )
    assert encoded[:40] not in str(error.value)
    assert not _scan_for_secret(tmp_path, encoded[:40])


@pytest.mark.parametrize("shape", ["native", "chat", "responses", "mcp"])
def test_transcript_omits_native_document_payload(tmp_path: pathlib.Path, shape: str) -> None:
    """A PDF attachment follows the same metadata-only retention rule as an image."""
    data = b"%PDF-1.7\nprivate document"
    encoded = base64.b64encode(data).decode()
    url = "data:application/pdf;base64," + encoded
    block = {
        "native": {
            "type": "document",
            "source": {"type": "base64", "media_type": "application/pdf", "data": encoded},
        },
        "chat": {"type": "file", "file": {"filename": "page.pdf", "file_data": url}},
        "responses": {"type": "input_file", "filename": "page.pdf", "file_data": url},
        "mcp": {
            "type": "resource",
            "resource": {"uri": "file:///page.pdf", "mimeType": "application/pdf", "blob": encoded},
        },
    }[shape]
    path = TranscriptSink(tmp_path).record(
        request_headers={},
        request_body={
            "content": [
                {"type": "text", "text": "document: page.pdf, 1 KiB"},
                block,
            ]
        },
        response_status=200,
        response_body={},
    )
    text = path.read_text()
    assert encoded not in text
    assert hashlib.sha256(data).hexdigest() in text
    assert "payload_omitted" in text and "page.pdf" in text


@pytest.mark.parametrize(
    "code, modality, reason",
    [
        ("image_input_not_supported", "image", "unsupported"),
        ("invalid_image", "image", "invalid"),
        ("image_too_large", "image", "too_large"),
        ("document_input_not_supported", "document", "unsupported"),
        ("invalid_file", "document", "invalid"),
    ],
)
def test_media_specific_errors_are_typed(
    monkeypatch: pytest.MonkeyPatch, code: str, modality: str, reason: str
) -> None:
    """Recognized media rejections carry a reason without echoing the provider body."""
    body = {"error": {"code": code, "message": "private upstream detail"}}

    def fake_post(*args: Any, **kwargs: Any) -> _FakeResponse:
        return _FakeResponse(status_code=400, text=json.dumps(body))

    monkeypatch.setattr("agent6.providers._transport.http_post", fake_post)
    provider = AnthropicProvider(api_key="key", model="m")
    mime = "application/pdf" if modality == "document" else "image/png"
    with pytest.raises(types.MediaInputError) as caught:
        provider.call(
            system="s",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": modality,
                            "source": {"type": "base64", "media_type": mime, "data": "eA=="},
                        }
                    ],
                }
            ],
        )
    assert caught.value.modality == modality and caught.value.reason == reason
    assert "private upstream detail" not in str(caught.value)


@pytest.mark.parametrize(
    "status, message",
    [
        (401, "This model does not support images"),
        (429, "image input is unsupported"),
        (400, "quota exhausted for image request"),
        (400, "invalid model"),
    ],
)
def test_unrelated_errors_are_not_media_rejections(
    monkeypatch: pytest.MonkeyPatch, status: int, message: str
) -> None:
    """Authentication, quota and unrelated client errors cannot mark a route unsupported."""

    def fake_post(*args: Any, **kwargs: Any) -> _FakeResponse:
        return _FakeResponse(status_code=status, text=json.dumps({"error": {"message": message}}))

    monkeypatch.setattr("agent6.providers._transport.http_post", fake_post)
    provider = AnthropicProvider(api_key="key", model="m")
    with pytest.raises(ProviderError) as caught:
        provider.call(
            system="s",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/png", "data": "eA=="},
                        }
                    ],
                }
            ],
        )
    assert not isinstance(caught.value, types.MediaInputError)


@pytest.mark.parametrize("encoded", ["sk-", base64.b64encode(b"x" * 66).decode()])
@pytest.mark.parametrize("target", ["sink", "off", "status", "frame"])
def test_credentials_are_scrubbed_before_payload_echoes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, encoded: str, target: str
) -> None:
    """A media payload matching a credential prefix cannot expose its secret suffix."""
    secret = encoded + "SECRET123"
    block = {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": encoded},
    }
    sink = TranscriptSink(tmp_path)
    text = "upstream echoed " + secret
    if target == "sink":
        path = sink.record(
            request_headers={"x-api-key": secret},
            request_body={"content": [block]},
            response_status=400,
            response_body=text,
        )
        safe = path.read_text()
    else:

        def post(*args: Any, **kwargs: Any) -> _FakeResponse:
            return _FakeResponse(status_code=400, text=text)

        def stream(*args: Any, **kwargs: Any) -> contextlib.AbstractContextManager[httpx2.Response]:
            if target == "frame":
                body = {"type": "error", "error": {"message": text}}
                return contextlib.nullcontext(
                    httpx2.Response(200, text="event: error\ndata: " + json.dumps(body) + "\n\n")
                )
            return contextlib.nullcontext(httpx2.Response(400, text=text))

        monkeypatch.setattr("agent6.providers._transport.http_post", post)
        monkeypatch.setattr("agent6.providers._stream.http_stream", stream)
        provider = AnthropicProvider(api_key=secret, model="m", transcript_sink=sink)
        with pytest.raises(ProviderError) as caught:
            provider.call(
                system="s",
                messages=[{"role": "user", "content": [block]}],
                text_delta_callback=None if target == "off" else lambda _: None,
            )
        safe = str(caught.value)
    assert "SECRET123" not in safe
    assert "<REDACTED>" in safe
    assert not _scan_for_secret(tmp_path, "SECRET123")


def test_a_short_attachment_echo_is_omitted_too() -> None:
    """A document small enough to fit an error echo is scrubbed like any other payload."""
    import base64

    encoded = base64.b64encode(b"%PDF-1.7\nsecret=abc\n%%EOF").decode("ascii")
    block = {
        "type": "document",
        "source": {"type": "base64", "media_type": "application/pdf", "data": encoded},
    }
    out = media.omit_payloads(
        {"request": {"content": [block]}, "error": f"upstream echoed {encoded} back"}
    )
    assert encoded not in json.dumps(out)
    assert out["error"] == "upstream echoed [media payload omitted] back"


def test_payload_omission_preserves_json_tokens() -> None:
    """Short base64 equal to a JSON token cannot corrupt the metadata document."""
    clean = media.omit_payloads(
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "true"}}
    )
    assert json.loads(json.dumps(clean))["source"]["payload_omitted"] is True
    assert "data" not in clean["source"]


@pytest.mark.parametrize("block_type", [None, "other"])
def test_typeless_file_data_is_plain_data(block_type: str | None) -> None:
    """Only known media block types turn a data URL into media metadata."""
    value: dict[str, Any] = {"file_data": "data:application/pdf;base64,eA=="}
    if block_type:
        value["type"] = block_type
    assert media.omit_payloads(value) == value


def test_payload_echo_replacement_changes_values_not_keys() -> None:
    """Echo omission walks string values without rewriting JSON object keys."""
    encoded = base64.b64encode(b"x" * 66).decode()
    value = {
        encoded: ["echo " + encoded],
        "image": {"type": "image", "mimeType": "image/png", "data": encoded},
    }
    assert media.omit_payloads(value)[encoded] == ["echo [media payload omitted]"]


@pytest.mark.parametrize("responses", [False, True])
@pytest.mark.parametrize("mime, kind", [("image/png", "image"), ("application/pdf", "document")])
def test_synthetic_media_labels_retain_source_path(
    tmp_path: pathlib.Path, responses: bool, mime: str, kind: str
) -> None:
    """OpenAI adapter labels supply the source path for metadata-only transcripts."""
    from agent6.providers import _openai_messages, _responses

    encoded = base64.b64encode(b"private data" * 10).decode()
    path = "nested/page:one"
    canonical = [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "c1", "name": "read_file", "input": {"path": path}}
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "c1",
                    "content": [
                        {"type": "text", "text": f"{kind}: {path}, 1 KiB"},
                        {
                            "type": kind,
                            "source": {"type": "base64", "media_type": mime, "data": encoded},
                        },
                    ],
                }
            ],
        },
    ]
    request = (
        {"input": _responses.responses_input(canonical, route="r")}
        if responses
        else {"messages": _openai_messages.anthropic_to_openai_messages("s", canonical)}
    )
    saved = TranscriptSink(tmp_path).record(
        request_headers={},
        request_body=request,
        response_status=200,
        response_body={},
    )
    text = saved.read_text()
    assert encoded not in text
    body = json.loads(text)["request"]["body"]
    last = body["input" if responses else "messages"][-1]["content"][-1]
    assert last["source"]["source_path"] == path
