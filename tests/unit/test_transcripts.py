# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the Anthropic provider transcript writer.

The critical security property: the literal `x-api-key` value must never
land on disk. The http_post seam is stubbed so no network call is made.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import httpx2
import pytest

from agent6 import portable
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
