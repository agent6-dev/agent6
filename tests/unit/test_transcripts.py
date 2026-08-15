# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the Anthropic provider transcript writer.

The critical security property: the literal `x-api-key` value must never
land on disk. The http_post seam is stubbed so no network call is made.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx2
import pytest

from agent6.providers import AnthropicProvider, ProviderError, TranscriptSink
from agent6.providers.types import _redact_headers  # pyright: ignore[reportPrivateUsage]


class _FakeResponse:
    def __init__(self, *, status_code: int, payload: dict[str, Any] | None = None, text: str = ""):
        self.status_code = status_code
        self.headers: dict[str, str] = {}
        self._payload = payload
        self.text = text

    def json(self) -> dict[str, Any]:
        assert self._payload is not None
        return self._payload


def _scan_for_secret(transcripts_dir: Path, secret: str) -> list[Path]:
    matches: list[Path] = []
    for p in transcripts_dir.rglob("*"):
        if not p.is_file():
            continue
        if secret in p.read_text(encoding="utf-8"):
            matches.append(p)
    return matches


def test_transcript_redacts_api_key_on_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
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
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
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
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
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
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Some gateways echo the received key in a 401 body; the echo rode
    verbatim into the transcript body and the ProviderError text, which header
    redaction never touches. Both must carry the marker, not the value."""
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


def test_record_scrubs_credential_values_from_the_bodies(tmp_path: Path) -> None:
    """The sink-level scrub: a body string equal to a credential riding in the
    request's auth headers (the bare token behind `Bearer ` included) is
    replaced at the one serialization point."""
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
    out = _redact_headers(  # pyright: ignore[reportPrivateUsage]
        {"x-api-key": "secret", "Authorization": "Bearer t", "Other": "keep"}
    )
    assert out["x-api-key"] == "<REDACTED>"
    assert out["Authorization"] == "<REDACTED>"
    assert out["Other"] == "keep"


def test_seq_continues_across_resume_legs(tmp_path: Path) -> None:
    """seq is per-RUN, not per-sink: a resume builds a fresh TranscriptSink over
    the same <run>/transcripts/ dir, and restarting at 1 produced duplicate seqs
    whose seq-primary sort interleaved the legs -- `sessions transcript` rendered a
    scrambled conversation with a false 'context summarised' marker and ended on
    stale leg-1 content. A new sink must continue from the highest seq present."""
    d = tmp_path / "transcripts"
    leg1 = TranscriptSink(d)
    for _ in range(2):
        leg1.record(request_headers={}, request_body={}, response_status=200, response_body={})
    leg2 = TranscriptSink(d)  # the resume's fresh sink over the same dir
    p = leg2.record(request_headers={}, request_body={}, response_status=200, response_body={})
    assert json.loads(p.read_text(encoding="utf-8"))["seq"] == 3
    from agent6.viewmodel.transcript_render import load_transcripts

    assert [t["seq"] for t in load_transcripts(d)] == [1, 2, 3]


def test_transcript_record_publishes_via_atomic_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The writer used a predictable `<name>.json.tmp` + write_text that would
    follow a planted symlink; it now publishes through atomic_write (mkstemp,
    unpredictable name, O_EXCL). Spying the primitive is the regression: the old
    write_text path never called it. The record still lands, headers redacted."""
    import agent6.providers.types as types_mod

    calls: list[Path] = []
    real = types_mod.atomic_write

    def spy(path: Path, data: str | bytes) -> None:
        calls.append(path)
        real(path, data)

    monkeypatch.setattr(types_mod, "atomic_write", spy)
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
