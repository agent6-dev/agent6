# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the Anthropic provider transcript writer.

The critical security property: the literal `x-api-key` value must never
land on disk. We monkeypatch httpx2.post so no network call is made.
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

    monkeypatch.setattr(httpx2, "post", fake_post)
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

    monkeypatch.setattr(httpx2, "post", fake_post)
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

    monkeypatch.setattr(httpx2, "post", fake_post)
    with pytest.raises(ProviderError):
        provider.call(system="sys", messages=[{"role": "user", "content": "x"}])
    leaks = _scan_for_secret(tmp_path / "transcripts", api_key)
    assert leaks == []


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
    whose seq-primary sort interleaved the legs -- `runs transcript` rendered a
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
