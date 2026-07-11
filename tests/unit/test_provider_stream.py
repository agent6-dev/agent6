# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the shared SSE stream lifecycle (`providers/_stream.py`).

The per-provider suites (test_openai_streaming, test_anthropic_streaming,
test_providers_bugfix) exercise event parsing through their providers; these
pin the shared lifecycle itself: the two-phase idle watchdog, operator
stop/steer classification, 4xx surfacing, and transcript records.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from unittest import mock

import httpx2
import pytest

from agent6.providers import _stream as stream_mod
from agent6.providers._stream import SseCall, StreamClock
from agent6.providers.types import (
    ProviderAborted,
    ProviderError,
    ProviderInterrupted,
    TranscriptSink,
)


def _serve(resp: object) -> Callable[..., object]:
    """side_effect factory returning ``resp`` for any httpx2.stream call."""

    def factory(*_a: object, **_kw: object) -> object:
        return resp

    return factory


def _call(sink: TranscriptSink | None = None, **overrides: Any) -> SseCall:
    kwargs: dict[str, Any] = {
        "api_label": "OpenAI",
        "api_format": "openai",
        "url": "https://api.test/v1/chat/completions",
        "headers": {"accept": "text/event-stream"},
        "body": {"model": "m", "stream": True},
        "timeout_s": 5.0,
        "transcript_sink": sink,
        "should_abort": None,
        "should_interrupt": None,
    }
    kwargs.update(overrides)
    return SseCall(**kwargs)


def _drain(resp: httpx2.Response, clock: StreamClock) -> None:
    for _line in resp.iter_lines():
        clock.mark_data()


class _ParkedResponse:
    """A 200 stream that emits ``lead_lines`` then parks until the watchdog
    calls ``close()``, at which point it raises like httpx2 does when the
    socket is closed mid-read."""

    def __init__(self, lead_lines: list[str] | None = None) -> None:
        self.status_code = 200
        self.headers: dict[str, str] = {}
        self._lead = lead_lines or []
        self._closed = threading.Event()

    def __enter__(self) -> _ParkedResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def close(self) -> None:
        self._closed.set()

    def read(self) -> bytes:
        return b""

    def iter_lines(self) -> Iterator[str]:
        yield from self._lead
        if not self._closed.wait(timeout=10.0):
            raise AssertionError("watchdog never closed the response")
        raise httpx2.ReadError("connection closed")


class _ErrorResponse:
    def __init__(self, *, status_code: int, body: str, headers: dict[str, str]) -> None:
        self.status_code = status_code
        self._body = body
        self.headers = headers

    def __enter__(self) -> _ErrorResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body.encode("utf-8")


def test_idle_kill_before_output_reports_prefill_and_records(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(stream_mod, "STREAM_FIRST_DATA_TIMEOUT_S", 0.05)
    monkeypatch.setattr(stream_mod, "STREAM_WATCHDOG_TICK_S", 0.01)
    sink = TranscriptSink(tmp_path / "transcripts")

    with (
        mock.patch("httpx2.stream", side_effect=_serve(_ParkedResponse())),
        pytest.raises(ProviderError, match="prefill"),
    ):
        _call(sink).run(_drain)

    files = list((tmp_path / "transcripts").glob("*.json"))
    assert len(files) == 1
    doc = json.loads(files[0].read_text(encoding="utf-8"))
    assert doc["response"]["status"] == 0
    assert "idle watchdog" in doc["response"]["body"]


def test_idle_kill_after_output_reports_mid_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stream_mod, "STREAM_IDLE_TIMEOUT_S", 0.05)
    monkeypatch.setattr(stream_mod, "STREAM_FIRST_DATA_TIMEOUT_S", 30.0)
    monkeypatch.setattr(stream_mod, "STREAM_WATCHDOG_TICK_S", 0.01)

    def consume(resp: httpx2.Response, clock: StreamClock) -> None:
        for _line in resp.iter_lines():
            clock.mark_data()
            clock.mark_output()

    with (
        mock.patch("httpx2.stream", side_effect=_serve(_ParkedResponse(["data: x"]))),
        pytest.raises(ProviderError, match=r"Anthropic SSE stream idle .* mid-stream"),
    ):
        _call(api_label="Anthropic", api_format="anthropic").run(consume)


def test_idle_budget_prefers_the_thinking_phase_over_mid_stream() -> None:
    # A display:omitted thinking block streams only pings; the patient thinking
    # budget must win over the tight mid-stream budget while it is open, else a
    # long reason false-kills (the sonnet stylebook regression).
    clock = StreamClock()
    assert clock.idle_budget() == (
        stream_mod.STREAM_FIRST_DATA_TIMEOUT_S,
        "before any data (prefill)",
    )
    clock.mark_output()
    assert clock.idle_budget() == (stream_mod.STREAM_IDLE_TIMEOUT_S, "mid-stream")
    clock.enter_thinking()
    assert clock.idle_budget() == (stream_mod.STREAM_THINKING_IDLE_TIMEOUT_S, "mid-thinking")
    clock.exit_thinking()
    assert clock.idle_budget() == (stream_mod.STREAM_IDLE_TIMEOUT_S, "mid-stream")


def test_idle_kill_in_thinking_reports_mid_thinking(monkeypatch: pytest.MonkeyPatch) -> None:
    # With output already seen, a wedged thinking block is bounded by the
    # thinking budget, not the (here deliberately huge) mid-stream one.
    monkeypatch.setattr(stream_mod, "STREAM_THINKING_IDLE_TIMEOUT_S", 0.05)
    monkeypatch.setattr(stream_mod, "STREAM_IDLE_TIMEOUT_S", 30.0)
    monkeypatch.setattr(stream_mod, "STREAM_FIRST_DATA_TIMEOUT_S", 30.0)
    monkeypatch.setattr(stream_mod, "STREAM_WATCHDOG_TICK_S", 0.01)

    def consume(resp: httpx2.Response, clock: StreamClock) -> None:
        for _line in resp.iter_lines():
            clock.mark_data()
            clock.mark_output()
            clock.enter_thinking()

    with (
        mock.patch("httpx2.stream", side_effect=_serve(_ParkedResponse(["data: x"]))),
        pytest.raises(ProviderError, match=r"Anthropic SSE stream idle .* mid-thinking"),
    ):
        _call(api_label="Anthropic", api_format="anthropic").run(consume)


def test_abort_classifies_as_provider_aborted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stream_mod, "STREAM_WATCHDOG_TICK_S", 0.01)
    with (
        mock.patch("httpx2.stream", side_effect=_serve(_ParkedResponse())),
        pytest.raises(ProviderAborted),
    ):
        _call(should_abort=lambda: True).run(_drain)


def test_interrupt_classifies_as_provider_interrupted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stream_mod, "STREAM_WATCHDOG_TICK_S", 0.01)
    with (
        mock.patch("httpx2.stream", side_effect=_serve(_ParkedResponse())),
        pytest.raises(ProviderInterrupted),
    ):
        _call(should_interrupt=lambda: True).run(_drain)


def test_4xx_threads_status_retry_after_and_label(tmp_path: Path) -> None:
    sink = TranscriptSink(tmp_path / "transcripts")
    resp = _ErrorResponse(status_code=429, body='{"error":"rate"}', headers={"retry-after": "7"})

    with (
        mock.patch("httpx2.stream", side_effect=_serve(resp)),
        pytest.raises(ProviderError) as ei,
    ):
        _call(sink).run(_drain)

    assert str(ei.value).startswith("OpenAI API error 429")
    assert ei.value.status_code == 429
    assert ei.value.retry_after_s == 7.0
    files = list((tmp_path / "transcripts").glob("*.json"))
    assert len(files) == 1


def test_consume_provider_error_propagates_unchanged() -> None:
    def consume(resp: httpx2.Response, clock: StreamClock) -> None:
        raise ProviderError("mid-stream error frame")

    with (
        mock.patch("httpx2.stream", side_effect=_serve(_ParkedResponse([]))),
        pytest.raises(ProviderError, match="mid-stream error frame"),
    ):
        _call().run(consume)


def test_transport_error_names_the_wire_format() -> None:
    with (
        mock.patch("httpx2.stream", side_effect=httpx2.ReadTimeout("boom")),
        pytest.raises(ProviderError, match=r"HTTP error streaming from .* \(anthropic format\)"),
    ):
        _call(api_label="Anthropic", api_format="anthropic").run(_drain)
