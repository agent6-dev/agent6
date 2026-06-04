# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Unit tests for the provider egress transport seam."""

from __future__ import annotations

import contextlib
from collections.abc import Generator

import httpx
import pytest

from agent6.providers import egress


def test_parse_endpoint_https_default_port() -> None:
    assert egress.parse_endpoint("https://api.anthropic.com/v1/messages") == (
        "api.anthropic.com",
        443,
    )


def test_parse_endpoint_http_default_port() -> None:
    assert egress.parse_endpoint("http://localhost/v1") == ("localhost", 80)


def test_parse_endpoint_explicit_port() -> None:
    assert egress.parse_endpoint("http://127.0.0.1:11434/v1") == ("127.0.0.1", 11434)


def test_parse_endpoint_no_host_raises() -> None:
    with pytest.raises(ValueError):
        egress.parse_endpoint("not-a-url")


def test_http_post_without_route_uses_plain_httpx(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake_post(
        url: str, *, headers: dict[str, str], content: bytes, timeout: float
    ) -> httpx.Response:
        seen["url"] = url
        seen["uds"] = "plain"
        return httpx.Response(200, text="ok")

    monkeypatch.setattr(egress.httpx, "post", fake_post)
    resp = egress.http_post(
        "https://api.openai.com/v1/chat",
        headers={"a": "b"},
        content=b"{}",
        timeout=5.0,
    )
    assert resp.status_code == 200
    assert seen["uds"] == "plain"


def test_http_post_with_route_dials_unix_socket(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeClient:
        def __init__(self, *, transport: object, timeout: float) -> None:
            captured["transport"] = transport

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def post(self, url: str, *, headers: dict[str, str], content: bytes) -> httpx.Response:
            captured["url"] = url
            return httpx.Response(200, text="routed")

    transports: dict[str, object] = {}

    def fake_transport(*, uds: str) -> str:
        transports["uds"] = uds
        return f"transport:{uds}"

    monkeypatch.setattr(egress.httpx, "Client", FakeClient)
    monkeypatch.setattr(egress.httpx, "HTTPTransport", fake_transport)
    egress.register_route("api.openai.com", 443, "/tmp/egress-0.sock")

    try:
        resp = egress.http_post(
            "https://api.openai.com/v1/chat",
            headers={},
            content=b"{}",
            timeout=5.0,
        )
    finally:
        egress.clear_routes()
    assert resp.text == "routed"
    assert transports["uds"] == "/tmp/egress-0.sock"
    assert captured["transport"] == "transport:/tmp/egress-0.sock"


def test_http_stream_without_route_uses_plain_httpx(monkeypatch: pytest.MonkeyPatch) -> None:
    used: dict[str, object] = {}

    @contextlib.contextmanager
    def fake_stream(
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        content: bytes,
        timeout: float,
    ) -> Generator[httpx.Response]:
        used["method"] = method
        used["url"] = url
        yield httpx.Response(200, text="streamed")

    monkeypatch.setattr(egress.httpx, "stream", fake_stream)
    with egress.http_stream(
        "POST",
        "https://api.openai.com/v1/chat",
        headers={},
        content=b"{}",
        timeout=5.0,
    ) as resp:
        assert resp.text == "streamed"
    assert used["method"] == "POST"
