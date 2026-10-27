# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Where a credential may travel: no redirect, no ambient proxy to loopback, no userinfo URL."""

from __future__ import annotations

import http.server
import pathlib
import threading
from collections.abc import Iterator
from typing import Any

import httpx2
import pytest

from agent6.config import load_config, validate_base_url
from agent6.providers import _oauth_protocol, _stream, _transport, types


@pytest.mark.parametrize(
    ("url", "word"),
    [
        ("https://user:pw@models.example/v1", "userinfo"),
        ("https://user@models.example/v1", "userinfo"),
        ("https://models.example/v1#frag", "fragment"),
    ],
)
def test_a_base_url_carries_no_userinfo_or_fragment(url: str, word: str) -> None:
    """A credential-bearing URL with embedded credentials or a fragment refuses at config load."""
    with pytest.raises(ValueError, match=word):
        validate_base_url(url)


def test_a_provider_block_with_userinfo_refuses_at_load(tmp_path: pathlib.Path) -> None:
    cfg = tmp_path / "agent6.toml"
    cfg.write_text(
        '[providers.p]\napi_format = "openai"\nbase_url = "https://u:p@models.example/v1"\n',
        encoding="utf-8",
    )
    with pytest.raises(Exception, match="userinfo"):
        load_config(cfg)


@pytest.fixture
def loopback_server() -> Iterator[str]:
    """A loopback HTTP server answering every POST and GET with a small JSON body."""

    class Handler(http.server.BaseHTTPRequestHandler):
        def _answer(self) -> None:
            length = int(self.headers.get("content-length") or 0)
            if length:
                self.rfile.read(length)
            body = b'{"ok": true, "choices": []}'
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        do_POST = _answer  # noqa: N815  # BaseHTTPRequestHandler API names
        do_GET = _answer  # noqa: N815

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def test_a_loopback_destination_ignores_the_ambient_proxy(
    loopback_server: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A plaintext loopback endpoint's bearer never goes to `HTTP_PROXY`."""
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")
    resp = _transport.http_post(
        f"{loopback_server}/v1/chat/completions",
        headers={"authorization": "Bearer t"},
        content=b"{}",
        timeout=5.0,
    )
    assert resp.status_code == 200
    with _stream.http_stream(
        "POST", f"{loopback_server}/v1/chat/completions", headers={}, content=b"{}", timeout=5.0
    ) as streamed:
        assert streamed.status_code == 200
    assert _transport.trust_env_for(loopback_server) is False
    assert _transport.trust_env_for("https://models.example/v1") is True


def test_credential_requests_follow_no_redirect(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every request that carries a credential is sent with redirects off."""
    seen: list[dict[str, Any]] = []

    class _Resp:
        status_code = 200
        text = "{}"
        headers = httpx2.Headers()

        def json(self) -> dict[str, Any]:
            return {"access_token": "AT", "expires_in": 60}

    def fake_post(url: str, **kwargs: Any) -> _Resp:
        seen.append({"url": url, **kwargs})
        return _Resp()

    monkeypatch.setattr(httpx2, "post", fake_post)
    _oauth_protocol.post_form("https://login.example/token", {"a": "b"}, 5.0)
    _oauth_protocol.post_json("https://login.example/device", {"a": "b"}, 5.0)
    _oauth_protocol.revoke_tokens(
        "https://login.example/revoke",
        "client",
        __import__("agent6.secret_store", fromlist=["x"]).OAuthTokens("a", "r", 1.0),
    )
    assert len(seen) == 3
    assert all(call["follow_redirects"] is False for call in seen)
    assert all(call["trust_env"] is True for call in seen)


def test_a_redirecting_token_endpoint_is_refused_without_a_second_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class _Redirect:
        status_code = 302
        text = ""
        headers = httpx2.Headers({"location": "https://elsewhere.example/token"})

        def json(self) -> dict[str, Any]:
            raise ValueError("no body")

    def fake_post(url: str, data: dict[str, str], timeout_s: float) -> _Redirect:
        calls.append(url)
        return _Redirect()

    monkeypatch.setattr(_oauth_protocol, "post_form", fake_post)
    with pytest.raises(types.ProviderError, match="redirect"):
        _oauth_protocol.refresh_grant(
            "https://login.example/token", "client", "RT", provider="p", label="p"
        )
    assert calls == ["https://login.example/token"]


def test_a_redirecting_model_endpoint_is_refused_without_a_second_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class _Redirect:
        status_code = 307
        text = ""
        headers = httpx2.Headers({"location": "https://elsewhere.example/v1/chat/completions"})

        def json(self) -> dict[str, Any]:
            raise ValueError("no body")

    def fake_post(url: str, **_kw: Any) -> _Redirect:
        calls.append(url)
        return _Redirect()

    monkeypatch.setattr(_transport, "http_post", fake_post)
    call = _transport.ProviderCall(
        api_label="OpenAI",
        api_format="openai",
        url="https://api.test/v1/chat/completions",
        body={"model": "m"},
        timeout_s=5.0,
        api_key="k",
        credential=None,
        transcript_sink=None,
        budget=None,
        model="m",
        build_headers=lambda _k: {},
        adapt_400=lambda _s, _t, _b: False,
        adapt_attempts=1,
        require_metered=lambda _d: None,
        parse=lambda _d: types.ProviderResponse(
            text="",
            tool_uses=(),
            stop_reason="end_turn",
            input_tokens=0,
            output_tokens=0,
            cache_read_tokens=0,
            cache_creation_tokens=0,
            raw={},
        ),
    )
    with pytest.raises(types.ProviderError, match="redirect"):
        call.run()
    assert calls == ["https://api.test/v1/chat/completions"]
