# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""No credential value rides an error text, a redirect line or the terminal."""

from __future__ import annotations

from typing import Any

import httpx2
import pytest

from agent6.config import OAuthExchangeConfig
from agent6.providers import ProviderError, _oauth_protocol, _stream, _transport, oauth


class _Resp:
    def __init__(self, status: int, body: object, text: str | None = None, **headers: str) -> None:
        self.status_code = status
        self._body = body
        self.text = text if text is not None else str(body)
        self.headers = httpx2.Headers(headers)

    def json(self) -> object:
        return self._body


def test_a_grant_whose_token_cannot_ride_a_header_is_refused_without_its_value() -> None:
    """A trailing newline in an access token would otherwise surface in h11's error text."""
    secret = "SECRET-TOKEN-VALUE\n"
    with pytest.raises(ProviderError, match="cannot ride a header") as refused:
        _oauth_protocol.grant_from_response(
            _Resp(200, {"access_token": secret, "expires_in": 60}),  # type: ignore[arg-type]
            operation="exchange",
            label="work",
        )
    assert "SECRET-TOKEN" not in str(refused.value)


def test_a_token_error_scrubs_the_whole_body_before_clipping() -> None:
    """A secret straddling the old clip boundary kept its prefix in the message."""
    rt = "refresh-token-value-0123456789abcdef"
    text = ("<" + rt + ">,") * 60 + rt
    err = _oauth_protocol.token_error(
        _Resp(400, {}, text=text),  # type: ignore[arg-type]
        operation="refresh",
        provider="work",
        label="work",
        secrets=(rt,),
    )
    assert rt[:10] not in str(err) and "<REDACTED>" in str(err)


def test_a_redirect_line_names_the_host_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Location that reflects the request would carry the credential into the error."""
    location = "https://evil.example/?echo=Bearer%20SECRETVALUE"
    err = _oauth_protocol.redirect_refusal(
        _Resp(302, {}, location=location),  # type: ignore[arg-type]
        operation="refresh",
        label="work",
    )
    assert "evil.example" in str(err) and "SECRETVALUE" not in str(err)

    def fake_post(url: str, **_kw: Any) -> _Resp:
        return _Resp(307, {}, location=location)

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
        facts=_transport._headers.RequestFacts(target="inference", model="m"),
        build_headers=lambda _k, _f: {},
        adapt_400=lambda _s, _t, _b: False,
        adapt_attempts=0,
        require_metered=lambda _d: None,
        parse=lambda _d: None,  # type: ignore[arg-type, return-value]
    )
    with pytest.raises(ProviderError, match=r"evil\.example") as refused:
        call.run()
    assert "SECRETVALUE" not in str(refused.value)
    assert _transport.redirect_host(_Resp(302, {}, location="/relative")) == "?"  # type: ignore[arg-type]
    assert _stream is not None  # the stream's refusal shares `redirect_host`


def test_a_revocation_answered_with_a_redirect_is_not_a_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def redirecting(*_a: object, **_k: object) -> _Resp:
        return _Resp(302, {}, location="https://x/")

    monkeypatch.setattr(_oauth_protocol, "post_form", redirecting)
    warn = _oauth_protocol.revoke_tokens(
        "https://login.example/revoke", "c", access_token="a", refresh_token="r", encoding="form"
    )
    assert warn is not None and "not followed" in warn


def test_issuer_text_reaches_the_terminal_printable_only() -> None:
    with pytest.raises(ValueError, match="refused: denied") as refused:
        _oauth_protocol.parse_callback(
            "state=S&error=access_denied&error_description=denied%1b%5b31m", state="S"
        )
    assert "\x1b" not in str(refused.value)


def test_a_device_grant_with_a_control_character_in_its_uri_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_post(
        url: str, data: dict[str, str], timeout_s: float, headers: dict[str, str] | None = None
    ) -> _Resp:
        return _Resp(
            200,
            {
                "device_code": "D",
                "user_code": "AB",
                "verification_uri": "https://login.example/activate\x1b[31m",
            },
        )

    monkeypatch.setattr(_oauth_protocol, "post_form", fake_post)
    with pytest.raises(ProviderError, match="unusable verification URI") as refused:
        _oauth_protocol.start_device_grant(
            "https://login.example/device", "c", scopes=(), encoding="form", label="work"
        )
    assert "\x1b" not in str(refused.value)


def test_an_exchange_transport_error_scrubs_the_issuer_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = "issuer-bearer-0123456789"

    def failing_post(*_a: Any, **_k: Any) -> Any:
        raise httpx2.LocalProtocolError(f"Illegal header value b'Bearer {token}'")

    monkeypatch.setattr(_transport, "http_request", failing_post)
    with pytest.raises(ProviderError, match="could not reach") as refused:
        oauth.run_exchange(OAuthExchangeConfig(endpoint="https://credentials.example/token"), token)
    assert token not in str(refused.value) and "<REDACTED>" in str(refused.value)
