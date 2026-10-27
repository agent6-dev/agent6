# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for agent6.providers.chatgpt_oauth (PKCE, grants, credential)."""

from __future__ import annotations

import base64
import json
from typing import Any
from urllib import parse

import pytest

from agent6.providers import _oauth_protocol, _transport, chatgpt_oauth, types


class _Resp:
    def __init__(self, status_code: int, body: object) -> None:
        self.status_code = status_code
        self._body = body
        self.text = body if isinstance(body, str) else json.dumps(body)

    def json(self) -> object:
        if isinstance(self._body, str):
            return json.loads(self._body)
        return self._body


def _jwt(claims: dict[str, Any]) -> str:
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=").decode()
    return f"h.{payload}.s"


_AUTH_CLAIM = "https://api.openai.com/auth"


def test_pkce_challenge_matches_rfc7636_vector() -> None:
    """RFC 7636 appendix B: the S256 transform of the sample verifier."""
    assert (
        _oauth_protocol.pkce_challenge("dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk")
        == "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
    )
    verifier, challenge = _oauth_protocol.pkce_pair()
    assert 43 <= len(verifier) <= 128 and "=" not in challenge
    assert challenge == _oauth_protocol.pkce_challenge(verifier)


def test_authorize_url_carries_the_registered_params() -> None:
    profile = chatgpt_oauth.PROFILE
    assert profile.authorization_endpoint is not None
    url = _oauth_protocol.authorization_url(
        profile.authorization_endpoint,
        profile.client_id,
        challenge="C",
        state="S",
        redirect_uri=profile.redirect_uri,
        scopes=profile.scopes,
        extra=dict(profile.authorization_params),
    )
    parts = parse.urlsplit(url)
    assert (parts.hostname, parts.path) == ("auth.openai.com", "/oauth/authorize")
    q = dict(parse.parse_qsl(parts.query))
    assert q["response_type"] == "code" and q["client_id"] == profile.client_id
    assert q["redirect_uri"] == chatgpt_oauth.REDIRECT_URI
    assert q["code_challenge"] == "C" and q["code_challenge_method"] == "S256"
    assert q["state"] == "S" and q["scope"] == "openid profile email offline_access"
    assert q["codex_cli_simplified_flow"] == "true" and q["originator"] == "agent6"


def test_parse_callback_accepts_url_or_query_and_checks_state() -> None:
    assert (
        _oauth_protocol.parse_callback(f"{chatgpt_oauth.REDIRECT_URI}?code=abc&state=S", state="S")
        == "abc"
    )
    assert _oauth_protocol.parse_callback("code=abc&state=S", state="S") == "abc"
    with pytest.raises(ValueError, match="state mismatch"):
        _oauth_protocol.parse_callback(
            f"{chatgpt_oauth.REDIRECT_URI}?code=abc&state=OTHER", state="S"
        )
    with pytest.raises(ValueError, match="no `code`"):
        _oauth_protocol.parse_callback(f"{chatgpt_oauth.REDIRECT_URI}?state=S", state="S")
    with pytest.raises(ValueError, match="access_denied"):
        _oauth_protocol.parse_callback(
            f"{chatgpt_oauth.REDIRECT_URI}?error=access_denied&state=S", state="S"
        )


@pytest.mark.parametrize(
    "payload",
    ["é", base64.urlsafe_b64encode(b'{"claim":true}').rstrip(b"=").decode() + "!!!!"],
    ids=["non-ascii", "invalid-alphabet"],
)
def test_jwt_claims_rejects_malformed_base64(payload: str) -> None:
    assert _oauth_protocol.jwt_claims(f"h.{payload}.s") == {}


def test_account_id_prefers_access_token_claim() -> None:
    access = _jwt({_AUTH_CLAIM: {"chatgpt_account_id": "acct-access"}})
    id_tok = _jwt({_AUTH_CLAIM: {"chatgpt_account_id": "acct-id"}})
    assert (
        chatgpt_oauth.account_id_of(_oauth_protocol.TokenGrant(access, "r", 60.0, id_token=id_tok))
        == "acct-access"
    )
    assert (
        chatgpt_oauth.account_id_of(
            _oauth_protocol.TokenGrant("opaque-token", "r", 60.0, id_token=id_tok)
        )
        == "acct-id"
    )
    assert chatgpt_oauth.account_id_of(_oauth_protocol.TokenGrant("garbage", "r", 60.0)) == ""
    assert _oauth_protocol.jwt_claims("not-a-jwt") == {}


def test_exchange_and_refresh_post_the_right_grants(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict[str, str]]] = []

    def fake_post(
        url: str, data: dict[str, str], timeout_s: float, headers: dict[str, str] | None = None
    ) -> _Resp:
        calls.append((url, data))
        return _Resp(
            200,
            {"access_token": "AT", "refresh_token": "RT", "expires_in": 1200, "id_token": "IT"},
        )

    monkeypatch.setattr("agent6.providers._oauth_protocol.post_form", fake_post)
    grant = _oauth_protocol.exchange_code(
        "https://auth.example/oauth/token",
        "app_X",
        label="ChatGPT",
        redirect_uri=chatgpt_oauth.PROFILE.redirect_uri,
        code="C0",
        verifier="V0",
        provider="chatgpt",
    )
    assert grant == _oauth_protocol.TokenGrant("AT", "RT", 1200.0, id_token="IT")
    url, data = calls[0]
    assert url == "https://auth.example/oauth/token"
    assert data["grant_type"] == "authorization_code"
    assert data["code_verifier"] == "V0" and data["redirect_uri"] == chatgpt_oauth.REDIRECT_URI

    _oauth_protocol.refresh_grant(
        "https://auth.example/oauth/token",
        "app_X",
        "RT",
        provider="chatgpt",
        label="ChatGPT",
    )
    _, data = calls[1]
    assert data == {"grant_type": "refresh_token", "refresh_token": "RT", "client_id": "app_X"}


@pytest.mark.parametrize(("field", "value"), [("refresh_token", 3), ("id_token", {})])
def test_grant_rejects_non_string_token_fields(
    monkeypatch: pytest.MonkeyPatch, field: str, value: object
) -> None:
    def malformed(
        url: str, data: dict[str, str], timeout_s: float, headers: dict[str, str] | None = None
    ) -> _Resp:
        body: dict[str, object] = {
            "access_token": "AT",
            "refresh_token": "RT",
            "expires_in": 3600,
        }
        body[field] = value
        return _Resp(200, body)

    monkeypatch.setattr("agent6.providers._oauth_protocol.post_form", malformed)
    with pytest.raises(types.ProviderError, match=field):
        _oauth_protocol.refresh_grant(
            "https://auth.example/oauth/token",
            "app_X",
            "RT",
            provider="chatgpt",
            label="ChatGPT",
        )


def test_dead_refresh_token_names_connect(monkeypatch: pytest.MonkeyPatch) -> None:
    """A dead refresh token names `agent6 connect chatgpt` and carries a 401.

    It is never retried.
    """

    def dead(
        url: str, data: dict[str, str], timeout_s: float, headers: dict[str, str] | None = None
    ) -> _Resp:
        return _Resp(400, {"error": {"code": "refresh_token_expired"}})

    monkeypatch.setattr("agent6.providers._oauth_protocol.post_form", dead)
    with pytest.raises(types.ProviderError) as exc:
        _oauth_protocol.refresh_grant(
            "https://auth.example/oauth/token",
            "app_X",
            "RT",
            provider="chatgpt",
            label="ChatGPT",
        )
    assert "agent6 connect chatgpt" in str(exc.value) and exc.value.status_code == 401

    def down(
        url: str, data: dict[str, str], timeout_s: float, headers: dict[str, str] | None = None
    ) -> _Resp:
        return _Resp(503, "upstream down")

    monkeypatch.setattr("agent6.providers._oauth_protocol.post_form", down)
    with pytest.raises(types.ProviderError) as exc:
        _oauth_protocol.refresh_grant(
            "https://auth.example/oauth/token",
            "app_X",
            "RT",
            provider="chatgpt",
            label="ChatGPT",
        )
    assert exc.value.status_code == 503


def test_device_auth_start_and_poll(monkeypatch: pytest.MonkeyPatch) -> None:
    """The device flow: usercode POST starts it, the poll waits on pending codes, success exchanges.

    A 404 on the start means disabled (None); slow_down backs off; the exchange uses the device
    redirect.
    """
    posts: list[tuple[str, dict[str, str]]] = []
    replies = [
        _Resp(200, {"device_auth_id": "da_1", "user_code": "AB-12", "interval": "5"}),
        _Resp(403, "pending"),
        _Resp(400, {"error": {"code": "deviceauth_authorization_pending"}}),
        _Resp(400, {"error": {"code": "slow_down"}}),
        _Resp(200, {"authorization_code": "AC", "code_verifier": "SERVER-V"}),
    ]

    def fake_json(
        url: str, data: dict[str, str], timeout_s: float, headers: dict[str, str] | None = None
    ) -> _Resp:
        posts.append((url, data))
        return replies.pop(0)

    exchanges: list[dict[str, str]] = []

    def fake_form(
        url: str, data: dict[str, str], timeout_s: float, headers: dict[str, str] | None = None
    ) -> _Resp:
        exchanges.append(data)
        return _Resp(200, {"access_token": "AT", "refresh_token": "RT", "expires_in": 60})

    monkeypatch.setattr("agent6.providers._oauth_protocol.post_json", fake_json)
    monkeypatch.setattr("agent6.providers._oauth_protocol.post_form", fake_form)
    naps: list[float] = []

    device = chatgpt_oauth.start_device_auth("https://auth.example", "app_X")
    assert device is not None and device.user_code == "AB-12" and device.interval_s == 5.0
    grant = chatgpt_oauth.poll_device_auth(
        "https://auth.example", "app_X", device, provider="chatgpt", sleep=naps.append
    )
    assert grant.access_token == "AT"
    assert posts[0] == (
        "https://auth.example/api/accounts/deviceauth/usercode",
        {"client_id": "app_X"},
    )
    assert posts[1][1] == {"device_auth_id": "da_1", "user_code": "AB-12"}
    assert naps == [5.0, 5.0, 10.0]  # two pendings, then slow_down backs off
    assert exchanges == [
        {
            "grant_type": "authorization_code",
            "client_id": "app_X",
            "code": "AC",
            "code_verifier": "SERVER-V",
            "redirect_uri": "https://auth.example/deviceauth/callback",
        }
    ]


def test_device_auth_disabled_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    def gone(
        url: str, data: dict[str, str], timeout_s: float, headers: dict[str, str] | None = None
    ) -> _Resp:
        return _Resp(404, "not enabled")

    monkeypatch.setattr("agent6.providers._oauth_protocol.post_json", gone)
    assert chatgpt_oauth.start_device_auth("https://auth.example", "app_X") is None


def test_refresh_error_scrubs_an_echoed_refresh_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """A refresh error that echoes the received credential is scrubbed before it reaches error text.

    Those messages land in retry events and logs; the oauth wire scrubs its own in-flight values.
    """
    secret = "rt-veryverysecretvalue123"

    def echoing_post(
        url: str, data: dict[str, str], timeout_s: float, headers: dict[str, str] | None = None
    ) -> _Resp:
        return _Resp(500, {"error": "boom", "received": secret})

    monkeypatch.setattr(_oauth_protocol, "post_form", echoing_post)
    with pytest.raises(types.ProviderError) as ei:
        _oauth_protocol.refresh_grant(
            "https://auth.openai.com/oauth/token",
            "cid",
            secret,
            provider="chatgpt",
            label="ChatGPT",
        )
    assert secret not in str(ei.value)
    assert "<REDACTED>" in str(ei.value)


def test_device_poll_failure_scrubs_a_device_id_split_across_the_clip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A device poll failure scrubs a device id that straddles the 200-char body clip."""
    device_id = "da-longenoughtomatterandbeused"
    body = "x" * 190 + device_id

    def fake_json(
        url: str, data: dict[str, str], timeout_s: float, headers: dict[str, str] | None = None
    ) -> _Resp:
        return _Resp(500, body)

    monkeypatch.setattr("agent6.providers._oauth_protocol.post_json", fake_json)
    device = chatgpt_oauth.DeviceAuth(device_auth_id=device_id, user_code="AB-12", interval_s=5.0)
    with pytest.raises(types.ProviderError) as ei:
        chatgpt_oauth.poll_device_auth("https://auth.example", "app_X", device, provider="chatgpt")
    assert device_id[:10] not in str(ei.value)


def test_revoke_warning_scrubs_the_token(monkeypatch: pytest.MonkeyPatch) -> None:
    tok = "at-echoedtokenvalue456789"

    def echoing_post(*args: object, **kwargs: object) -> _Resp:
        return _Resp(500, {"error": "boom", "received": tok})

    monkeypatch.setattr(_transport, "http_request", echoing_post)
    warn = _oauth_protocol.revoke_tokens(
        chatgpt_oauth.PROFILE.revocation_endpoint or "",
        "cid",
        access_token=tok,
        refresh_token="",
        encoding="json",
    )
    assert warn is not None and tok not in warn and "<REDACTED>" in warn


def test_revoke_warning_scrubs_a_token_split_across_the_clip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A revoke warning scrubs a token that straddles the 200-char body clip."""
    tok = "at-echoedtokenvalue456789"
    body = "x" * 190 + tok  # tok starts at 190: its first 10 chars sit before the 200 clip

    def echoing_post(*args: object, **kwargs: object) -> _Resp:
        return _Resp(500, body)

    monkeypatch.setattr(_transport, "http_request", echoing_post)
    warn = _oauth_protocol.revoke_tokens(
        chatgpt_oauth.PROFILE.revocation_endpoint or "",
        "cid",
        access_token=tok,
        refresh_token="",
        encoding="json",
    )
    assert warn is not None
    assert tok[:10] not in warn


def test_account_id_never_guesses_from_user_id() -> None:
    """The account id is never guessed from `user_id`.

    An account-less grant demands a re-connect.
    """
    tok = _jwt({_AUTH_CLAIM: {"user_id": "user-123"}})
    assert chatgpt_oauth.account_id_of(_oauth_protocol.TokenGrant(tok, "", 100.0, tok)) == ""
    good = _jwt({_AUTH_CLAIM: {"chatgpt_account_id": "acct-9"}})
    assert (
        chatgpt_oauth.account_id_of(_oauth_protocol.TokenGrant(good, "", 100.0, good)) == "acct-9"
    )


def test_callback_state_checked_before_the_error_param() -> None:
    """The callback's state check outranks its error param.

    A foreign request is not processed.
    """
    with pytest.raises(ValueError, match="state mismatch"):
        _oauth_protocol.parse_callback(
            "error=x&error_description=<script>alert(1)</script>", state="S"
        )


def test_callback_error_page_escapes_the_description() -> None:
    """The callback error page escapes the description, so error_description never runs script."""
    import urllib.error
    import urllib.request

    from agent6.ui.cli import _oauth

    srv = _oauth.CallbackServer("STATE", port=0)
    try:
        port = srv.port
        url = (
            f"http://127.0.0.1:{port}/auth/callback"
            "?state=STATE&error=x&error_description=<script>alert(1)</script>"
        )
        with pytest.raises(urllib.error.HTTPError) as ei:
            urllib.request.urlopen(url, timeout=5)
        body = ei.value.read().decode()
        assert "<script>" not in body
        assert "&lt;script&gt;" in body
        assert ei.value.headers.get("content-security-policy") == "default-src 'none'"
    finally:
        srv.close()


@pytest.mark.parametrize(
    "expires_in", ["soon", int("1" + "0" * 400), 0, -1, float("nan"), float("inf"), True]
)
def test_an_unusable_expires_in_is_a_provider_error(
    monkeypatch: pytest.MonkeyPatch, expires_in: object
) -> None:
    """A non-numeric, non-finite, boolean or non-positive expires_in is a provider error."""

    def odd(
        url: str, data: dict[str, str], timeout_s: float, headers: dict[str, str] | None = None
    ) -> _Resp:
        return _Resp(200, {"access_token": "AT", "refresh_token": "RT", "expires_in": expires_in})

    monkeypatch.setattr("agent6.providers._oauth_protocol.post_form", odd)
    with pytest.raises(types.ProviderError, match="expires_in"):
        _oauth_protocol.exchange_code(
            "https://auth.example/oauth/token",
            "app_X",
            label="ChatGPT",
            redirect_uri=chatgpt_oauth.PROFILE.redirect_uri,
            code="C",
            verifier="V",
            provider="chatgpt",
        )
    with pytest.raises(types.ProviderError, match="expires_in"):
        _oauth_protocol.refresh_grant(
            "https://auth.example/oauth/token",
            "app_X",
            "RT",
            provider="chatgpt",
            label="ChatGPT",
        )
