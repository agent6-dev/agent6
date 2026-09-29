# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for agent6.providers.chatgpt_oauth (PKCE, grants, credential)."""

from __future__ import annotations

import base64
import json
import pathlib
import time
from collections.abc import Generator
from typing import Any
from urllib import parse

import pytest

from agent6 import secrets
from agent6.providers import chatgpt_oauth, types


@pytest.fixture
def gcfg(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> pathlib.Path:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "g"))
    return tmp_path / "g"


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
        chatgpt_oauth.pkce_challenge("dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk")
        == "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
    )
    verifier, challenge = chatgpt_oauth.pkce_pair()
    assert 43 <= len(verifier) <= 128 and "=" not in challenge
    assert challenge == chatgpt_oauth.pkce_challenge(verifier)


def test_authorize_url_carries_the_registered_params() -> None:
    url = chatgpt_oauth.authorize_url("https://auth.example/", "app_X", challenge="C", state="S")
    parts = parse.urlsplit(url)
    assert (parts.hostname, parts.path) == ("auth.example", "/oauth/authorize")
    q = dict(parse.parse_qsl(parts.query))
    assert q["response_type"] == "code" and q["client_id"] == "app_X"
    assert q["redirect_uri"] == chatgpt_oauth.REDIRECT_URI
    assert q["code_challenge"] == "C" and q["code_challenge_method"] == "S256"
    assert q["state"] == "S" and q["scope"] == "openid profile email offline_access"
    assert q["codex_cli_simplified_flow"] == "true" and q["originator"] == "agent6"


def test_parse_callback_accepts_url_or_query_and_checks_state() -> None:
    assert (
        chatgpt_oauth.parse_callback(f"{chatgpt_oauth.REDIRECT_URI}?code=abc&state=S", state="S")
        == "abc"
    )
    assert chatgpt_oauth.parse_callback("code=abc&state=S", state="S") == "abc"
    with pytest.raises(ValueError, match="state mismatch"):
        chatgpt_oauth.parse_callback(
            f"{chatgpt_oauth.REDIRECT_URI}?code=abc&state=OTHER", state="S"
        )
    with pytest.raises(ValueError, match="no `code`"):
        chatgpt_oauth.parse_callback(f"{chatgpt_oauth.REDIRECT_URI}?state=S", state="S")
    with pytest.raises(ValueError, match="access_denied"):
        chatgpt_oauth.parse_callback(
            f"{chatgpt_oauth.REDIRECT_URI}?error=access_denied&state=S", state="S"
        )


@pytest.mark.parametrize(
    "payload",
    ["é", base64.urlsafe_b64encode(b'{"claim":true}').rstrip(b"=").decode() + "!!!!"],
    ids=["non-ascii", "invalid-alphabet"],
)
def test_jwt_claims_rejects_malformed_base64(payload: str) -> None:
    assert chatgpt_oauth.jwt_claims(f"h.{payload}.s") == {}


def test_account_id_prefers_access_token_claim() -> None:
    access = _jwt({_AUTH_CLAIM: {"chatgpt_account_id": "acct-access"}})
    id_tok = _jwt({_AUTH_CLAIM: {"chatgpt_account_id": "acct-id"}})
    assert (
        chatgpt_oauth.account_id_of(chatgpt_oauth.TokenGrant(access, "r", 60.0, id_token=id_tok))
        == "acct-access"
    )
    assert (
        chatgpt_oauth.account_id_of(
            chatgpt_oauth.TokenGrant("opaque-token", "r", 60.0, id_token=id_tok)
        )
        == "acct-id"
    )
    assert chatgpt_oauth.account_id_of(chatgpt_oauth.TokenGrant("garbage", "r", 60.0)) == ""
    assert chatgpt_oauth.jwt_claims("not-a-jwt") == {}


def test_exchange_and_refresh_post_the_right_grants(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict[str, str]]] = []

    def fake_post(url: str, data: dict[str, str], timeout_s: float) -> _Resp:
        calls.append((url, data))
        return _Resp(
            200,
            {"access_token": "AT", "refresh_token": "RT", "expires_in": 1200, "id_token": "IT"},
        )

    monkeypatch.setattr("agent6.providers.chatgpt_oauth._post_form", fake_post)
    grant = chatgpt_oauth.exchange_code(
        "https://auth.example", "app_X", code="C0", verifier="V0", provider="chatgpt"
    )
    assert grant == chatgpt_oauth.TokenGrant("AT", "RT", 1200.0, id_token="IT")
    url, data = calls[0]
    assert url == "https://auth.example/oauth/token"
    assert data["grant_type"] == "authorization_code"
    assert data["code_verifier"] == "V0" and data["redirect_uri"] == chatgpt_oauth.REDIRECT_URI

    chatgpt_oauth.refresh_grant("https://auth.example", "app_X", "RT", provider="chatgpt")
    _, data = calls[1]
    assert data == {"grant_type": "refresh_token", "refresh_token": "RT", "client_id": "app_X"}


def test_initial_exchange_requires_a_refresh_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Connect must not report success for a grant that cannot be loaded later."""

    def incomplete(url: str, data: dict[str, str], timeout_s: float) -> _Resp:
        return _Resp(200, {"access_token": "AT", "expires_in": 3600})

    monkeypatch.setattr("agent6.providers.chatgpt_oauth._post_form", incomplete)
    with pytest.raises(types.ProviderError, match="refresh_token"):
        chatgpt_oauth.exchange_code(
            "https://auth.example", "app_X", code="C", verifier="V", provider="chatgpt"
        )


@pytest.mark.parametrize(("field", "value"), [("refresh_token", 3), ("id_token", {})])
def test_grant_rejects_non_string_token_fields(
    monkeypatch: pytest.MonkeyPatch, field: str, value: object
) -> None:
    def malformed(url: str, data: dict[str, str], timeout_s: float) -> _Resp:
        body: dict[str, object] = {
            "access_token": "AT",
            "refresh_token": "RT",
            "expires_in": 3600,
        }
        body[field] = value
        return _Resp(200, body)

    monkeypatch.setattr("agent6.providers.chatgpt_oauth._post_form", malformed)
    with pytest.raises(types.ProviderError, match=field):
        chatgpt_oauth.refresh_grant("https://auth.example", "app_X", "RT", provider="chatgpt")


def test_dead_refresh_token_names_connect(monkeypatch: pytest.MonkeyPatch) -> None:
    """A dead refresh token names `agent6 connect chatgpt` and carries a 401.

    It is never retried.
    """

    def dead(url: str, data: dict[str, str], timeout_s: float) -> _Resp:
        return _Resp(400, {"error": {"code": "refresh_token_expired"}})

    monkeypatch.setattr("agent6.providers.chatgpt_oauth._post_form", dead)
    with pytest.raises(types.ProviderError) as exc:
        chatgpt_oauth.refresh_grant("https://auth.example", "app_X", "RT", provider="chatgpt")
    assert "agent6 connect chatgpt" in str(exc.value) and exc.value.status_code == 401

    def down(url: str, data: dict[str, str], timeout_s: float) -> _Resp:
        return _Resp(503, "upstream down")

    monkeypatch.setattr("agent6.providers.chatgpt_oauth._post_form", down)
    with pytest.raises(types.ProviderError) as exc:
        chatgpt_oauth.refresh_grant("https://auth.example", "app_X", "RT", provider="chatgpt")
    assert exc.value.status_code == 503


def test_every_remedy_names_the_provider_it_diagnosed(
    gcfg: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every remedy names the provider it diagnosed.

    `chatgpt` for a provider under another name.
    """

    def dead(url: str, data: dict[str, str], timeout_s: float) -> _Resp:
        return _Resp(400, {"error": {"code": "refresh_token_expired"}})

    monkeypatch.setattr("agent6.providers.chatgpt_oauth._post_form", dead)
    cred = chatgpt_oauth.ChatGPTCredential(
        "codex", issuer="https://auth.example", client_id="app_X"
    )
    with pytest.raises(types.ProviderError, match="agent6 connect codex"):
        cred.token()  # nothing stored
    secrets.save_oauth_tokens(
        "codex", secrets.OAuthTokens("AT0", "RT1", time.time() + 3600, "acct-1")
    )
    assert cred.token() == "AT0"
    secrets.save_oauth_tokens(
        "codex", secrets.OAuthTokens("AT1", "RT1", time.time() + 3600, "acct-2")
    )
    cred.invalidate()
    with pytest.raises(types.ProviderError, match="agent6 connect codex"):
        cred.token()  # the stored sign-in moved to another account
    with pytest.raises(types.ProviderError, match="agent6 connect codex"):
        chatgpt_oauth.refresh_grant(
            "https://auth.example", "app_X", "RT", provider="codex"
        )  # dead grant


def test_tokens_from_grant_keeps_previous_on_partial_refresh() -> None:
    prev = secrets.OAuthTokens("old-a", "old-r", 1.0, account_id="acct-1")
    fresh = chatgpt_oauth.tokens_from_grant(
        chatgpt_oauth.TokenGrant("new-a", "", 600.0), previous=prev
    )
    assert fresh.access_token == "new-a"
    assert fresh.refresh_token == "old-r" and fresh.account_id == "acct-1"
    assert fresh.expires_at > time.time() + 500


def test_credential_caches_refreshes_and_persists(
    gcfg: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    refreshes: list[str] = []

    def fake_post(url: str, data: dict[str, str], timeout_s: float) -> _Resp:
        refreshes.append(data["refresh_token"])
        return _Resp(
            200, {"access_token": f"AT{len(refreshes)}", "refresh_token": "RT2", "expires_in": 3600}
        )

    monkeypatch.setattr("agent6.providers.chatgpt_oauth._post_form", fake_post)
    cred = chatgpt_oauth.ChatGPTCredential(
        "chatgpt", issuer="https://auth.example", client_id="app_X"
    )
    with pytest.raises(types.ProviderError, match="agent6 connect chatgpt"):
        cred.token()

    secrets.save_oauth_tokens(
        "chatgpt", secrets.OAuthTokens("AT0", "RT1", time.time() + 3600, "acct")
    )
    assert cred.token() == "AT0" and refreshes == []

    secrets.save_oauth_tokens(
        "chatgpt", secrets.OAuthTokens("AT0", "RT1", time.time() + 10, "acct")
    )
    cred2 = chatgpt_oauth.ChatGPTCredential(
        "chatgpt", issuer="https://auth.example", client_id="app_X"
    )
    assert cred2.token() == "AT1" and refreshes == ["RT1"]
    stored = secrets.load_oauth_tokens("chatgpt")
    assert stored is not None and stored.refresh_token == "RT2" and stored.account_id == "acct"
    assert cred2.token() == "AT1" and len(refreshes) == 1  # cached until expiry
    assert cred2.account_id() == "acct"

    cred2.invalidate()
    assert cred2.token() == "AT2" and refreshes == ["RT1", "RT2"]


def test_credential_adopts_a_sibling_process_rotation(
    gcfg: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The credential adopts a sibling process's rotation.

    Replaying the old refresh token would hit a `refresh_token_reused` dead end.
    """

    def never(url: str, data: dict[str, str], timeout_s: float) -> _Resp:
        pytest.fail("refresh must not run")

    monkeypatch.setattr("agent6.providers.chatgpt_oauth._post_form", never)
    clock = {"now": 1000.0}
    fake_time = type("T", (), {"time": staticmethod(lambda: clock["now"])})
    monkeypatch.setattr("agent6.providers.chatgpt_oauth.time", fake_time)
    cred = chatgpt_oauth.ChatGPTCredential(
        "chatgpt", issuer="https://auth.example", client_id="app_X"
    )
    secrets.save_oauth_tokens("chatgpt", secrets.OAuthTokens("stale", "RT1", 5000.0, "acct"))
    assert cred.token() == "stale"
    # The cached copy ages out; a sibling has meanwhile stored a fresher grant.
    clock["now"] = 4800.0
    secrets.save_oauth_tokens("chatgpt", secrets.OAuthTokens("rotated", "RT2", 9000.0, "acct"))
    assert cred.token() == "rotated"


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

    def fake_json(url: str, data: dict[str, str], timeout_s: float) -> _Resp:
        posts.append((url, data))
        return replies.pop(0)

    exchanges: list[dict[str, str]] = []

    def fake_form(url: str, data: dict[str, str], timeout_s: float) -> _Resp:
        exchanges.append(data)
        return _Resp(200, {"access_token": "AT", "refresh_token": "RT", "expires_in": 60})

    monkeypatch.setattr("agent6.providers.chatgpt_oauth._post_json", fake_json)
    monkeypatch.setattr("agent6.providers.chatgpt_oauth._post_form", fake_form)
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
    assert exchanges[0]["code_verifier"] == "SERVER-V"
    assert exchanges[0]["redirect_uri"] == "https://auth.example/deviceauth/callback"


def test_device_auth_disabled_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    def gone(url: str, data: dict[str, str], timeout_s: float) -> _Resp:
        return _Resp(404, "not enabled")

    monkeypatch.setattr("agent6.providers.chatgpt_oauth._post_json", gone)
    assert chatgpt_oauth.start_device_auth("https://auth.example", "app_X") is None


def test_refresh_error_scrubs_an_echoed_refresh_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """A refresh error that echoes the received credential is scrubbed before it reaches error text.

    Those messages land in retry events and logs; the oauth wire scrubs its own in-flight values.
    """
    secret = "rt-veryverysecretvalue123"

    def echoing_post(url: str, data: dict[str, str], timeout_s: float) -> _Resp:
        return _Resp(500, {"error": "boom", "received": secret})

    monkeypatch.setattr(chatgpt_oauth, "_post_form", echoing_post)
    with pytest.raises(types.ProviderError) as ei:
        chatgpt_oauth.refresh_grant("https://auth.openai.com", "cid", secret, provider="chatgpt")
    assert secret not in str(ei.value)
    assert "<REDACTED>" in str(ei.value)


def test_device_poll_failure_scrubs_a_device_id_split_across_the_clip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A device poll failure scrubs a device id that straddles the 200-char body clip."""
    device_id = "da-longenoughtomatterandbeused"
    body = "x" * 190 + device_id

    def fake_json(url: str, data: dict[str, str], timeout_s: float) -> _Resp:
        return _Resp(500, body)

    monkeypatch.setattr("agent6.providers.chatgpt_oauth._post_json", fake_json)
    device = chatgpt_oauth.DeviceAuth(device_auth_id=device_id, user_code="AB-12", interval_s=5.0)
    with pytest.raises(types.ProviderError) as ei:
        chatgpt_oauth.poll_device_auth("https://auth.example", "app_X", device, provider="chatgpt")
    assert device_id[:10] not in str(ei.value)


def test_revoke_warning_scrubs_the_token(monkeypatch: pytest.MonkeyPatch) -> None:
    tok = "at-echoedtokenvalue456789"

    def echoing_post(*args: object, **kwargs: object) -> _Resp:
        return _Resp(500, {"error": "boom", "received": tok})

    monkeypatch.setattr(chatgpt_oauth.httpx2, "post", echoing_post)
    tokens = secrets.OAuthTokens(access_token=tok, refresh_token="", expires_at=0.0)
    warn = chatgpt_oauth.revoke_tokens("https://auth.openai.com", "cid", tokens)
    assert warn is not None and tok not in warn and "<REDACTED>" in warn


def test_revoke_warning_scrubs_a_token_split_across_the_clip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A revoke warning scrubs a token that straddles the 200-char body clip."""
    tok = "at-echoedtokenvalue456789"
    body = "x" * 190 + tok  # tok starts at 190: its first 10 chars sit before the 200 clip

    def echoing_post(*args: object, **kwargs: object) -> _Resp:
        return _Resp(500, body)

    monkeypatch.setattr(chatgpt_oauth.httpx2, "post", echoing_post)
    tokens = secrets.OAuthTokens(access_token=tok, refresh_token="", expires_at=0.0)
    warn = chatgpt_oauth.revoke_tokens("https://auth.openai.com", "cid", tokens)
    assert warn is not None
    assert tok[:10] not in warn


def test_account_id_never_guesses_from_user_id() -> None:
    """The account id is never guessed from `user_id`.

    An account-less grant demands a re-connect.
    """
    tok = _jwt({_AUTH_CLAIM: {"user_id": "user-123"}})
    assert chatgpt_oauth.account_id_of(chatgpt_oauth.TokenGrant(tok, "", 100.0, tok)) == ""
    good = _jwt({_AUTH_CLAIM: {"chatgpt_account_id": "acct-9"}})
    assert chatgpt_oauth.account_id_of(chatgpt_oauth.TokenGrant(good, "", 100.0, good)) == "acct-9"


def test_credential_refuses_an_account_swap(
    gcfg: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stored grant bound to a different account refuses with the connect hint.

    The first read pins the account.
    """
    clock = {"now": 1000.0}
    fake_time = type("T", (), {"time": staticmethod(lambda: clock["now"])})
    monkeypatch.setattr("agent6.providers.chatgpt_oauth.time", fake_time)
    cred = chatgpt_oauth.ChatGPTCredential(
        "chatgpt", issuer="https://auth.example", client_id="app_X"
    )
    secrets.save_oauth_tokens("chatgpt", secrets.OAuthTokens("tokA", "RT1", 5000.0, "acct-A"))
    assert cred.token() == "tokA"
    secrets.save_oauth_tokens("chatgpt", secrets.OAuthTokens("tokB", "RT2", 9000.0, "acct-B"))
    cred.invalidate(401)
    with pytest.raises(types.ProviderError, match="different account"):
        cred.token()


def test_credential_pins_a_claim_when_the_stored_account_is_empty(
    gcfg: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty stored account field must not hide a claimed account switch."""
    clock = {"now": 1000.0}
    fake_time = type("T", (), {"time": staticmethod(lambda: clock["now"])})
    monkeypatch.setattr("agent6.providers.chatgpt_oauth.time", fake_time)
    first = _jwt({_AUTH_CLAIM: {"chatgpt_account_id": "account-a"}})
    second = _jwt({_AUTH_CLAIM: {"chatgpt_account_id": "account-b"}})
    cred = chatgpt_oauth.ChatGPTCredential(
        "chatgpt", issuer="https://auth.example", client_id="app_X"
    )
    secrets.save_oauth_tokens("chatgpt", secrets.OAuthTokens(first, "RT1", 5000.0, "account-a"))
    assert cred.token() == first
    secrets.save_oauth_tokens("chatgpt", secrets.OAuthTokens(second, "RT2", 9000.0))
    cred.invalidate(401)
    with pytest.raises(types.ProviderError, match="different account"):
        cred.token()


def test_post_401_recovery_adopts_a_sibling_grant_first(
    gcfg: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After a 401 the credential adopts a newer stored grant first and rotates only without one."""

    def never(url: str, data: dict[str, str], timeout_s: float) -> _Resp:
        pytest.fail("refresh must not run when a fresh sibling grant exists")

    monkeypatch.setattr("agent6.providers.chatgpt_oauth._post_form", never)
    clock = {"now": 1000.0}
    fake_time = type("T", (), {"time": staticmethod(lambda: clock["now"])})
    monkeypatch.setattr("agent6.providers.chatgpt_oauth.time", fake_time)
    cred = chatgpt_oauth.ChatGPTCredential(
        "chatgpt", issuer="https://auth.example", client_id="app_X"
    )
    secrets.save_oauth_tokens("chatgpt", secrets.OAuthTokens("revoked", "RT1", 5000.0, "acct"))
    assert cred.token() == "revoked"
    secrets.save_oauth_tokens("chatgpt", secrets.OAuthTokens("fresh", "RT2", 9000.0, "acct"))
    cred.invalidate(401)
    assert cred.token() == "fresh"


def test_403_does_not_arm_a_refresh(gcfg: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A 403 does not arm a refresh: it is permission or entitlement.

    The bearer stays cached.
    """

    def never(url: str, data: dict[str, str], timeout_s: float) -> _Resp:
        pytest.fail("a 403 must not trigger a refresh")

    monkeypatch.setattr("agent6.providers.chatgpt_oauth._post_form", never)
    clock = {"now": 1000.0}
    fake_time = type("T", (), {"time": staticmethod(lambda: clock["now"])})
    monkeypatch.setattr("agent6.providers.chatgpt_oauth.time", fake_time)
    cred = chatgpt_oauth.ChatGPTCredential(
        "chatgpt", issuer="https://auth.example", client_id="app_X"
    )
    secrets.save_oauth_tokens("chatgpt", secrets.OAuthTokens("tok", "RT1", 5000.0, "acct"))
    assert cred.token() == "tok"
    cred.invalidate(403)
    assert cred.token() == "tok"


def test_invalid_grant_is_a_dead_signin(
    gcfg: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`invalid_grant` is a dead sign-in whose error names connect, not a retryable HTTP 400."""

    def dead(url: str, data: dict[str, str], timeout_s: float) -> _Resp:
        return _Resp(400, {"error": "invalid_grant"})

    monkeypatch.setattr("agent6.providers.chatgpt_oauth._post_form", dead)
    clock = {"now": 6000.0}
    fake_time = type("T", (), {"time": staticmethod(lambda: clock["now"])})
    monkeypatch.setattr("agent6.providers.chatgpt_oauth.time", fake_time)
    cred = chatgpt_oauth.ChatGPTCredential(
        "chatgpt", issuer="https://auth.example", client_id="app_X"
    )
    secrets.save_oauth_tokens("chatgpt", secrets.OAuthTokens("old", "RT1", 5000.0, "acct"))
    with pytest.raises(types.ProviderError, match="no longer valid"):
        cred.token()


def test_reused_rotation_rereads_once(gcfg: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`refresh_token_reused` with a fresher grant on disk adopts that grant instead of dying."""

    def reused(url: str, data: dict[str, str], timeout_s: float) -> _Resp:
        # A sibling's rotation lands between our read and the endpoint's answer.
        secrets.save_oauth_tokens("chatgpt", secrets.OAuthTokens("winner", "RT9", 9000.0, "acct"))
        return _Resp(401, {"error": {"code": "refresh_token_reused"}})

    monkeypatch.setattr("agent6.providers.chatgpt_oauth._post_form", reused)

    def no_sleep(_s: float) -> None:
        return None

    clock = {"now": 6000.0}
    fake_time = type(
        "T",
        (),
        {"time": staticmethod(lambda: clock["now"]), "sleep": staticmethod(no_sleep)},
    )
    monkeypatch.setattr("agent6.providers.chatgpt_oauth.time", fake_time)
    cred = chatgpt_oauth.ChatGPTCredential(
        "chatgpt", issuer="https://auth.example", client_id="app_X"
    )
    secrets.save_oauth_tokens("chatgpt", secrets.OAuthTokens("old", "RT1", 5000.0, "acct"))
    assert cred.token() == "winner"


def test_reused_rotation_does_not_adopt_an_expired_sibling(
    gcfg: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A changed access token is not a rescue when it is already stale."""

    def reused(url: str, data: dict[str, str], timeout_s: float) -> _Resp:
        secrets.save_oauth_tokens(
            "chatgpt", secrets.OAuthTokens("stale-sibling", "RT9", 6200.0, "acct")
        )
        return _Resp(401, {"error": {"code": "refresh_token_reused"}})

    monkeypatch.setattr("agent6.providers.chatgpt_oauth._post_form", reused)

    def no_sleep(_s: float) -> None:
        return None

    fake_time = type(
        "T",
        (),
        {"time": staticmethod(lambda: 6000.0), "sleep": staticmethod(no_sleep)},
    )
    monkeypatch.setattr("agent6.providers.chatgpt_oauth.time", fake_time)
    cred = chatgpt_oauth.ChatGPTCredential(
        "chatgpt", issuer="https://auth.example", client_id="app_X"
    )
    secrets.save_oauth_tokens("chatgpt", secrets.OAuthTokens("old", "RT1", 5000.0, "acct"))
    with pytest.raises(types.ProviderError, match="refresh_token_reused"):
        cred.token()


def test_callback_state_checked_before_the_error_param() -> None:
    """The callback's state check outranks its error param.

    A foreign request is not processed.
    """
    with pytest.raises(ValueError, match="state mismatch"):
        chatgpt_oauth.parse_callback(
            "error=x&error_description=<script>alert(1)</script>", state="S"
        )


def test_callback_error_page_escapes_the_description() -> None:
    """The callback error page escapes the description, so error_description never runs script."""
    import urllib.error
    import urllib.request

    from agent6.ui.cli import connect  # pyright: ignore[reportPrivateUsage]

    srv = connect._CallbackServer("STATE", port=0)
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


def test_unheld_refresh_lock_refuses_the_rotation(
    gcfg: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the interprocess refresh lock not held, the rotation is refused retryably."""
    import contextlib

    @contextlib.contextmanager
    def unheld(_path: pathlib.Path) -> Generator[bool]:
        yield False

    monkeypatch.setattr("agent6.portable.locked_file", unheld)
    clock = {"now": 6000.0}
    fake_time = type("T", (), {"time": staticmethod(lambda: clock["now"])})
    monkeypatch.setattr("agent6.providers.chatgpt_oauth.time", fake_time)
    cred = chatgpt_oauth.ChatGPTCredential(
        "chatgpt", issuer="https://auth.example", client_id="app_X"
    )
    secrets.save_oauth_tokens("chatgpt", secrets.OAuthTokens("old", "RT1", 5000.0, "acct"))
    with pytest.raises(types.ProviderError, match="refresh lock"):
        cred.token()


def test_stored_account_must_match_the_tokens_own_claim(
    gcfg: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stored account must match the token's own claim, or the credential refuses.

    An entry written by an older parser can hold a user id in the account field.
    """
    clock = {"now": 1000.0}
    fake_time = type("T", (), {"time": staticmethod(lambda: clock["now"])})
    monkeypatch.setattr("agent6.providers.chatgpt_oauth.time", fake_time)
    tok = _jwt({_AUTH_CLAIM: {"chatgpt_account_id": "acct-real"}})
    secrets.save_oauth_tokens("chatgpt", secrets.OAuthTokens(tok, "RT1", 5000.0, "user-legacy"))
    cred = chatgpt_oauth.ChatGPTCredential(
        "chatgpt", issuer="https://auth.example", client_id="app_X"
    )
    with pytest.raises(types.ProviderError, match="does not match its own token"):
        cred.token()


def test_403_reports_no_retry_worthwhile() -> None:
    cred = chatgpt_oauth.ChatGPTCredential(
        "chatgpt", issuer="https://auth.example", client_id="app_X"
    )
    assert cred.invalidate(403) is False
    assert cred.invalidate(401) is True


@pytest.mark.parametrize(
    "expires_in", ["soon", int("1" + "0" * 400), 0, -1, float("nan"), float("inf"), True]
)
def test_an_unusable_expires_in_is_a_provider_error(
    monkeypatch: pytest.MonkeyPatch, expires_in: object
) -> None:
    """A non-numeric, non-finite, boolean or non-positive expires_in is a provider error."""

    def odd(url: str, data: dict[str, str], timeout_s: float) -> _Resp:
        return _Resp(200, {"access_token": "AT", "refresh_token": "RT", "expires_in": expires_in})

    monkeypatch.setattr("agent6.providers.chatgpt_oauth._post_form", odd)
    with pytest.raises(types.ProviderError, match="expires_in"):
        chatgpt_oauth.exchange_code(
            "https://auth.example", "app_X", code="C", verifier="V", provider="chatgpt"
        )
    with pytest.raises(types.ProviderError, match="expires_in"):
        chatgpt_oauth.refresh_grant("https://auth.example", "app_X", "RT", provider="chatgpt")
