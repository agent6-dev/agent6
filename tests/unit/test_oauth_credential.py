# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The configured-OAuth credential: stored, bound, refreshed, exchanged, recovered once."""

from __future__ import annotations

import base64
import dataclasses
import json
import pathlib
import stat
import threading
import time
from concurrent import futures
from typing import Any

import httpx2
import pytest

from agent6 import secret_store
from agent6.config import ChatGPTProviderEntry, Config, OpenAIProviderEntry
from agent6.providers import _oauth_protocol, _transport, chatgpt_oauth, oauth, types

_ENTRY: dict[str, Any] = {
    "api_format": "openai",
    "base_url": "https://models.example/v1",
    "oauth": {
        "issuer": "https://login.example",
        "client_id": "client-1",
        "device_authorization_endpoint": "https://login.example/device",
        "token_endpoint": "https://login.example/token",
        "refresh_skew_s": 60.0,
    },
}


@pytest.fixture
def gcfg(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> pathlib.Path:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "g"))
    return tmp_path / "g"


def _entry(**overrides: Any) -> OpenAIProviderEntry:
    data = {**_ENTRY, "oauth": {**_ENTRY["oauth"], **overrides.pop("oauth", {})}, **overrides}
    cfg = Config.model_validate({"providers": {"work": data}})
    entry = cfg.providers["work"]
    assert isinstance(entry, OpenAIProviderEntry)
    return entry


class _Resp:
    def __init__(self, status: int, body: object, text: str | None = None) -> None:
        self.status_code = status
        self._body = body
        self.text = text if text is not None else str(body)
        self.headers = httpx2.Headers()

    def json(self) -> object:
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


def _store(
    entry: OpenAIProviderEntry | ChatGPTProviderEntry, **fields: Any
) -> secret_store.StoredGrant:
    grant = secret_store.StoredGrant(
        binding=oauth.binding_fingerprint(entry),
        access_token="AT1",
        refresh_token="RT1",
        issued_at=1_000.0,
        expires_at=4_600.0,
        exchange_access_token="",
        exchange_issued_at=0.0,
        exchange_expires_at=None,
    )
    grant = dataclasses.replace(grant, **fields)
    secret_store.save_oauth_grant("work", grant)
    return grant


def test_a_grant_round_trips_the_store_at_0600(gcfg: pathlib.Path) -> None:
    entry = _entry()
    stored = _store(entry, expires_at=None, refresh_token="")
    path = secret_store.paths.secrets_path()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert secret_store.load_oauth_grant("work") == stored
    _store(entry, refresh_token="RT-full")
    text = path.read_text(encoding="utf-8")
    assert "oauth_version" in text and "oauth_binding" in text


def test_shared_holder_accepts_a_bound_chatgpt_profile(gcfg: pathlib.Path) -> None:
    entry = ChatGPTProviderEntry(api_format="chatgpt")
    _store(entry)
    assert oauth.OAuthCredential("work", entry, now=lambda: 2_000.0).token() == "AT1"


def test_a_configured_grant_keeps_its_original_binding(gcfg: pathlib.Path) -> None:
    """Profile resolution must not invalidate an existing configured sign-in."""
    fingerprint = "e7efc70de66a76e1f169a28681e239b3"
    entry = _entry()
    assert oauth.binding_fingerprint(entry) == fingerprint
    stored = _store(entry, binding=fingerprint)
    assert oauth.OAuthCredential("work", entry, now=lambda: 2_000.0).token() == stored.access_token


def test_configured_profile_preserves_settings_without_chatgpt_policies() -> None:
    entry = _entry(oauth={"request_encoding": "json"})
    profile = oauth.profile_for(entry)
    assert profile.token_endpoint == "https://login.example/token"
    assert profile.request_encoding == profile.revocation_encoding == "json"
    assert profile.device_flow == "rfc8628"
    assert profile.default_expires_in is None
    assert profile.cap_refresh_skew is True
    assert profile.require_refresh_token is False
    assert profile.account_id_of is None
    assert profile.refresh_skew_s == 60.0


def test_chatgpt_profile_keeps_registration_and_mixed_encodings() -> None:
    profile = oauth.profile_for(ChatGPTProviderEntry(api_format="chatgpt"))
    assert profile is chatgpt_oauth.PROFILE
    assert profile.redirect_uri == "http://localhost:1455/auth/callback"
    assert profile.callback_host == "127.0.0.1"
    assert profile.request_encoding == "form" and profile.revocation_encoding == "json"
    assert profile.device_flow == "chatgpt"
    assert (
        profile.device_authorization_endpoint
        == "https://auth.openai.com/api/accounts/deviceauth/usercode"
    )
    assert profile.device_poll_endpoint == "https://auth.openai.com/api/accounts/deviceauth/token"
    assert profile.device_redirect_uri == "https://auth.openai.com/deviceauth/callback"
    assert dict(profile.authorization_params) == {
        "id_token_add_organizations": "true",
        "codex_cli_simplified_flow": "true",
        "originator": "agent6",
    }
    assert profile.label == "ChatGPT"
    assert profile.account_id_of is chatgpt_oauth.account_id_of
    assert profile.require_refresh_token is True
    assert profile.default_expires_in == 3600.0
    assert profile.refresh_skew_s == 300.0 and profile.cap_refresh_skew is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("token_endpoint", "https://other.example/token"),
        ("device_poll_endpoint", "https://other.example/poll"),
        ("redirect_uri", "http://localhost:1455/other"),
        ("revocation_encoding", "form"),
    ],
)
def test_chatgpt_binding_covers_effective_profile_settings(
    monkeypatch: pytest.MonkeyPatch, field: str, value: str
) -> None:
    entry = ChatGPTProviderEntry(api_format="chatgpt")
    fingerprint = oauth.binding_fingerprint(entry)
    monkeypatch.setattr(
        chatgpt_oauth, "PROFILE", dataclasses.replace(chatgpt_oauth.PROFILE, **{field: value})
    )
    assert oauth.binding_fingerprint(entry) != fingerprint


def test_the_binding_follows_every_destination_setting() -> None:
    base = oauth.binding_fingerprint(_entry())
    assert base == oauth.binding_fingerprint(_entry())
    assert base != oauth.binding_fingerprint(_entry(base_url="https://other.example/v1"))
    assert base != oauth.binding_fingerprint(
        _entry(oauth={"issuer": "https://other-login.example"})
    )
    assert base != oauth.binding_fingerprint(
        _entry(oauth={"exchange": {"endpoint": "https://credentials.example/token"}})
    )
    assert base != oauth.binding_fingerprint(_entry(extra_query={"api-version": "1"}))
    assert base == oauth.binding_fingerprint(_entry(oauth={"refresh_skew_s": 5.0}))  # a margin


def test_a_grant_made_under_other_settings_refuses_before_anything_is_sent(
    gcfg: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _store(_entry())

    def never(*_a: object, **_k: object) -> None:
        raise AssertionError("no request may leave")

    monkeypatch.setattr(_oauth_protocol, "post_form", never)
    changed = _entry(base_url="https://other.example/v1")
    with pytest.raises(types.ProviderError, match=r"other provider settings.*agent6 connect work"):
        oauth.OAuthCredential("work", changed).token()


def test_a_fresh_grant_is_sent_as_it_is_and_refreshed_inside_its_skew(
    gcfg: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = _entry()
    _store(entry)
    clock = {"now": 2_000.0}
    refreshes: list[dict[str, str]] = []

    def fake_post(
        url: str, data: dict[str, str], timeout_s: float, headers: dict[str, str] | None = None
    ) -> _Resp:
        refreshes.append(data)
        return _Resp(200, {"access_token": "AT2", "expires_in": 3600})  # no rotated refresh token

    monkeypatch.setattr(_oauth_protocol, "post_form", fake_post)
    cred = oauth.OAuthCredential("work", entry, now=lambda: clock["now"])
    assert cred.token() == "AT1" and refreshes == []
    clock["now"] = 4_550.0  # inside the 60 s skew
    assert cred.token() == "AT2"
    assert refreshes[0]["grant_type"] == "refresh_token" and refreshes[0]["refresh_token"] == "RT1"
    stored = secret_store.load_oauth_grant("work")
    assert stored is not None
    assert (stored.access_token, stored.refresh_token) == ("AT2", "RT1")  # the previous one kept
    assert stored.expires_at == 4_550.0 + 3600


def test_a_non_expiring_grant_without_a_refresh_token_is_used_until_refused(
    gcfg: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = _entry()
    _store(entry, refresh_token="", expires_at=None)
    cred = oauth.OAuthCredential("work", entry, now=lambda: 9_999_999.0)
    assert cred.token() == "AT1"
    assert cred.invalidate(403, "AT1") is False
    assert cred.token() == "AT1"
    assert cred.invalidate(401, "AT1") is True
    with pytest.raises(types.ProviderError, match=r"no refresh token.*agent6 connect work"):
        cred.token()


@pytest.mark.parametrize(
    ("status", "rejected_bearer", "armed"),
    [(401, "AT1", False), (401, "AT2", True), (403, "AT2", False)],
)
@pytest.mark.parametrize("chatgpt", [False, True])
def test_only_a_current_bearers_401_arms_a_refresh(
    gcfg: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    rejected_bearer: str,
    armed: bool,
    chatgpt: bool,
) -> None:
    """A late refusal of an old bearer must not rotate the shared holder's current grant."""
    entry = ChatGPTProviderEntry(api_format="chatgpt") if chatgpt else _entry()
    _store(entry)
    clock = {"now": 2_000.0}
    refreshes: list[str] = []

    def fake_post(
        url: str, data: dict[str, str], timeout_s: float, headers: dict[str, str] | None = None
    ) -> _Resp:
        refreshes.append(data["refresh_token"])
        version = len(refreshes) + 1
        return _Resp(
            200,
            {
                "access_token": f"AT{version}",
                "refresh_token": f"RT{version}",
                "expires_in": 3600,
            },
        )

    monkeypatch.setattr(_oauth_protocol, "post_form", fake_post)
    cred = oauth.OAuthCredential("work", entry, now=lambda: clock["now"])
    assert cred.token() == "AT1"
    clock["now"] = 4_550.0
    assert cred.token() == "AT2"
    assert refreshes == ["RT1"]

    assert cred.invalidate(status, rejected_bearer) is armed
    assert cred.token() == ("AT3" if armed else "AT2")
    assert refreshes == (["RT1", "RT2"] if armed else ["RT1"])


def test_the_second_exchange_runs_first_and_refreshes_the_issuer_once_on_its_401(
    gcfg: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = _entry(
        oauth={
            "exchange": {
                "endpoint": "https://credentials.example/session-token",
                "method": "GET",
                "access_token_field": "token",
                "expires_field": "expires_at",
                "expires_format": "unix_seconds",
            }
        }
    )
    _store(entry)
    clock = {"now": 2_000.0}
    seen: list[dict[str, Any]] = []
    answers: list[_Resp] = [
        _Resp(200, {"token": "EX1", "expires_at": 2_600.0}),
        _Resp(401, {"message": "token AT1 rejected"}),
        _Resp(200, {"token": "EX2", "expires_at": 9_000.0}),
    ]

    def fake_request(method: str, url: str, **kwargs: Any) -> _Resp:
        seen.append({"url": url, "method": method, **kwargs})
        return answers.pop(0)

    def fake_post(
        url: str, data: dict[str, str], timeout_s: float, headers: dict[str, str] | None = None
    ) -> _Resp:
        seen.append({"url": url, "data": data})
        return _Resp(200, {"access_token": "AT2", "refresh_token": "RT2", "expires_in": 7200})

    monkeypatch.setattr(_transport, "http_request", fake_request)
    monkeypatch.setattr(_oauth_protocol, "post_form", fake_post)
    cred = oauth.OAuthCredential("work", entry, now=lambda: clock["now"])
    assert cred.token() == "EX1"
    assert seen[0]["headers"]["authorization"] == "Bearer AT1"
    assert seen[0]["method"] == "GET" and seen[0]["cap"] == _transport.CREDENTIAL_BODY_BYTES
    stored = secret_store.load_oauth_grant("work")
    assert stored is not None and stored.exchange_access_token == "EX1"
    clock["now"] = 2_590.0  # inside the exchanged token's skew: re-exchange, which answers 401
    assert cred.token() == "EX2"
    urls = [s["url"] for s in seen]
    assert urls == [
        "https://credentials.example/session-token",
        "https://credentials.example/session-token",
        "https://login.example/token",
        "https://credentials.example/session-token",
    ]
    assert seen[3]["headers"]["authorization"] == "Bearer AT2"
    stored = secret_store.load_oauth_grant("work")
    assert stored is not None
    assert (stored.access_token, stored.refresh_token, stored.exchange_access_token) == (
        "AT2",
        "RT2",
        "EX2",
    )


def test_an_exchange_error_never_echoes_the_issuer_token(
    gcfg: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = _entry(oauth={"exchange": {"endpoint": "https://credentials.example/token"}})
    _store(entry, access_token="AT-SECRET-VALUE-1")

    def fake_request(method: str, url: str, **_kw: Any) -> _Resp:
        return _Resp(
            500, {"error": "boom AT-SECRET-VALUE-1"}, text='{"error": "boom AT-SECRET-VALUE-1"}'
        )

    monkeypatch.setattr(_transport, "http_request", fake_request)
    with pytest.raises(types.ProviderError) as failed:
        oauth.OAuthCredential("work", entry, now=lambda: 2_000.0).token()
    assert "AT-SECRET-VALUE-1" not in str(failed.value) and "REDACTED" in str(failed.value)


def test_the_device_grant_waits_through_pending_and_slow_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RFC 8628 polling, including an issuer that answers pending as 200 with an error body."""
    posts: list[tuple[str, dict[str, str]]] = []
    answers: list[_Resp] = [
        _Resp(
            200,
            {
                "device_code": "DC",
                "user_code": "ABCD-EFGH",
                "verification_uri": "https://login.example/device",
                "interval": 1,
                "expires_in": 600,
            },
        ),
        _Resp(400, {"error": "authorization_pending"}),
        _Resp(200, {"error": "authorization_pending"}),
        _Resp(400, {"error": "slow_down"}),
        _Resp(200, {"access_token": "AT", "token_type": "bearer", "scope": "x"}),
    ]
    sleeps: list[float] = []

    def fake_post(
        url: str, data: dict[str, str], timeout_s: float, headers: dict[str, str] | None = None
    ) -> _Resp:
        posts.append((url, data))
        return answers.pop(0)

    monkeypatch.setattr(_oauth_protocol, "post_form", fake_post)
    started = _oauth_protocol.start_device_grant(
        "https://login.example/device", "client-1", scopes=("a", "b"), encoding="form", label="work"
    )
    assert (started.user_code, started.verification_uri) == (
        "ABCD-EFGH",
        "https://login.example/device",
    )
    assert posts[0][1] == {"client_id": "client-1", "scope": "a b"}
    grant = _oauth_protocol.poll_device_grant(
        "https://login.example/token",
        "client-1",
        started,
        encoding="form",
        label="work",
        sleep=sleeps.append,
        now=lambda: 0.0,
    )
    assert grant.access_token == "AT" and grant.refresh_token == "" and grant.expires_in is None
    assert posts[1][1]["grant_type"] == "urn:ietf:params:oauth:grant-type:device_code"
    assert posts[1][1]["device_code"] == "DC"
    assert sleeps == [1.0, 1.0, 6.0]


@pytest.mark.parametrize(
    ("code", "word"), [("access_denied", "refused"), ("expired_token", "expired")]
)
def test_a_denied_or_expired_device_grant_ends_by_name(
    monkeypatch: pytest.MonkeyPatch, code: str, word: str
) -> None:
    def fake_post(
        url: str, data: dict[str, str], timeout_s: float, headers: dict[str, str] | None = None
    ) -> _Resp:
        return _Resp(400, {"error": code})

    monkeypatch.setattr(_oauth_protocol, "post_form", fake_post)
    started = _oauth_protocol.DeviceGrant(
        "DC", "AB", "https://login.example/device", "", 1.0, 600.0
    )
    with pytest.raises(types.ProviderError, match=word):
        _oauth_protocol.poll_device_grant(
            "https://login.example/token",
            "client-1",
            started,
            encoding="form",
            label="work",
            sleep=lambda _s: None,
            now=lambda: 0.0,
        )


def test_the_listing_goes_through_the_credential(
    gcfg: pathlib.Path, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    from agent6.models import cache

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "c"))
    entry = _entry()
    _store(entry, expires_at=None)  # the listing runs on the real clock
    seen: dict[str, Any] = {}

    def fake_get(
        method: str, url: str, headers: dict[str, str], timeout: float, **_kw: Any
    ) -> httpx2.Response:
        seen["url"] = url
        seen["headers"] = headers
        return httpx2.Response(
            200, json={"data": [{"id": "m1"}]}, request=httpx2.Request("GET", url)
        )

    monkeypatch.setattr(_transport, "http_request", fake_get)
    assert cache.fetch_models_live("work", entry, None, timeout_s=5.0) == ["m1"]
    assert seen["headers"]["authorization"] == "Bearer AT1"


def test_a_device_poll_never_sleeps_past_the_grant_expiry(monkeypatch: pytest.MonkeyPatch) -> None:
    """An issuer asking for a huge interval cannot hold `connect` past the code's expiry."""
    clock = {"t": 0.0}
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock["t"] += seconds

    def fake_post(
        url: str, data: dict[str, str], timeout_s: float, headers: dict[str, str] | None = None
    ) -> _Resp:
        return _Resp(400, {"error": "slow_down"})

    monkeypatch.setattr(_oauth_protocol, "post_form", fake_post)
    started = _oauth_protocol.DeviceGrant("DC", "AB", "https://l.example/d", "", 1e9, 5.0)
    with pytest.raises(types.ProviderError, match="expired"):
        _oauth_protocol.poll_device_grant(
            "https://l.example/token",
            "c",
            started,
            encoding="form",
            label="work",
            sleep=sleep,
            now=lambda: clock["t"],
        )
    assert sleeps == [5.0]


def test_a_device_start_caps_the_issuers_interval(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(
        url: str, data: dict[str, str], timeout_s: float, headers: dict[str, str] | None = None
    ) -> _Resp:
        return _Resp(
            200,
            {
                "device_code": "DC",
                "user_code": "AB",
                "verification_uri": "https://l.example/d",
                "interval": 100000,
            },
        )

    monkeypatch.setattr(_oauth_protocol, "post_form", fake_post)
    started = _oauth_protocol.start_device_grant(
        "https://l.example/device", "c", scopes=(), encoding="form", label="work"
    )
    assert started.interval_s == _oauth_protocol.POLL_INTERVAL_CEILING_S


def _account_token(account: str) -> str:
    payload = json.dumps({"https://api.openai.com/auth": {"chatgpt_account_id": account}})
    return "e30." + base64.urlsafe_b64encode(payload.encode()).decode().rstrip("=") + ".sig"


def test_id_token_only_account_is_persisted_without_identity_token(gcfg: pathlib.Path) -> None:
    entry = ChatGPTProviderEntry(api_format="chatgpt")
    identity = _account_token("account-a")
    grant = oauth.stored_grant_for(
        _oauth_protocol.TokenGrant("opaque", "RT", 3600, identity), entry, now=1000
    )
    assert grant.account_id == "account-a"
    path = secret_store.save_oauth_grant("work", grant)
    assert identity not in path.read_text() and "id_token" not in path.read_text()
    holder = oauth.OAuthCredential("work", entry, now=lambda: 2000)
    assert holder.account_id() == "account-a"
    assert holder.token() == "opaque"


@pytest.mark.parametrize("refresh", ["", " "])
def test_initial_chatgpt_grant_requires_nonblank_refresh_token(refresh: str) -> None:
    grant = _oauth_protocol.TokenGrant("AT", refresh, None)
    with pytest.raises(types.ProviderError, match=r"ChatGPT.*no refresh_token"):
        oauth.stored_grant_for(grant, ChatGPTProviderEntry(api_format="chatgpt"), now=1000)
    assert oauth.stored_grant_for(grant, _entry(), now=1000).refresh_token == refresh


def test_chatgpt_omitted_expiry_uses_an_hour_but_configured_expiry_is_indefinite() -> None:
    grant = _oauth_protocol.TokenGrant("AT", "RT", None)
    assert (
        oauth.stored_grant_for(
            grant, ChatGPTProviderEntry(api_format="chatgpt"), now=1000
        ).expires_at
        == 4600
    )
    assert oauth.stored_grant_for(grant, _entry(), now=1000).expires_at is None


def test_chatgpt_uses_full_skew_and_keeps_account_on_partial_refresh(
    gcfg: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = ChatGPTProviderEntry(api_format="chatgpt")
    _store(entry, account_id="account-a", expires_at=1400)
    calls: list[dict[str, str]] = []

    def refresh(url: str, data: dict[str, str], *_a: object, **_kw: object) -> _Resp:
        calls.append(data)
        return _Resp(200, {"access_token": "AT2"})

    monkeypatch.setattr(_oauth_protocol, "post_form", refresh)
    holder = oauth.OAuthCredential("work", entry, now=lambda: 1100)
    assert holder.token() == "AT2"
    stored = secret_store.load_oauth_grant("work")
    assert stored is not None
    assert (stored.account_id, stored.refresh_token, stored.expires_at) == (
        "account-a",
        "RT1",
        4700,
    )
    assert calls == [
        {
            "grant_type": "refresh_token",
            "refresh_token": "RT1",
            "client_id": chatgpt_oauth.CHATGPT_CLIENT_ID,
        }
    ]


def test_shared_holder_refuses_stored_account_contradiction(gcfg: pathlib.Path) -> None:
    entry = ChatGPTProviderEntry(api_format="chatgpt")
    _store(entry, access_token=_account_token("claimed"), account_id="stored")
    with pytest.raises(
        types.ProviderError, match=r"ChatGPT.*does not match its own token.*agent6 connect work"
    ):
        oauth.OAuthCredential("work", entry, now=lambda: 2000).token()


@pytest.mark.parametrize("source", ["refresh", "sibling"])
@pytest.mark.parametrize("stored_account", [None, "account-a"])
def test_shared_holder_pins_first_account_before_refresh_or_adoption(
    gcfg: pathlib.Path, monkeypatch: pytest.MonkeyPatch, source: str, stored_account: str | None
) -> None:
    entry = ChatGPTProviderEntry(api_format="chatgpt")
    first = _account_token("account-a")
    _store(entry, access_token=first, account_id=stored_account)
    holder = oauth.OAuthCredential("work", entry, now=lambda: 2000)
    assert holder.token() == first
    second = _account_token("account-b")

    def refresh(*_a: object, **_kw: object) -> _Resp:
        return _Resp(200, {"access_token": second, "refresh_token": "RT2", "expires_in": 3600})

    monkeypatch.setattr(_oauth_protocol, "post_form", refresh)
    if source == "sibling":
        _store(entry, access_token=second, account_id=None)
    before = secret_store.paths.secrets_path().read_bytes()
    holder.invalidate(401, first)
    with pytest.raises(types.ProviderError, match=r"different account.*agent6 connect work"):
        holder.token()
    assert secret_store.paths.secrets_path().read_bytes() == before


@pytest.mark.parametrize("kind", ["missing", "legacy", "rebound", "contradiction", "valid"])
def test_read_only_bound_grant_validation(gcfg: pathlib.Path, kind: str) -> None:
    entry = ChatGPTProviderEntry(api_format="chatgpt")
    if kind == "legacy":
        secret_store.save_secret(
            "work",
            "",
            extra={
                "oauth_access_token": "AT",
                "oauth_refresh_token": "RT",
                "oauth_expires_at": "4600",
                "oauth_account_id": "a",
            },
        )
    elif kind != "missing":
        _store(
            entry,
            binding="other" if kind == "rebound" else oauth.binding_fingerprint(entry),
            access_token=_account_token("a"),
            account_id="b" if kind == "contradiction" else None,
        )
    secrets = secret_store.load_secrets()
    path = secret_store.paths.secrets_path()
    before = path.read_bytes() if path.exists() else None
    if kind == "valid":
        assert oauth.load_bound_grant("work", entry, secrets=secrets).account_id == "a"
    else:
        message = {
            "missing": "No ChatGPT sign-in",
            "legacy": "predates credential binding",
            "rebound": "other provider settings",
            "contradiction": "does not match its own token",
        }[kind]
        with pytest.raises(types.ProviderError, match=message) as refused:
            oauth.load_bound_grant("work", entry, secrets=secrets)
        assert refused.value.status_code == 401 and "agent6 connect work" in str(refused.value)
    assert (path.read_bytes() if path.exists() else None) == before


@pytest.mark.parametrize(
    "code",
    ["refresh_token_reused", "refresh_token_expired", "refresh_token_invalidated", "invalid_grant"],
)
def test_refresh_errors_have_classified_codes(code: str) -> None:
    response = httpx2.Response(400, json={"error": {"code": code}})
    error = _oauth_protocol.token_error(
        response, operation="refresh", provider="work", label="ChatGPT", secrets=("r",)
    )
    assert isinstance(error, _oauth_protocol.OAuthError)
    assert error.code == code and error.status_code == 401


@pytest.mark.parametrize("replacement", ["fresh", "stale", "same", "rebound", "other-account"])
def test_shared_holder_rereads_once_after_reused_refresh(
    gcfg: pathlib.Path, monkeypatch: pytest.MonkeyPatch, replacement: str
) -> None:
    entry = ChatGPTProviderEntry(api_format="chatgpt")
    original = _store(entry, account_id="account-a")
    sleeps: list[float] = []
    calls: list[str] = []

    def reused(*_a: object, **_kw: object) -> httpx2.Response:
        calls.append("refresh")
        return httpx2.Response(400, json={"error": {"code": "refresh_token_reused"}})

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        sibling = dataclasses.replace(
            original,
            binding="other" if replacement == "rebound" else original.binding,
            access_token=original.access_token if replacement == "same" else "winner",
            refresh_token="RT2",
            issued_at=6000,
            expires_at=6200 if replacement == "stale" else 9600,
            account_id="account-b" if replacement == "other-account" else "account-a",
        )
        secret_store.save_oauth_grant("work", sibling)

    monkeypatch.setattr(_oauth_protocol, "post_form", reused)
    monkeypatch.setattr(oauth.time, "sleep", sleep)
    holder = oauth.OAuthCredential("work", entry, now=lambda: 6000)
    if replacement == "fresh":
        assert holder.token() == "winner"
        assert holder.account_id() == "account-a"
    else:
        match = {
            "stale": "refresh_token_reused",
            "same": "refresh_token_reused",
            "rebound": "other provider settings",
            "other-account": "different account",
        }[replacement]
        with pytest.raises(types.ProviderError, match=match):
            holder.token()
    assert sleeps == [1.0] and calls == ["refresh"]


def test_refresh_rescue_ignores_reused_token_words_in_prose(
    gcfg: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = ChatGPTProviderEntry(api_format="chatgpt")
    _store(entry)
    sleeps: list[float] = []

    def refused(*_a: object, **_kw: object) -> httpx2.Response:
        return httpx2.Response(
            500, json={"error": "server_error", "description": "refresh_token_reused"}
        )

    monkeypatch.setattr(_oauth_protocol, "post_form", refused)
    monkeypatch.setattr(oauth.time, "sleep", sleeps.append)
    with pytest.raises(types.ProviderError):
        oauth.OAuthCredential("work", entry, now=lambda: 6000).token()
    assert sleeps == []


@pytest.mark.parametrize("shared", [False, True])
def test_concurrent_holders_rotate_once_under_real_locks(
    gcfg: pathlib.Path, monkeypatch: pytest.MonkeyPatch, shared: bool
) -> None:
    """Concurrent calls and independent holders consume a single-use token only once."""
    entry = ChatGPTProviderEntry(api_format="chatgpt")
    _store(entry, account_id="account-a")
    first = oauth.OAuthCredential("work", entry, now=lambda: 6000)
    second = first if shared else oauth.OAuthCredential("work", entry, now=lambda: 6000)
    calls: list[str] = []
    start = threading.Barrier(2)

    def refresh(url: str, data: dict[str, str], *_a: object, **_kw: object) -> _Resp:
        calls.append(data["refresh_token"])
        time.sleep(0.1)
        return _Resp(200, {"access_token": "AT2", "refresh_token": "RT2", "expires_in": 3600})

    def token(holder: oauth.OAuthCredential) -> str:
        start.wait(timeout=5)
        return holder.token()

    monkeypatch.setattr(_oauth_protocol, "post_form", refresh)
    with futures.ThreadPoolExecutor(max_workers=2) as pool:
        pending = [pool.submit(token, holder) for holder in (first, second)]
        assert [result.result(timeout=10) for result in pending] == ["AT2", "AT2"]
    assert calls == ["RT1"]
    stored = secret_store.load_oauth_grant("work")
    assert stored is not None and stored.refresh_token == "RT2" and stored.account_id == "account-a"


def test_shared_holder_refuses_an_unopenable_real_lock(
    gcfg: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = ChatGPTProviderEntry(api_format="chatgpt")
    _store(entry)
    path = secret_store.paths.secrets_path()
    path.with_name(path.name + ".lock").mkdir()

    def never(*_a: object, **_kw: object) -> _Resp:
        pytest.fail("no request may leave without the refresh lock")

    monkeypatch.setattr(_oauth_protocol, "post_form", never)
    with pytest.raises(types.ProviderError, match="credential-refresh lock"):
        oauth.OAuthCredential("work", entry, now=lambda: 6000).token()


def test_shared_holder_adopts_same_account_sibling_after_401_without_rotation(
    gcfg: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = ChatGPTProviderEntry(api_format="chatgpt")
    _store(entry, account_id="account-a")
    holder = oauth.OAuthCredential("work", entry, now=lambda: 2000)
    assert holder.token() == "AT1"
    _store(entry, access_token="AT2", refresh_token="RT2", account_id="account-a")

    def never(*_a: object, **_kw: object) -> _Resp:
        pytest.fail("a fresh sibling avoids rotation")

    monkeypatch.setattr(_oauth_protocol, "post_form", never)
    assert holder.invalidate(403, "AT1") is False
    assert holder.token() == "AT1"
    assert holder.invalidate(401, "AT1") is True
    assert holder.token() == "AT2"


def test_partial_refresh_preserves_the_pinned_account(
    gcfg: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = ChatGPTProviderEntry(api_format="chatgpt")
    _store(entry, account_id="account-a")
    holder = oauth.OAuthCredential("work", entry, now=lambda: 2000)
    assert holder.account_id() == "account-a"
    assert holder.token() == "AT1"

    def refresh(*_a: object, **_kw: object) -> _Resp:
        return _Resp(200, {"access_token": "AT2", "expires_in": 3600})

    monkeypatch.setattr(_oauth_protocol, "post_form", refresh)
    holder.invalidate(401, "AT1")
    assert holder.token() == "AT2"
    stored = secret_store.load_oauth_grant("work")
    assert stored is not None and stored.account_id == "account-a"
    assert stored.refresh_token == "RT1"


@pytest.mark.parametrize("source", ["sibling", "rescue"])
def test_pinned_holder_refuses_an_unidentified_stored_grant(
    gcfg: pathlib.Path, monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    """A different sign-in without identity cannot inherit this run's pinned account."""
    entry = ChatGPTProviderEntry(api_format="chatgpt")
    _store(entry, account_id="account-a")
    holder = oauth.OAuthCredential("work", entry, now=lambda: 2000)
    assert holder.token() == "AT1"
    assert holder.account_id() == "account-a"
    holder.invalidate(401, "AT1")
    sleeps: list[float] = []

    def publish(seconds: float) -> None:
        sleeps.append(seconds)
        _store(entry, access_token="unidentified", refresh_token="RT2", account_id=None)

    def refresh(*_args: object, **_kwargs: object) -> httpx2.Response:
        assert source == "rescue", "sibling adoption must not refresh"
        return httpx2.Response(400, json={"error": "refresh_token_reused"})

    monkeypatch.setattr(_oauth_protocol, "post_form", refresh)
    monkeypatch.setattr(oauth.time, "sleep", publish)
    if source == "sibling":
        publish(0)
    with pytest.raises(types.ProviderError, match="unidentified sign-in") as refused:
        holder.token()
    assert "agent6 connect work" in str(refused.value)
    assert refused.value.status_code == 401
    assert sleeps == ([0] if source == "sibling" else [1])
    stored = secret_store.load_oauth_grant("work")
    assert stored is not None and stored.account_id is None
    assert holder.account_id() == "account-a"
