# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The configured-OAuth credential: stored, bound, refreshed, exchanged, recovered once."""

from __future__ import annotations

import dataclasses
import pathlib
import stat
from typing import Any

import httpx2
import pytest

from agent6 import secret_store
from agent6.config import Config, OpenAIProviderEntry
from agent6.providers import _oauth_protocol, oauth, types

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


def _store(entry: OpenAIProviderEntry, **fields: Any) -> secret_store.StoredGrant:
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
    assert secret_store.load_oauth_tokens("work") is None  # the ChatGPT shape is another
    _store(entry, refresh_token="RT-full")
    assert secret_store.load_oauth_tokens("work") is None  # a refresh token changes nothing
    text = path.read_text(encoding="utf-8")
    assert "oauth_version" in text and "oauth_binding" in text


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

    def fake_post(url: str, data: dict[str, str], timeout_s: float) -> _Resp:
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
    assert cred.invalidate(403) is False
    assert cred.invalidate(401) is True
    with pytest.raises(types.ProviderError, match=r"no refresh token.*agent6 connect work"):
        cred.token()


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

    def fake_get(url: str, **kwargs: Any) -> _Resp:
        seen.append({"url": url, **kwargs})
        return answers.pop(0)

    def fake_post(url: str, data: dict[str, str], timeout_s: float) -> _Resp:
        seen.append({"url": url, "data": data})
        return _Resp(200, {"access_token": "AT2", "refresh_token": "RT2", "expires_in": 7200})

    monkeypatch.setattr(httpx2, "get", fake_get)
    monkeypatch.setattr(_oauth_protocol, "post_form", fake_post)
    cred = oauth.OAuthCredential("work", entry, now=lambda: clock["now"])
    assert cred.token() == "EX1"
    assert seen[0]["headers"]["authorization"] == "Bearer AT1"
    assert seen[0]["follow_redirects"] is False
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

    def fake_post(url: str, **_kw: Any) -> _Resp:
        return _Resp(
            500, {"error": "boom AT-SECRET-VALUE-1"}, text='{"error": "boom AT-SECRET-VALUE-1"}'
        )

    monkeypatch.setattr(httpx2, "post", fake_post)
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

    def fake_post(url: str, data: dict[str, str], timeout_s: float) -> _Resp:
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
    def fake_post(url: str, data: dict[str, str], timeout_s: float) -> _Resp:
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

    def fake_get(url: str, headers: dict[str, str], timeout: float, **_kw: Any) -> httpx2.Response:
        seen["url"] = url
        seen["headers"] = headers
        return httpx2.Response(
            200, json={"data": [{"id": "m1"}]}, request=httpx2.Request("GET", url)
        )

    monkeypatch.setattr(httpx2, "get", fake_get)
    assert cache.fetch_models_live("work", entry, None, timeout_s=5.0) == ["m1"]
    assert seen["headers"]["authorization"] == "Bearer AT1"


def test_a_device_poll_never_sleeps_past_the_grant_expiry(monkeypatch: pytest.MonkeyPatch) -> None:
    """An issuer asking for a huge interval cannot hold `connect` past the code's expiry."""
    clock = {"t": 0.0}
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock["t"] += seconds

    def fake_post(url: str, data: dict[str, str], timeout_s: float) -> _Resp:
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
    def fake_post(url: str, data: dict[str, str], timeout_s: float) -> _Resp:
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
