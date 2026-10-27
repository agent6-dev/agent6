# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""One authentication recovery per provider call, whatever other retry slots the call holds."""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Any

import httpx2
import pytest

from agent6 import config, secret_store
from agent6.providers import _headers, _oauth_protocol, _transport, oauth, types


class _AlwaysRefreshing:
    """A credential whose every recovery claims to be worth a retry."""

    def __init__(self) -> None:
        self.minted = 0
        self.invalidated: list[tuple[int, str]] = []

    def token(self) -> str:
        self.minted += 1
        return f"t{self.minted}"

    def invalidate(self, status: int, rejected_bearer: str) -> bool:
        self.invalidated.append((status, rejected_bearer))
        return True


def _call(cred: types.BearerCredential, *, adapt_attempts: int) -> _transport.ProviderCall:
    return _transport.ProviderCall(
        api_label="OpenAI",
        api_format="openai",
        url="https://api.test/v1/chat/completions",
        body={"model": "m"},
        timeout_s=5.0,
        api_key="",
        credential=cred,
        transcript_sink=None,
        budget=None,
        model="m",
        facts=_headers.RequestFacts(target="inference", model="m"),
        build_headers=lambda token, _facts: {"authorization": f"Bearer {token}"},
        adapt_400=lambda _s, _t, _b: False,
        adapt_attempts=adapt_attempts,
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


def test_a_401_recovers_the_credential_once_even_beside_an_adaptation_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An adaptation slot is for a 400 body rewrite; it never buys a second credential rotation."""
    sent: list[str] = []

    class _Unauthorized:
        status_code = 401
        text = '{"error": "bad token"}'
        headers = httpx2.Headers()

        def json(self) -> dict[str, Any]:
            return {"error": "bad token"}

    def fake_post(url: str, *, headers: dict[str, str], **_kw: Any) -> _Unauthorized:
        sent.append(headers["authorization"])
        return _Unauthorized()

    monkeypatch.setattr(_transport, "http_post", fake_post)
    cred = _AlwaysRefreshing()
    with pytest.raises(types.ProviderError, match="401"):
        _call(cred, adapt_attempts=1).run()
    assert sent == ["Bearer t1", "Bearer t2"]
    assert cred.invalidated == [(401, "t1")]


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("final_status", [200, 401])
def test_a_stale_401_retries_once_with_the_current_bearer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    streaming: bool,
    final_status: int,
) -> None:
    """A shared holder's rotation during an attempt supplies the retry without another refresh."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "g"))
    entry = config.OpenAIProviderEntry.model_validate(
        {
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
    )
    secret_store.save_oauth_grant(
        "work",
        oauth.stored_grant_for(_oauth_protocol.TokenGrant("AT1", "RT1", 3600.0), entry, now=1000.0),
    )
    clock = {"now": 2000.0}
    cred = oauth.OAuthCredential("work", entry, now=lambda: clock["now"])
    refreshes: list[str] = []
    sent: list[str] = []

    def fake_refresh(
        url: str, data: dict[str, str], timeout_s: float, headers: dict[str, str] | None = None
    ) -> httpx2.Response:
        refreshes.append(data["refresh_token"])
        version = len(refreshes) + 1
        return httpx2.Response(
            200,
            json={
                "access_token": f"AT{version}",
                "refresh_token": f"RT{version}",
                "expires_in": 3600,
            },
        )

    def fake_post(url: str, *, headers: dict[str, str], **_kw: Any) -> httpx2.Response:
        sent.append(headers["authorization"])
        if len(sent) == 1:
            clock["now"] = 4550.0
            assert cred.token() == "AT2"  # another seat refreshes while this attempt is in flight
            return httpx2.Response(401, json={})
        return httpx2.Response(final_status, json={})

    call = _call(cred, adapt_attempts=1)

    def fake_stream(headers: dict[str, str]) -> types.ProviderResponse:
        response = fake_post(call.url, headers=headers)
        if response.status_code == 401:
            raise types.ProviderError("refused", status_code=401)
        return call.parse({})

    monkeypatch.setattr(_oauth_protocol, "post_form", fake_refresh)
    monkeypatch.setattr(_transport, "http_post", fake_post)
    if streaming:
        call = dataclasses.replace(call, stream=fake_stream)
    if final_status == 401:
        with pytest.raises(types.ProviderError) as failed:
            call.run()
        assert failed.value.status_code == 401
    else:
        assert call.run().stop_reason == "end_turn"
    assert sent == ["Bearer AT1", "Bearer AT2"]
    assert refreshes == ["RT1"]


def test_a_403_that_the_credential_cannot_mend_is_final(monkeypatch: pytest.MonkeyPatch) -> None:
    """A credential answering False to a 403 ends the call on the first refusal."""
    sent: list[str] = []

    class _Forbidden:
        status_code = 403
        text = "no"
        headers = httpx2.Headers()

        def json(self) -> dict[str, Any]:
            return {}

    def fake_post(url: str, *, headers: dict[str, str], **_kw: Any) -> _Forbidden:
        sent.append(headers["authorization"])
        return _Forbidden()

    class _Fixed(_AlwaysRefreshing):
        def invalidate(self, status: int, rejected_bearer: str) -> bool:
            self.invalidated.append((status, rejected_bearer))
            return False

    monkeypatch.setattr(_transport, "http_post", fake_post)
    cred = _Fixed()
    with pytest.raises(types.ProviderError, match="403"):
        _call(cred, adapt_attempts=1).run()
    assert sent == ["Bearer t1"]
    assert cred.invalidated == [(403, "t1")]
