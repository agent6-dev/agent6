# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""One authentication recovery per provider call, whatever other retry slots the call holds."""

from __future__ import annotations

from typing import Any

import httpx2
import pytest

from agent6.providers import _headers, _transport, types


class _AlwaysRefreshing:
    """A credential whose every recovery claims to be worth a retry."""

    def __init__(self) -> None:
        self.minted = 0
        self.invalidated: list[int] = []

    def token(self) -> str:
        self.minted += 1
        return f"t{self.minted}"

    def invalidate(self, status: int = 401) -> bool:
        self.invalidated.append(status)
        return True


def _call(cred: _AlwaysRefreshing, *, adapt_attempts: int) -> _transport.ProviderCall:
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
    assert cred.invalidated == [401]


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
        def invalidate(self, status: int = 401) -> bool:
            self.invalidated.append(status)
            return False

    monkeypatch.setattr(_transport, "http_post", fake_post)
    cred = _Fixed()
    with pytest.raises(types.ProviderError, match="403"):
        _call(cred, adapt_attempts=1).run()
    assert sent == ["Bearer t1"]
    assert cred.invalidated == [403]
