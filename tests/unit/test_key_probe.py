# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`probe_provider_key`, the `agent6 connect` key-validation probe."""

from __future__ import annotations

import contextlib
from typing import Any
from unittest import mock

import httpx2
import pytest

from agent6.config import AnthropicProviderEntry, OpenAIProviderEntry
from agent6.models import cache
from agent6.providers import _transport


class _FakeResp:
    def __init__(self, status_code: int, payload: Any = None) -> None:
        self.status_code = status_code
        self._payload = payload if payload is not None else {"data": []}

    def json(self) -> Any:
        return self._payload


def _patch_get(monkeypatch: pytest.MonkeyPatch, resp_or_exc: Any) -> list[str]:
    """Stub the request seam and record each URL."""
    urls: list[str] = []

    def _get(method: str, url: str, **_kw: Any) -> _FakeResp:
        urls.append(url)
        if isinstance(resp_or_exc, Exception):
            raise resp_or_exc
        return resp_or_exc

    monkeypatch.setattr(_transport, "http_request", _get)
    return urls


def test_probe_ok_with_models(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_get(monkeypatch, _FakeResp(200, {"data": [{"id": "a"}, {"id": "b"}]}))
    r = cache.probe_provider_key(AnthropicProviderEntry(api_format="anthropic"), "sk-ant-real")
    assert r.ok and r.status == "ok"
    assert "2 models" in r.detail


@pytest.mark.parametrize("cap", [16, 1024])
def test_key_probe_uses_the_capped_request(monkeypatch: pytest.MonkeyPatch, cap: int) -> None:
    """A probe uses the default cap and reports an oversized reply as unreachable."""
    entry = OpenAIProviderEntry(api_format="openai", base_url="https://models.example/v1")
    url = "https://models.example/v1/models"
    response = httpx2.Response(
        200, json={"data": [{"id": "m1"}]}, request=httpx2.Request("GET", url)
    )
    monkeypatch.setattr(_transport, "MAX_RESPONSE_BYTES", cap)
    with (
        mock.patch.object(httpx2, "get", return_value=response),
        mock.patch.object(httpx2, "stream", return_value=contextlib.nullcontext(response)),
        mock.patch.object(_transport, "http_request", wraps=_transport.http_request) as request,
    ):
        result = cache.probe_provider_key(entry, "AT", timeout_s=7.0)
    if cap == 16:
        assert result == cache.KeyProbeResult(
            ok=False,
            status="unreachable",
            detail="provider response exceeded 16 bytes; refusing to buffer it",
        )
    else:
        assert result == cache.KeyProbeResult(
            ok=True, status="ok", detail="provider returned 1 models"
        )
    request.assert_called_once_with("GET", url, headers={"authorization": "Bearer AT"}, timeout=7.0)


def test_probe_ok_empty_list(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_get(monkeypatch, _FakeResp(200, {"data": []}))
    r = cache.probe_provider_key(OpenAIProviderEntry(api_format="openai"), "sk-real")
    assert r.ok and r.status == "ok"
    assert "accepted the key" in r.detail


@pytest.mark.parametrize("code", [401, 403])
def test_probe_auth_failed(monkeypatch: pytest.MonkeyPatch, code: int) -> None:
    _patch_get(monkeypatch, _FakeResp(code))
    r = cache.probe_provider_key(AnthropicProviderEntry(api_format="anthropic"), "sk-ant-bad")
    assert not r.ok and r.status == "auth_failed"
    assert str(code) in r.detail


def test_probe_other_4xx_5xx_is_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_get(monkeypatch, _FakeResp(500))
    r = cache.probe_provider_key(OpenAIProviderEntry(api_format="openai"), "sk-real")
    assert not r.ok and r.status == "unreachable"


def test_probe_network_error_is_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_get(monkeypatch, httpx2.ConnectError("refused"))
    r = cache.probe_provider_key(OpenAIProviderEntry(api_format="openai"), "sk-real")
    assert not r.ok and r.status == "unreachable"


def test_probe_openrouter_uses_auth_gated_key_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    # OpenRouter's /models is public, so the probe must hit /key (auth-gated).
    urls = _patch_get(monkeypatch, _FakeResp(200, {"data": {"label": "k"}}))
    entry = OpenAIProviderEntry(api_format="openai", base_url="https://openrouter.ai/api/v1")
    r = cache.probe_provider_key(entry, "sk-or-real")
    assert r.ok
    assert urls == ["https://openrouter.ai/api/v1/key"]


def test_probe_non_openrouter_uses_models(monkeypatch: pytest.MonkeyPatch) -> None:
    urls = _patch_get(monkeypatch, _FakeResp(200, {"data": []}))
    entry = OpenAIProviderEntry(api_format="openai", base_url="https://api.openai.com/v1")
    cache.probe_provider_key(entry, "sk-real")
    assert urls == ["https://api.openai.com/v1/models"]


def test_probe_openrouter_match_is_host_not_substring(monkeypatch: pytest.MonkeyPatch) -> None:
    # A URL merely containing "openrouter.ai" in its path uses /models, not OpenRouter's /key.
    urls = _patch_get(monkeypatch, _FakeResp(200, {"data": []}))
    proxy = OpenAIProviderEntry(
        api_format="openai", base_url="https://myproxy.com/openrouter.ai/v1"
    )
    cache.probe_provider_key(proxy, "sk")
    assert urls == ["https://myproxy.com/openrouter.ai/v1/models"]
    urls.clear()
    sub = OpenAIProviderEntry(api_format="openai", base_url="https://gateway.openrouter.ai/api/v1")
    cache.probe_provider_key(sub, "sk")
    assert urls == ["https://gateway.openrouter.ai/api/v1/key"]


def test_probe_unsupported_deployment_skips_network(monkeypatch: pytest.MonkeyPatch) -> None:
    # Vertex/Azure have no uniform /models; the probe must NOT make a request.
    urls = _patch_get(monkeypatch, _FakeResp(200))
    entry = AnthropicProviderEntry(
        api_format="anthropic",
        deployment="vertex",
        base_url="https://example.com",
        token_command=("true",),
    )
    r = cache.probe_provider_key(entry, "tok")
    assert r.status == "unsupported"
    assert urls == []  # no network call
