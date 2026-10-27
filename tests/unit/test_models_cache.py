# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the live + cached provider model listing (agent6.models.cache)."""

from __future__ import annotations

import contextlib
import json
import pathlib
import time
from unittest import mock

import httpx2
import pytest

from agent6 import secret_store
from agent6.config import AnthropicProviderEntry, ChatGPTProviderEntry, OpenAIProviderEntry
from agent6.models import cache as models_cache
from agent6.models import registry as models_registry
from agent6.providers import _oauth_protocol, _transport, oauth, types


def test_chatgpt_listing_refuses_legacy_before_sending_a_bearer(
    cache_home: pathlib.Path, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A legacy listing cannot send the stored bearer to an unbound destination."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "g"))
    secret_store.save_secret(
        "work",
        "",
        extra={
            "oauth_access_token": "AT",
            "oauth_refresh_token": "RT",
            "oauth_expires_at": "9999999999",
        },
    )
    with (
        mock.patch.object(_transport, "http_request") as request,
        pytest.raises(types.ProviderError, match="predates credential binding"),
    ):
        models_cache._fetch("work", ChatGPTProviderEntry(api_format="chatgpt"), None, 1)
    request.assert_not_called()


def test_chatgpt_listing_uses_holder_without_refresh_and_falls_back_to_cache(
    cache_home: pathlib.Path, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An expired bound bearer goes through the holder without rotating during a listing."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "g"))
    entry = ChatGPTProviderEntry(api_format="chatgpt")
    secret_store.save_oauth_grant(
        "work", oauth.stored_grant_for(_oauth_protocol.TokenGrant("expired", "RT", 1), entry, now=1)
    )
    path = cache_home / "models" / "work.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"models": ["cached"]}))
    real_token = oauth.OAuthCredential.token
    seen: list[dict[str, object]] = []

    def token(self: oauth.OAuthCredential, **kwargs: object) -> str:
        seen.append(kwargs)
        return real_token(self, **kwargs)  # type: ignore[arg-type]

    def refused(method: str, url: str, **kwargs: object) -> httpx2.Response:
        assert kwargs["headers"]["authorization"] == "Bearer expired"  # type: ignore[index]
        return httpx2.Response(401, request=httpx2.Request(method, url))

    monkeypatch.setattr(oauth.OAuthCredential, "token", token)
    monkeypatch.setattr(_transport, "http_request", refused)
    assert models_cache.list_models("work", entry, None, ttl_s=0) == ["cached"]
    assert seen == [{"refresh": False}]


@pytest.fixture
def cache_home(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> pathlib.Path:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    return tmp_path / "cache" / "agent6"


def _ok_response(ids: list[str]) -> object:
    def _get(
        method: str, url: str, headers: dict[str, str], timeout: float, **_kw: object
    ) -> httpx2.Response:
        return httpx2.Response(
            200, json={"data": [{"id": i} for i in ids]}, request=httpx2.Request("GET", url)
        )

    return _get


def test_fetches_and_caches_openai(
    cache_home: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_transport, "http_request", _ok_response(["gpt-x", "gpt-y"]))
    entry = OpenAIProviderEntry(api_format="openai", base_url="https://api.openai.com/v1")
    out = models_cache.list_models("openai", entry, "sk-test")
    assert out == ["gpt-x", "gpt-y"]
    cached = json.loads((cache_home / "models" / "openai.json").read_text())
    assert cached["models"] == ["gpt-x", "gpt-y"]


@pytest.mark.parametrize(
    "entry",
    [
        OpenAIProviderEntry(api_format="openai", base_url="https://models.example/v1"),
        ChatGPTProviderEntry(api_format="chatgpt", base_url="https://models.example/v1"),
    ],
)
@pytest.mark.parametrize("cap", [16, 1024])
def test_model_listings_use_the_capped_request(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    entry: OpenAIProviderEntry | ChatGPTProviderEntry,
    cap: int,
) -> None:
    """Both listing formats use the default cap and propagate its retryable refusal."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "g"))
    if isinstance(entry, ChatGPTProviderEntry):
        secret_store.save_oauth_grant(
            "p",
            oauth.stored_grant_for(
                _oauth_protocol.TokenGrant("AT", "RT", 3600), entry, now=time.time()
            ),
        )
    url = "https://models.example/v1/models"
    if isinstance(entry, ChatGPTProviderEntry):
        url += "?client_version=1.0.0"
    response = httpx2.Response(
        200,
        json={"data": [{"id": "m1"}], "models": [{"slug": "m1"}]},
        request=httpx2.Request("GET", url),
    )
    monkeypatch.setattr(_transport, "MAX_RESPONSE_BYTES", cap)
    with (
        mock.patch.object(httpx2, "get", return_value=response),
        mock.patch.object(httpx2, "stream", return_value=contextlib.nullcontext(response)),
        mock.patch.object(_transport, "http_request", wraps=_transport.http_request) as request,
    ):
        if cap == 16:
            with pytest.raises(types.ProviderError) as refused:
                models_cache._fetch("p", entry, "AT", 7.0)
            assert str(refused.value) == (
                "provider response exceeded 16 bytes; refusing to buffer it"
            )
            assert refused.value.status_code is None and not refused.value.fatal
        else:
            assert models_cache._fetch("p", entry, "AT", 7.0) == models_cache.Listing(
                ["m1"], {}, {}, {}
            )
    request.assert_called_once()
    assert request.call_args.args == ("GET", url)
    assert request.call_args.kwargs["headers"]["authorization"] == "Bearer AT"
    assert request.call_args.kwargs["timeout"] == 7.0
    assert request.call_args.kwargs.get("cap") is None


def test_fresh_cache_skips_network(
    cache_home: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = cache_home / "models" / "anthropic.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"models": ["claude-cached"]}), encoding="utf-8")

    def _boom(*a: object, **k: object) -> httpx2.Response:
        raise AssertionError("network must not be hit when cache is fresh")

    monkeypatch.setattr(_transport, "http_request", _boom)
    entry = AnthropicProviderEntry(api_format="anthropic")
    assert models_cache.list_models("anthropic", entry, "sk-test") == ["claude-cached"]


def test_stale_cache_used_on_network_error(
    cache_home: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = cache_home / "models" / "openrouter.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"models": ["stale-1"]}), encoding="utf-8")
    # Make the cache look old so a refresh is attempted.
    old = time.time() - 10_000
    import os

    os.utime(path, (old, old))

    def _fail(*a: object, **k: object) -> httpx2.Response:
        raise httpx2.ConnectError("no route")

    monkeypatch.setattr(_transport, "http_request", _fail)
    entry = OpenAIProviderEntry(api_format="openai", base_url="https://openrouter.ai/api/v1")
    assert models_cache.list_models("openrouter", entry, None) == ["stale-1"]


def test_no_cache_network_error_returns_empty(
    cache_home: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _fail(*a: object, **k: object) -> httpx2.Response:
        raise httpx2.ConnectTimeout("slow")

    monkeypatch.setattr(_transport, "http_request", _fail)
    entry = OpenAIProviderEntry(api_format="openai")
    assert models_cache.list_models("openai", entry, "sk") == []


def test_never_raises_on_bad_payload(
    cache_home: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _garbage(*a: object, **k: object) -> httpx2.Response:
        return httpx2.Response(
            200, text="not json", request=httpx2.Request("GET", "http://x/models")
        )

    monkeypatch.setattr(_transport, "http_request", _garbage)
    entry = OpenAIProviderEntry(api_format="openai")
    assert models_cache.list_models("openai", entry, "sk") == []


def test_unsafe_provider_name_has_no_cache_path() -> None:
    # A provider name with path separators / traversal must not form a cache
    # path (no writing the cache outside cache_dir/models).
    cache_path = models_cache._cache_path  # pyright: ignore[reportPrivateUsage]
    assert cache_path("../../etc/cron") is None
    assert cache_path("a/b") is None
    assert cache_path("..") is None
    assert cache_path("openrouter") is not None


def test_unsafe_provider_name_still_fetches(
    cache_home: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An unsafe name skips the cache but still fetches live (never raises).
    monkeypatch.setattr(_transport, "http_request", _ok_response(["m1"]))
    entry = OpenAIProviderEntry(api_format="openai", base_url="https://api.openai.com/v1")
    assert models_cache.list_models("../evil", entry, "sk") == ["m1"]
    assert not (cache_home / "models").exists()  # nothing written outside


# --- context-window reads (file format owned here, policy in registry) -----


def _ok_full(models: list[dict[str, object]]) -> object:
    def _get(
        method: str, url: str, headers: dict[str, str], timeout: float, **_kw: object
    ) -> httpx2.Response:
        return httpx2.Response(200, json={"data": models}, request=httpx2.Request("GET", url))

    return _get


def test_caches_context_length_and_reads_it_back(
    cache_home: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        _transport,
        "http_request",
        _ok_full(
            [
                {"id": "vendor/big", "context_length": 200000},
                {"id": "vendor/small", "context_length": 8192},
                {"id": "vendor/nocontext"},  # missing -> absent (unknown beats wrong)
            ]
        ),
    )
    entry = OpenAIProviderEntry(api_format="openai", base_url="https://x/v1")
    models_cache.list_models("vendorx", entry, "k")
    cached = json.loads((cache_home / "models" / "vendorx.json").read_text())
    assert cached["context"] == {"vendor/big": 200000, "vendor/small": 8192}
    # The narrow reader and the registry both see the write (no network).
    assert models_cache.cached_context_window("vendorx", ("vendor/big",)) == 200000
    assert models_registry.context_window("vendorx", "vendor/big") == 200000
    assert models_registry.context_window("vendorx", "vendor/nocontext") is None


def test_fetch_models_live_bypasses_ttl_and_signals_failure(
    cache_home: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The validation seam: a FRESH cache is ignored (the point is live
    # evidence), success rewrites the cache, and failure returns None -- never
    # a stale fallback -- so the caller can tell evidence from leftovers.
    path = cache_home / "models" / "o.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"models": ["stale-1"]}), encoding="utf-8")
    entry = OpenAIProviderEntry(api_format="openai", base_url="https://x/v1")
    monkeypatch.setattr(_transport, "http_request", _ok_response(["fresh-1"]))
    assert models_cache.fetch_models_live("o", entry, None) == ["fresh-1"]
    assert json.loads(path.read_text())["models"] == ["fresh-1"]

    def _fail(*a: object, **k: object) -> httpx2.Response:
        raise httpx2.ConnectError("down")

    monkeypatch.setattr(_transport, "http_request", _fail)
    assert models_cache.fetch_models_live("o", entry, None) is None


def test_boolean_context_and_pricing_values_are_rejected(tmp_path: pathlib.Path) -> None:
    """Boolean context and pricing values are rejected.

    Bool subclasses int: a provider entry with `context_length: true` caches a 1-token window
    (collapsing the compaction thresholds every turn), and `pricing: true` coerces to $1/MTok. Both
    read as absent.
    """
    payload = {
        "data": [
            {
                "id": "vendor/x",
                "context_length": True,
                "pricing": {"prompt": True, "completion": "1"},
            },
            {
                "id": "vendor/y",
                "context_length": 200_000,
                "pricing": {"prompt": "3", "completion": "15"},
            },
        ]
    }
    ctx = models_cache._parse_context(payload)
    assert "vendor/x" not in ctx
    assert ctx["vendor/y"] == 200_000
    pricing = models_cache._parse_pricing(payload)
    assert "vendor/x" not in pricing
    assert "vendor/y" in pricing


def test_pricing_catalog_refresh_prices_a_bare_claude_id(
    cache_home: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pricing catalog refresh prices a bare claude id.

    A config with only `[providers.anthropic]` must still fetch the OpenRouter catalog, or every
    claude-* run is unpriced on a cold cache and the $ cap never binds (an unmetered opus run inside
    a container).
    """
    from agent6.models import pricing as models_pricing

    def _get(
        method: str, url: str, headers: dict[str, str], timeout: float, **_kw: object
    ) -> httpx2.Response:
        assert "openrouter.ai" in url
        assert "Authorization" not in headers, "catalog fetch must be keyless"
        return httpx2.Response(
            200,
            json={
                "data": [
                    {
                        "id": "anthropic/claude-opus-5",
                        "pricing": {"prompt": "0.000005", "completion": "0.000025"},
                    }
                ]
            },
            request=httpx2.Request("GET", url),
        )

    monkeypatch.setattr(_transport, "http_request", _get)
    assert models_pricing.lookup_price("claude-opus-5") is None
    models_cache.refresh_pricing_catalog()
    price = models_pricing.lookup_price("claude-opus-5")
    assert price is not None
    assert price.input > 0 and price.output > 0


def test_chatgpt_listing_fetches_with_the_sign_in(
    cache_home: pathlib.Path, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """The chatgpt listing fetches with the sign-in.

    It comes from the backend's own /models with the stored bearer, the account header and a ceiling
    client_version; hidden entries stay out of completion; context windows land in the cache.
    """
    from agent6 import secret_store
    from agent6.config import ChatGPTProviderEntry

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "g"))
    entry = ChatGPTProviderEntry(api_format="chatgpt")
    secret_store.save_oauth_grant(
        "chatgpt",
        secret_store.StoredGrant(
            oauth.binding_fingerprint(entry),
            "AT",
            "RT",
            time.time(),
            time.time() + 3600,
            "",
            0,
            None,
            "acct-1",
        ),
    )
    seen: dict[str, object] = {}

    def _get(
        method: str, url: str, headers: dict[str, str], timeout: float, **_kw: object
    ) -> httpx2.Response:
        seen["url"] = url
        seen["headers"] = headers
        body = {
            "models": [
                {"slug": "gpt-5.6-sol", "context_window": 272000, "visibility": "list"},
                {"slug": "codex-auto-review", "context_window": 272000, "visibility": "hide"},
            ]
        }
        return httpx2.Response(200, json=body, request=httpx2.Request("GET", url))

    monkeypatch.setattr(_transport, "http_request", _get)
    entry = ChatGPTProviderEntry(api_format="chatgpt")
    ids = models_cache.list_models("chatgpt", entry, None, ttl_s=0)
    assert ids == ["gpt-5.6-sol"]
    assert str(seen["url"]).endswith("/models?client_version=1.0.0")
    headers = seen["headers"]
    assert isinstance(headers, dict)
    assert headers["authorization"] == "Bearer AT"
    assert headers["chatgpt-account-id"] == "acct-1"
    assert models_cache.cached_context_window("chatgpt", ("gpt-5.6-sol",)) == 272000


def test_chatgpt_listing_without_sign_in_fails_soft(
    cache_home: pathlib.Path, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    from agent6.config import ChatGPTProviderEntry

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))

    def _never(method: str, url: str, headers: dict[str, str], timeout: float) -> httpx2.Response:
        pytest.fail("no sign-in: nothing to fetch with")

    monkeypatch.setattr(_transport, "http_request", _never)
    entry = ChatGPTProviderEntry(api_format="chatgpt")
    assert models_cache.list_models("chatgpt", entry, None, ttl_s=0) == []


def test_input_modalities_are_explicit_bound_and_fresh(
    cache_home: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the exact endpoint, entry and model can consume a fresh listing's modalities."""
    entry = OpenAIProviderEntry(api_format="openai", base_url="https://models.example/v1")
    payload = {
        "data": [
            {"id": "vision", "architecture": {"input_modalities": ["text", "image", "file"]}},
            {"id": "text", "architecture": {"input_modalities": ["text"]}},
            {"id": "unknown-vision"},
            {"id": "bad", "architecture": {"input_modalities": "image"}},
        ]
    }

    def fake_get(*args: object, **kwargs: object) -> httpx2.Response:
        return httpx2.Response(
            200, json=payload, request=httpx2.Request("GET", "https://models.example/v1/models")
        )

    monkeypatch.setattr(_transport, "http_request", fake_get)
    assert models_cache.fetch_models_live("p", entry, None)
    fact = models_cache.cached_input_capabilities("p", entry, "vision")
    assert (fact.image, fact.document) == ("supported", "supported")
    assert fact.route.spec == "p/vision"
    fact = models_cache.cached_input_capabilities("p", entry, "text")
    assert (fact.image, fact.document) == ("unsupported", "unsupported")
    for name in ("unknown-vision", "bad", "VISION", "provider/vision"):
        assert models_cache.cached_input_capabilities("p", entry, name).image == "unknown"
    assert models_cache.cached_input_capabilities("other", entry, "vision").image == "unknown"
    for changed in (
        entry.model_copy(update={"base_url": "https://elsewhere/v1"}),
        entry.model_copy(update={"extra_body": {"route": "other"}}),
    ):
        assert models_cache.cached_input_capabilities("p", changed, "vision").image == "unknown"
    monkeypatch.setattr(time, "time", lambda: (cache_home / "models/p.json").stat().st_mtime + 601)
    assert models_cache.cached_input_capabilities("p", entry, "vision").image == "unknown"


def test_legacy_listing_has_unknown_modalities(cache_home: pathlib.Path) -> None:
    """A stale or unbound capability cache cannot turn an unknown route into unsupported."""
    path = cache_home / "models/p.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"models": ["m"], "modalities": {"m": ["text"]}}))
    entry = OpenAIProviderEntry(api_format="openai", base_url="https://models.example/v1")
    fact = models_cache.cached_input_capabilities("p", entry, "m")
    assert (fact.image, fact.document) == ("unknown", "unknown")


def test_review_seats_resolve_their_own_modality_facts(
    cache_home: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review seats receive their exact route's capability, not the worker's state."""
    from agent6 import budget
    from agent6.app import providers
    from agent6.config import Config

    cfg = Config.model_validate(
        {
            "providers": {"p": {"api_format": "openai", "base_url": "https://models.example/v1"}},
            "models": {"worker": {"provider": "p", "model": "text"}},
            "review": {"reviewers": [{"focus": "ui", "provider": "p", "model": "vision"}]},
        }
    )
    payload = {
        "data": [
            {"id": "vision", "architecture": {"input_modalities": ["image", "file"]}},
            {"id": "text", "architecture": {"input_modalities": ["text"]}},
        ]
    }

    def fake_get(*args: object, **kwargs: object) -> httpx2.Response:
        return httpx2.Response(
            200, json=payload, request=httpx2.Request("GET", "https://models.example/v1/models")
        )

    monkeypatch.setattr(_transport, "http_request", fake_get)
    models_cache.fetch_models_live("p", cfg.providers["p"], None)
    worker = providers.input_capabilities(cfg, cfg.models.resolve("worker"))
    seats = providers.build_review_seats(
        cfg,
        transcript_sink=types.TranscriptSink(cache_home / "transcripts"),
        budget=budget.BudgetTracker(max_usd=1.0, max_tokens_fallback=10000, max_plan_points=0),
        n=1,
    )
    assert worker.image == "unsupported"
    assert seats[0].media_input.image == "supported"
    assert seats[0].media_input.route.spec == "p/vision"
