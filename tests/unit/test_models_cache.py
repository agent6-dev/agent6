# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the live + cached provider model listing (agent6.models_cache)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import pytest

from agent6 import models_cache
from agent6.config import AnthropicProviderEntry, OpenAIProviderEntry


@pytest.fixture
def cache_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("AGENT6_CACHE_HOME", str(tmp_path / "cache"))
    return tmp_path / "cache"


def _ok_response(ids: list[str]) -> object:
    def _get(url: str, headers: dict[str, str], timeout: float) -> httpx.Response:
        return httpx.Response(
            200, json={"data": [{"id": i} for i in ids]}, request=httpx.Request("GET", url)
        )

    return _get


def test_fetches_and_caches_openai(cache_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "get", _ok_response(["gpt-x", "gpt-y"]))
    entry = OpenAIProviderEntry(kind="openai", base_url="https://api.openai.com/v1")
    out = models_cache.list_models("openai", entry, "sk-test")
    assert out == ["gpt-x", "gpt-y"]
    cached = json.loads((cache_home / "models" / "openai.json").read_text())
    assert cached["models"] == ["gpt-x", "gpt-y"]


def test_fresh_cache_skips_network(cache_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = cache_home / "models" / "anthropic.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"models": ["claude-cached"]}), encoding="utf-8")

    def _boom(*a: object, **k: object) -> httpx.Response:
        raise AssertionError("network must not be hit when cache is fresh")

    monkeypatch.setattr(httpx, "get", _boom)
    entry = AnthropicProviderEntry(kind="anthropic")
    assert models_cache.list_models("anthropic", entry, "sk-test") == ["claude-cached"]


def test_stale_cache_used_on_network_error(
    cache_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = cache_home / "models" / "openrouter.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"models": ["stale-1"]}), encoding="utf-8")
    # Make the cache look old so a refresh is attempted.
    old = time.time() - 10_000
    import os

    os.utime(path, (old, old))

    def _fail(*a: object, **k: object) -> httpx.Response:
        raise httpx.ConnectError("no route")

    monkeypatch.setattr(httpx, "get", _fail)
    entry = OpenAIProviderEntry(kind="openai", base_url="https://openrouter.ai/api/v1")
    assert models_cache.list_models("openrouter", entry, None) == ["stale-1"]


def test_no_cache_network_error_returns_empty(
    cache_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _fail(*a: object, **k: object) -> httpx.Response:
        raise httpx.ConnectTimeout("slow")

    monkeypatch.setattr(httpx, "get", _fail)
    entry = OpenAIProviderEntry(kind="openai")
    assert models_cache.list_models("openai", entry, "sk") == []


def test_never_raises_on_bad_payload(cache_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _garbage(*a: object, **k: object) -> httpx.Response:
        return httpx.Response(200, text="not json", request=httpx.Request("GET", "http://x/models"))

    monkeypatch.setattr(httpx, "get", _garbage)
    entry = OpenAIProviderEntry(kind="openai")
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
    cache_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An unsafe name skips the cache but still fetches live (never raises).
    monkeypatch.setattr(httpx, "get", _ok_response(["m1"]))
    entry = OpenAIProviderEntry(kind="openai", base_url="https://api.openai.com/v1")
    assert models_cache.list_models("../evil", entry, "sk") == ["m1"]
    assert not (cache_home / "models").exists()  # nothing written outside
