# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the live + cached provider model listing (agent6.models_cache)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx2
import pytest

from agent6 import models_cache
from agent6.config import AnthropicProviderEntry, OpenAIProviderEntry


@pytest.fixture
def cache_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("AGENT6_CACHE_HOME", str(tmp_path / "cache"))
    return tmp_path / "cache"


def _ok_response(ids: list[str]) -> object:
    def _get(url: str, headers: dict[str, str], timeout: float) -> httpx2.Response:
        return httpx2.Response(
            200, json={"data": [{"id": i} for i in ids]}, request=httpx2.Request("GET", url)
        )

    return _get


def test_fetches_and_caches_openai(cache_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx2, "get", _ok_response(["gpt-x", "gpt-y"]))
    entry = OpenAIProviderEntry(api_format="openai", base_url="https://api.openai.com/v1")
    out = models_cache.list_models("openai", entry, "sk-test")
    assert out == ["gpt-x", "gpt-y"]
    cached = json.loads((cache_home / "models" / "openai.json").read_text())
    assert cached["models"] == ["gpt-x", "gpt-y"]


def test_fresh_cache_skips_network(cache_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = cache_home / "models" / "anthropic.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"models": ["claude-cached"]}), encoding="utf-8")

    def _boom(*a: object, **k: object) -> httpx2.Response:
        raise AssertionError("network must not be hit when cache is fresh")

    monkeypatch.setattr(httpx2, "get", _boom)
    entry = AnthropicProviderEntry(api_format="anthropic")
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

    def _fail(*a: object, **k: object) -> httpx2.Response:
        raise httpx2.ConnectError("no route")

    monkeypatch.setattr(httpx2, "get", _fail)
    entry = OpenAIProviderEntry(api_format="openai", base_url="https://openrouter.ai/api/v1")
    assert models_cache.list_models("openrouter", entry, None) == ["stale-1"]


def test_no_cache_network_error_returns_empty(
    cache_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _fail(*a: object, **k: object) -> httpx2.Response:
        raise httpx2.ConnectTimeout("slow")

    monkeypatch.setattr(httpx2, "get", _fail)
    entry = OpenAIProviderEntry(api_format="openai")
    assert models_cache.list_models("openai", entry, "sk") == []


def test_never_raises_on_bad_payload(cache_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _garbage(*a: object, **k: object) -> httpx2.Response:
        return httpx2.Response(
            200, text="not json", request=httpx2.Request("GET", "http://x/models")
        )

    monkeypatch.setattr(httpx2, "get", _garbage)
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
    cache_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An unsafe name skips the cache but still fetches live (never raises).
    monkeypatch.setattr(httpx2, "get", _ok_response(["m1"]))
    entry = OpenAIProviderEntry(api_format="openai", base_url="https://api.openai.com/v1")
    assert models_cache.list_models("../evil", entry, "sk") == ["m1"]
    assert not (cache_home / "models").exists()  # nothing written outside


# --- context window + adaptive compaction sizing --------------------------


def _ok_full(models: list[dict[str, object]]) -> object:
    def _get(url: str, headers: dict[str, str], timeout: float) -> httpx2.Response:
        return httpx2.Response(200, json={"data": models}, request=httpx2.Request("GET", url))

    return _get


def test_caches_context_length_and_reads_it_back(
    cache_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        httpx2,
        "get",
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
    # context_window reads the cache (no network), for a model not in the table
    assert models_cache.context_window("vendorx", "vendor/big") == 200000
    assert models_cache.context_window("vendorx", "vendor/nocontext") is None


def test_context_window_bundled_and_normalized(cache_home: Path) -> None:
    # Bundled table: exact + a dated/tagged id normalised to the canonical key.
    assert models_cache.context_window("anthropic", "claude-sonnet-4-6") == 200_000
    assert models_cache.context_window("anthropic", "claude-haiku-4-5-20251001") == 200_000
    assert models_cache.context_window("openrouter", "qwen/qwen3-coder:free") == 1_048_576
    assert models_cache.context_window("openrouter", "vendor/totally-unknown") is None


def test_compaction_thresholds_explicit_override_wins(cache_home: Path) -> None:
    assert models_cache.compaction_thresholds(
        "openrouter", "moonshotai/kimi-k2.6", drop_override=111, summarise_override=222
    ) == (111, 222)


def test_compaction_thresholds_adaptive_from_window(cache_home: Path) -> None:
    drop, summarise = models_cache.compaction_thresholds(
        "openrouter", "moonshotai/kimi-k2.6", drop_override=None, summarise_override=None
    )
    assert drop == int(262_144 * 4 * 0.45)
    assert summarise == int(262_144 * 4 * 0.80)
    assert drop < summarise  # tier-2 escalates above tier-1


def test_compaction_thresholds_fixed_fallback_when_unknown(cache_home: Path) -> None:
    # No bundled entry, empty cache -> historical 256k/768k (behaviour preserved).
    assert models_cache.compaction_thresholds(
        "openrouter", "vendor/totally-unknown", drop_override=None, summarise_override=None
    ) == (256_000, 768_000)
