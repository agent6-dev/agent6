# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for cache-fetched model pricing (agent6.pricing + models_cache)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent6.models_cache import _parse_pricing  # pyright: ignore[reportPrivateUsage]
from agent6.pricing import lookup_price


def _write_pricing(cache: Path, name: str, pricing: dict[str, list[float]]) -> None:
    (cache / "models").mkdir(parents=True, exist_ok=True)
    (cache / "models" / f"{name}.json").write_text(
        json.dumps({"models": list(pricing), "pricing": pricing}), encoding="utf-8"
    )


def test_lookup_price_reads_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AGENT6_CACHE_HOME", str(tmp_path))
    _write_pricing(tmp_path, "openrouter", {"a/model": [0.5, 2.5]})
    assert lookup_price("a/model") == (0.5, 2.5)
    assert lookup_price("nobody/else") is None


def test_lookup_price_sees_cache_written_after_first_miss(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The CLI preflight refreshes the cache AFTER config construction already
    # did a lookup; the memo must not pin the early empty result.
    monkeypatch.setenv("AGENT6_CACHE_HOME", str(tmp_path))
    assert lookup_price("a/model") is None
    _write_pricing(tmp_path, "openrouter", {"a/model": [0.5, 2.5]})
    assert lookup_price("a/model") == (0.5, 2.5)


def test_lookup_price_ignores_malformed_entries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("AGENT6_CACHE_HOME", str(tmp_path))
    (tmp_path / "models").mkdir(parents=True)
    (tmp_path / "models" / "bad.json").write_text(
        json.dumps(
            {
                "pricing": {
                    "neg/model": [-1.0, 2.0],
                    "str/model": ["x", "y"],
                    "short/model": [1.0],
                    "ok/model": [1.0, 2.0],
                }
            }
        ),
        encoding="utf-8",
    )
    assert lookup_price("ok/model") == (1.0, 2.0)
    for bad in ("neg/model", "str/model", "short/model"):
        assert lookup_price(bad) is None


def test_parse_pricing_openrouter_shape() -> None:
    # OpenRouter reports USD per TOKEN as strings; normalized to per-MTok.
    payload = {
        "data": [
            {
                "id": "moonshotai/kimi-k2.6",
                "pricing": {"prompt": "0.00000068", "completion": "0.00000341"},
            },
            {"id": "no-pricing-model"},
            {"id": "bad-pricing", "pricing": {"prompt": "free", "completion": "0"}},
        ]
    }
    got = _parse_pricing(payload)
    assert got["moonshotai/kimi-k2.6"] == (pytest.approx(0.68), pytest.approx(3.41))
    assert "no-pricing-model" not in got
    assert "bad-pricing" not in got
