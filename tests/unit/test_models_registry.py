# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the model-capability registry (agent6.models.registry)."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.models import registry as models_registry


@pytest.fixture
def cache_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("AGENT6_CACHE_HOME", str(tmp_path / "cache"))
    return tmp_path / "cache"


def test_context_window_bundled_and_normalized(cache_home: Path) -> None:
    # Bundled table: exact + a dated/tagged id normalised to the canonical key.
    assert models_registry.context_window("anthropic", "claude-sonnet-4-6") == 200_000
    assert models_registry.context_window("anthropic", "claude-haiku-4-5-20251001") == 200_000
    assert models_registry.context_window("openrouter", "qwen/qwen3-coder:free") == 1_048_576
    assert models_registry.context_window("openrouter", "vendor/totally-unknown") is None


def test_compaction_thresholds_explicit_override_wins(cache_home: Path) -> None:
    assert models_registry.compaction_thresholds(
        "openrouter", "moonshotai/kimi-k2.6", drop_override=111, summarise_override=222
    ) == (111, 222)


def test_compaction_thresholds_adaptive_from_window(cache_home: Path) -> None:
    drop, summarise = models_registry.compaction_thresholds(
        "openrouter", "moonshotai/kimi-k2.6", drop_override=None, summarise_override=None
    )
    assert drop == int(262_144 * 4 * 0.45)
    assert summarise == int(262_144 * 4 * 0.80)
    assert drop < summarise  # tier-2 escalates above tier-1


def test_compaction_thresholds_fixed_fallback_when_unknown(cache_home: Path) -> None:
    # No bundled entry, empty cache -> historical 256k/768k (behaviour preserved).
    assert models_registry.compaction_thresholds(
        "openrouter", "vendor/totally-unknown", drop_override=None, summarise_override=None
    ) == (256_000, 768_000)
