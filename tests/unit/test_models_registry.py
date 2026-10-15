# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the model-capability registry (agent6.models.registry)."""

from __future__ import annotations

import pathlib
from collections.abc import Callable

import pytest

from agent6.app import providers
from agent6.config import Config
from agent6.models import registry as models_registry


@pytest.fixture
def cache_home(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> pathlib.Path:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    return tmp_path / "cache"


def _cfg(model: str, decompose: str = "auto") -> Config:
    return Config.model_validate(
        {
            "providers": {"o": {"api_format": "openai", "base_url": "https://x/v1"}},
            "models": {"worker": {"provider": "o", "model": model}},
            "prompt": {"decompose": decompose},
        }
    )


def test_context_window_bundled_and_normalized(cache_home: pathlib.Path) -> None:
    # Bundled table: exact + a dated/tagged id normalised to the canonical key.
    assert models_registry.context_window("anthropic", "claude-sonnet-4-6") == 200_000
    # The 5-family's 1M default: a missing row silently fell back to the fixed
    # 256k thresholds, over-compacting flagship models (caught live on a
    # sonnet-5 dogfood run).
    assert models_registry.context_window("anthropic", "claude-sonnet-5") == 1_000_000
    assert models_registry.context_window("anthropic", "claude-fable-5") == 1_000_000
    assert models_registry.context_window("anthropic", "claude-haiku-4-5-20251001") == 200_000
    assert models_registry.context_window("openrouter", "qwen/qwen3-coder:free") == 1_048_576
    assert models_registry.context_window("openrouter", "vendor/totally-unknown") is None


@pytest.mark.parametrize(
    ("model", "suffix"),
    [
        ("claude-fable-5-1", ""),
        ("claude-opus-5-5", "-20260101"),
        ("claude-sonnet-5-5", ":tag"),
        ("claude-fable-5-1", "-20260101:tag"),
    ],
)
def test_claude_minor_versions_size_compaction_from_family(
    cache_home: pathlib.Path, model: str, suffix: str
) -> None:
    """Claude minor ids inherit the family's window instead of fixed compaction defaults."""
    cfg = Config.model_validate(
        {
            "providers": {"claude": {"api_format": "claude_code"}},
            "models": {"worker": {"provider": "claude", "model": model + suffix}},
        }
    )
    logs: list[str] = []
    assert models_registry.context_window("claude", model + suffix) == 1_000_000
    drop, summarise, _ = providers.resolve_compaction_thresholds(
        cfg, cfg.models.resolve("worker"), log=logs.append
    )
    assert (drop, summarise) == (1_800_000, 3_934_464)
    assert any(f"family fallback from {model.rsplit('-', 1)[0]}" in line for line in logs)


@pytest.mark.parametrize("suffix", ["", "-20260101:tag"])
def test_specific_claude_window_wins_over_family(
    cache_home: pathlib.Path, monkeypatch: pytest.MonkeyPatch, suffix: str
) -> None:
    """Specific minor windows beat family windows even with date and tag suffixes."""
    monkeypatch.setitem(models_registry.BUNDLED_CONTEXT_WINDOWS, "claude-fable-5-1", 200_000)
    assert models_registry.context_window("claude", "claude-fable-5-1" + suffix) == 200_000


@pytest.mark.parametrize(
    "model",
    [
        "claude-fable-6-1",
        "claude-haiku-5-5",
        "claude-unknown-5-1",
        "claude-fable-5-1-preview",
        "other-5-1",
    ],
)
def test_family_fallback_does_not_guess_unknown_windows(
    cache_home: pathlib.Path, model: str
) -> None:
    """Family fallback requires a known table key and a numeric minor suffix."""
    assert models_registry.context_window("claude", model) is None


def test_compaction_thresholds_explicit_override_wins(cache_home: pathlib.Path) -> None:
    assert models_registry.compaction_thresholds(
        "openrouter", "moonshotai/kimi-k2.6", drop_override=111, summarise_override=222
    ) == (111, 222)


def test_compaction_thresholds_adaptive_from_window(cache_home: pathlib.Path) -> None:
    drop, summarise = models_registry.compaction_thresholds(
        "openrouter", "moonshotai/kimi-k2.6", drop_override=None, summarise_override=None
    )
    assert drop == int(262_144 * 4 * 0.45)
    # Tier-2 is a near-edge valve: the window minus a 16k-token reserve
    # (pi's reserveTokens shape), not a mid-window fraction.
    assert summarise == (262_144 - 16_384) * 4
    assert drop < summarise  # tier-2 escalates above tier-1


def _window(tokens: int) -> Callable[[str, str], int]:
    """A stand-in for the live model listing's context window."""

    def window(_provider: str, _model: str) -> int:
        return tokens

    return window


def test_a_small_window_clamps_the_verbatim_tail_and_says_so(
    cache_home: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The config validator refuses an explicit pair whose verbatim tail alone re-triggers tier 2.

    A small window sizes exactly that pair from the 80,000-char default, and under it the restart
    kept NO verbatim turns at all, the freshest tool results, paraphrased away, every restart.
    """
    from agent6.config import Config, RoleModel

    monkeypatch.setattr(models_registry, "context_window", _window(32_768))
    logs: list[str] = []

    drop, summarise, keep = providers.resolve_compaction_thresholds(
        Config(), RoleModel(provider="p", model="m"), log=logs.append
    )

    assert drop < summarise
    assert keep == summarise // 2, "the tail must leave room for the restart to shrink"
    assert any("keep_recent_chars 80,000 ->" in line for line in logs), logs

    # A window the default already fits inside keeps the operator's number.
    monkeypatch.setattr(models_registry, "context_window", _window(262_144))
    assert (
        providers.resolve_compaction_thresholds(Config(), RoleModel(provider="p", model="m"))[2]
        == 80_000
    )


def test_compaction_thresholds_fixed_fallback_when_unknown(cache_home: pathlib.Path) -> None:
    # No bundled entry, empty cache -> historical 256k/768k (behaviour preserved).
    assert models_registry.compaction_thresholds(
        "openrouter", "vendor/totally-unknown", drop_override=None, summarise_override=None
    ) == (256_000, 768_000)


# --- decompose capability ---------------------------------------------------


def test_decompose_default_measured_families_only() -> None:
    dd = models_registry.decompose_default
    # The measured winner matches with org prefix, :tag, and a snapshot suffix.
    assert dd("mistralai/mistral-small-3.2-24b-instruct")
    assert dd("mistralai/mistral-small-3.2-24b-instruct:free")
    assert dd("mistral-small-3.2-24b-instruct-2506")
    # Models benched at the ceiling (or never benched) stay off.
    assert not dd("claude-haiku-4-5")
    assert not dd("qwen/qwen3-coder-30b-a3b-instruct")
    assert not dd("vendor/unknown-model")


def test_resolve_decompose_pins_auto_from_registry() -> None:
    win = _cfg("mistralai/mistral-small-3.2-24b-instruct")
    assert providers.resolve_decompose(win, win.models.resolve("worker")).prompt.decompose == "on"
    ceiling = _cfg("claude-haiku-4-5")
    assert (
        providers.resolve_decompose(ceiling, ceiling.models.resolve("worker")).prompt.decompose
        == "off"
    )
    # Unresolvable model pins off (never leaves "auto" for the engine).
    assert providers.resolve_decompose(_cfg("x"), None).prompt.decompose == "off"


def test_resolve_decompose_explicit_setting_passes_through() -> None:
    forced_on = _cfg("claude-haiku-4-5", decompose="on")
    assert providers.resolve_decompose(forced_on, forced_on.models.resolve("worker")) is forced_on
    forced_off = _cfg("mistralai/mistral-small-3.2-24b-instruct", decompose="off")
    assert (
        providers.resolve_decompose(forced_off, forced_off.models.resolve("worker")) is forced_off
    )


def test_with_decompose_pins_value() -> None:
    assert Config().with_decompose("on").prompt.decompose == "on"
    assert Config().with_decompose("off").prompt.decompose == "off"


def test_resolved_adaptive_values_reports_auto_decompose(cache_home: pathlib.Path) -> None:
    win = models_registry.resolved_adaptive_values(_cfg("mistralai/mistral-small-3.2-24b-instruct"))
    assert win["prompt.decompose"] == "on"
    ceiling = models_registry.resolved_adaptive_values(_cfg("claude-haiku-4-5"))
    assert ceiling["prompt.decompose"] == "off"
    # Explicitly pinned config has nothing to resolve.
    pinned = models_registry.resolved_adaptive_values(_cfg("claude-haiku-4-5", decompose="on"))
    assert "prompt.decompose" not in pinned
