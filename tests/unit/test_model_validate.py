# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Pre-spawn model validation for `/parallel` specs (agent6.models.validate)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent6.config import Config
from agent6.models import validate


@pytest.fixture
def cache_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("AGENT6_CACHE_HOME", str(tmp_path / "cache"))
    return tmp_path / "cache"


def _write_cache(cache_home: Path, provider: str, models: list[str]) -> None:
    p = cache_home / "models" / f"{provider}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"models": models}), encoding="utf-8")


def _cfg(model: str = "kimi-k2") -> Config:
    return Config.model_validate(
        {
            "providers": {"o": {"api_format": "openai", "base_url": "https://x/v1"}},
            "models": {"worker": {"provider": "o", "model": model}},
        }
    )


def test_known_role_model_ok_without_cache(cache_home: Path) -> None:
    v = validate.validate_spec_models(["kimi-k2"], _cfg("kimi-k2"))
    assert v.unknown == ()
    assert not v.refused and not v.warned


def test_known_cached_model_ok(cache_home: Path) -> None:
    _write_cache(cache_home, "o", ["gpt-x", "gpt-y"])
    v = validate.validate_spec_models(["gpt-x"], _cfg())
    assert v.unknown == ()
    assert not v.refused


def test_unknown_with_cache_refuses_with_suggestions(cache_home: Path) -> None:
    _write_cache(cache_home, "o", ["moonshotai/kimi-k2.6", "z-ai/glm-4.6"])
    v = validate.validate_spec_models(["moonshotai/kimi-k2.7"], _cfg())
    assert v.refused and not v.warned
    assert v.can_validate
    assert v.unknown == ("moonshotai/kimi-k2.7",)
    assert "moonshotai/kimi-k2.6" in v.suggestions["moonshotai/kimi-k2.7"]
    msg = validate.refusal_message(v, directive=True)
    assert "unknown model 'moonshotai/kimi-k2.7'" in msg
    assert "closest: moonshotai/kimi-k2.6" in msg
    assert "backtick" in msg


def test_bare_nickname_typo_suggests_closest_bare_model(cache_home: Path) -> None:
    # The natural typo shape is a short nickname (`glm`, `kimi`), not a full
    # provider-prefixed id. Matching only the full ids scored these below difflib's
    # cutoff (the `z-ai/` prefix dominates the ratio), so the did-you-mean was dead.
    # Now the un-prefixed segment is matched too, and the suggestion maps back to
    # the full, runnable id.
    _write_cache(cache_home, "o", ["moonshotai/kimi-k2.6", "z-ai/glm-4.6", "z-ai/glm-4.7"])
    v = validate.validate_spec_models(["kimi", "glm"], _cfg("moonshotai/kimi-k2.6"))
    assert v.refused
    assert v.unknown == ("kimi", "glm")
    assert "moonshotai/kimi-k2.6" in v.suggestions["kimi"]
    assert "z-ai/glm-4.6" in v.suggestions["glm"]
    msg = validate.refusal_message(v, directive=True)
    assert "closest: moonshotai/kimi-k2.6" in msg
    assert "z-ai/glm-4.6" in msg


def test_bare_nickname_match_stays_worker_scoped(cache_home: Path) -> None:
    # A bare-nickname hit must still map only to WORKER-provider ids: a sibling
    # provider's model is unrunnable in a lane, so it can never be suggested.
    _write_cache(cache_home, "w", ["w/glm-4.6"])
    _write_cache(cache_home, "s", ["s/glm-4.7"])
    v = validate.validate_spec_models(["glm"], _two_provider_cfg())
    assert v.refused
    assert all(m.startswith("w/") for m in v.suggestions["glm"])


def test_unknown_no_cache_warns_and_proceeds(cache_home: Path) -> None:
    # A role model exists but no on-disk cache: cannot validate, so warn.
    v = validate.validate_spec_models(["totally-made-up"], _cfg())
    assert v.warned and not v.refused
    assert not v.can_validate
    assert "totally-made-up" in validate.warning_message(v)


def test_none_lanes_skipped(cache_home: Path) -> None:
    _write_cache(cache_home, "o", ["gpt-x"])
    v = validate.validate_spec_models([None, None], _cfg())
    assert v.unknown == ()
    assert not v.refused and not v.warned


def test_unknown_deduped_in_spec_order(cache_home: Path) -> None:
    _write_cache(cache_home, "o", ["gpt-x"])
    v = validate.validate_spec_models(["bad-b", "bad-a", "bad-b"], _cfg())
    assert v.unknown == ("bad-b", "bad-a")


def test_refusal_message_non_directive_omits_backtick_hint(cache_home: Path) -> None:
    _write_cache(cache_home, "o", ["gpt-x"])
    v = validate.validate_spec_models(["gpt-z"], _cfg())
    assert "backtick" not in validate.refusal_message(v, directive=False)


def test_known_models_is_worker_model_plus_worker_cache(cache_home: Path) -> None:
    _write_cache(cache_home, "o", ["gpt-x"])
    known = validate.known_models(_cfg("kimi-k2"))
    assert known == {"kimi-k2", "gpt-x"}


# --- worker-provider scoping: lanes inherit the WORKER provider (only the model
# --- is overridden per lane), so a sibling provider's catalog is unrunnable.


def _two_provider_cfg() -> Config:
    return Config.model_validate(
        {
            "providers": {
                "w": {"api_format": "openai", "base_url": "https://w.example/v1"},
                "s": {"api_format": "openai", "base_url": "https://s.example/v1"},
            },
            "models": {"worker": {"provider": "w", "model": "w/base-model"}},
        }
    )


def test_sibling_provider_model_refused_with_worker_suggestions(cache_home: Path) -> None:
    # FALSE-ACCEPT guard: a model served only by a NON-worker provider cannot run
    # (the lane inherits the worker provider), so it must refuse, with the
    # did-you-mean drawn from the WORKER universe only.
    _write_cache(cache_home, "w", ["w/model-a", "w/model-b"])
    _write_cache(cache_home, "s", ["s/only-model"])
    v = validate.validate_spec_models(["s/only-model"], _two_provider_cfg())
    assert v.refused
    assert v.unknown == ("s/only-model",)
    assert all(m.startswith("w/") for m in v.suggestions["s/only-model"])


def test_worker_uncached_sibling_cached_warns_and_proceeds(cache_home: Path) -> None:
    # can_validate keys on the WORKER provider's cache alone: a sibling's cache
    # proves nothing about what the worker provider serves.
    _write_cache(cache_home, "s", ["s/only-model"])
    v = validate.validate_spec_models(["anything/at-all"], _two_provider_cfg())
    assert v.warned and not v.refused
    assert not v.can_validate


def test_known_models_excludes_sibling_provider_catalog(cache_home: Path) -> None:
    _write_cache(cache_home, "w", ["w/model-a"])
    _write_cache(cache_home, "s", ["s/only-model"])
    assert validate.known_models(_two_provider_cfg()) == {"w/base-model", "w/model-a"}


def test_no_worker_role_cannot_validate(cache_home: Path) -> None:
    cfg = Config.model_validate(
        {"providers": {"w": {"api_format": "openai", "base_url": "https://w.example/v1"}}}
    )
    _write_cache(cache_home, "w", ["w/model-a"])
    v = validate.validate_spec_models(["w/model-a"], cfg)
    # No worker role -> no lane universe to check against: warn, never refuse.
    assert not v.refused
    assert validate.known_models(cfg) == set()
