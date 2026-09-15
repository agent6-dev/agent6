# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Pre-spawn model validation for `/parallel` specs (agent6.models.validate)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent6.config import Config
from agent6.models import validate
from agent6.types import ModelRoute


@pytest.fixture
def cache_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    return tmp_path / "cache" / "agent6"


@pytest.fixture(autouse=True)
def no_live_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    # A miss against an existing cache re-checks the LIVE listing before any
    # refusal; tests opt in explicitly via _fresh below. An unpatched fetch
    # here would be real network -- fail loudly instead.
    def _fail(cfg: Config, provider: str) -> list[str] | None:
        pytest.fail("unexpected live fetch")

    monkeypatch.setattr(validate, "_fresh_listing", _fail)


def _fresh(monkeypatch: pytest.MonkeyPatch, result: list[str] | None) -> None:
    """Stub the miss-path live re-fetch: *result* is the fresh listing, None a
    failed fetch (offline / provider down)."""

    def _stub(cfg: Config, provider: str) -> list[str] | None:
        return result

    monkeypatch.setattr(validate, "_fresh_listing", _stub)


def _write_cache(cache_home: Path, provider: str, models: list[str]) -> None:
    p = cache_home / "models" / f"{provider}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"models": models}), encoding="utf-8")


def _r(model: str, provider: str = "o") -> ModelRoute:
    return ModelRoute(provider, model)


def _cfg(model: str = "kimi-k2") -> Config:
    return Config.model_validate(
        {
            "providers": {"o": {"api_format": "openai", "base_url": "https://x/v1"}},
            "models": {"worker": {"provider": "o", "model": model}},
        }
    )


def test_known_role_model_ok_without_cache(cache_home: Path) -> None:
    v = validate.validate_spec_models([_r("kimi-k2")], _cfg("kimi-k2"))
    assert v.unknown == ()
    assert not v.refused and not v.warned


def test_known_cached_model_ok(cache_home: Path) -> None:
    _write_cache(cache_home, "o", ["gpt-x", "gpt-y"])
    v = validate.validate_spec_models([_r("gpt-x")], _cfg())
    assert v.unknown == ()
    assert not v.refused


def test_unknown_with_cache_refuses_with_suggestions(
    cache_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_cache(cache_home, "o", ["moonshotai/kimi-k2.6", "z-ai/glm-4.6"])
    _fresh(monkeypatch, ["moonshotai/kimi-k2.6", "z-ai/glm-4.6"])
    v = validate.validate_spec_models([_r("moonshotai/kimi-k2.7")], _cfg())
    assert v.refused and not v.warned
    assert v.can_validate
    assert v.unknown == ("o/moonshotai/kimi-k2.7",)
    assert "o/moonshotai/kimi-k2.6" in v.suggestions["o/moonshotai/kimi-k2.7"]
    msg = validate.refusal_message(v, directive=True)
    assert "unknown model 'o/moonshotai/kimi-k2.7'" in msg
    assert "closest: o/moonshotai/kimi-k2.6" in msg
    assert "backtick" in msg


def test_bare_nickname_typo_suggests_closest_bare_model(
    cache_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The natural typo shape is a short nickname (`glm`, `kimi`), not a full
    # provider-prefixed id. Matching only the full ids scored these below difflib's
    # cutoff (the `z-ai/` prefix dominates the ratio), so the did-you-mean was dead.
    # Now the un-prefixed segment is matched too, and the suggestion maps back to
    # the full, runnable id.
    _write_cache(cache_home, "o", ["moonshotai/kimi-k2.6", "z-ai/glm-4.6", "z-ai/glm-4.7"])
    _fresh(monkeypatch, ["moonshotai/kimi-k2.6", "z-ai/glm-4.6", "z-ai/glm-4.7"])
    v = validate.validate_spec_models([_r("kimi"), _r("glm")], _cfg("moonshotai/kimi-k2.6"))
    assert v.refused
    assert v.unknown == ("o/kimi", "o/glm")
    assert "o/moonshotai/kimi-k2.6" in v.suggestions["o/kimi"]
    assert "o/z-ai/glm-4.6" in v.suggestions["o/glm"]
    msg = validate.refusal_message(v, directive=True)
    assert "closest: o/moonshotai/kimi-k2.6" in msg
    assert "o/z-ai/glm-4.6" in msg


def test_bare_nickname_match_stays_on_the_routes_provider(
    cache_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A bare-nickname hit maps only to ids on the route's own provider: a
    # sibling provider's model would run elsewhere, so it is never suggested.
    _write_cache(cache_home, "w", ["w/glm-4.6"])
    _write_cache(cache_home, "s", ["s/glm-4.7"])
    _fresh(monkeypatch, ["w/glm-4.6"])
    v = validate.validate_spec_models([_r("glm", "w")], _two_provider_cfg())
    assert v.refused
    assert all(m.startswith("w/w/") for m in v.suggestions["w/glm"])


def test_unknown_no_cache_warns_and_proceeds(cache_home: Path) -> None:
    # A role model exists but no on-disk cache: cannot validate, so warn.
    v = validate.validate_spec_models([_r("totally-made-up")], _cfg())
    assert v.warned and not v.refused
    assert not v.can_validate
    assert "o/totally-made-up" in validate.warning_message(v)


def test_none_lanes_skipped(cache_home: Path) -> None:
    _write_cache(cache_home, "o", ["gpt-x"])
    v = validate.validate_spec_models([None, None], _cfg())
    assert v.unknown == ()
    assert not v.refused and not v.warned


def test_unknown_deduped_in_spec_order(cache_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_cache(cache_home, "o", ["gpt-x"])
    _fresh(monkeypatch, ["gpt-x"])
    v = validate.validate_spec_models([_r("bad-b"), _r("bad-a"), _r("bad-b")], _cfg())
    assert v.unknown == ("o/bad-b", "o/bad-a")


def test_refusal_message_non_directive_omits_backtick_hint(
    cache_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_cache(cache_home, "o", ["gpt-x"])
    _fresh(monkeypatch, ["gpt-x"])
    v = validate.validate_spec_models([_r("gpt-z")], _cfg())
    assert "backtick" not in validate.refusal_message(v, directive=False)


def test_known_models_is_the_roles_models_plus_the_providers_cache(cache_home: Path) -> None:
    _write_cache(cache_home, "o", ["gpt-x"])
    known = validate._known_models(_cfg("kimi-k2"), "o")  # pyright: ignore[reportPrivateUsage]
    assert known == {"kimi-k2", "gpt-x"}


# --- per-provider scoping: a route is checked against its own provider.


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


def test_a_route_on_a_sibling_provider_checks_that_providers_listing(
    cache_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_cache(cache_home, "w", ["w/model-a", "w/model-b"])
    _write_cache(cache_home, "s", ["s/only-model"])
    _fresh(monkeypatch, ["s/only-model"])
    v = validate.validate_spec_models([_r("s/only-model", "s")], _two_provider_cfg())
    assert v.unknown == () and not v.refused
    v = validate.validate_spec_models([_r("s/typo", "s")], _two_provider_cfg())
    assert v.refused
    assert v.unknown == ("s/s/typo",)
    assert all(m.startswith("s/s/") for m in v.suggestions["s/s/typo"])


def test_a_route_on_an_uncached_provider_warns_and_proceeds(cache_home: Path) -> None:
    # can_validate keys on the route's own provider's cache: a sibling's cache
    # proves nothing about what this provider serves.
    _write_cache(cache_home, "s", ["s/only-model"])
    v = validate.validate_spec_models([_r("anything/at-all", "w")], _two_provider_cfg())
    assert v.warned and not v.refused
    assert not v.can_validate


def test_a_confirmed_miss_refuses_even_beside_an_unvalidated_route(
    cache_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_cache(cache_home, "s", ["s/only-model"])
    _fresh(monkeypatch, ["s/only-model"])
    v = validate.validate_spec_models([_r("anything", "w"), _r("s/typo", "s")], _two_provider_cfg())
    assert v.refused
    assert v.unknown == ("s/s/typo",)


# --- configured-model validation (a typo'd models.<role>.model, U3) -----------


def test_configured_model_ok_when_in_cache(cache_home: Path) -> None:
    _write_cache(cache_home, "o", ["moonshotai/kimi-k2.6", "z-ai/glm-4.6"])
    v = validate.validate_configured_model(_cfg("moonshotai/kimi-k2.6"), "worker")
    assert v.unknown == () and not v.refused and v.can_validate


def test_configured_model_typo_refuses_with_suggestion(
    cache_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_cache(cache_home, "o", ["moonshotai/kimi-k2.6", "z-ai/glm-4.6"])
    _fresh(monkeypatch, ["moonshotai/kimi-k2.6", "z-ai/glm-4.6"])
    v = validate.validate_configured_model(_cfg("moonshotai/kimi-k2.7"), "worker")
    assert v.refused
    assert v.unknown == ("moonshotai/kimi-k2.7",)
    assert "moonshotai/kimi-k2.6" in v.suggestions["moonshotai/kimi-k2.7"]
    msg = validate.configured_model_refusal(v, "worker")
    assert "models.worker.model 'moonshotai/kimi-k2.7'" in msg
    assert "moonshotai/kimi-k2.6" in msg
    # The listing was re-fetched live before refusing, so the old "refresh the
    # cache" remediation would be misleading.
    assert "checked live" in msg and "agent6 model" not in msg


def test_refusal_names_the_entry_the_user_wrote_on_worker_fallback(
    cache_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `plan` validates the planner role, but with no [models.planner] the model
    # comes from the worker entry; the refusal must point at models.worker.model
    # (the key that exists in the user's config), not a phantom models.planner.
    _write_cache(cache_home, "o", ["moonshotai/kimi-k2.6"])
    _fresh(monkeypatch, ["moonshotai/kimi-k2.6"])
    cfg = _cfg("moonshotai/kimi-k2.7")
    assert cfg.models.source_role("planner") == "worker"
    v = validate.validate_configured_model(cfg, "planner")
    assert v.refused
    msg = validate.configured_model_refusal(v, cfg.models.source_role("planner"))
    assert "models.worker.model" in msg
    assert "models.planner" not in msg


def test_configured_model_no_cache_never_refuses(cache_home: Path) -> None:
    # No cached listing (fresh machine, or a provider that lists nothing like
    # Anthropic): must proceed, never block a configured model.
    v = validate.validate_configured_model(_cfg("anything-goes"), "worker")
    assert not v.refused and not v.can_validate


def test_known_models_excludes_sibling_provider_catalog(cache_home: Path) -> None:
    _write_cache(cache_home, "w", ["w/model-a"])
    _write_cache(cache_home, "s", ["s/only-model"])
    known = validate._known_models(_two_provider_cfg(), "w")  # pyright: ignore[reportPrivateUsage]
    assert known == {"w/base-model", "w/model-a"}


def test_a_route_needs_no_worker_role(cache_home: Path) -> None:
    cfg = Config.model_validate(
        {"providers": {"w": {"api_format": "openai", "base_url": "https://w.example/v1"}}}
    )
    _write_cache(cache_home, "w", ["w/model-a"])
    v = validate.validate_spec_models([_r("w/model-a", "w")], cfg)
    assert v.unknown == () and not v.refused and v.can_validate


# --- fresh-evidence refusals: a hard stop needs a listing fetched NOW ---------


def test_variant_of_listed_model_passes_without_fetch(cache_home: Path) -> None:
    # A dated/tagged variant of a listed id is provider-plausible (the registry
    # normalizes the same way); it must never hard-refuse -- and it matches from
    # the cache alone (the autouse guard proves no fetch happened).
    _write_cache(cache_home, "o", ["qwen/qwen3-coder", "claude-haiku-4-5"])
    v = validate.validate_configured_model(_cfg("qwen/qwen3-coder:free"), "worker")
    assert v.unknown == () and not v.refused
    v = validate.validate_configured_model(_cfg("claude-haiku-4-5-20251001"), "worker")
    assert v.unknown == () and not v.refused
    v = validate.validate_spec_models([_r("qwen/qwen3-coder:free")], _cfg())
    assert v.unknown == () and not v.refused


def test_stale_cache_heals_via_live_listing(
    cache_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # THE false-refusal fix: a just-pulled/just-published model missing from the
    # snapshot must not refuse -- the live listing has it, so the run proceeds.
    _write_cache(cache_home, "o", ["old-model"])
    _fresh(monkeypatch, ["old-model", "just-pulled"])
    v = validate.validate_configured_model(_cfg("just-pulled"), "worker")
    assert v.unknown == () and not v.refused and v.can_validate
    v = validate.validate_spec_models([_r("just-pulled")], _cfg())
    assert v.unknown == () and not v.refused


def test_failed_live_fetch_downgrades_refusal_to_warning(
    cache_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Offline / provider down: the snapshot could not be freshened, so a miss
    # must warn and proceed, never refuse on stale evidence.
    _write_cache(cache_home, "o", ["old-model"])
    _fresh(monkeypatch, None)
    v = validate.validate_configured_model(_cfg("just-pulled"), "worker")
    assert v.warned and not v.refused
    assert "just-pulled" in validate.warning_message(v)
    v = validate.validate_spec_models([_r("just-pulled")], _cfg())
    assert v.warned and not v.refused


def test_refusal_suggestions_come_from_the_fresh_listing(
    cache_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The did-you-mean must reflect the listing the refusal rests on, not the
    # stale snapshot it replaced.
    _write_cache(cache_home, "o", ["stale-only-model"])
    _fresh(monkeypatch, ["fresh-model-a"])
    v = validate.validate_configured_model(_cfg("fresh-model-b"), "worker")
    assert v.refused
    assert v.suggestions["fresh-model-b"] == ("fresh-model-a",)
