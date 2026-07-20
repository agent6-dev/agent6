# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Unit tests for USD budget enforcement guards.

USD is a single runtime bound (`BudgetTracker.max_usd`), not a load-time
token conversion: the config-side guards here refuse an unenforceable
`--max-usd` flag and warn on an unenforceable config limit.

Pricing has no static table: it comes from the provider-fetched models cache
(agent6.models.pricing reads $AGENT6_CACHE_HOME/models/*.json). Tests inject prices
by writing a real cache file, exercising the same path production uses.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

PRICED_MODEL = "test/priced-model"
CHEAP_MODEL = "test/cheap-model"


@pytest.fixture
def price_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    cache = tmp_path / "cache"
    (cache / "models").mkdir(parents=True)
    (cache / "models" / "testprovider.json").write_text(
        json.dumps(
            {
                "models": [PRICED_MODEL, CHEAP_MODEL],
                "pricing": {PRICED_MODEL: [3.0, 15.0], CHEAP_MODEL: [0.27, 1.10]},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("AGENT6_CACHE_HOME", str(cache))
    return cache


def test_max_usd_override_does_not_ratchet_token_caps(price_cache: Path) -> None:
    """A later --max-usd (via with_budget_overrides) must not be bounded by an
    earlier USD limit. The token ceilings are the operator's DECLARED values,
    never rewritten from best_effort_usd_limit; USD is enforced by the runtime
    BudgetTracker.max_usd. Previously a load-time USD->token conversion
    overwrote the declared caps, and with_budget_overrides re-fed the tightened
    value as if operator-set, so raising the USD limit could never win."""
    from agent6.config import Config

    cfg = Config.model_validate(
        {
            "providers": {"p": {"api_format": "openai", "base_url": "http://localhost:1"}},
            "models": {"worker": {"provider": "p", "model": PRICED_MODEL}},
            "budget": {
                "best_effort_usd_limit": 5.0,
                "max_input_tokens": 999_999_999,
                "max_output_tokens": 999_999_999,
            },
        }
    )
    # Declared ceilings are preserved verbatim (priced worker, USD limit set).
    assert cfg.budget.max_input_tokens == 999_999_999
    assert cfg.budget.max_output_tokens == 999_999_999
    # Raising the USD limit takes full effect; nothing ratchets it down.
    out = cfg.with_budget_overrides(max_usd=50.0)
    assert out.budget.best_effort_usd_limit == 50.0
    assert out.budget.max_input_tokens == 999_999_999


def test_explicit_usd_flag_refused_when_unpriced(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # An explicit --max-usd is a promise for the run; with no price data for
    # the worker model it cannot be kept, so the CLI refuses to start.
    from agent6.app._setup import (
        explicit_usd_flag_error as _explicit_usd_flag_error,
    )
    from agent6.config import Config

    monkeypatch.setenv("AGENT6_CACHE_HOME", str(tmp_path / "empty-cache"))
    cfg = Config.model_validate(
        {
            "providers": {"p": {"api_format": "openai", "base_url": "http://localhost:1"}},
            "models": {"worker": {"provider": "p", "model": "nobody/unpriced"}},
        }
    )
    err = _explicit_usd_flag_error(2.5, cfg)
    assert err is not None and "no price data" in err
    # no flag, or a priced model, passes
    assert _explicit_usd_flag_error(None, cfg) is None
    assert _explicit_usd_flag_error(0, cfg) is None


def test_explicit_usd_flag_ok_when_priced(price_cache: Path) -> None:
    from agent6.app._setup import (
        explicit_usd_flag_error as _explicit_usd_flag_error,
    )
    from agent6.config import Config

    cfg = Config.model_validate(
        {
            "providers": {"p": {"api_format": "openai", "base_url": "http://localhost:1"}},
            "models": {"worker": {"provider": "p", "model": PRICED_MODEL}},
        }
    )
    assert _explicit_usd_flag_error(2.5, cfg) is None


def testwarn_if_usd_unenforceable(price_cache: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A TOML best_effort_usd_limit on an unpriced worker can't enforce (Anthropic
    publishes no pricing), so run startup must warn instead of silently no-op'ing.
    The --max-usd *flag* is already guarded by _explicit_usd_flag_error; this is
    the config-path complement."""
    from agent6.app.preflight import warn_if_usd_unenforceable
    from agent6.config import Config

    def _cfg(usd: float, worker: str, reviewer: str | None = None) -> Config:
        models: dict[str, Any] = {"worker": {"provider": "p", "model": worker}}
        if reviewer is not None:
            models["reviewer"] = {"provider": "p", "model": reviewer}
        return Config.model_validate(
            {
                "providers": {"p": {"api_format": "openai", "base_url": "http://localhost:1"}},
                "models": models,
                "budget": {"best_effort_usd_limit": usd},
            }
        )

    # unpriced worker + usd>0 -> warns, names the model, points at token ceilings
    warn_if_usd_unenforceable(_cfg(1.0, "nobody/unpriced"))
    warned = capsys.readouterr().err
    assert "cannot be enforced" in warned and "nobody/unpriced" in warned

    # priced worker -> the USD ceiling works, so stay silent
    warn_if_usd_unenforceable(_cfg(1.0, PRICED_MODEL))
    assert capsys.readouterr().err == ""

    # priced worker but UNPRICED reviewer -> still warns (its spend is invisible
    # to the dollar ceiling). Names the reviewer, not the priced worker.
    warn_if_usd_unenforceable(_cfg(1.0, PRICED_MODEL, reviewer="nobody/unpriced-reviewer"))
    warned = capsys.readouterr().err
    assert "cannot be enforced" in warned and "nobody/unpriced-reviewer" in warned
    assert PRICED_MODEL not in warned

    # no USD limit -> nothing to enforce, stay silent
    warn_if_usd_unenforceable(_cfg(0.0, "nobody/unpriced"))
    assert capsys.readouterr().err == ""
