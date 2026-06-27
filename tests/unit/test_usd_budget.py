# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Unit tests for the USD-to-tokens budget converter (~budget.py).

Pricing has no static table: it comes from the provider-fetched models cache
(agent6.pricing reads $AGENT6_CACHE_HOME/models/*.json). Tests inject prices
by writing a real cache file, exercising the same path production uses.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agent6.budget import usd_budget_to_tokens

# $3/M in, $15/M out: the worked example in the usd_budget_to_tokens docstring.
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


def _convert(max_usd: float, model: str) -> tuple[int, int]:
    converted = usd_budget_to_tokens(max_usd, worker_model=model)
    assert converted is not None
    return converted


def test_usd_converter_priced_model(price_cache: Path) -> None:
    """$3/M input + $15/M output, $5 budget. Each axis is sized to the FULL
    budget (the runtime USD ceiling bounds combined spend), so input alone is
    ~$5 of input tokens and output alone is ~$5 of output tokens."""
    max_in, max_out = _convert(5.0, PRICED_MODEL)
    assert abs(max_in - int(5.0 * 1_000_000 / 3.0)) <= 1  # 1_666_666
    assert abs(max_out - int(5.0 * 1_000_000 / 15.0)) <= 1  # 333_333


def test_usd_converter_output_axis_gets_full_budget(price_cache: Path) -> None:
    """Regression: an output-heavy (e.g. reasoning) workload must be able to
    spend the whole USD budget on output. The output cap must equal max_usd /
    output_price, NOT a small ratio-split fraction of it -- the bug that halted
    a $3 GLM 5.2 run at $0.66 because its 1:1 reasoning I/O blew a 1/6 output
    cap while the input cap sat 94% unused."""
    # $15/M out, $5 budget -> the full budget buys 333_333 output tokens.
    _, max_out = _convert(5.0, PRICED_MODEL)
    assert max_out == int(5.0 * 1_000_000 / 15.0)
    # ...not the old 1/6 split (which would have been ~55_555).
    assert max_out > 5 * 55_555


def test_usd_converter_unknown_model_returns_none(price_cache: Path) -> None:
    """No cached price means NO conversion: unknown beats a wrong fallback."""
    assert usd_budget_to_tokens(5.0, worker_model="nobody/unpriced-model") is None


def test_usd_converter_cheap_model_buys_more_tokens(price_cache: Path) -> None:
    # Same $1 budget buys 11x more input tokens at $0.27/M vs $3/M.
    cheap_in, _ = _convert(1.0, CHEAP_MODEL)
    priced_in, _ = _convert(1.0, PRICED_MODEL)
    assert cheap_in > 10 * priced_in


def test_usd_converter_rejects_zero_or_negative(price_cache: Path) -> None:
    with pytest.raises(ValueError):
        usd_budget_to_tokens(0.0, worker_model=PRICED_MODEL)
    with pytest.raises(ValueError):
        usd_budget_to_tokens(-1.0, worker_model=PRICED_MODEL)


def test_usd_converter_scale_linearity(price_cache: Path) -> None:
    """Doubling the budget doubles both token ceilings (within rounding)."""
    in_5, out_5 = _convert(5.0, PRICED_MODEL)
    in_10, out_10 = _convert(10.0, PRICED_MODEL)
    assert abs((in_10 / in_5) - 2.0) < 0.01
    assert abs((out_10 / out_5) - 2.0) < 0.01


def test_explicit_usd_flag_refused_when_unpriced(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # An explicit --max-usd is a promise for the run; with no price data for
    # the worker model it cannot be kept, so the CLI refuses to start.
    from agent6.cli._common import (
        _explicit_usd_flag_error,  # pyright: ignore[reportPrivateUsage]
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
    from agent6.cli._common import (
        _explicit_usd_flag_error,  # pyright: ignore[reportPrivateUsage]
    )
    from agent6.config import Config

    cfg = Config.model_validate(
        {
            "providers": {"p": {"api_format": "openai", "base_url": "http://localhost:1"}},
            "models": {"worker": {"provider": "p", "model": PRICED_MODEL}},
        }
    )
    assert _explicit_usd_flag_error(2.5, cfg) is None


def test_warn_if_usd_unenforceable(price_cache: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A TOML best_effort_usd_limit on an unpriced worker can't enforce (Anthropic
    publishes no pricing), so run startup must warn instead of silently no-op'ing.
    The --max-usd *flag* is already guarded by _explicit_usd_flag_error; this is
    the config-path complement."""
    from agent6.cli.run import _warn_if_usd_unenforceable  # pyright: ignore[reportPrivateUsage]
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
    _warn_if_usd_unenforceable(_cfg(1.0, "nobody/unpriced"))
    warned = capsys.readouterr().err
    assert "cannot be enforced" in warned and "nobody/unpriced" in warned

    # priced worker -> the USD ceiling works, so stay silent
    _warn_if_usd_unenforceable(_cfg(1.0, PRICED_MODEL))
    assert capsys.readouterr().err == ""

    # priced worker but UNPRICED reviewer -> still warns (its spend is invisible
    # to the dollar ceiling). Names the reviewer, not the priced worker.
    _warn_if_usd_unenforceable(_cfg(1.0, PRICED_MODEL, reviewer="nobody/unpriced-reviewer"))
    warned = capsys.readouterr().err
    assert "cannot be enforced" in warned and "nobody/unpriced-reviewer" in warned
    assert PRICED_MODEL not in warned

    # no USD limit -> nothing to enforce, stay silent
    _warn_if_usd_unenforceable(_cfg(0.0, "nobody/unpriced"))
    assert capsys.readouterr().err == ""
