from __future__ import annotations

import pathlib

import pytest
from bench.longhorizon import run

from agent6 import config


@pytest.mark.parametrize("condition", sorted(run.CONDITIONS))
def test_each_condition_pins_the_test_model(tmp_path: pathlib.Path, condition: str) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        run._provider_block("openrouter", "vendor/model") + run.CONDITIONS[condition].toml,
        encoding="utf-8",
    )
    cfg = config.load_config(path)
    for route in (cfg.models.default, cfg.models.worker, cfg.models.planner, cfg.models.reviewer):
        assert route is not None
        assert (route.provider, route.model) == ("openrouter", "vendor/model")
