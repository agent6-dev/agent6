# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""One lane selector resolver: a role name brings its settings whole, a route runs the worker's."""

from __future__ import annotations

import pathlib

import pytest

from agent6 import kinds, lane_records
from agent6.app import worktrees
from agent6.config import Config, ConfigError, layer
from agent6.ui.cli import completers


def _cfg() -> Config:
    return Config.model_validate(
        {
            "providers": {
                "o": {"backend": "openai", "base_url": "https://o.example/v1"},
                "p": {"backend": "openai", "base_url": "https://p.example/v1"},
            },
            "models": {
                "default": {"model": "o/base", "effort": "low", "temperature": 0.3},
                "worker": {"model": "o/worker-model", "effort": "medium"},
                "reviewer": {"model": "p/review-model", "effort": "xhigh", "temperature": 0.0},
            },
        }
    )


def test_a_selector_resolves_to_complete_settings() -> None:
    cfg = _cfg()
    worker = kinds.ModelSettings(kinds.ModelRoute("o", "worker-model"), "medium", 0.3)
    # No selection: the lane runs the worker's own settings, its config unchanged.
    assert cfg.lane_settings(None) is None and cfg.lane_settings("") is None
    assert cfg.lane_settings("worker") == worker
    # A role name brings the role's own effort and temperature, the explicit zero included.
    assert cfg.lane_settings("reviewer") == kinds.ModelSettings(
        kinds.ModelRoute("p", "review-model"), "xhigh", 0.0
    )
    assert cfg.lane_settings("default") == kinds.ModelSettings(
        kinds.ModelRoute("o", "base"), "low", 0.3
    )
    # A route runs the worker's effort and temperature; a bare id stays on the worker's provider.
    assert cfg.lane_settings("p/other") == kinds.ModelSettings(
        kinds.ModelRoute("p", "other"), "medium", 0.3
    )
    assert cfg.lane_settings("vendor/model@tag") == kinds.ModelSettings(
        kinds.ModelRoute("o", "vendor/model@tag"), "medium", 0.3
    )
    # A model named like a role is written with its provider.
    literal = cfg.lane_settings("p/reviewer")
    assert literal is not None and literal.route == kinds.ModelRoute("p", "reviewer")


def test_a_role_without_a_route_refuses() -> None:
    assert Config().lane_settings("") is None  # the worker's own, whatever it is
    with pytest.raises(ConfigError, match=r"models\.worker: no model route"):
        Config().lane_settings("worker")
    with pytest.raises(ConfigError, match="no model id"):
        _cfg().lane_settings("p/")


def test_the_lane_config_reads_back_the_selected_settings(tmp_path: pathlib.Path) -> None:
    """The written lane config materializes every selected field; none re-inherits the worker."""
    cfg = _cfg()
    chosen = cfg.lane_settings("reviewer")
    spec = lane_records.LaneSpec(
        lane=1, session_id="fan-l1", worktree=tmp_path / "lanes" / "lane-1", settings=chosen
    )
    written = worktrees.write_lane_config(
        cfg, spec, explicit_leaves=frozenset(), into=spec.worktree.parent / "launch"
    ).path
    lane = layer.load_effective(spec.worktree, explicit_paths=(written,))
    assert lane.config.role_settings("worker") == chosen
    assert lane.config.models.worker is not None
    assert (
        lane.config.models.worker.temperature == 0.0 and lane.config.models.worker.effort == "xhigh"
    )


def test_the_parallel_completer_offers_role_names_before_routes() -> None:
    offered = completers._complete_parallel_models("rev")  # pyright: ignore[reportPrivateUsage]
    assert offered[:1] == ["reviewer"]
    assert completers._complete_parallel_models("o/x,wor")[:1] == ["o/x,worker"]  # pyright: ignore[reportPrivateUsage]
