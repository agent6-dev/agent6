# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The `[parallel]` config section: defaults + repo override."""

from __future__ import annotations

import pathlib
import tomllib

import pytest

from agent6 import kinds, paths
from agent6.app import _workdirs, confine
from agent6.config import Config, ParallelConfig, layer
from agent6.harness import subrun


def _write_repo_config(repo: pathlib.Path, toml: str) -> None:
    p = paths.repo_config_path(repo)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(toml, encoding="utf-8")


@pytest.fixture
def repo(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    r = tmp_path / "repo"
    r.mkdir()
    return r


def test_parallel_defaults() -> None:
    cfg = Config()
    assert cfg.parallel.max_lanes == 4
    assert cfg.parallel.workdir == ""


def test_parallel_max_lanes_bounds() -> None:
    # le=1024: the cap itself is capped, or a huge max_lanes re-opens the
    # huge-count allocation the spec grammar refuses against.
    with pytest.raises(ValueError):
        ParallelConfig(max_lanes=0)
    with pytest.raises(ValueError):
        ParallelConfig(max_lanes=1025)
    assert ParallelConfig(max_lanes=1024).max_lanes == 1024


def test_parallel_override_via_repo_config(repo: pathlib.Path) -> None:
    _write_repo_config(repo, '[parallel]\nmax_lanes = 8\nworkdir = "/tmp/lanes"\n')
    cfg = layer.load_effective(repo).config
    assert cfg.parallel.max_lanes == 8
    assert cfg.parallel.workdir == "/tmp/lanes"


@pytest.mark.parametrize("explicit_protect_git", [False, True])
def test_lane_config_preserves_protect_git_provenance(
    repo: pathlib.Path, explicit_protect_git: bool
) -> None:
    """A partial lane degrades default protect_git but refuses an explicit demand."""
    toml = '[sandbox]\nisolation = "partial"\n'
    if explicit_protect_git:
        toml += "protect_git = true\n"
    _write_repo_config(repo, toml)
    origin = layer.load_effective(repo)
    spec = subrun.LaneSpec(
        lane=1, session_id="fan-l1", workdir=repo.parent / "lanes" / "lane-1", route=None
    )

    config_path = _workdirs.write_lane_config(
        origin.config, spec, explicit_leaves=origin.explicit_leaves
    )
    lane = layer.load_effective(spec.workdir, explicit_paths=(config_path,))

    assert lane.config.sandbox.isolation == "partial"
    assert lane.config.sandbox.protect_git is True
    assert ("sandbox.protect_git" in lane.explicit_leaves) is explicit_protect_git
    assert "sandbox.home" not in lane.explicit_leaves
    refusal = confine.check_protect_git_support(
        lane.config, "partial", explicitly_set="sandbox.protect_git" in lane.explicit_leaves
    )
    if explicit_protect_git:
        assert refusal is not None and "requires the full isolation" in refusal
    else:
        assert refusal is None


def test_lane_config_keeps_an_explicit_reset_to_default(repo: pathlib.Path) -> None:
    """A repo's reset to default beats the global value when a lane reloads it."""
    global_path = paths.global_config_path()
    global_path.parent.mkdir(parents=True, exist_ok=True)
    global_path.write_text("[sandbox]\nprotect_git = false\n", encoding="utf-8")
    _write_repo_config(repo, "[sandbox]\nprotect_git = true\n")
    origin = layer.load_effective(repo)
    spec = subrun.LaneSpec(
        lane=1, session_id="fan-l1", workdir=repo.parent / "lanes" / "lane-1", route=None
    )

    config_path = _workdirs.write_lane_config(
        origin.config, spec, explicit_leaves=origin.explicit_leaves
    )
    written = tomllib.loads(config_path.read_text(encoding="utf-8"))
    lane = layer.load_effective(spec.workdir, explicit_paths=(config_path,))

    assert written["sandbox"]["protect_git"] is True
    assert lane.config.sandbox.protect_git is True
    assert lane.sources["sandbox.protect_git"] == f"flag ({config_path})"


def test_lane_config_keeps_worker_route_and_budget_override(repo: pathlib.Path) -> None:
    """Lane routing and post-load budget changes survive a lane config reload."""
    _write_repo_config(
        repo,
        '[providers.p]\napi_format = "openai"\nbase_url = "https://p.example/v1"\n'
        '[models.worker]\nprovider = "p"\nmodel = "origin-model"\n',
    )
    origin = layer.load_effective(repo)
    cfg = origin.config.with_budget_overrides(max_usd=3.5)
    spec = subrun.LaneSpec(
        lane=1,
        session_id="fan-l1",
        workdir=repo.parent / "lanes" / "lane-1",
        route=kinds.ModelRoute("p", "lane-model"),
    )

    config_path = _workdirs.write_lane_config(cfg, spec, explicit_leaves=origin.explicit_leaves)
    written = tomllib.loads(config_path.read_text(encoding="utf-8"))
    lane = layer.load_effective(spec.workdir, explicit_paths=(config_path,))

    assert written["models"]["worker"]["model"] == "lane-model"
    assert written["budget"]["max_usd"] == 3.5
    assert lane.config.model_route("worker", "lane-model") == kinds.ModelRoute("p", "lane-model")
    assert lane.config.budget.max_usd == 3.5
