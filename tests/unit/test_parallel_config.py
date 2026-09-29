# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The `[parallel]` config section: defaults + repo override."""

from __future__ import annotations

import pathlib

import pytest

from agent6 import paths
from agent6.config import Config, ParallelConfig, layer


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
