# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The `[harness.parallel]` and `[git.worktree]` config tables: defaults and repo overrides."""

from __future__ import annotations

import pathlib
import tomllib

import pytest

from agent6 import child_records, kinds, paths
from agent6.app import confine, worktrees
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


def test_parallel_and_workspace_defaults() -> None:
    cfg = Config()
    assert cfg.harness.parallel.max_children == 4
    assert cfg.git.worktree.base_dir == ""


def test_parallel_max_children_bounds() -> None:
    # le=1024: the cap itself is capped, or a huge max_children re-opens the
    # huge-count allocation the spec grammar refuses against.
    with pytest.raises(ValueError):
        ParallelConfig(max_children=0)
    with pytest.raises(ValueError):
        ParallelConfig(max_children=1025)
    assert ParallelConfig(max_children=1024).max_children == 1024


def test_parallel_and_workspace_override_via_repo_config(repo: pathlib.Path) -> None:
    _write_repo_config(
        repo, '[harness.parallel]\nmax_children = 8\n[git.worktree]\nbase_dir = "/tmp/children"\n'
    )
    cfg = layer.load_effective(repo).config
    assert cfg.harness.parallel.max_children == 8
    assert cfg.git.worktree.base_dir == "/tmp/children"


@pytest.mark.parametrize("word", ["auto", "on"])
def test_child_config_carries_the_protect_git_word(repo: pathlib.Path, word: str) -> None:
    """A child under `partial` degrades `auto` and refuses `on`, as the origin run would."""
    toml = '[sandbox]\nisolation = "partial"\n'
    if word == "on":
        toml += "protect_git = 'on'\n"
    _write_repo_config(repo, toml)
    origin = layer.load_effective(repo)
    spec = child_records.ChildSpec(
        number=1,
        session_id="fan-l1",
        worktree=repo.parent / "children" / "child-1",
        settings=None,
    )

    config_path = worktrees.write_child_config(
        origin.config,
        spec,
        explicit_leaves=origin.explicit_leaves,
        into=spec.worktree.parent / "launch",
    ).path
    child = layer.load_effective(spec.worktree, explicit_paths=(config_path,))

    assert child.config.sandbox.isolation == "partial"
    assert child.config.sandbox.protect_git == word
    assert ("sandbox.protect_git" in child.explicit_leaves) is (word == "on")
    assert "sandbox.home" not in child.explicit_leaves
    refusal = confine.check_protect_git_support(child.config, "partial")
    if word == "on":
        assert refusal is not None and "requires sandbox.isolation = 'full'" in refusal
    else:
        assert refusal is None


def test_child_config_keeps_an_explicit_reset_to_default(repo: pathlib.Path) -> None:
    """A repo's reset to default beats the global value when a child reloads it."""
    global_path = paths.global_config_path()
    global_path.parent.mkdir(parents=True, exist_ok=True)
    global_path.write_text("[sandbox]\nprotect_git = 'off'\n", encoding="utf-8")
    _write_repo_config(repo, "[sandbox]\nprotect_git = 'on'\n")
    origin = layer.load_effective(repo)
    spec = child_records.ChildSpec(
        number=1,
        session_id="fan-l1",
        worktree=repo.parent / "children" / "child-1",
        settings=None,
    )

    config_path = worktrees.write_child_config(
        origin.config,
        spec,
        explicit_leaves=origin.explicit_leaves,
        into=spec.worktree.parent / "launch",
    ).path
    written = tomllib.loads(config_path.read_text(encoding="utf-8"))
    child = layer.load_effective(spec.worktree, explicit_paths=(config_path,))

    assert written["sandbox"]["protect_git"] == "on"
    assert child.config.sandbox.protect_git == "on"
    assert child.sources["sandbox.protect_git"] == f"flag ({config_path})"


def test_child_config_keeps_worker_route_and_budget_override(repo: pathlib.Path) -> None:
    """Child routing and post-load budget changes survive a child config reload."""
    _write_repo_config(
        repo,
        '[providers.p]\nbackend = "openai"\nbase_url = "https://p.example/v1"\n'
        '[models.worker]\nmodel = "p/origin-model"\n',
    )
    origin = layer.load_effective(repo)
    cfg = origin.config.with_budget_overrides(max_usd=3.5)
    spec = child_records.ChildSpec(
        number=1,
        session_id="fan-l1",
        worktree=repo.parent / "children" / "child-1",
        settings=kinds.ModelSettings(kinds.ModelRoute("p", "child-model")),
    )

    config_path = worktrees.write_child_config(
        cfg, spec, explicit_leaves=origin.explicit_leaves, into=spec.worktree.parent / "launch"
    ).path
    written = tomllib.loads(config_path.read_text(encoding="utf-8"))
    child = layer.load_effective(spec.worktree, explicit_paths=(config_path,))

    assert written["models"]["worker"]["model"] == "p/child-model"
    assert written["budget"]["max_usd"] == 3.5
    assert child.config.model_route("worker", "child-model") == kinds.ModelRoute("p", "child-model")
    assert child.config.budget.max_usd == 3.5
