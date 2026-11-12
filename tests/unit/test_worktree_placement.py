# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Where agent6 puts the worktrees it makes, and how a base inside a repository stays out of git."""

from __future__ import annotations

import pathlib
import subprocess

import pydantic
import pytest

from agent6 import errors, path_safety, paths
from agent6.app import worktrees
from agent6.config import Config, ContainerConfig
from agent6.tools import index

_CONTAINER = Config.model_validate({"sandbox": {"isolation": "container"}})


@pytest.fixture
def repo(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    root = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", "-b", "main", str(root)], check=True)
    return root


def test_local_levels_keep_the_cache_base_and_write_nothing_to_the_repository(
    repo: pathlib.Path,
) -> None:
    """Only a base inside the repository is excluded; the default sits in the cache."""
    placed = worktrees.place(Config(), repo, "run-1")
    assert placed == paths.cache_dir() / "worktrees" / paths.repo_id(repo) / "run-1"
    assert "worktrees" not in (repo / ".git/info/exclude").read_text(encoding="utf-8")


def test_under_a_container_the_base_is_inside_the_checkout_and_excluded_once(
    repo: pathlib.Path,
) -> None:
    """A tree in the cache has no path inside the container that mounts the workspace."""
    first = worktrees.place(_CONTAINER, repo, "run-1")
    worktrees.place(_CONTAINER, repo, "run-2")
    assert first == repo.resolve() / ".agent6/worktrees" / paths.repo_id(repo) / "run-1"
    exclude = (repo / ".git/info/exclude").read_text(encoding="utf-8").splitlines()
    assert exclude.count("/.agent6/worktrees/") == 1
    assert not (repo / ".gitignore").exists()


def test_a_rule_that_un_ignores_the_base_refuses_naming_its_key(repo: pathlib.Path) -> None:
    """Left visible, the run's own commits would take every worktree in."""
    (repo / ".gitignore").write_text("!/.agent6/\n!/.agent6/**\n", encoding="utf-8")
    with pytest.raises(errors.OperatorError, match=r"\[container\]\.worktree_base_dir"):
        worktrees.place(_CONTAINER, repo, "run-1")


def test_the_mounted_workspace_is_the_checkout_above_every_base(tmp_path: pathlib.Path) -> None:
    """A lane of a lane sits two bases deep; the container mounts the outermost checkout."""
    root = tmp_path / "repo"
    lane = root / ".agent6/worktrees/repo-1/run-p1/lane-1"
    reviewer = lane / ".agent6/worktrees/lane-2/panel/lane-3"
    for tree in (lane, reviewer):
        assert worktrees.mounted_workspace(_CONTAINER, tree) == root
        assert worktrees.holds(_CONTAINER, tree)
    assert worktrees.mounted_workspace(_CONTAINER, root) == root
    assert not worktrees.holds(_CONTAINER, root)
    assert not worktrees.holds(_CONTAINER, root / ".agent6")


def test_a_local_run_holds_only_trees_under_its_configured_base(tmp_path: pathlib.Path) -> None:
    cfg = Config.model_validate({"git": {"worktree": {"base_dir": str(tmp_path / "wt")}}})
    assert worktrees.holds(cfg, tmp_path / "wt/repo-1/run-p1/lane-1")
    assert not worktrees.holds(cfg, tmp_path / "repo/.agent6/worktrees/repo-1/run-p1/lane-1")


@pytest.mark.parametrize("value", ["", "/abs/trees", "../trees", ".agent6/../../trees"])
def test_the_container_base_must_stay_under_the_project_root(value: str) -> None:
    """The container reaches only the workspace it mounts."""
    with pytest.raises(pydantic.ValidationError, match=r"container\.worktree_base_dir"):
        ContainerConfig(worktree_base_dir=value)
    assert ContainerConfig().worktree_base_dir == ".agent6/worktrees"


def test_the_symbol_index_skips_the_directory_as_it_skips_git(tmp_path: pathlib.Path) -> None:
    """A parent's index read every lane's copy of the repository."""
    (tmp_path / "mod.py").write_text("def kept():\n    pass\n", encoding="utf-8")
    copy = tmp_path / ".agent6/worktrees/repo-1/run-p1/lane-1"
    copy.mkdir(parents=True)
    (copy / "mod.py").write_text("def kept():\n    pass\n", encoding="utf-8")
    symbols = index.SymbolIndex(path_safety.Workspace(tmp_path))
    assert [d.path for d in symbols.find_definition("kept")] == [tmp_path / "mod.py"]
    inside = index.SymbolIndex(path_safety.Workspace(copy))
    assert [d.path for d in inside.find_definition("kept")] == [copy / "mod.py"]
