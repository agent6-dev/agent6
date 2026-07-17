# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for agent6.workflows.subrun on temporary git repositories."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from agent6.git_ops import branch_exists, commit_all, create_branch, status
from agent6.workflows.subrun import SubrunError, clone_workspace, import_run, join_branch


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _init_repo(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / "README.md").write_text("hi\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")


def test_clone_workspace_produces_independent_clone(tmp_path: Path) -> None:
    origin = tmp_path / "origin"
    _init_repo(origin)
    dest = tmp_path / "lane-1"

    clone_workspace(origin, dest)

    assert (dest / "README.md").read_text(encoding="utf-8") == "hi\n"
    (dest / "README.md").write_text("edited in the lane\n", encoding="utf-8")
    assert origin.joinpath("README.md").read_text(encoding="utf-8") == "hi\n"


def test_clone_workspace_missing_origin_raises_subrun_error(tmp_path: Path) -> None:
    with pytest.raises(SubrunError):
        clone_workspace(tmp_path / "does-not-exist", tmp_path / "lane-1")


def test_clone_workspace_raises_subrun_error_on_failure(tmp_path: Path) -> None:
    origin = tmp_path / "origin"
    _init_repo(origin)
    dest = tmp_path / "lane-1"
    dest.mkdir()
    (dest / "existing.txt").write_text("occupied\n", encoding="utf-8")

    with pytest.raises(SubrunError):
        clone_workspace(origin, dest)


def test_import_run_lands_branch_and_moves_run_dir(tmp_path: Path) -> None:
    origin = tmp_path / "origin"
    _init_repo(origin)
    lane_repo = tmp_path / "lane-1"
    clone_workspace(origin, lane_repo)

    branch = "agent6/lane-1"
    create_branch(lane_repo, branch)
    (lane_repo / "feature.txt").write_text("new stuff\n", encoding="utf-8")
    commit_all(lane_repo, "lane change")

    lane_run_dir = tmp_path / "lane-state" / "runs" / "01ABC"
    lane_run_dir.mkdir(parents=True)
    (lane_run_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
    origin_state = tmp_path / "origin-state"

    imported = import_run(origin, lane_repo, branch, lane_run_dir, origin_state)

    assert imported == origin_state / "runs" / "01ABC"
    assert (imported / "manifest.json").read_text(encoding="utf-8") == "{}\n"
    assert not lane_run_dir.exists()
    assert branch_exists(origin, branch)


def test_import_run_refuses_existing_branch(tmp_path: Path) -> None:
    origin = tmp_path / "origin"
    _init_repo(origin)
    branch = "agent6/lane-1"
    create_branch(origin, branch)
    _git(origin, "checkout", "main")

    lane_repo = tmp_path / "lane-1"
    clone_workspace(origin, lane_repo)
    lane_run_dir = tmp_path / "lane-state" / "runs" / "01ABC"
    lane_run_dir.mkdir(parents=True)
    (lane_run_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
    origin_state = tmp_path / "origin-state"

    with pytest.raises(SubrunError):
        import_run(origin, lane_repo, branch, lane_run_dir, origin_state)
    # Refused before moving anything.
    assert lane_run_dir.exists()
    assert not (origin_state / "runs" / "01ABC").exists()


def test_import_run_refuses_existing_run_dir(tmp_path: Path) -> None:
    origin = tmp_path / "origin"
    _init_repo(origin)
    lane_repo = tmp_path / "lane-1"
    clone_workspace(origin, lane_repo)
    branch = "agent6/lane-1"
    create_branch(lane_repo, branch)
    (lane_repo / "feature.txt").write_text("new stuff\n", encoding="utf-8")
    commit_all(lane_repo, "lane change")

    lane_run_dir = tmp_path / "lane-state" / "runs" / "01ABC"
    lane_run_dir.mkdir(parents=True)
    (lane_run_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
    origin_state = tmp_path / "origin-state"
    (origin_state / "runs" / "01ABC").mkdir(parents=True)  # already imported

    with pytest.raises(SubrunError):
        import_run(origin, lane_repo, branch, lane_run_dir, origin_state)
    # Refused before fetching or moving anything.
    assert not branch_exists(origin, branch)
    assert lane_run_dir.exists()


def test_join_branch_merges_cleanly(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    _init_repo(workspace)
    branch = "agent6/lane-1"
    create_branch(workspace, branch)
    (workspace / "feature.txt").write_text("new stuff\n", encoding="utf-8")
    commit_all(workspace, "lane change")
    _git(workspace, "checkout", "main")

    sha = join_branch(workspace, branch)

    assert sha
    assert (workspace / "feature.txt").read_text(encoding="utf-8") == "new stuff\n"
    assert status(workspace).is_clean


def test_join_branch_conflict_returns_none_and_leaves_workspace_clean(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    _init_repo(workspace)
    branch = "agent6/lane-1"
    create_branch(workspace, branch)
    (workspace / "README.md").write_text("branch version\n", encoding="utf-8")
    commit_all(workspace, "branch change")
    _git(workspace, "checkout", "main")
    (workspace / "README.md").write_text("main version\n", encoding="utf-8")
    commit_all(workspace, "main change")

    result = join_branch(workspace, branch)

    assert result is None
    porcelain = subprocess.run(
        ["git", "-C", str(workspace), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert porcelain == ""
