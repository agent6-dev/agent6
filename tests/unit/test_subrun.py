# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for agent6.harness.subrun on temporary git repositories."""

from __future__ import annotations

import pathlib
import subprocess

import pytest

from agent6.harness import subrun
from agent6.vcs.git import ops as git_ops
from agent6.vcs.git import read as git_read


def _git(repo: pathlib.Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _init_repo(repo: pathlib.Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / "README.md").write_text("hi\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")


def test_clone_workspace_produces_independent_clone(tmp_path: pathlib.Path) -> None:
    origin = tmp_path / "origin"
    _init_repo(origin)
    dest = tmp_path / "lane-1"

    subrun.clone_workspace(origin, dest)

    assert (dest / "README.md").read_text(encoding="utf-8") == "hi\n"
    (dest / "README.md").write_text("edited in the lane\n", encoding="utf-8")
    assert origin.joinpath("README.md").read_text(encoding="utf-8") == "hi\n"


def test_clone_workspace_missing_origin_raises_subrun_error(tmp_path: pathlib.Path) -> None:
    with pytest.raises(subrun.SubrunError):
        subrun.clone_workspace(tmp_path / "does-not-exist", tmp_path / "lane-1")


def test_clone_workspace_raises_subrun_error_on_failure(tmp_path: pathlib.Path) -> None:
    origin = tmp_path / "origin"
    _init_repo(origin)
    dest = tmp_path / "lane-1"
    dest.mkdir()
    (dest / "existing.txt").write_text("occupied\n", encoding="utf-8")

    with pytest.raises(subrun.SubrunError):
        subrun.clone_workspace(origin, dest)


def test_import_run_lands_the_branch_and_leaves_the_record_in_place(tmp_path: pathlib.Path) -> None:
    """The lane's record is in the origin's state from its first event: the import moves none."""
    origin = tmp_path / "origin"
    _init_repo(origin)
    lane_repo = tmp_path / "lane-1"
    subrun.clone_workspace(origin, lane_repo)

    branch = "agent6/lane-1"
    git_ops.create_branch(lane_repo, branch)
    (lane_repo / "feature.txt").write_text("new stuff\n", encoding="utf-8")
    _git(lane_repo, "add", "-A")
    _git(lane_repo, "commit", "-q", "-m", "lane change")

    lane_session_dir = tmp_path / "origin-state" / "sessions" / "runs" / "01ABC"
    lane_session_dir.mkdir(parents=True)
    (lane_session_dir / "manifest.json").write_text("{}\n", encoding="utf-8")

    result = subrun.import_run(origin, lane_repo, branch, lane_session_dir)

    assert result.session_dir == lane_session_dir
    assert result.ref == branch
    assert result.tip == git_read.branch_tip_sha(origin, branch)
    assert (lane_session_dir / "manifest.json").read_text(encoding="utf-8") == "{}\n"
    assert git_read.branch_exists(origin, branch)


def test_import_run_lands_a_lane_that_never_committed(tmp_path: pathlib.Path) -> None:
    """A lane stopped before its first commit has no branch to land.

    Reported as a failed fetch, every lane of a fleet stopped by an expired key reads "couldn't find
    remote ref", with the reason left in a session dir the operator cannot reach.
    """
    origin = tmp_path / "origin"
    _init_repo(origin)
    lane_repo = tmp_path / "lane-1"
    subrun.clone_workspace(origin, lane_repo)
    lane_session_dir = tmp_path / "lane-state" / "sessions" / "runs" / "01DEAD"
    lane_session_dir.mkdir(parents=True)
    (lane_session_dir / "manifest.json").write_text("{}\n", encoding="utf-8")

    result = subrun.import_run(origin, lane_repo, "agent6/lane-1", lane_session_dir)

    assert (result.session_dir / "manifest.json").is_file()
    assert (result.ref, result.tip) == ("", "")
    assert not git_read.branch_exists(origin, "agent6/lane-1")


def test_import_run_under_model_control_lands_the_hidden_chain_ref(
    tmp_path: pathlib.Path,
) -> None:
    """A model-controlled lane's HEAD lands on `refs/agent6/<id>/head`; the record names it.

    The lane's own branch name is created in the origin when free, and the returned ref and tip
    are the chain ref and the lane's HEAD, so a join merges what the lane committed.
    """
    origin = tmp_path / "origin"
    _init_repo(origin)
    lane_repo = tmp_path / "lane-1"
    subrun.clone_workspace(origin, lane_repo)
    _git(lane_repo, "checkout", "-q", "-b", "feature")
    (lane_repo / "new.txt").write_text("lane\n", encoding="utf-8")
    _git(lane_repo, "add", "-A")
    _git(lane_repo, "commit", "-q", "-m", "lane work")
    head = git_read.status(lane_repo).head_sha
    lane_session_dir = tmp_path / "lane-state" / "sessions" / "runs" / "01MC"
    lane_session_dir.mkdir(parents=True)
    (lane_session_dir / "manifest.json").write_text("{}\n", encoding="utf-8")

    result = subrun.import_run(origin, lane_repo, "", lane_session_dir, model_control=True)

    chain = git_read.chain_ref_for("01MC")
    assert (result.ref, result.tip) == (chain, head)
    landed = subprocess.run(
        ["git", "-C", str(origin), "rev-parse", chain], capture_output=True, text=True, check=True
    )
    assert landed.stdout.strip() == head
    assert git_read.branch_tip_sha(origin, "feature") == head


def test_import_run_refuses_existing_branch(tmp_path: pathlib.Path) -> None:
    origin = tmp_path / "origin"
    _init_repo(origin)
    branch = "agent6/lane-1"
    git_ops.create_branch(origin, branch)
    _git(origin, "checkout", "main")

    lane_repo = tmp_path / "lane-1"
    subrun.clone_workspace(origin, lane_repo)
    lane_session_dir = tmp_path / "lane-state" / "sessions" / "runs" / "01ABC"
    lane_session_dir.mkdir(parents=True)
    (lane_session_dir / "manifest.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(subrun.SubrunError):
        subrun.import_run(origin, lane_repo, branch, lane_session_dir)
