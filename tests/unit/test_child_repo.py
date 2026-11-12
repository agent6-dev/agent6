# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for agent6.harness.child_repo on temporary git repositories."""

from __future__ import annotations

import pathlib
import subprocess

import pytest

from agent6.harness import child_repo
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


def test_clone_worktree_produces_independent_clone(tmp_path: pathlib.Path) -> None:
    origin = tmp_path / "origin"
    _init_repo(origin)
    dest = tmp_path / "child-1"

    child_repo.clone_worktree(origin, dest)

    assert (dest / "README.md").read_text(encoding="utf-8") == "hi\n"
    (dest / "README.md").write_text("edited in the child\n", encoding="utf-8")
    assert origin.joinpath("README.md").read_text(encoding="utf-8") == "hi\n"


def test_clone_worktree_missing_origin_raises_child_repo_error(tmp_path: pathlib.Path) -> None:
    with pytest.raises(child_repo.ChildRepoError):
        child_repo.clone_worktree(tmp_path / "does-not-exist", tmp_path / "child-1")


def test_clone_worktree_raises_child_repo_error_on_failure(tmp_path: pathlib.Path) -> None:
    origin = tmp_path / "origin"
    _init_repo(origin)
    dest = tmp_path / "child-1"
    dest.mkdir()
    (dest / "existing.txt").write_text("occupied\n", encoding="utf-8")

    with pytest.raises(child_repo.ChildRepoError):
        child_repo.clone_worktree(origin, dest)


def test_import_run_lands_the_branch_and_leaves_the_record_in_place(tmp_path: pathlib.Path) -> None:
    """The child's record is in the origin's state from its first event: the import moves none."""
    origin = tmp_path / "origin"
    _init_repo(origin)
    child_worktree = tmp_path / "child-1"
    child_repo.clone_worktree(origin, child_worktree)

    branch = "agent6/child-1"
    git_ops.create_branch(child_worktree, branch)
    (child_worktree / "feature.txt").write_text("new stuff\n", encoding="utf-8")
    _git(child_worktree, "add", "-A")
    _git(child_worktree, "commit", "-q", "-m", "child change")

    child_session_dir = tmp_path / "origin-state" / "sessions" / "runs" / "01ABC"
    child_session_dir.mkdir(parents=True)
    (child_session_dir / "manifest.json").write_text("{}\n", encoding="utf-8")

    result = child_repo.import_run(origin, child_worktree, branch, child_session_dir)

    assert result.session_dir == child_session_dir
    assert result.ref == branch
    assert result.tip == git_read.branch_tip_sha(origin, branch)
    assert (child_session_dir / "manifest.json").read_text(encoding="utf-8") == "{}\n"
    assert git_read.branch_exists(origin, branch)


def test_import_run_lands_a_child_that_never_committed(tmp_path: pathlib.Path) -> None:
    """A child stopped before its first commit has no branch to land.

    Reported as a failed fetch, each child of a fleet stopped by an expired key reads "couldn't find
    remote ref", with the reason left in a session dir the operator cannot reach.
    """
    origin = tmp_path / "origin"
    _init_repo(origin)
    child_worktree = tmp_path / "child-1"
    child_repo.clone_worktree(origin, child_worktree)
    child_session_dir = tmp_path / "child-state" / "sessions" / "runs" / "01DEAD"
    child_session_dir.mkdir(parents=True)
    (child_session_dir / "manifest.json").write_text("{}\n", encoding="utf-8")

    result = child_repo.import_run(origin, child_worktree, "agent6/child-1", child_session_dir)

    assert (result.session_dir / "manifest.json").is_file()
    assert (result.ref, result.tip) == ("", "")
    assert not git_read.branch_exists(origin, "agent6/child-1")


def test_import_run_under_model_control_lands_the_hidden_chain_ref(
    tmp_path: pathlib.Path,
) -> None:
    """A model-controlled child's HEAD lands on `refs/agent6/<id>/head`; the record names it.

    The child's own branch name is created in the origin when free, and the returned ref and tip
    are the chain ref and the child's HEAD, so a join merges what the child committed.
    """
    origin = tmp_path / "origin"
    _init_repo(origin)
    child_worktree = tmp_path / "child-1"
    child_repo.clone_worktree(origin, child_worktree)
    _git(child_worktree, "checkout", "-q", "-b", "feature")
    (child_worktree / "new.txt").write_text("child\n", encoding="utf-8")
    _git(child_worktree, "add", "-A")
    _git(child_worktree, "commit", "-q", "-m", "child work")
    head = git_read.status(child_worktree).head_sha
    child_session_dir = tmp_path / "child-state" / "sessions" / "runs" / "01MC"
    child_session_dir.mkdir(parents=True)
    (child_session_dir / "manifest.json").write_text("{}\n", encoding="utf-8")

    result = child_repo.import_run(
        origin, child_worktree, "", child_session_dir, model_control=True
    )

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
    branch = "agent6/child-1"
    git_ops.create_branch(origin, branch)
    _git(origin, "checkout", "main")

    child_worktree = tmp_path / "child-1"
    child_repo.clone_worktree(origin, child_worktree)
    child_session_dir = tmp_path / "child-state" / "sessions" / "runs" / "01ABC"
    child_session_dir.mkdir(parents=True)
    (child_session_dir / "manifest.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(child_repo.ChildRepoError):
        child_repo.import_run(origin, child_worktree, branch, child_session_dir)
