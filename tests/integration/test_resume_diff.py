# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for `GraphCurator.compute_resume_diff` against a real git repo."""

from __future__ import annotations

import subprocess
from pathlib import Path

from agent6.graph.curator import GraphCurator, hash_uncommitted
from agent6.graph.models import (
    AddSubtaskIntent,
    SetCursorIntent,
    SnapshotNodeIntent,
    TaskNodeDraft,
)
from agent6.graph.storage import RunLayout


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    (repo / "a.txt").write_text("hello\n")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-q", "-m", "init")
    return repo


def _setup_curator_with_snapshot(repo: Path) -> tuple[GraphCurator, str]:
    layout = RunLayout(state_dir=repo / ".agent6", run_id="run1")
    curator = GraphCurator(layout)
    node = curator.add_subtask(
        AddSubtaskIntent(
            parent_id=None,
            draft=TaskNodeDraft(title="t", created_by="planner"),
        )
    )
    curator.set_cursor(SetCursorIntent(id=node.id))
    head = _git(repo, "rev-parse", "HEAD")
    touched = hash_uncommitted(repo, ("a.txt",))
    curator.snapshot_node(
        SnapshotNodeIntent(id=node.id, head_sha=head, branch="main", uncommitted_touched=touched)
    )
    return curator, node.id


def test_resume_diff_clean_workspace(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    curator, _ = _setup_curator_with_snapshot(repo)
    diff = curator.compute_resume_diff("run1", repo)
    assert diff.snapshot_missing is False
    assert diff.committed_delta.files == ()
    assert diff.uncommitted_diff == ()


def test_resume_diff_detects_user_commit(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    curator, _ = _setup_curator_with_snapshot(repo)
    (repo / "b.txt").write_text("new\n")
    _git(repo, "add", "b.txt")
    _git(repo, "commit", "-q", "-m", "user added b")
    diff = curator.compute_resume_diff("run1", repo)
    assert "b.txt" in diff.committed_delta.files
    assert diff.snapshot_missing is False


def test_resume_diff_detects_uncommitted_modification(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    curator, _ = _setup_curator_with_snapshot(repo)
    (repo / "a.txt").write_text("modified by user\n")
    diff = curator.compute_resume_diff("run1", repo)
    assert len(diff.uncommitted_diff) == 1
    assert diff.uncommitted_diff[0].path == "a.txt"
    assert diff.uncommitted_diff[0].note == "modified since snapshot"


def test_resume_diff_snapshot_commit_gone(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    curator, _ = _setup_curator_with_snapshot(repo)
    # Rewrite history so the snapshot SHA is unreachable, then GC it.
    (repo / "a.txt").write_text("changed\n")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "--amend", "-q", "-m", "amended")
    subprocess.run(
        ["git", "reflog", "expire", "--expire=now", "--all"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "gc", "--prune=now", "--quiet"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    diff = curator.compute_resume_diff("run1", repo)
    assert diff.snapshot_missing is True
    assert "--force-resume" in diff.guard_summary
