# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the resume head guard (`snapshot_head_mismatch`) against a real git repo.

The guard is what makes `agent6 resume` refuse when the workspace HEAD moved
since the run's last `loop_state.json` write. It reads the snapshot's
`head_sha` field directly (best-effort) and compares it to the current HEAD.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from agent6.cli.run import snapshot_head_mismatch


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


def _write_snapshot(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "loop_state.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_aligned_head_passes(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    head = _git(repo, "rev-parse", "HEAD")
    snap = _write_snapshot(tmp_path, {"head_sha": head})
    assert snapshot_head_mismatch(snap, repo) is None


def test_own_forward_commit_is_allowed(tmp_path: Path) -> None:
    # The run's own per-step commit advances HEAD forward from the snapshot on
    # the same line (a kill after the commit but before the next snapshot
    # refresh). HEAD is a descendant of snap_head, so resume must proceed.
    repo = _init_repo(tmp_path)
    old_head = _git(repo, "rev-parse", "HEAD")
    snap = _write_snapshot(tmp_path, {"head_sha": old_head})
    (repo / "b.txt").write_text("new\n")
    _git(repo, "add", "b.txt")
    _git(repo, "commit", "-q", "-m", "the run's own step commit")
    assert snapshot_head_mismatch(snap, repo) is None


def test_diverged_head_is_reported(tmp_path: Path) -> None:
    # An amend rewrites HEAD to a new sha and orphans the snapshot commit, so
    # HEAD is NOT a descendant of snap_head: genuine divergence, refuse.
    repo = _init_repo(tmp_path)
    old_head = _git(repo, "rev-parse", "HEAD")
    snap = _write_snapshot(tmp_path, {"head_sha": old_head})
    (repo / "a.txt").write_text("rewritten\n")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-q", "--amend", "-m", "amended init")
    mismatch = snapshot_head_mismatch(snap, repo)
    assert mismatch is not None
    snap_head, current_head = mismatch
    assert snap_head == old_head
    assert current_head == _git(repo, "rev-parse", "HEAD")


def test_reset_backward_is_reported(tmp_path: Path) -> None:
    # Operator reset the branch back: HEAD is an ancestor of snap_head, not a
    # descendant, so the snapshot's work is no longer present. Refuse.
    repo = _init_repo(tmp_path)
    (repo / "b.txt").write_text("second\n")
    _git(repo, "add", "b.txt")
    _git(repo, "commit", "-q", "-m", "second")
    snap_head = _git(repo, "rev-parse", "HEAD")
    snap = _write_snapshot(tmp_path, {"head_sha": snap_head})
    _git(repo, "reset", "--hard", "HEAD~1")
    mismatch = snapshot_head_mismatch(snap, repo)
    assert mismatch is not None
    assert mismatch[0] == snap_head


def test_blank_head_sha_skips_check(tmp_path: Path) -> None:
    # A snapshot written while git was unreadable records "": no basis to
    # refuse, resume proceeds.
    repo = _init_repo(tmp_path)
    snap = _write_snapshot(tmp_path, {"head_sha": ""})
    assert snapshot_head_mismatch(snap, repo) is None


def test_pre_head_sha_snapshot_skips_check(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    snap = _write_snapshot(tmp_path, {"version": 1, "messages": []})
    assert snapshot_head_mismatch(snap, repo) is None


def test_corrupt_snapshot_skips_check(tmp_path: Path) -> None:
    # The guard stays quiet on a corrupt or missing file; the resume snapshot
    # load reports it loudly right after.
    repo = _init_repo(tmp_path)
    snap = tmp_path / "loop_state.json"
    snap.write_text("{not json", encoding="utf-8")
    assert snapshot_head_mismatch(snap, repo) is None
    assert snapshot_head_mismatch(tmp_path / "missing.json", repo) is None


def test_non_dict_snapshot_skips_check(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    snap = _write_snapshot(tmp_path, ["not", "a", "dict"])
    assert snapshot_head_mismatch(snap, repo) is None
