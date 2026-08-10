# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the resume head guard (`snapshot_head_mismatch`) against a real git repo.

The guard is what makes `agent6 resume` refuse when the run's chain ref
(`refs/agent6/<id>`) moved OFF the line recorded by the last `loop_state.json`
write. It reads the snapshot's `head_sha` field directly (best-effort) and
compares it to the chain tip; the operator's checkout is never read or touched.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from agent6.app.resume import snapshot_head_mismatch

_REF = "refs/agent6/r1"


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


def _commit(repo: Path, name: str, content: str, msg: str) -> str:
    (repo / name).write_text(content)
    _git(repo, "add", name)
    _git(repo, "commit", "-q", "-m", msg)
    return _git(repo, "rev-parse", "HEAD")


def _write_snapshot(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "loop_state.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_aligned_chain_tip_passes(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    head = _git(repo, "rev-parse", "HEAD")
    _git(repo, "update-ref", _REF, head)
    snap = _write_snapshot(tmp_path, {"head_sha": head})
    assert snapshot_head_mismatch(snap, repo, chain_ref=_REF) is None


def test_own_forward_commit_is_allowed(tmp_path: Path) -> None:
    # The run's own per-step commit advances the chain forward from the
    # snapshot on the same line (a kill after the commit but before the next
    # snapshot refresh). The tip descends from snap_head, so resume proceeds.
    repo = _init_repo(tmp_path)
    old_head = _git(repo, "rev-parse", "HEAD")
    snap = _write_snapshot(tmp_path, {"head_sha": old_head})
    tip = _commit(repo, "b.txt", "new\n", "the run's own step commit")
    _git(repo, "update-ref", _REF, tip)
    assert snapshot_head_mismatch(snap, repo, chain_ref=_REF) is None


def test_replaced_chain_is_reported(tmp_path: Path) -> None:
    # The ref points at a commit that does NOT descend from the snapshot head
    # (someone rewrote or replaced the chain): genuine divergence, refuse.
    repo = _init_repo(tmp_path)
    old_head = _git(repo, "rev-parse", "HEAD")
    snap = _write_snapshot(tmp_path, {"head_sha": old_head})
    (repo / "a.txt").write_text("rewritten\n")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-q", "--amend", "-m", "amended init")
    other_line = _git(repo, "rev-parse", "HEAD")
    _git(repo, "update-ref", _REF, other_line)
    assert snapshot_head_mismatch(snap, repo, chain_ref=_REF) == (old_head, other_line)


def test_reset_backward_is_reported(tmp_path: Path) -> None:
    # The ref was moved back to an ancestor: the snapshot's work is no longer
    # on the chain. Refuse.
    repo = _init_repo(tmp_path)
    base = _git(repo, "rev-parse", "HEAD")
    snap_head = _commit(repo, "b.txt", "second\n", "second")
    snap = _write_snapshot(tmp_path, {"head_sha": snap_head})
    _git(repo, "update-ref", _REF, base)
    mismatch = snapshot_head_mismatch(snap, repo, chain_ref=_REF)
    assert mismatch == (snap_head, base)


def test_missing_ref_skips_check(tmp_path: Path) -> None:
    # An unborn chain (the run never committed, or a pre-chain session):
    # resume continues from the snapshot head itself, nothing to compare.
    repo = _init_repo(tmp_path)
    snap = _write_snapshot(tmp_path, {"head_sha": _git(repo, "rev-parse", "HEAD")})
    assert snapshot_head_mismatch(snap, repo, chain_ref="refs/agent6/gone") is None


def test_blank_head_sha_skips_check(tmp_path: Path) -> None:
    # A snapshot written while git was unreadable records "": no basis to
    # refuse, resume proceeds.
    repo = _init_repo(tmp_path)
    snap = _write_snapshot(tmp_path, {"head_sha": ""})
    assert snapshot_head_mismatch(snap, repo, chain_ref=_REF) is None


def test_pre_head_sha_snapshot_skips_check(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    snap = _write_snapshot(tmp_path, {"version": 1, "messages": []})
    assert snapshot_head_mismatch(snap, repo, chain_ref=_REF) is None


def test_corrupt_snapshot_skips_check(tmp_path: Path) -> None:
    # The guard stays quiet on a corrupt or missing file; the resume snapshot
    # load reports it loudly right after.
    repo = _init_repo(tmp_path)
    snap = tmp_path / "loop_state.json"
    snap.write_text("{not json", encoding="utf-8")
    assert snapshot_head_mismatch(snap, repo, chain_ref=_REF) is None
    assert snapshot_head_mismatch(tmp_path / "missing.json", repo, chain_ref=_REF) is None


def test_non_dict_snapshot_skips_check(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    snap = _write_snapshot(tmp_path, ["not", "a", "dict"])
    assert snapshot_head_mismatch(snap, repo, chain_ref=_REF) is None


def test_guard_never_reads_the_checkout(tmp_path: Path) -> None:
    """Divergence is detected from the ref alone while HEAD sits anywhere: the
    operator's checkout is on a rewritten main, the chain is aligned, and the
    guard still passes (an implementation comparing HEAD would refuse)."""
    repo = _init_repo(tmp_path)
    base = _git(repo, "rev-parse", "HEAD")
    _git(repo, "update-ref", _REF, base)
    (repo / "a.txt").write_text("moved on\n")
    _git(repo, "commit", "-aqm", "advance main")
    (repo / "a.txt").write_text("rewritten\n")
    _git(repo, "commit", "-aqm", "rewrite main", "--amend")  # HEAD now diverged
    snap = _write_snapshot(tmp_path, {"head_sha": base})
    assert snapshot_head_mismatch(snap, repo, chain_ref=_REF) is None
    assert _git(repo, "rev-parse", "--abbrev-ref", "HEAD") == "main"
