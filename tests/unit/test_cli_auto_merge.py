# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for `git.auto_merge` (run.py's _finalize_auto_merge)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from agent6.cli import run as runmod
from agent6.config_layer import load_effective, resolved_state_dir
from agent6.graph.storage import RunLayout


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _setup_run_on_branch(
    tmp_path: Path, run_id: str, *, commits: list[tuple[str, str, str]], run_branch: str | None
) -> str:
    """Init a repo and cut agent6/<run_id> off main with *commits*, leaving the
    checkout ON the run branch (the end-of-run state). Writes the manifest with
    *run_branch* recorded (None to simulate branch_per_run off). Returns base sha."""
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "README.md").write_text("base\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "init")
    base_sha = _git(tmp_path, "rev-parse", "HEAD")
    branch = f"agent6/{run_id}"
    _git(tmp_path, "checkout", "-q", "-b", branch)
    for name, content, msg in commits:
        (tmp_path / name).write_text(content, encoding="utf-8")
        _git(tmp_path, "add", "-A")
        _git(tmp_path, "commit", "-q", "-m", msg)
    layout = RunLayout(state_dir=resolved_state_dir(tmp_path), run_id=run_id)
    layout.ensure()
    layout.manifest_path.write_text(
        json.dumps(
            {
                "version": 2,
                "run_id": run_id,
                "base_sha": base_sha,
                "base_branch": "main",
                "run_branch": run_branch,
                "user_task": "implement the thing",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return base_sha


def test_auto_merge_squashes_and_lands_on_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    base = _setup_run_on_branch(
        tmp_path,
        "run-AM1111",
        commits=[
            ("a.txt", "a\n", "agent6 iter 1: add a"),
            ("b.txt", "b\n", "agent6 iter 2: add b"),
        ],
        run_branch="agent6/run-AM1111",
    )
    cfg = load_effective(tmp_path, None).config
    runmod._finalize_auto_merge(  # pyright: ignore[reportPrivateUsage]
        tmp_path, layout=RunLayout(resolved_state_dir(tmp_path), "run-AM1111"), cfg=cfg
    )
    assert _git(tmp_path, "rev-parse", "--abbrev-ref", "HEAD") == "main"  # ends on base
    assert _git(tmp_path, "rev-list", "--count", f"{base}..main") == "1"  # one squash commit
    m = json.loads(
        (resolved_state_dir(tmp_path) / "runs" / "run-AM1111" / "manifest.json").read_text()
    )
    assert m["merged_into"] == "main"
    assert m.get("merged_sha")


def test_auto_merge_noop_without_run_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    base = _setup_run_on_branch(
        tmp_path,
        "run-AM2222",
        commits=[("a.txt", "a\n", "agent6 iter 1: add a")],
        run_branch=None,  # branch_per_run was off
    )
    cfg = load_effective(tmp_path, None).config
    # On main (no run branch); the helper must no-op without crashing.
    _git(tmp_path, "checkout", "-q", "main")
    runmod._finalize_auto_merge(  # pyright: ignore[reportPrivateUsage]
        tmp_path, layout=RunLayout(resolved_state_dir(tmp_path), "run-AM2222"), cfg=cfg
    )
    assert _git(tmp_path, "rev-list", "--count", f"{base}..main") == "0"  # nothing merged


def test_auto_merge_conflict_keeps_run_branch_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    base = _setup_run_on_branch(
        tmp_path,
        "run-AM3333",
        commits=[("conflict.txt", "from-run\n", "agent6 iter 1: edit")],
        run_branch="agent6/run-AM3333",
    )
    # Make base diverge so the squash conflicts on the same file.
    _git(tmp_path, "checkout", "-q", "main")
    (tmp_path / "conflict.txt").write_text("from-base\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "base edits the same file")
    _git(tmp_path, "checkout", "-q", "agent6/run-AM3333")
    cfg = load_effective(tmp_path, None).config
    runmod._finalize_auto_merge(  # pyright: ignore[reportPrivateUsage]
        tmp_path, layout=RunLayout(resolved_state_dir(tmp_path), "run-AM3333"), cfg=cfg
    )
    err = capsys.readouterr().err
    assert "conflict" in err.lower()
    assert _git(tmp_path, "status", "--porcelain") == ""  # clean tree, no partial merge
    assert _git(tmp_path, "rev-parse", "--abbrev-ref", "HEAD") == "main"  # ends on base, clean
    # the run branch still has its commit
    assert "agent6 iter 1: edit" in _git(tmp_path, "log", "--oneline", "agent6/run-AM3333")
    _ = base


def test_auto_merge_skips_when_base_branch_is_gone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _setup_run_on_branch(
        tmp_path,
        "run-GONE11",
        commits=[("a.txt", "a\n", "agent6 iter 1: add a")],
        run_branch="agent6/run-GONE11",
    )
    # operator deleted the base branch mid-run (we're on the run branch, so -D works)
    _git(tmp_path, "branch", "-D", "main")
    cfg = load_effective(tmp_path, None).config
    runmod._finalize_auto_merge(  # pyright: ignore[reportPrivateUsage]
        tmp_path, layout=RunLayout(resolved_state_dir(tmp_path), "run-GONE11"), cfg=cfg
    )
    assert _git(tmp_path, "branch", "--list", "main") == ""  # base NOT fabricated
    manifest = json.loads(
        (resolved_state_dir(tmp_path) / "runs" / "run-GONE11" / "manifest.json").read_text()
    )
    assert "merged_into" not in manifest  # no phantom merge recorded
    assert "failed" in capsys.readouterr().err.lower()


def test_auto_prune_deletes_reachable_merge_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _setup_run_on_branch(
        tmp_path,
        "run-AP1111",
        commits=[("a.txt", "a\n", "agent6 iter 1: add a")],
        run_branch="agent6/run-AP1111",
    )
    cfg = load_effective(tmp_path, None).config
    git2 = cfg.git.model_copy(
        update={"auto_merge": True, "auto_prune": True, "merge_strategy": "merge"}
    )
    cfg2 = cfg.model_copy(update={"git": git2})
    runmod._finalize_auto_merge(  # pyright: ignore[reportPrivateUsage]
        tmp_path, layout=RunLayout(resolved_state_dir(tmp_path), "run-AP1111"), cfg=cfg2
    )
    assert _git(tmp_path, "branch", "--list", "agent6/run-AP1111") == ""  # pruned (reachable)


def test_auto_prune_keeps_squash_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _setup_run_on_branch(
        tmp_path,
        "run-AP2222",
        commits=[("a.txt", "a\n", "agent6 iter 1: add a")],
        run_branch="agent6/run-AP2222",
    )
    cfg = load_effective(tmp_path, None).config
    git2 = cfg.git.model_copy(
        update={"auto_merge": True, "auto_prune": True, "merge_strategy": "squash"}
    )
    cfg2 = cfg.model_copy(update={"git": git2})
    runmod._finalize_auto_merge(  # pyright: ignore[reportPrivateUsage]
        tmp_path, layout=RunLayout(resolved_state_dir(tmp_path), "run-AP2222"), cfg=cfg2
    )
    assert _git(tmp_path, "branch", "--list", "agent6/run-AP2222")  # kept (squash unreachable)
    assert "git branch -D" in capsys.readouterr().err
