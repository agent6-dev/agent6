# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for `agent6 runs prune`."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from agent6.config_layer import resolved_state_dir
from agent6.graph.storage import RunLayout
from agent6.ui.cli import main


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _branch_exists(repo: Path, name: str) -> bool:
    return bool(_git(repo, "branch", "--list", name))


def _make_branch(repo: Path, run_id: str, fname: str) -> None:
    _git(repo, "checkout", "-q", "-b", f"agent6/{run_id}", "main")
    (repo / fname).write_text("x\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", f"work {run_id}")
    _git(repo, "checkout", "-q", "main")


def _manifest(repo: Path, run_id: str, base: str, *, merged: bool) -> None:
    layout = RunLayout(state_dir=resolved_state_dir(repo), run_id=run_id)
    layout.ensure()
    data = {
        "version": 2,
        "run_id": run_id,
        "base_sha": base,
        "base_branch": "main",
        "run_branch": f"agent6/{run_id}",
        "user_task": "t",
    }
    if merged:
        data["merged_into"] = "main"
        data["merged_sha"] = "0" * 40
    layout.manifest_path.write_text(json.dumps(data) + "\n", encoding="utf-8")


def test_runs_prune_classifies_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "README.md").write_text("base\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "init")
    base = _git(tmp_path, "rev-parse", "HEAD")

    # reachable-merged (--no-ff): git branch -d can delete it
    _make_branch(tmp_path, "reach11", "r.txt")
    _git(tmp_path, "merge", "--no-ff", "-m", "merge reach", "agent6/reach11")
    _manifest(tmp_path, "reach11", base, merged=True)
    # squash-merged: content in main but the branch is unreachable
    _make_branch(tmp_path, "sqush11", "s.txt")
    _git(tmp_path, "merge", "--squash", "agent6/sqush11")
    _git(tmp_path, "commit", "-q", "-m", "squash sqush11")
    _manifest(tmp_path, "sqush11", base, merged=True)
    # genuinely unmerged
    _make_branch(tmp_path, "unmrg11", "u.txt")
    _manifest(tmp_path, "unmrg11", base, merged=False)

    rc = main(["runs", "prune"])
    cap = capsys.readouterr()
    text = cap.out + cap.err
    assert rc == 0
    assert not _branch_exists(tmp_path, "agent6/reach11")  # safely deleted
    assert _branch_exists(tmp_path, "agent6/sqush11")  # kept (unreachable squash)
    assert _branch_exists(tmp_path, "agent6/unmrg11")  # kept (unmerged)
    assert "deleted agent6/reach11" in text
    assert "squash-merged" in text  # sqush11 classification
    assert "NOT merged" in text  # unmrg11 classification
    assert cap.out.index("kept agent6/sqush11") < cap.out.index("[agent6] deleted 1; kept 2")


def test_runs_prune_from_non_base_does_not_mislabel_merge_as_squash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "README.md").write_text("base\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "init")
    base = _git(tmp_path, "rev-parse", "HEAD")
    # merge-merged into main
    _make_branch(tmp_path, "reach22", "r.txt")
    _git(tmp_path, "merge", "--no-ff", "-m", "merge reach", "agent6/reach22")
    _manifest(tmp_path, "reach22", base, merged=True)
    # switch to a branch cut from the ORIGINAL base, so reach22 is unreachable here
    _git(tmp_path, "checkout", "-q", "-b", "feature", base)

    rc = main(["runs", "prune"])
    cap = capsys.readouterr()
    text = cap.out + cap.err
    assert rc == 0
    assert _branch_exists(tmp_path, "agent6/reach22")  # not reachable from feature, so kept
    assert "not reachable from 'feature'" in text  # accurate reason
    assert "squash-merged" not in text  # the merge must NOT be mislabeled as squash


def test_runs_prune_no_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "README.md").write_text("base\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "init")
    rc = main(["runs", "prune"])
    assert rc == 0
    assert "no agent6/* run branches" in capsys.readouterr().out
