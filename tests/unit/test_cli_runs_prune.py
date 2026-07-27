# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for `agent6 runs prune`."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from agent6.config.layer import resolved_state_dir
from agent6.runs.layout import RunLayout
from agent6.ui.cli import main


def _git(repo: Path, *args: str, check: bool = True) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=check, capture_output=True, text=True
    ).stdout.strip()


def _branch_exists(repo: Path, name: str) -> bool:
    return bool(_git(repo, "branch", "--list", name))


def _make_branch(repo: Path, run_id: str, fname: str) -> None:
    _git(repo, "checkout", "-q", "-b", f"agent6/{run_id}", "main")
    (repo / fname).write_text("x\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", f"work {run_id}")
    _git(repo, "checkout", "-q", "main")


def _manifest(repo: Path, run_id: str, base: str, *, merged: bool, merged_tip: str = "") -> None:
    layout = RunLayout(state_dir=resolved_state_dir(repo), run_id=run_id)
    layout.ensure()
    data: dict[str, object] = {
        "version": 2,
        "run_id": run_id,
        "base_sha": base,
        "base_branch": "main",
        "run_branch": f"agent6/{run_id}",
        "user_task": "t",
    }
    if merged:
        tip = merged_tip or _git(repo, "rev-parse", f"agent6/{run_id}", check=False)
        data["merged"] = {"into": "main", "sha": "0" * 40, "tip": tip}
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


def test_runs_commits_and_diff_after_prune_say_where_the_work_went(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A pruned (deleted) run branch: diff/commits must not leak a raw git fatal.
    # The manifest recorded the squash merge, so report it instead.
    monkeypatch.chdir(tmp_path)
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "README.md").write_text("base\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "init")
    base = _git(tmp_path, "rev-parse", "HEAD")
    # Manifest says the run branch existed and was squash-merged, but the branch
    # itself is gone (never created here = pruned).
    _manifest(tmp_path, "gone11", base, merged=True)

    assert main(["runs", "commits", "gone11"]) == 0
    out = capsys.readouterr().out
    assert "was pruned" in out and "squash-merged into main" in out

    assert main(["runs", "diff", "gone11"]) == 0
    assert "was pruned" in capsys.readouterr().out


def test_runs_prune_delete_squashed_removes_only_confirmed_squash_merged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # --delete-squashed force-deletes a manifest-confirmed squash-merged branch
    # (content-safe in the base commit) and prints an undelete hint; an unmerged
    # branch is NEVER force-deleted.
    monkeypatch.chdir(tmp_path)
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "README.md").write_text("base\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "init")
    base = _git(tmp_path, "rev-parse", "HEAD")

    _make_branch(tmp_path, "sqush22", "s.txt")
    sha = _git(tmp_path, "rev-parse", "agent6/sqush22")
    _git(tmp_path, "merge", "--squash", "agent6/sqush22")
    _git(tmp_path, "commit", "-q", "-m", "squash sqush22")
    _manifest(tmp_path, "sqush22", base, merged=True)
    _make_branch(tmp_path, "unmrg22", "u.txt")
    _manifest(tmp_path, "unmrg22", base, merged=False)

    rc = main(["runs", "prune", "--delete-squashed"])
    cap = capsys.readouterr()
    text = cap.out + cap.err
    assert rc == 0
    assert not _branch_exists(tmp_path, "agent6/sqush22")  # force-deleted (content safe)
    assert _branch_exists(tmp_path, "agent6/unmrg22")  # unmerged: never force-deleted
    assert "deleted agent6/sqush22 (squash-merged into main)" in text
    assert f"undelete: git branch agent6/sqush22 {sha[:12]}" in text  # recoverable
    assert "(1 squash-merged)" in text


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


def test_runs_prune_delete_squashed_keeps_a_branch_that_advanced_after_the_merge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The one sanctioned force-delete must prove the CURRENT tip is what was
    merged. A run that is squash-merged and then resumed keeps committing on the
    same branch under a stale merge stamp; force-deleting it destroys commits
    that exist in no other ref (reflog-only recovery)."""
    monkeypatch.chdir(tmp_path)
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "README.md").write_text("base\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "init")
    base = _git(tmp_path, "rev-parse", "HEAD")

    _make_branch(tmp_path, "resmd33", "s.txt")
    merged_tip = _git(tmp_path, "rev-parse", "agent6/resmd33")
    _git(tmp_path, "merge", "--squash", "agent6/resmd33")
    _git(tmp_path, "commit", "-q", "-m", "squash resmd33")
    _manifest(tmp_path, "resmd33", base, merged=True, merged_tip=merged_tip)
    # The operator resumes the run: a new commit lands on the run branch only.
    _git(tmp_path, "checkout", "-q", "agent6/resmd33")
    (tmp_path / "after.txt").write_text("post-merge work\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "agent6 iter 2: post-merge follow-up work")
    after = _git(tmp_path, "rev-parse", "HEAD")
    _git(tmp_path, "checkout", "-q", "main")

    rc = main(["runs", "prune", "--delete-squashed"])
    text = "".join(capsys.readouterr())
    assert rc == 0
    assert _branch_exists(tmp_path, "agent6/resmd33")  # the post-merge commit survives
    assert _git(tmp_path, "rev-parse", "agent6/resmd33") == after
    assert "advanced since the merge" in text


def test_runs_prune_says_why_a_pre_tip_manifest_is_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A run merged before agent6 recorded the merged tip cannot be confirmed,
    so --delete-squashed keeps it. The message must say that and name the manual
    command -- it told the operator to run `runs prune --delete-squashed`, the
    very command that had just skipped the branch."""
    monkeypatch.chdir(tmp_path)
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "README.md").write_text("base\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "init")
    base = _git(tmp_path, "rev-parse", "HEAD")

    _make_branch(tmp_path, "pretip1", "s.txt")
    _git(tmp_path, "merge", "--squash", "agent6/pretip1")
    _git(tmp_path, "commit", "-q", "-m", "squash pretip1")
    # A manifest written before MergeStamp.tip existed: merged, but no tip.
    layout = RunLayout(state_dir=resolved_state_dir(tmp_path), run_id="pretip1")
    layout.ensure()
    layout.manifest_path.write_text(
        json.dumps(
            {
                "version": 2,
                "run_id": "pretip1",
                "base_sha": base,
                "base_branch": "main",
                "run_branch": "agent6/pretip1",
                "user_task": "t",
                "merged": {"into": "main", "sha": "0" * 40},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert main(["runs", "prune", "--delete-squashed"]) == 0
    text = "".join(capsys.readouterr())
    assert _branch_exists(tmp_path, "agent6/pretip1")  # unconfirmed: kept
    assert "no recorded merge tip" in text
    assert "git branch -D agent6/pretip1" in text
    # It must NOT tell the operator to re-run the command that just skipped it.
    assert "remove with: runs prune --delete-squashed" not in text
