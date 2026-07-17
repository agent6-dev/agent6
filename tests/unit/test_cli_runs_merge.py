# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for `agent6 runs merge` and `agent6 runs commits`."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from agent6.config.layer import resolved_state_dir
from agent6.runs.layout import RunLayout
from agent6.ui.cli import main


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _setup_run(
    tmp_path: Path,
    run_id: str,
    *,
    commits: list[tuple[str, str, str]],
    run_branch: str | None = "<auto>",
) -> str:
    """Init a repo, cut agent6/<run_id> off main with *commits* (name, content,
    message), return to main, and write the run manifest. Returns the base sha."""
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
    _git(tmp_path, "checkout", "-q", "main")
    layout = RunLayout(state_dir=resolved_state_dir(tmp_path), run_id=run_id)
    layout.ensure()
    recorded_branch = branch if run_branch == "<auto>" else run_branch
    layout.manifest_path.write_text(
        json.dumps(
            {
                "version": 2,
                "run_id": run_id,
                "base_sha": base_sha,
                "base_branch": "main",
                "run_branch": recorded_branch,
                "user_task": "implement the thing",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return base_sha


def test_runs_merge_squash_is_one_commit_and_records_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    base = _setup_run(
        tmp_path,
        "run-AAAA11",
        commits=[
            ("a.txt", "a\n", "agent6 iter 1: add a"),
            ("b.txt", "b\n", "agent6 iter 2: add b"),
        ],
    )
    rc = main(["runs", "merge", "run-AAAA11", "--strategy", "squash"])
    assert rc == 0
    assert (tmp_path / "a.txt").exists() and (tmp_path / "b.txt").exists()
    # exactly one new commit on main (the squash), not the two per-step commits
    assert _git(tmp_path, "rev-list", "--count", f"{base}..main") == "1"
    layout = RunLayout(state_dir=resolved_state_dir(tmp_path), run_id="run-AAAA11")
    m = json.loads(layout.manifest_path.read_text(encoding="utf-8"))
    assert m["merged"]["into"] == "main"
    assert m["merged"]["sha"]


def test_runs_merge_strategy_merge_keeps_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _setup_run(tmp_path, "run-MERG11", commits=[("a.txt", "a\n", "agent6 iter 1: add a")])
    rc = main(["runs", "merge", "run-MERG11", "--strategy", "merge"])
    assert rc == 0
    assert (tmp_path / "a.txt").exists()  # the merge landed the work on main
    log = _git(tmp_path, "log", "--oneline")
    assert "agent6 iter 1: add a" in log  # --no-ff keeps the per-step commit reachable


def test_runs_merge_squash_honors_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _setup_run(tmp_path, "run-MSG111", commits=[("a.txt", "a\n", "agent6 iter 1: add a")])
    rc = main(["runs", "merge", "run-MSG111", "--strategy", "squash", "-m", "custom subject"])
    assert rc == 0
    assert _git(tmp_path, "log", "-1", "--format=%s", "main") == "custom subject"


def test_runs_merge_refuses_when_no_branch_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _setup_run(tmp_path, "run-NOBR11", commits=[], run_branch=None)
    rc = main(["runs", "merge", "run-NOBR11"])
    assert rc == 2
    assert "no branch to merge" in capsys.readouterr().err


def test_runs_merge_refuses_dirty_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _setup_run(tmp_path, "run-DIRT11", commits=[("a.txt", "a\n", "agent6 iter 1: add a")])
    (tmp_path / "wip.txt").write_text("uncommitted\n", encoding="utf-8")
    rc = main(["runs", "merge", "run-DIRT11"])
    assert rc == 2
    assert "not clean" in capsys.readouterr().err


def test_runs_merge_refuses_unknown_into_without_creating_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _setup_run(tmp_path, "run-INTO11", commits=[("a.txt", "a\n", "agent6 iter 1: add a")])
    rc = main(["runs", "merge", "run-INTO11", "--into", "nonexistent-branch"])
    assert rc == 2
    assert "does not exist" in capsys.readouterr().err
    branches = _git(tmp_path, "branch", "--format=%(refname:short)")
    assert "nonexistent-branch" not in branches  # a typo must not fabricate a branch


def test_runs_merge_refuses_self_merge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _setup_run(tmp_path, "run-SELF11", commits=[("a.txt", "a\n", "agent6 iter 1: add a")])
    rc = main(["runs", "merge", "run-SELF11", "--into", "agent6/run-SELF11"])
    assert rc == 2
    assert "run branch itself" in capsys.readouterr().err


def test_runs_merge_restores_original_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _setup_run(tmp_path, "run-REST11", commits=[("a.txt", "a\n", "agent6 iter 1: add a")])
    _git(tmp_path, "checkout", "-q", "-b", "feature")  # user is on a third branch
    rc = main(["runs", "merge", "run-REST11", "--into", "main"])
    assert rc == 0
    assert _git(tmp_path, "rev-parse", "--abbrev-ref", "HEAD") == "feature"  # restored
    assert "a.txt" in _git(tmp_path, "show", "--stat", "main")  # merge still landed on main


def test_runs_merge_from_the_run_branch_lands_on_the_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A run strands the checkout on agent6/<id> (branch_per_run never switches
    # back), so `runs merge` is typically invoked FROM the run branch. Restoring
    # to it would leave the user on a squash-dead branch whose tree no longer
    # matches main. They should land on the merge target instead.
    monkeypatch.chdir(tmp_path)
    _setup_run(tmp_path, "run-STRAND1", commits=[("a.txt", "a\n", "agent6 iter 1: add a")])
    _git(tmp_path, "checkout", "-q", "agent6/run-STRAND1")  # stranded on the run branch
    rc = main(["runs", "merge", "run-STRAND1", "--into", "main"])
    assert rc == 0
    assert _git(tmp_path, "rev-parse", "--abbrev-ref", "HEAD") == "main"  # landed on target
    assert "a.txt" in _git(tmp_path, "show", "--stat", "main")


def test_runs_merge_without_identity_refuses_with_clean_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # No git identity anywhere: isolate from the real ~/.gitconfig, then drop the
    # local identity that _setup_run configured for its commits.
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    monkeypatch.chdir(tmp_path)
    _setup_run(tmp_path, "run-NOID11", commits=[("a.txt", "a\n", "agent6 iter 1: add a")])
    _git(tmp_path, "config", "--unset", "user.name")
    _git(tmp_path, "config", "--unset", "user.email")
    rc = main(["runs", "merge", "run-NOID11", "--strategy", "squash"])
    assert rc == 2
    assert "identity not configured" in capsys.readouterr().err.lower()
    assert _git(tmp_path, "status", "--porcelain") == ""  # nothing staged
    assert not (tmp_path / "a.txt").exists()  # nothing leaked onto main


def test_runs_commits_lists_per_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _setup_run(
        tmp_path,
        "run-COMM11",
        commits=[
            ("a.txt", "a\n", "agent6 iter 1: add a"),
            ("b.txt", "b\n", "agent6 iter 2: add b"),
        ],
    )
    rc = main(["runs", "commits", "run-COMM11"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "agent6 iter 1: add a" in out
    assert "agent6 iter 2: add b" in out


def test_runs_merge_zero_commit_branch_is_a_stated_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A run branch with no commits used to print a success line
    # indistinguishable from a real merge.
    monkeypatch.chdir(tmp_path)
    _setup_run(tmp_path, "run-EMPTY1", commits=[])
    head_before = _git(tmp_path, "rev-parse", "main")
    rc = main(["runs", "merge", "run-EMPTY1"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "nothing to merge" in out
    assert "[agent6] merged" not in out
    assert _git(tmp_path, "rev-parse", "main") == head_before  # no commit made


def test_runs_diff_zero_commit_branch_prints_no_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _setup_run(tmp_path, "run-EMPTY2", commits=[])
    rc = main(["runs", "diff", "run-EMPTY2"])
    assert rc == 0
    assert "(no changes)" in capfd.readouterr().out


def test_runs_diff_notes_uncommitted_work_on_the_live_run_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    # A live run mid-work has uncommitted edits on its branch (a run commits
    # only after a verify pass), so base..HEAD shows no committed changes. If
    # that branch is the current checkout and dirty, say so instead of a bare
    # "(no changes)" that reads as "the agent did nothing".
    monkeypatch.chdir(tmp_path)
    _setup_run(tmp_path, "run-LIVE01", commits=[])
    _git(tmp_path, "checkout", "-q", "agent6/run-LIVE01")  # the run's own checkout
    (tmp_path / "work.py").write_text("in progress\n", encoding="utf-8")  # uncommitted
    rc = main(["runs", "diff", "run-LIVE01"])
    assert rc == 0
    out = capfd.readouterr().out
    assert "no committed changes yet" in out
    assert "1 file modified" in out


def test_runs_diff_stays_silent_when_dirty_tree_is_a_different_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    # The note only fires when the CURRENT branch is the diffed run's branch;
    # uncommitted work on main (or another run) is not attributed to this run.
    monkeypatch.chdir(tmp_path)
    _setup_run(tmp_path, "run-OTHER1", commits=[])
    (tmp_path / "unrelated.py").write_text("x\n", encoding="utf-8")  # dirty, but on main
    rc = main(["runs", "diff", "run-OTHER1"])
    assert rc == 0
    assert "(no changes)" in capfd.readouterr().out


def test_runs_diff_with_commits_prints_the_patch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _setup_run(tmp_path, "run-DIFF01", commits=[("a.txt", "a\n", "agent6 iter 1: add a")])
    rc = main(["runs", "diff", "run-DIFF01"])
    assert rc == 0
    out = capfd.readouterr().out
    assert "(no changes)" not in out
    assert "+a" in out  # the real patch still prints


def test_runs_diff_neutralizes_poisoned_diff_external(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    # A checkout with `[diff] external = CMD` in .git/config must not execute
    # CMD on the host when the operator runs `agent6 runs diff`; the -c
    # hardening overrides force the builtin diff.
    monkeypatch.chdir(tmp_path)
    _setup_run(tmp_path, "run-EVIL01", commits=[("a.txt", "a\n", "agent6 iter 1: add a")])
    marker = tmp_path / "pwned"
    script = tmp_path / "evil.sh"
    script.write_text(f"#!/bin/sh\ntouch {marker}\n", encoding="utf-8")
    script.chmod(0o755)
    _git(tmp_path, "config", "diff.external", str(script))
    rc = main(["runs", "diff", "run-EVIL01"])
    assert rc == 0
    assert not marker.exists()  # the payload never ran
    assert "+a" in capfd.readouterr().out  # builtin diff still printed the patch
