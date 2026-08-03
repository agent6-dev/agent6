# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for `agent6 sessions merge` and `agent6 sessions commits`."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from agent6.config.layer import resolved_state_dir
from agent6.sessions.layout import SessionLayout
from agent6.ui.cli import main


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _setup_run(
    tmp_path: Path,
    session_id: str,
    *,
    commits: list[tuple[str, str, str]],
    run_branch: str | None = "<auto>",
) -> str:
    """Init a repo, cut agent6/<session_id> off main with *commits* (name, content,
    message), return to main, and write the run manifest. Returns the base sha."""
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "README.md").write_text("base\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "init")
    base_sha = _git(tmp_path, "rev-parse", "HEAD")
    branch = f"agent6/{session_id}"
    _git(tmp_path, "checkout", "-q", "-b", branch)
    for name, content, msg in commits:
        (tmp_path / name).write_text(content, encoding="utf-8")
        _git(tmp_path, "add", "-A")
        _git(tmp_path, "commit", "-q", "-m", msg)
    _git(tmp_path, "checkout", "-q", "main")
    layout = SessionLayout(state_dir=resolved_state_dir(tmp_path), session_id=session_id)
    layout.ensure()
    recorded_branch = branch if run_branch == "<auto>" else run_branch
    layout.manifest_path.write_text(
        json.dumps(
            {
                "version": 2,
                "session_id": session_id,
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
    rc = main(["sessions", "merge", "run-AAAA11", "--strategy", "squash"])
    assert rc == 0
    assert (tmp_path / "a.txt").exists() and (tmp_path / "b.txt").exists()
    # exactly one new commit on main (the squash), not the two per-step commits
    assert _git(tmp_path, "rev-list", "--count", f"{base}..main") == "1"
    layout = SessionLayout(state_dir=resolved_state_dir(tmp_path), session_id="run-AAAA11")
    m = json.loads(layout.manifest_path.read_text(encoding="utf-8"))
    assert m["merged"]["into"] == "main"
    assert m["merged"]["sha"]


def test_runs_merge_refuses_while_the_worker_is_alive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Merging a LIVE run hijacks the shared checkout: execute_merge switches
    it to the base branch and the still-running worker's next auto-commit then
    lands mid-run WIP directly on the base. A run's tree is clean for the whole
    duration of every provider call, so every other _plan_merge guard passes
    mid-run; the liveness gate is the one that must refuse (matching
    stop/resume/compact). Killing the worker (stale pid) restores the merge."""
    from agent6.sessions.ipc import write_worker_pid

    monkeypatch.chdir(tmp_path)
    base = _setup_run(tmp_path, "run-LIVE11", commits=[("a.txt", "a\n", "agent6 iter 1: add a")])
    layout = SessionLayout(state_dir=resolved_state_dir(tmp_path), session_id="run-LIVE11")
    write_worker_pid(layout.session_dir, os.getpid())  # this test process = a live worker

    rc = main(["sessions", "merge", "run-LIVE11"])
    assert rc == 2
    assert "still live" in capsys.readouterr().err
    assert _git(tmp_path, "rev-list", "--count", f"{base}..main") == "0"  # nothing landed

    write_worker_pid(layout.session_dir, 999_999_999)  # dead pid -> a finished/crashed run merges
    rc = main(["sessions", "merge", "run-LIVE11"])
    assert rc == 0
    assert _git(tmp_path, "rev-list", "--count", f"{base}..main") == "1"


def test_runs_merge_strategy_merge_keeps_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _setup_run(tmp_path, "run-MERG11", commits=[("a.txt", "a\n", "agent6 iter 1: add a")])
    rc = main(["sessions", "merge", "run-MERG11", "--strategy", "merge"])
    assert rc == 0
    assert (tmp_path / "a.txt").exists()  # the merge landed the work on main
    log = _git(tmp_path, "log", "--oneline")
    assert "agent6 iter 1: add a" in log  # --no-ff keeps the per-step commit reachable


def test_runs_merge_squash_honors_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _setup_run(tmp_path, "run-MSG111", commits=[("a.txt", "a\n", "agent6 iter 1: add a")])
    rc = main(["sessions", "merge", "run-MSG111", "--strategy", "squash", "-m", "custom subject"])
    assert rc == 0
    assert _git(tmp_path, "log", "-1", "--format=%s", "main") == "custom subject"


def test_runs_merge_refuses_when_no_branch_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _setup_run(tmp_path, "run-NOBR11", commits=[], run_branch=None)
    rc = main(["sessions", "merge", "run-NOBR11"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "no branch to merge" in err
    assert "branch_per_run was off, so the work already landed on your current branch" in err


def test_runs_merge_refuses_dirty_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _setup_run(tmp_path, "run-DIRT11", commits=[("a.txt", "a\n", "agent6 iter 1: add a")])
    (tmp_path / "wip.txt").write_text("uncommitted\n", encoding="utf-8")
    rc = main(["sessions", "merge", "run-DIRT11"])
    assert rc == 2
    assert "not clean" in capsys.readouterr().err


def test_runs_merge_refuses_unknown_into_without_creating_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _setup_run(tmp_path, "run-INTO11", commits=[("a.txt", "a\n", "agent6 iter 1: add a")])
    rc = main(["sessions", "merge", "run-INTO11", "--into", "nonexistent-branch"])
    assert rc == 2
    assert "does not exist" in capsys.readouterr().err
    branches = _git(tmp_path, "branch", "--format=%(refname:short)")
    assert "nonexistent-branch" not in branches  # a typo must not fabricate a branch


def test_runs_merge_refuses_self_merge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _setup_run(tmp_path, "run-SELF11", commits=[("a.txt", "a\n", "agent6 iter 1: add a")])
    rc = main(["sessions", "merge", "run-SELF11", "--into", "agent6/run-SELF11"])
    assert rc == 2
    assert "run branch itself" in capsys.readouterr().err


def test_runs_merge_restores_original_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _setup_run(tmp_path, "run-REST11", commits=[("a.txt", "a\n", "agent6 iter 1: add a")])
    _git(tmp_path, "checkout", "-q", "-b", "feature")  # user is on a third branch
    rc = main(["sessions", "merge", "run-REST11", "--into", "main"])
    assert rc == 0
    assert _git(tmp_path, "rev-parse", "--abbrev-ref", "HEAD") == "feature"  # restored
    assert "a.txt" in _git(tmp_path, "show", "--stat", "main")  # merge still landed on main


def test_runs_merge_from_the_run_branch_lands_on_the_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A run strands the checkout on agent6/<id> (branch_per_run never switches
    # back), so `sessions merge` is typically invoked FROM the run branch. Restoring
    # to it would leave the user on a squash-dead branch whose tree no longer
    # matches main. They should land on the merge target instead.
    monkeypatch.chdir(tmp_path)
    _setup_run(tmp_path, "run-STRAND1", commits=[("a.txt", "a\n", "agent6 iter 1: add a")])
    _git(tmp_path, "checkout", "-q", "agent6/run-STRAND1")  # stranded on the run branch
    rc = main(["sessions", "merge", "run-STRAND1", "--into", "main"])
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
    rc = main(["sessions", "merge", "run-NOID11", "--strategy", "squash"])
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
    rc = main(["sessions", "commits", "run-COMM11"])
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
    rc = main(["sessions", "merge", "run-EMPTY1"])
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
    rc = main(["sessions", "diff", "run-EMPTY2"])
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
    rc = main(["sessions", "diff", "run-LIVE01"])
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
    rc = main(["sessions", "diff", "run-OTHER1"])
    assert rc == 0
    assert "(no changes)" in capfd.readouterr().out


def test_runs_diff_with_commits_prints_the_patch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _setup_run(tmp_path, "run-DIFF01", commits=[("a.txt", "a\n", "agent6 iter 1: add a")])
    rc = main(["sessions", "diff", "run-DIFF01"])
    assert rc == 0
    out = capfd.readouterr().out
    assert "(no changes)" not in out
    assert "+a" in out  # the real patch still prints


def test_runs_diff_neutralizes_poisoned_diff_external(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    # A checkout with `[diff] external = CMD` in .git/config must not execute
    # CMD on the host when the operator runs `agent6 sessions diff`; the -c
    # hardening overrides force the builtin diff.
    monkeypatch.chdir(tmp_path)
    _setup_run(tmp_path, "run-EVIL01", commits=[("a.txt", "a\n", "agent6 iter 1: add a")])
    marker = tmp_path / "pwned"
    script = tmp_path / "evil.sh"
    script.write_text(f"#!/bin/sh\ntouch {marker}\n", encoding="utf-8")
    script.chmod(0o755)
    _git(tmp_path, "config", "diff.external", str(script))
    rc = main(["sessions", "diff", "run-EVIL01"])
    assert rc == 0
    assert not marker.exists()  # the payload never ran
    assert "+a" in capfd.readouterr().out  # builtin diff still printed the patch


def test_ff_merge_of_a_diverged_base_refuses_with_a_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # `git merge --ff-only` would spew `fatal: Not possible to fast-forward`
    # plus rebase hints with no agent6 framing; the pre-check refuses with the
    # reason and the way out instead.
    monkeypatch.chdir(tmp_path)
    _setup_run(tmp_path, "run-FFDIV1", commits=[("a.txt", "a\n", "agent6 iter 1: add a")])
    # main moves after the branch was cut: ff is now impossible.
    (tmp_path / "moved.txt").write_text("m\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "main moves on")
    rc = main(["sessions", "merge", "run-FFDIV1", "--strategy", "ff"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "fast-forward is impossible" in err
    assert "--strategy merge or squash" in err
    assert "Not possible to fast-forward" not in err  # no raw git spew


def test_ff_merge_lands_when_the_base_has_not_moved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Pins the pre-check's is_ancestor argument order: an unmoved base IS an
    # ancestor of the run branch, so the ff must not be falsely refused.
    monkeypatch.chdir(tmp_path)
    _setup_run(tmp_path, "run-FFOK1", commits=[("a.txt", "a\n", "agent6 iter 1: add a")])
    rc = main(["sessions", "merge", "run-FFOK1", "--strategy", "ff"])
    assert rc == 0
    assert "fast-forward is impossible" not in capsys.readouterr().err
    # main now points at the run branch tip: a true fast-forward.
    assert _git(tmp_path, "rev-parse", "main") == _git(tmp_path, "rev-parse", "agent6/run-FFOK1")


def test_ff_merge_of_an_already_contained_branch_is_a_clean_no_op(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The run branch's commits are already in main (merged earlier) and main
    # has moved on: `git merge --ff-only` says "Already up to date" (rc 0), so
    # the moved-base pre-check must not refuse it.
    monkeypatch.chdir(tmp_path)
    _setup_run(tmp_path, "run-FFIN1", commits=[("a.txt", "a\n", "agent6 iter 1: add a")])
    _git(tmp_path, "merge", "-q", "--ff-only", "agent6/run-FFIN1")  # contain it
    (tmp_path / "moved.txt").write_text("m\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "main moves on")
    before = _git(tmp_path, "rev-parse", "main")
    rc = main(["sessions", "merge", "run-FFIN1", "--strategy", "ff"])
    assert rc == 0
    assert "fast-forward is impossible" not in capsys.readouterr().err
    assert _git(tmp_path, "rev-parse", "main") == before  # no-op, nothing rewound


def test_merging_an_already_merged_run_does_not_claim_a_second_merge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A second `sessions merge` of the same run is a no-op, and says so.

    `git merge --squash` on an up-to-date branch stages nothing and leaves HEAD
    alone, so the merge helpers return the TARGET'S CURRENT HEAD. That was
    printed as the merge sha and stamped into the manifest, so an unrelated
    commit made between the two merges was reported as the run's work and
    overwrote the real merge record -- which `sessions diff` then pointed at
    after a prune.

    Worded "nothing left to merge" rather than "already merged": git also
    stages nothing when the branch's CONTENT arrived by another route, and the
    branch is then not an ancestor, so `prune` would call it unmerged in the
    same minute."""
    monkeypatch.chdir(tmp_path)
    _setup_run(tmp_path, "run-AAAA77", commits=[("a.txt", "a\n", "agent6 iter 1: add a")])
    assert main(["sessions", "merge", "run-AAAA77", "--strategy", "squash"]) == 0
    layout = SessionLayout(state_dir=resolved_state_dir(tmp_path), session_id="run-AAAA77")
    real_sha = json.loads(layout.manifest_path.read_text(encoding="utf-8"))["merged"]["sha"]
    capsys.readouterr()

    # The operator commits something of their own on top.
    (tmp_path / "human.txt").write_text("mine\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "human: totally unrelated")
    unrelated = _git(tmp_path, "rev-parse", "HEAD")

    assert main(["sessions", "merge", "run-AAAA77", "--strategy", "squash"]) == 0
    out = capsys.readouterr().out
    assert "nothing left to merge" in out
    assert unrelated[:12] not in out
    # The real merge record survives; the operator's commit never replaces it.
    stamped = json.loads(layout.manifest_path.read_text(encoding="utf-8"))["merged"]["sha"]
    assert stamped == real_sha


def test_diff_on_a_session_that_cannot_commit_does_not_show_your_own_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A plan has no run branch, and diffing `base..HEAD` for one presented the
    operator's own commits as the plan's work. A plan cannot write to the repo
    at all, so the honest answer is that it made none."""
    monkeypatch.chdir(tmp_path)
    base = _setup_run(tmp_path, "plan-AAA044", commits=[], run_branch=None)
    (tmp_path / "human.txt").write_text("mine\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "human: my own work")
    layout = SessionLayout(state_dir=resolved_state_dir(tmp_path), session_id="plan-AAA044")
    m = json.loads(layout.manifest_path.read_text(encoding="utf-8"))
    m["mode"] = "plan"
    layout.manifest_path.write_text(json.dumps(m) + "\n", encoding="utf-8")
    assert base

    assert main(["sessions", "diff", "plan-AAA044", "--stat"]) == 0
    out = capsys.readouterr().out
    assert "made no commits" in out
    assert "human.txt" not in out


def test_a_parked_run_does_not_claim_the_run_it_was_parked_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A parked run has mode="run" and no branch, so keying only on whether the
    MODE can edit let it fall through to `base..HEAD` -- and parking happens
    because another run holds the checkout, so those commits are that run's."""
    monkeypatch.chdir(tmp_path)
    _setup_run(tmp_path, "run-PARK01", commits=[], run_branch=None)
    (tmp_path / "other.txt").write_text("theirs\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "the other run's commit")
    layout = SessionLayout(state_dir=resolved_state_dir(tmp_path), session_id="run-PARK01")
    m = json.loads(layout.manifest_path.read_text(encoding="utf-8"))
    m["mode"], m["parked_task"] = "run", "do the thing"
    layout.manifest_path.write_text(json.dumps(m) + "\n", encoding="utf-8")

    assert main(["sessions", "diff", "run-PARK01", "--stat"]) == 0
    assert "parked before it started" in capsys.readouterr().out


def test_commits_explains_a_plan_the_same_way_merge_does(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`sessions commits` kept the claim its sibling `merge` was corrected for."""
    monkeypatch.chdir(tmp_path)
    _setup_run(tmp_path, "plan-AAA055", commits=[], run_branch=None)
    layout = SessionLayout(state_dir=resolved_state_dir(tmp_path), session_id="plan-AAA055")
    m = json.loads(layout.manifest_path.read_text(encoding="utf-8"))
    m["mode"] = "plan"
    layout.manifest_path.write_text(json.dumps(m) + "\n", encoding="utf-8")

    assert main(["sessions", "commits", "plan-AAA055"]) == 2
    err = capsys.readouterr().err
    assert "does not write to the repo" in err
    assert "branch_per_run was off?" not in err


def _set_manifest_field(tmp_path: Path, session_id: str, **fields: str) -> None:
    layout = SessionLayout(state_dir=resolved_state_dir(tmp_path), session_id=session_id)
    m = json.loads(layout.manifest_path.read_text(encoding="utf-8"))
    m.update(fields)
    layout.manifest_path.write_text(json.dumps(m) + "\n", encoding="utf-8")


def test_diff_of_a_branch_per_run_off_run_shows_the_work_on_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    """branch_per_run off records no run branch and the run commits onto the
    checked-out branch, so diff (alone among the branch verbs) falls back to
    base..HEAD."""
    monkeypatch.chdir(tmp_path)
    _setup_run(tmp_path, "run-HEADF1", commits=[], run_branch=None)
    _set_manifest_field(tmp_path, "run-HEADF1", mode="run")
    (tmp_path / "work.txt").write_text("w\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "agent6 iter 1: work on HEAD")

    rc = main(["sessions", "diff", "run-HEADF1"])
    assert rc == 0
    out = capfd.readouterr().out
    assert "+w" in out
    assert "made no commits" not in out


def test_commits_of_a_branch_per_run_off_run_names_where_the_work_went(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _setup_run(tmp_path, "run-HEADF2", commits=[], run_branch=None)
    _set_manifest_field(tmp_path, "run-HEADF2", mode="run")

    assert main(["sessions", "commits", "run-HEADF2"]) == 2
    err = capsys.readouterr().err
    assert "no branch to list commits from" in err
    assert "branch_per_run was off, so the work already landed on your current branch" in err


def test_commits_with_a_branch_but_no_base_sha_does_not_blame_branch_per_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A manifest that records a run branch but lost base_sha (agent6 never
    writes that pair) is a base_sha problem; the combined guard called it
    "branch_per_run was off", a branch it plainly recorded."""
    monkeypatch.chdir(tmp_path)
    _setup_run(tmp_path, "run-NOBASE1", commits=[("a.txt", "a\n", "agent6 iter 1: add a")])
    _set_manifest_field(tmp_path, "run-NOBASE1", base_sha="")

    assert main(["sessions", "commits", "run-NOBASE1"]) == 2
    err = capsys.readouterr().err
    assert "no base_sha" in err
    assert "branch_per_run" not in err


def _head_message(repo: Path) -> str:
    return _git(repo, "log", "-1", "--format=%B")


def test_merge_squash_combine_style_uses_gits_own_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """[git.commit.squash].message = combine commits with git's squash message
    (the concatenated per-step log)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(tmp_path / "g"))
    (tmp_path / "g").mkdir()
    (tmp_path / "g" / "config.toml").write_text(
        '[git.commit.squash]\nmessage = "combine"\n', encoding="utf-8"
    )
    _setup_run(tmp_path, "run-CMB111", commits=[("a.txt", "a\n", "agent6 iter 1: add a")])
    assert main(["sessions", "merge", "run-CMB111", "--strategy", "squash"]) == 0
    msg = _head_message(tmp_path)
    assert "Squashed commit of the following" in msg
    assert "agent6 iter 1: add a" in msg


def test_merge_squash_conventional_style_derives_the_subject(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(tmp_path / "g"))
    (tmp_path / "g").mkdir()
    (tmp_path / "g" / "config.toml").write_text(
        '[git.commit.squash]\nmessage = "conventional"\n', encoding="utf-8"
    )
    _setup_run(tmp_path, "run-CNV111", commits=[("a.txt", "a\n", "agent6 iter 1: add a")])
    assert main(["sessions", "merge", "run-CNV111", "--strategy", "squash"]) == 0
    subject = _head_message(tmp_path).splitlines()[0]
    assert subject == "feat: implement the thing"  # an added file, no common scope


def test_merge_squash_model_style_degrades_with_a_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """No provider is reachable in this environment, so the model style must
    fall back to the agent6 message and say so, never fail the merge."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(tmp_path / "g"))
    (tmp_path / "g").mkdir()
    (tmp_path / "g" / "config.toml").write_text(
        '[git.commit.squash]\nmessage = "model"\n', encoding="utf-8"
    )
    _setup_run(tmp_path, "run-MDL111", commits=[("a.txt", "a\n", "agent6 iter 1: add a")])
    assert main(["sessions", "merge", "run-MDL111", "--strategy", "squash"]) == 0
    assert "model squash message failed" in capsys.readouterr().err
    assert _head_message(tmp_path).splitlines()[0] == "implement the thing"


def test_merge_squash_trailer_lands_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT6_CONFIG_HOME", str(tmp_path / "g"))
    (tmp_path / "g").mkdir()
    (tmp_path / "g" / "config.toml").write_text(
        '[git.commit]\ntrailer = "Assisted-by: agent6:{model}"\n', encoding="utf-8"
    )
    _setup_run(tmp_path, "run-TRL111", commits=[("a.txt", "a\n", "agent6 iter 1: add a")])
    assert main(["sessions", "merge", "run-TRL111", "--strategy", "squash"]) == 0
    assert _head_message(tmp_path).count("Assisted-by: agent6:") == 1
