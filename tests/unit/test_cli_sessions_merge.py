# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for `agent6 sessions merge` and `agent6 sessions commits`."""

from __future__ import annotations

import json
import os
import pathlib
import subprocess

import pytest

from agent6 import git_ops, paths
from agent6.sessions import layout as sessions_layout
from agent6.ui.cli import main


def _git(repo: pathlib.Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _setup_run(
    tmp_path: pathlib.Path,
    session_id: str,
    *,
    commits: list[tuple[str, str, str]],
    run_branch: str | None = "<auto>",
) -> str:
    """Init a repo with agent6/<session_id> cut off main and the run manifest written.

    Each commit is a (name, content, message) triple; the checkout returns to main. Returns the base
    sha.
    """
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
    layout = sessions_layout.SessionLayout(
        state_dir=paths.state_dir(tmp_path), session_id=session_id
    )
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


def test_merge_follows_the_chain_when_the_branch_stopped_tracking_it(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The chain is the run's record and the branch a view of it, so a merge follows the chain.

    An operator's commit on the run branch takes it off the chain; every later chain commit lands on
    the chain alone.
    """
    monkeypatch.chdir(tmp_path)
    base = _setup_run(tmp_path, "frozen-branch1", commits=[("a.txt", "one\n", "iter 1")])
    chain = git_ops.chain_ref_for("frozen-branch1")
    _git(tmp_path, "update-ref", chain, "agent6/frozen-branch1")
    # The operator's own commit moves the branch off the chain.
    _git(tmp_path, "checkout", "-q", "agent6/frozen-branch1")
    (tmp_path / "theirs.txt").write_text("the operator's\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "operator work")
    _git(tmp_path, "checkout", "-q", "main")
    # The run keeps committing on the chain, which the branch stopped covering.
    tree = _git(tmp_path, "rev-parse", f"{chain}^{{tree}}")
    later = subprocess.run(
        ["git", "-C", str(tmp_path), "commit-tree", tree, "-p", chain, "-m", "iter 2"],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t"},
    ).stdout.strip()
    _git(tmp_path, "update-ref", chain, later)

    assert main(["sessions", "merge", "frozen-branch1"]) == 0

    out = capsys.readouterr().out
    assert chain in out, out
    assert _git(tmp_path, "log", "--format=%s", "main").splitlines()[0] != "operator work"
    merged = _git(tmp_path, "rev-parse", "main")
    assert _git(tmp_path, "rev-list", "--count", f"{base}..{merged}") == "1"
    assert (tmp_path / "a.txt").name in _git(tmp_path, "show", "--name-only", "--format=", merged)


def test_commits_follows_the_chain_when_the_visible_branch_diverged(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Commits lists the chain when the visible branch diverged, not a pruned branch."""
    monkeypatch.chdir(tmp_path)
    _setup_run(tmp_path, "frozen-commits1", commits=[("a.txt", "one\n", "agent6 iter 1")])
    chain = git_ops.chain_ref_for("frozen-commits1")
    _git(tmp_path, "update-ref", chain, "agent6/frozen-commits1")
    _git(tmp_path, "checkout", "-q", "agent6/frozen-commits1")
    (tmp_path / "theirs.txt").write_text("the operator's\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "operator work")
    _git(tmp_path, "checkout", "-q", "main")
    tree = _git(tmp_path, "rev-parse", f"{chain}^{{tree}}")
    later = _git(tmp_path, "commit-tree", tree, "-p", chain, "-m", "agent6 iter 2")
    _git(tmp_path, "update-ref", chain, later)

    assert main(["sessions", "commits", "frozen-commits1"]) == 0

    captured = capsys.readouterr()
    assert "agent6 iter 1" in captured.out and "agent6 iter 2" in captured.out
    assert "is gone" not in captured.out
    assert chain in captured.err


def test_a_fork_of_a_squash_merged_run_lands_only_its_own_work(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A fork of a squash-merged run merges from A's landed tip, so only its own line lands.

    git relates the squash to nothing, so the commit before A would conflict on the file both sides
    added.
    """
    monkeypatch.chdir(tmp_path)
    guarded = "def f(x):\n    if not x:\n        return 0\n    return 1\n"
    base = _setup_run(tmp_path, "parent-run1", commits=[("f.py", guarded, "guard")])
    parent_tip = _git(tmp_path, "rev-parse", "agent6/parent-run1")
    _git(tmp_path, "update-ref", git_ops.chain_ref_for("parent-run1"), parent_tip)
    assert main(["sessions", "merge", "parent-run1"]) == 0
    # The fork continues the parent's chain past its merged tip.
    _git(tmp_path, "checkout", "-q", "-b", "agent6/fork-run1", "agent6/parent-run1")
    documented = guarded.replace("def f(x):\n", 'def f(x):\n    """Zero for empty."""\n')
    (tmp_path / "f.py").write_text(documented, encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "docstring")
    _git(tmp_path, "checkout", "-q", "main")
    fork_tip = _git(tmp_path, "rev-parse", "agent6/fork-run1")
    _git(tmp_path, "update-ref", git_ops.chain_ref_for("fork-run1"), fork_tip)
    layout = sessions_layout.SessionLayout(
        state_dir=paths.state_dir(tmp_path), session_id="fork-run1"
    )
    layout.ensure()
    layout.manifest_path.write_text(
        json.dumps(
            {
                "version": 2,
                "session_id": "fork-run1",
                "base_sha": base,
                "base_branch": "main",
                "run_branch": "agent6/fork-run1",
                "user_task": "add the docstring",
                "parent_session_id": "parent-run1",
                "forked_from_turn": 3,
                "forked_from_sha": parent_tip,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert main(["sessions", "merge", "fork-run1"]) == 0, capsys.readouterr().out

    landed = _git(tmp_path, "show", "main:f.py")
    assert '"""Zero for empty."""' in landed and "return 0" in landed
    assert _git(tmp_path, "rev-list", "--count", f"{base}..main") == "2"


def test_runs_merge_squash_is_one_commit_and_records_manifest(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
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
    layout = sessions_layout.SessionLayout(
        state_dir=paths.state_dir(tmp_path), session_id="run-AAAA11"
    )
    m = json.loads(layout.manifest_path.read_text(encoding="utf-8"))
    assert m["merged"]["into"] == "main"
    assert m["merged"]["sha"]


def test_runs_merge_refuses_while_the_worker_is_alive(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`sessions merge` refuses while the worker is alive; a stale pid restores the merge.

    A merge switches the shared checkout to the base branch, where the worker's next auto-commit
    would land mid-run WIP; every other guard passes on a run's clean tree.
    """
    from agent6.sessions import ipc

    monkeypatch.chdir(tmp_path)
    base = _setup_run(tmp_path, "run-LIVE11", commits=[("a.txt", "a\n", "agent6 iter 1: add a")])
    layout = sessions_layout.SessionLayout(
        state_dir=paths.state_dir(tmp_path), session_id="run-LIVE11"
    )
    ipc.write_worker_pid(layout.session_dir, os.getpid())  # this test process = a live worker

    rc = main(["sessions", "merge", "run-LIVE11"])
    assert rc == 2
    assert "still live" in capsys.readouterr().err
    assert _git(tmp_path, "rev-list", "--count", f"{base}..main") == "0"  # nothing landed

    ipc.write_worker_pid(
        layout.session_dir, 999_999_999
    )  # dead pid -> a finished/crashed run merges
    rc = main(["sessions", "merge", "run-LIVE11"])
    assert rc == 0
    assert _git(tmp_path, "rev-list", "--count", f"{base}..main") == "1"


def test_runs_merge_strategy_merge_keeps_history(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _setup_run(tmp_path, "run-MERG11", commits=[("a.txt", "a\n", "agent6 iter 1: add a")])
    rc = main(["sessions", "merge", "run-MERG11", "--strategy", "merge"])
    assert rc == 0
    assert (tmp_path / "a.txt").exists()  # the merge landed the work on main
    log = _git(tmp_path, "log", "--oneline")
    assert "agent6 iter 1: add a" in log  # --no-ff keeps the per-step commit reachable


def test_runs_merge_squash_honors_message(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _setup_run(tmp_path, "run-MSG111", commits=[("a.txt", "a\n", "agent6 iter 1: add a")])
    rc = main(["sessions", "merge", "run-MSG111", "--strategy", "squash", "-m", "custom subject"])
    assert rc == 0
    assert _git(tmp_path, "log", "-1", "--format=%s", "main") == "custom subject"


def test_runs_merge_refuses_when_no_branch_recorded(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _setup_run(tmp_path, "run-NOBR11", commits=[], run_branch=None)
    rc = main(["sessions", "merge", "run-NOBR11"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "no branch to merge" in err
    assert "this run recorded no commits" in err


def test_runs_merge_lands_over_a_worktree_carrying_the_run(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The merge is ref plumbing, so it lands over a worktree holding the run's uncommitted work."""
    monkeypatch.chdir(tmp_path)
    base = _setup_run(tmp_path, "run-DIRT11", commits=[("a.txt", "a\n", "agent6 iter 1: add a")])
    (tmp_path / "a.txt").write_text("a\n", encoding="utf-8")  # the run's work, uncommitted
    (tmp_path / "wip.txt").write_text("operator wip\n", encoding="utf-8")
    rc = main(["sessions", "merge", "run-DIRT11"])
    assert rc == 0
    assert _git(tmp_path, "rev-list", "--count", f"{base}..main") == "1"
    assert (tmp_path / "wip.txt").read_text(encoding="utf-8") == "operator wip\n"
    # Only the operator's own file remains as dirt; the landed work is clean.
    assert _git(tmp_path, "status", "--porcelain").split() == ["??", "wip.txt"]


def test_a_conflicting_merge_prints_a_by_hand_command_that_runs_from_this_checkout(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The by-hand command for a conflict runs from the checkout as it stands.

    It stashes first, since git refuses `merge` over modified tracked files, and moves a checkout on
    another branch to the target.
    """
    monkeypatch.chdir(tmp_path)
    _setup_run(tmp_path, "run-CONF11", commits=[("README.md", "theirs\n", "agent6 iter 1")])
    (tmp_path / "README.md").write_text("ours\n", encoding="utf-8")
    _git(tmp_path, "commit", "-qam", "main moved")
    (tmp_path / "README.md").write_text("theirs\n", encoding="utf-8")  # the run's work, uncommitted

    assert main(["sessions", "merge", "run-CONF11"]) == 1
    err = capsys.readouterr().err
    assert "CONFLICT" in err and "README.md" in err
    assert "    git stash && git merge --squash agent6/run-CONF11" in err

    _git(tmp_path, "checkout", "-q", "--", "README.md")
    _git(tmp_path, "checkout", "-q", "-b", "elsewhere")
    assert main(["sessions", "merge", "run-CONF11"]) == 1
    assert (
        "    git checkout main && git merge --squash agent6/run-CONF11" in capsys.readouterr().err
    )


def test_runs_merge_refuses_unknown_into_without_creating_it(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _setup_run(tmp_path, "run-INTO11", commits=[("a.txt", "a\n", "agent6 iter 1: add a")])
    rc = main(["sessions", "merge", "run-INTO11", "--into", "nonexistent-branch"])
    assert rc == 2
    assert "does not exist" in capsys.readouterr().err
    branches = _git(tmp_path, "branch", "--format=%(refname:short)")
    assert "nonexistent-branch" not in branches  # a typo must not fabricate a branch


def test_runs_merge_refuses_self_merge(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _setup_run(tmp_path, "run-SELF11", commits=[("a.txt", "a\n", "agent6 iter 1: add a")])
    rc = main(["sessions", "merge", "run-SELF11", "--into", "agent6/run-SELF11"])
    assert rc == 2
    assert "run branch itself" in capsys.readouterr().err


def test_runs_merge_restores_original_checkout(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _setup_run(tmp_path, "run-REST11", commits=[("a.txt", "a\n", "agent6 iter 1: add a")])
    _git(tmp_path, "checkout", "-q", "-b", "feature")  # user is on a third branch
    rc = main(["sessions", "merge", "run-REST11", "--into", "main"])
    assert rc == 0
    assert _git(tmp_path, "rev-parse", "--abbrev-ref", "HEAD") == "feature"  # restored
    assert "a.txt" in _git(tmp_path, "show", "--stat", "main")  # merge still landed on main


def test_runs_merge_never_switches_the_checkout(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The merge is ref plumbing: the operator's checkout stays put while the target gains the work.
    monkeypatch.chdir(tmp_path)
    _setup_run(tmp_path, "run-STRAND1", commits=[("a.txt", "a\n", "agent6 iter 1: add a")])
    _git(tmp_path, "checkout", "-q", "agent6/run-STRAND1")
    rc = main(["sessions", "merge", "run-STRAND1", "--into", "main"])
    assert rc == 0
    assert _git(tmp_path, "rev-parse", "--abbrev-ref", "HEAD") == "agent6/run-STRAND1"
    assert "a.txt" in _git(tmp_path, "show", "--stat", "main")


def test_runs_merge_without_identity_refuses_with_clean_tree(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # No git identity anywhere: isolate from ~/.gitconfig and drop the fixture's local identity.
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
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
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
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A run branch with no commits is not reported as a merge.
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
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _setup_run(tmp_path, "run-EMPTY2", commits=[])
    rc = main(["sessions", "diff", "run-EMPTY2"])
    assert rc == 0
    assert "(no changes)" in capfd.readouterr().out


def test_runs_diff_notes_uncommitted_work_on_the_live_run_branch(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    # A live run's edits are uncommitted, so the diff says the branch is dirty, not "(no changes)".
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
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    # The note fires only when the current branch is the diffed run's; main's dirt is not the run's.
    monkeypatch.chdir(tmp_path)
    _setup_run(tmp_path, "run-OTHER1", commits=[])
    (tmp_path / "unrelated.py").write_text("x\n", encoding="utf-8")  # dirty, but on main
    rc = main(["sessions", "diff", "run-OTHER1"])
    assert rc == 0
    assert "(no changes)" in capfd.readouterr().out


def test_runs_diff_with_commits_prints_the_patch(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _setup_run(tmp_path, "run-DIFF01", commits=[("a.txt", "a\n", "agent6 iter 1: add a")])
    rc = main(["sessions", "diff", "run-DIFF01"])
    assert rc == 0
    out = capfd.readouterr().out
    assert "(no changes)" not in out
    assert "+a" in out  # the real patch still prints


def test_the_repl_diff_keeps_gits_pager_out(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    """The REPL's /diff runs `git --no-pager diff`.

    Git's pager never takes over the prompt loop.
    """
    import subprocess

    from agent6.ui.cli import _repl

    monkeypatch.chdir(tmp_path)
    _setup_run(tmp_path, "run-DIFF02", commits=[("a.txt", "a\n", "agent6 iter 1: add a")])
    seen: list[list[str]] = []
    real_run = subprocess.run

    def _spy(argv: list[str], **kw: object) -> object:
        seen.append(list(argv))
        return real_run(argv, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(subprocess, "run", _spy)
    _repl.repl_run_diff("run-DIFF02")
    assert "+a" in capfd.readouterr().out
    diffs = [a for a in seen if a[0] == "git" and "diff" in a and "--quiet" not in a]
    assert diffs and all(a[1] == "--no-pager" for a in diffs)
    # The plain command keeps the pager (git's own tty rule).
    seen.clear()
    assert main(["sessions", "diff", "run-DIFF02"]) == 0
    capfd.readouterr()
    diffs = [a for a in seen if a[0] == "git" and "diff" in a and "--quiet" not in a]
    assert diffs and all("--no-pager" not in a for a in diffs)


def test_runs_diff_neutralizes_poisoned_diff_external(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    # `[diff] external = CMD` in .git/config never runs on `sessions diff`; the -c hardening wins.
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
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The pre-check refuses with the reason and the way out, not git's fast-forward fatal.
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
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Pins is_ancestor's argument order: an unmoved base is an ancestor of the run branch.
    monkeypatch.chdir(tmp_path)
    _setup_run(tmp_path, "run-FFOK1", commits=[("a.txt", "a\n", "agent6 iter 1: add a")])
    rc = main(["sessions", "merge", "run-FFOK1", "--strategy", "ff"])
    assert rc == 0
    assert "fast-forward is impossible" not in capsys.readouterr().err
    # main now points at the run branch tip: a true fast-forward.
    assert _git(tmp_path, "rev-parse", "main") == _git(tmp_path, "rev-parse", "agent6/run-FFOK1")


def test_ff_merge_of_an_already_contained_branch_is_a_clean_no_op(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Already-merged commits with a moved main: `--ff-only` says up to date; the pre-check agrees.
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
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A second `sessions merge` of the same run is a no-op and says "nothing left to merge".

    `git merge --squash` on an up-to-date branch stages nothing, so the target's current HEAD is not
    a merge sha to print or stamp. "Already merged" would be wrong when the content arrived by
    another route and the branch is not an ancestor.
    """
    monkeypatch.chdir(tmp_path)
    _setup_run(tmp_path, "run-AAAA77", commits=[("a.txt", "a\n", "agent6 iter 1: add a")])
    assert main(["sessions", "merge", "run-AAAA77", "--strategy", "squash"]) == 0
    layout = sessions_layout.SessionLayout(
        state_dir=paths.state_dir(tmp_path), session_id="run-AAAA77"
    )
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


def test_a_merge_that_adds_nothing_still_records_the_run_as_merged(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A merge that adds nothing still records the run as merged.

    The stamp names the all-zero sentinel (the target's own tip could be an operator's commit) and
    the branch tip it covers; a later merge leaves the record alone.
    """
    from agent6.sessions import manifest
    from agent6.viewmodel import summarize_session_dir

    monkeypatch.chdir(tmp_path)
    _setup_run(tmp_path, "run-SAME11", commits=[("a.txt", "a\n", "agent6 iter 1: add a")])
    (tmp_path / "a.txt").write_text("a\n", encoding="utf-8")  # the same edit, by hand
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "the same edit, by hand")
    main_tip = _git(tmp_path, "rev-parse", "main")
    branch_tip = _git(tmp_path, "rev-parse", "agent6/run-SAME11")

    assert main(["sessions", "merge", "run-SAME11"]) == 0
    out = capsys.readouterr().out
    assert "main already has its content" in out and "recorded as merged into main" in out
    assert _git(tmp_path, "rev-parse", "main") == main_tip  # nothing landed
    layout = sessions_layout.SessionLayout(
        state_dir=paths.state_dir(tmp_path), session_id="run-SAME11"
    )
    stamp = json.loads(layout.manifest_path.read_text(encoding="utf-8"))["merged"]
    assert (stamp["into"], stamp["sha"], stamp["tip"]) == (
        "main",
        manifest.NO_MERGE_COMMIT,
        branch_tip,
    )
    row = summarize_session_dir(layout.session_dir, branch_tips=git_ops.run_ref_tips(tmp_path))
    assert row.unmerged is False

    assert main(["sessions", "merge", "run-SAME11"]) == 0
    assert "nothing left to merge" in capsys.readouterr().out
    assert json.loads(layout.manifest_path.read_text(encoding="utf-8"))["merged"] == stamp

    assert main(["sessions", "prune", "--delete-squashed"]) == 0
    capsys.readouterr()
    assert main(["sessions", "commits", "run-SAME11"]) == 0
    out = capsys.readouterr().out
    assert "was pruned; already on main, no merge commit" in out
    assert main_tip[:12] not in out  # the by-hand commit is not the run's


def test_a_noop_merge_over_new_commits_restamps_the_tip_it_covers(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A no-op merge whose branch tip differs from the stamp's re-stamps the tip it covers.

    A no-op over the tip already recorded leaves the record alone.
    """
    from agent6.sessions import manifest
    from agent6.viewmodel import summarize_session_dir

    monkeypatch.chdir(tmp_path)
    _setup_run(tmp_path, "run-RSTP11", commits=[("a.txt", "a\n", "agent6 iter 1: add a")])
    assert main(["sessions", "merge", "run-RSTP11", "--strategy", "squash"]) == 0
    layout = sessions_layout.SessionLayout(
        state_dir=paths.state_dir(tmp_path), session_id="run-RSTP11"
    )
    first = json.loads(layout.manifest_path.read_text(encoding="utf-8"))["merged"]
    _git(tmp_path, "checkout", "-q", "agent6/run-RSTP11")
    (tmp_path / "b.txt").write_text("b\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "agent6 iter 2: add b")
    tip2 = _git(tmp_path, "rev-parse", "HEAD")
    _git(tmp_path, "checkout", "-q", "main")
    (tmp_path / "b.txt").write_text("b\n", encoding="utf-8")  # the same edit, by hand
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "the same edit, by hand")
    main_tip = _git(tmp_path, "rev-parse", "main")
    capsys.readouterr()

    assert main(["sessions", "merge", "run-RSTP11", "--strategy", "squash"]) == 0
    assert "recorded as merged into main" in capsys.readouterr().out
    stamp = json.loads(layout.manifest_path.read_text(encoding="utf-8"))["merged"]
    assert (stamp["into"], stamp["sha"], stamp["tip"]) == ("main", manifest.NO_MERGE_COMMIT, tip2)
    assert stamp["tip"] != first["tip"] and main_tip[:12] not in stamp["sha"]
    row = summarize_session_dir(layout.session_dir, branch_tips=git_ops.run_ref_tips(tmp_path))
    assert row.unmerged is False

    assert main(["sessions", "merge", "run-RSTP11", "--strategy", "squash"]) == 0
    assert "nothing left to merge" in capsys.readouterr().out
    assert json.loads(layout.manifest_path.read_text(encoding="utf-8"))["merged"] == stamp


def test_diff_on_a_session_that_cannot_commit_does_not_show_your_own_work(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Diff on a plan shows no work: it cannot write to the repo.

    `base..HEAD` is the operator's.
    """
    monkeypatch.chdir(tmp_path)
    base = _setup_run(tmp_path, "plan-AAA044", commits=[], run_branch=None)
    (tmp_path / "human.txt").write_text("mine\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "human: my own work")
    layout = sessions_layout.SessionLayout(
        state_dir=paths.state_dir(tmp_path), session_id="plan-AAA044"
    )
    m = json.loads(layout.manifest_path.read_text(encoding="utf-8"))
    m["mode"] = "plan"
    layout.manifest_path.write_text(json.dumps(m) + "\n", encoding="utf-8")
    assert base

    assert main(["sessions", "diff", "plan-AAA044", "--stat"]) == 0
    out = capsys.readouterr().out
    assert "made no commits" in out
    assert "human.txt" not in out


def test_a_parked_run_does_not_claim_the_run_it_was_parked_behind(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A parked run does not claim the commits of the run it was parked behind.

    It has mode run and no branch; `base..HEAD` is the other run's.
    """
    monkeypatch.chdir(tmp_path)
    _setup_run(tmp_path, "run-PARK01", commits=[], run_branch=None)
    (tmp_path / "other.txt").write_text("theirs\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "the other run's commit")
    layout = sessions_layout.SessionLayout(
        state_dir=paths.state_dir(tmp_path), session_id="run-PARK01"
    )
    m = json.loads(layout.manifest_path.read_text(encoding="utf-8"))
    m["mode"], m["parked_task"] = "run", "do the thing"
    layout.manifest_path.write_text(json.dumps(m) + "\n", encoding="utf-8")

    assert main(["sessions", "diff", "run-PARK01", "--stat"]) == 0
    assert "parked before it started" in capsys.readouterr().out


def test_commits_explains_a_plan_the_same_way_merge_does(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`sessions commits` kept the claim its sibling `merge` was corrected for."""
    monkeypatch.chdir(tmp_path)
    _setup_run(tmp_path, "plan-AAA055", commits=[], run_branch=None)
    layout = sessions_layout.SessionLayout(
        state_dir=paths.state_dir(tmp_path), session_id="plan-AAA055"
    )
    m = json.loads(layout.manifest_path.read_text(encoding="utf-8"))
    m["mode"] = "plan"
    layout.manifest_path.write_text(json.dumps(m) + "\n", encoding="utf-8")

    assert main(["sessions", "commits", "plan-AAA055"]) == 2
    err = capsys.readouterr().err
    assert "does not write to the repo" in err
    assert "branch_per_run was off?" not in err


def _set_manifest_field(tmp_path: pathlib.Path, session_id: str, **fields: str) -> None:
    layout = sessions_layout.SessionLayout(
        state_dir=paths.state_dir(tmp_path), session_id=session_id
    )
    m = json.loads(layout.manifest_path.read_text(encoding="utf-8"))
    m.update(fields)
    layout.manifest_path.write_text(json.dumps(m) + "\n", encoding="utf-8")


def test_diff_of_a_branch_per_run_off_run_reads_the_chain(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    """Diff of a branch_per_run-off run reads base..refs/agent6/<id>, never base..HEAD."""
    monkeypatch.chdir(tmp_path)
    _setup_run(tmp_path, "run-HEADF1", commits=[], run_branch=None)
    _set_manifest_field(tmp_path, "run-HEADF1", mode="run")
    (tmp_path / "work.txt").write_text("w\n", encoding="utf-8")
    git_ops.chain_commit(
        tmp_path,
        "agent6 iter 1: work",
        ref=git_ops.chain_ref_for("run-HEADF1"),
        fallback_parent="HEAD",
    )

    rc = main(["sessions", "diff", "run-HEADF1"])
    assert rc == 0
    out = capfd.readouterr().out
    assert "+w" in out
    assert "made no commits" not in out


def test_commits_of_a_branch_per_run_off_run_lists_the_chain_like_diff(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`sessions commits` reads a branchless run's chain ref as `sessions diff` does."""
    monkeypatch.chdir(tmp_path)
    _setup_run(tmp_path, "run-HEADF2", commits=[], run_branch=None)
    _set_manifest_field(tmp_path, "run-HEADF2", mode="run")
    (tmp_path / "work.txt").write_text("w\n", encoding="utf-8")
    git_ops.chain_commit(
        tmp_path,
        "agent6 iter 1: work",
        ref=git_ops.chain_ref_for("run-HEADF2"),
        fallback_parent="HEAD",
    )

    assert main(["sessions", "commits", "run-HEADF2"]) == 0
    captured = capsys.readouterr()
    assert "agent6 iter 1: work" in captured.out
    assert git_ops.chain_ref_for("run-HEADF2") in captured.err


def test_merge_with_a_branch_but_no_base_sha_refuses_before_mutating(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _setup_run(tmp_path, "run-NOBASE2", commits=[("a.txt", "a\n", "agent6 iter 1: add a")])
    _set_manifest_field(tmp_path, "run-NOBASE2", base_sha="")
    before = _git(tmp_path, "rev-parse", "main")

    assert main(["sessions", "merge", "run-NOBASE2"]) == 2
    err = capsys.readouterr().err
    assert "records no base_sha" in err
    assert _git(tmp_path, "rev-parse", "main") == before


def test_commits_with_a_branch_but_no_base_sha_does_not_blame_branch_per_run(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A manifest with a run branch but no base_sha is a base_sha problem.

    The guard does not blame branch_per_run for a branch it recorded.
    """
    monkeypatch.chdir(tmp_path)
    _setup_run(tmp_path, "run-NOBASE1", commits=[("a.txt", "a\n", "agent6 iter 1: add a")])
    _set_manifest_field(tmp_path, "run-NOBASE1", base_sha="")

    assert main(["sessions", "commits", "run-NOBASE1"]) == 2
    err = capsys.readouterr().err
    assert "no base_sha" in err
    assert "branch_per_run" not in err


def _head_message(repo: pathlib.Path) -> str:
    return _git(repo, "log", "-1", "--format=%B")


def test_merge_squash_combine_style_uses_gits_own_message(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """[git.commit.squash].message = combine commits with git's own squash message."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "g"))
    (tmp_path / "g" / "agent6").mkdir(parents=True, exist_ok=True)
    (tmp_path / "g" / "agent6" / "config.toml").write_text(
        '[git.commit.squash]\nmessage = "combine"\n', encoding="utf-8"
    )
    _setup_run(
        tmp_path,
        "run-CMB111",
        commits=[("a.txt", "a\n", "agent6 iter 1: add a\n\nThe reason for a.")],
    )
    assert main(["sessions", "merge", "run-CMB111", "--strategy", "squash"]) == 0
    msg = _head_message(tmp_path)
    assert "Squashed commit of the following" in msg
    assert "agent6 iter 1: add a" in msg
    assert "The reason for a." in msg


def test_merge_squash_conventional_style_derives_the_subject(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "g"))
    (tmp_path / "g" / "agent6").mkdir(parents=True, exist_ok=True)
    (tmp_path / "g" / "agent6" / "config.toml").write_text(
        '[git.commit.squash]\nmessage = "conventional"\n', encoding="utf-8"
    )
    _setup_run(tmp_path, "run-CNV111", commits=[("a.txt", "a\n", "agent6 iter 1: add a")])
    assert main(["sessions", "merge", "run-CNV111", "--strategy", "squash"]) == 0
    subject = _head_message(tmp_path).splitlines()[0]
    assert subject == "feat: implement the thing"  # an added file, no common scope


def test_merge_squash_default_style_writes_the_run_message(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default `agent6` squash style writes the run's task and its condensed steps."""
    monkeypatch.chdir(tmp_path)
    _setup_run(
        tmp_path,
        "run-DEF222",
        commits=[
            ("a.txt", "a\n", "agent6 iter 1: add a"),
            ("b.txt", "b\n", "agent6 iter 2: add b"),
        ],
    )
    assert main(["sessions", "merge", "run-DEF222", "--strategy", "squash"]) == 0
    head = _head_message(tmp_path)
    assert head.splitlines()[0] == "implement the thing"
    assert "add a" in head and "add b" in head


def test_merge_squash_model_style_degrades_on_a_budget_fault(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The model squash style degrades with its reason on every fault its setup can raise.

    auto_merge runs the draft in a finished run's teardown, where a BudgetExceededError must not
    escape.
    """
    from agent6 import budget as agent6_budget
    from agent6.app import merge as merge_mod

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "g"))
    (tmp_path / "g" / "agent6").mkdir(parents=True, exist_ok=True)
    (tmp_path / "g" / "agent6" / "config.toml").write_text(
        '[git.commit.squash]\nmessage = "model"\n', encoding="utf-8"
    )

    def over(*_a: object, **_k: object) -> object:
        raise agent6_budget.BudgetExceededError("the ceiling was crossed on the last call")

    monkeypatch.setattr(merge_mod, "build_role_provider", over)
    _setup_run(tmp_path, "run-BDG111", commits=[("a.txt", "a\n", "agent6 iter 1: add a")])
    assert main(["sessions", "merge", "run-BDG111", "--strategy", "squash"]) == 0
    assert "model squash message failed (the ceiling was crossed" in capsys.readouterr().err
    assert _head_message(tmp_path).splitlines()[0] == "implement the thing"


def test_merge_squash_model_style_degrades_with_a_warning(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """With no provider reachable, the model style falls back to the agent6 message and says so."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "g"))
    (tmp_path / "g" / "agent6").mkdir(parents=True, exist_ok=True)
    (tmp_path / "g" / "agent6" / "config.toml").write_text(
        '[git.commit.squash]\nmessage = "model"\n', encoding="utf-8"
    )
    _setup_run(tmp_path, "run-MDL111", commits=[("a.txt", "a\n", "agent6 iter 1: add a")])
    assert main(["sessions", "merge", "run-MDL111", "--strategy", "squash"]) == 0
    assert "model squash message failed" in capsys.readouterr().err
    assert _head_message(tmp_path).splitlines()[0] == "implement the thing"


def test_merge_squash_trailer_lands_once(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "g"))
    (tmp_path / "g" / "agent6").mkdir(parents=True, exist_ok=True)
    (tmp_path / "g" / "agent6" / "config.toml").write_text(
        '[git.commit]\ntrailer = "Assisted-by: agent6:{model}"\n', encoding="utf-8"
    )
    _setup_run(tmp_path, "run-TRL111", commits=[("a.txt", "a\n", "agent6 iter 1: add a")])
    assert main(["sessions", "merge", "run-TRL111", "--strategy", "squash"]) == 0
    assert _head_message(tmp_path).count("Assisted-by: agent6:") == 1


def test_merge_squash_trailer_names_every_code_writer(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The squash trailer names the journal's worker models in first-seen order.

    A message-writing role never appears.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "g"))
    (tmp_path / "g" / "agent6").mkdir(parents=True, exist_ok=True)
    (tmp_path / "g" / "agent6" / "config.toml").write_text(
        '[git.commit]\ntrailer = "Assisted-by: agent6:{model}"\n', encoding="utf-8"
    )
    _setup_run(tmp_path, "run-TRL222", commits=[("a.txt", "a\n", "agent6 iter 1: add a")])
    layout = sessions_layout.SessionLayout(
        state_dir=paths.state_dir(tmp_path), session_id="run-TRL222"
    )
    layout.logs_path.write_text(
        "".join(
            json.dumps(e) + "\n"
            for e in (
                {"type": "role.call", "role": "worker", "model": "m-one"},
                {"type": "role.call", "role": "reviewer", "model": "m-rev"},
                {"type": "role.call", "role": "worker", "model": "m-two"},
                {"type": "role.call", "role": "worker", "model": "m-one"},
            )
        ),
        encoding="utf-8",
    )
    assert main(["sessions", "merge", "run-TRL222", "--strategy", "squash"]) == 0
    head = _head_message(tmp_path)
    assert "Assisted-by: agent6:m-one, m-two" in head
    assert "m-rev" not in head


def test_model_squash_message_spends_the_runs_budget_and_reaches_the_log(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inside a run the squash draft spends the run's tracker and is instrumented.

    `sessions merge` passes neither and keeps its per-invocation ceiling.
    """
    from typing import Any

    from agent6 import budget as agent6_budget
    from agent6.app import merge as merge_mod
    from agent6.config import Config
    from agent6.providers import types

    seen: dict[str, Any] = {}

    class _Prov:
        def call(self, **_k: Any) -> types.ProviderResponse:
            return types.ProviderResponse(
                text="feat: x\n\nbody",
                tool_uses=(),
                stop_reason="end_turn",
                input_tokens=10,
                output_tokens=5,
                cache_read_tokens=0,
                cache_creation_tokens=0,
            )

    def _brp(_cfg: Any, _role: str, *, budget: Any, **_k: Any) -> Any:
        seen["budget"] = budget
        return _Prov()

    class _Sink:
        def __init__(self) -> None:
            self.types: list[str] = []

        def emit(self, event_type: str, **_f: Any) -> None:
            self.types.append(event_type)

    def _no_files(*_a: Any) -> list[tuple[str, str]]:
        return []

    monkeypatch.setattr(merge_mod, "build_role_provider", _brp)
    monkeypatch.setattr(merge_mod, "range_name_status", _no_files)
    sink = _Sink()
    tracker = agent6_budget.BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1)
    msg = merge_mod._model_squash_message(  # pyright: ignore[reportPrivateUsage]
        tmp_path,
        Config(),
        (),
        base_sha="0" * 40,
        run_branch="agent6/x",
        task="t",
        transcript_dir=tmp_path,
        budget=tracker,
        events=sink,  # pyright: ignore[reportArgumentType]
    )
    assert msg == "feat: x\n\nbody"
    assert seen["budget"] is tracker, "the squash call minted its own budget"
    assert "budget.update" in sink.types, "the squash spend never reached the log"


def test_merge_adopts_an_orphaned_fanout_lane(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`sessions merge` adopts an orphaned fan-out lane from its clone and lands it like any run.

    A coordinator death leaves a finished lane's branch only in its clone; `sessions prune` then
    sweeps the clone dir.
    """
    monkeypatch.chdir(tmp_path)
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "README.md").write_text("base\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "init")
    base_sha = _git(tmp_path, "rev-parse", "HEAD")

    # The lane clone as the spawner leaves it: workdir cache / repo-id / fanout / lane-1.
    workdir = tmp_path / "cache" / "parallel"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    (tmp_path / "cfg" / "agent6").mkdir(parents=True, exist_ok=True)
    (tmp_path / "cfg" / "agent6" / "config.toml").write_text(
        f'[parallel]\nworkdir = "{workdir}"\n', encoding="utf-8"
    )
    clone = workdir / paths.repo_id(tmp_path) / "fan" / "lane-1"
    clone.parent.mkdir(parents=True)
    _git(tmp_path, "clone", "-q", str(tmp_path), str(clone))
    _git(clone, "config", "user.email", "t@t")
    _git(clone, "config", "user.name", "t")
    _git(clone, "checkout", "-q", "-b", "agent6/fan-l1")
    (clone / "work.txt").write_text("lane work\n", encoding="utf-8")
    _git(clone, "add", "-A")
    _git(clone, "commit", "-q", "-m", "agent6 iter 1: work")

    # The lane's session dir in its bucket plus the origin's live-view symlink, lineage stamped.
    lane_state = tmp_path / "lane-state" / "sessions" / "runs" / "fan-l1"
    lane_state.mkdir(parents=True)
    (lane_state / "manifest.json").write_text(
        json.dumps(
            {
                "version": 2,
                "session_id": "fan-l1",
                "mode": "run",
                "user_task": "t",
                "base_sha": base_sha,
                "base_branch": "main",
                "run_branch": "agent6/fan-l1",
                "parallel": {"group": "fan", "lane": 1, "coordinator": "fan"},
            }
        ),
        encoding="utf-8",
    )
    (lane_state / "logs.jsonl").write_text(
        json.dumps({"type": "session.end", "reason": "finish_session", "all_passed": False}) + "\n",
        encoding="utf-8",
    )
    origin_runs = paths.state_dir(tmp_path) / "sessions" / "runs"
    origin_runs.mkdir(parents=True)
    (origin_runs / "fan-l1").symlink_to(lane_state)

    assert main(["sessions", "merge", "fan-l1", "--strategy", "squash"]) == 0
    out = capsys.readouterr().out
    assert "imported orphaned lane branch agent6/fan-l1" in out
    assert (tmp_path / "work.txt").read_text(encoding="utf-8") == "lane work\n"
    assert not (origin_runs / "fan-l1").is_symlink()  # the real dir replaced it

    # Every lane commit is now in the origin: prune sweeps the fan-out dir.
    assert main(["sessions", "prune", "--delete-squashed"]) == 0
    out = capsys.readouterr().out
    assert "fan-out clones: swept 1" in out
    assert not (workdir / "fan").exists()


def test_diff_explains_an_ask_the_same_way_merge_does(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`sessions diff` on an ask says an ask writes nothing, as merge does."""
    monkeypatch.chdir(tmp_path)
    _setup_run(tmp_path, "ask-AAA056", commits=[], run_branch=None)
    layout = sessions_layout.SessionLayout(
        state_dir=paths.state_dir(tmp_path), session_id="ask-AAA056"
    )
    m = json.loads(layout.manifest_path.read_text(encoding="utf-8"))
    m["mode"], m["base_sha"] = "ask", ""
    layout.manifest_path.write_text(json.dumps(m) + "\n", encoding="utf-8")

    assert main(["sessions", "diff", "ask-AAA056"]) == 0
    captured = capsys.readouterr()
    assert "an ask does not write to the repo" in captured.out
    assert "base_sha" not in captured.err


def test_a_resumed_run_merges_again_from_its_own_landed_tip(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A resumed run merges again from its own landed tip.

    The second execution's commits sit on the chain above a tip git relates to nothing on the base.
    """
    monkeypatch.chdir(tmp_path)
    guarded = "def f(x):\n    if not x:\n        return 0\n    return 1\n"
    base = _setup_run(tmp_path, "resume-run1", commits=[("f.py", guarded, "guard")])
    _git(tmp_path, "update-ref", git_ops.chain_ref_for("resume-run1"), "agent6/resume-run1")
    assert main(["sessions", "merge", "resume-run1"]) == 0
    # The resumed execution keeps committing on the same branch and chain.
    _git(tmp_path, "checkout", "-q", "agent6/resume-run1")
    documented = guarded.replace("def f(x):\n", 'def f(x):\n    """Zero for empty."""\n')
    (tmp_path / "f.py").write_text(documented, encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "docstring")
    _git(tmp_path, "checkout", "-q", "main")
    _git(tmp_path, "update-ref", git_ops.chain_ref_for("resume-run1"), "agent6/resume-run1")

    assert main(["sessions", "merge", "resume-run1"]) == 0, capsys.readouterr().out

    landed = _git(tmp_path, "show", "main:f.py")
    assert '"""Zero for empty."""' in landed and "return 0" in landed
    assert _git(tmp_path, "rev-list", "--count", f"{base}..main") == "2"


def test_the_branch_verbs_refuse_a_fan_out_coordinator_by_name(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The branch verbs refuse a fan-out coordinator by name: its lanes hold the work."""
    from agent6.ui.cli import sessions_cmds

    monkeypatch.chdir(tmp_path)
    layout = sessions_layout.SessionLayout(state_dir=paths.state_dir(tmp_path), session_id="fan")
    layout.ensure()
    layout.manifest_path.write_text(
        json.dumps(
            {"version": 3, "session_id": "fan", "mode": "run", "fanout": {"lanes": 2, "spec": "2"}}
        ),
        encoding="utf-8",
    )
    for bare in ("", "fan"):
        assert sessions_cmds._resolve_session_manifest(tmp_path, bare) == 2
        err = capsys.readouterr().err
        assert "fan is a fan-out" in err and "sessions show fan" in err


def test_diff_reads_the_chain_ref_when_the_branch_is_gone(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    """Diff reads the chain ref when the branch is gone, as merge does."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.chdir(tmp_path)
    _setup_run(tmp_path, "run-GONE01", commits=[("a.py", "x = 1\n", "step 1")])
    tip = _git(tmp_path, "rev-parse", "agent6/run-GONE01")
    _git(tmp_path, "update-ref", git_ops.chain_ref_for("run-GONE01"), tip)
    _git(tmp_path, "branch", "-D", "agent6/run-GONE01")

    assert main(["sessions", "diff", "run-GONE01"]) == 0
    out = capfd.readouterr().out
    assert "+x = 1" in out
    assert "is gone" not in out
