# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The end-of-run console headline folds the same session.end `agent6 sessions` reads.

A finish_session over a red or stale verify is all_passed=false, so both read "finished".
"""

from __future__ import annotations

import json
import pathlib

import pytest

from agent6 import budget, git_ops
from agent6.app import finalize as _finalize
from agent6.app import reporter
from agent6.harness import _snapshot
from agent6.sessions import layout as sessions_layout


def _layout(
    tmp_path: pathlib.Path, session_id: str, events: list[dict[str, object]]
) -> sessions_layout.SessionLayout:
    rd = tmp_path / "sessions" / "runs" / session_id
    rd.mkdir(parents=True)
    (rd / "logs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    return sessions_layout.SessionLayout(state_dir=tmp_path, session_id=session_id)


def test_finish_session_over_red_verify_is_not_headlined_passed(
    tmp_path: pathlib.Path, capsys: object
) -> None:
    layout = _layout(
        tmp_path,
        "r1",
        [
            {"type": "session.start", "session_id": "r1", "user_task": "t"},
            {"type": "session.end", "reason": "finish_session", "all_passed": False},
        ],
    )
    result = _snapshot.SessionResult(
        completed=True,
        reason="finish_session",
        summary="all tests pass",
        iterations=3,
        tool_calls=5,
    )
    _finalize.print_session_end(
        result,
        layout=layout,
        cwd=tmp_path,
        budget=budget.BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1),
        console_stream=False,
        reporter=reporter.STDIO_REPORTER,
    )
    out = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "finished" in out
    assert "passed" not in out.split("\n")[1]  # the headline line, not the agent's summary


def test_all_green_finish_is_headlined_passed(tmp_path: pathlib.Path, capsys: object) -> None:
    layout = _layout(
        tmp_path,
        "r2",
        [
            {"type": "session.start", "session_id": "r2", "user_task": "t"},
            {"type": "session.end", "reason": "finish_session", "all_passed": True},
        ],
    )
    result = _snapshot.SessionResult(
        completed=True, reason="finish_session", summary="done", iterations=2, tool_calls=3
    )
    _finalize.print_session_end(
        result,
        layout=layout,
        cwd=tmp_path,
        budget=budget.BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1),
        console_stream=False,
        reporter=reporter.STDIO_REPORTER,
    )
    out = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "passed" in out


def _end_output(
    tmp_path: pathlib.Path,
    session_id: str,
    result: _snapshot.SessionResult,
    capsys: pytest.CaptureFixture[str],
    manifest: dict[str, object] | None = None,
) -> str:
    layout = _layout(
        tmp_path,
        session_id,
        [
            {"type": "session.start", "session_id": session_id, "user_task": "t"},
            {"type": "session.end", "reason": result.reason, "all_passed": False},
        ],
    )
    if manifest is not None:
        (layout.session_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    _finalize.print_session_end(
        result,
        layout=layout,
        cwd=tmp_path,
        budget=budget.BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1),
        console_stream=False,
        reporter=reporter.STDIO_REPORTER,
    )
    return capsys.readouterr().out


def test_the_red_gate_errand_is_only_printed_over_a_real_red(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The "nothing checked it before this run started" line needs a verify that ran and failed.

    An unverified finish says what is missing instead.
    """
    manifest: dict[str, object] = {
        "version": 3,
        "session_id": "r-red",
        "mode": "run",
        "user_task": "t",
        "base_sha": "a" * 40,
        "harness": {"verify_command": ["pytest", "-q"], "verify_origin": "configured"},
    }
    result = _snapshot.SessionResult(
        completed=True,
        reason="finish_session",
        summary="",
        iterations=1,
        tool_calls=1,
        verified="failed",
    )
    out = _end_output(tmp_path, "r-red", result, capsys, manifest)
    assert "the gate is red" in out

    unverified = _snapshot.SessionResult(
        completed=True,
        reason="finish_session",
        summary="",
        iterations=1,
        tool_calls=1,
        verified="unverified",
    )
    out = _end_output(tmp_path, "r-unv", unverified, capsys, manifest)
    assert "the gate is red" not in out
    assert "no verify ran this execution" in out


def test_the_stale_gate_proposal_survives_an_unverified_verdict(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A gate declared stale with no verify observation still prints the proposal."""
    result = _snapshot.SessionResult(
        completed=True,
        reason="gate_stale",
        summary="",
        iterations=1,
        tool_calls=1,
        stale_gate="pytest -q tests/",
        verified="unverified",
    )
    out = _end_output(tmp_path, "r-stale", result, capsys)
    assert "it proposes: pytest -q tests/" in out


def test_the_stale_gate_remedy_is_a_command_that_installs_that_gate(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`harness.verify_command` is argv and takes no shell; a proposed pipeline wraps as `sh -c`."""
    result = _snapshot.SessionResult(
        completed=True,
        reason="gate_stale",
        summary="",
        iterations=1,
        tool_calls=1,
        stale_gate="pytest -q tests/ && ruff check",
        verified="failed",
    )

    out = _end_output(tmp_path, "r-stale2", result, capsys)

    assert '\'["sh", "-c", "pytest -q tests/ && ruff check"]\'' in out


def test_end_banner_does_not_claim_merged_from_a_prior_executions_stamp(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The end block claims a merge only while the branch still points at the stamped tip.

    A resumed run keeps committing under the first execution's merged stamp; the comparison is the
    one `sessions prune` trusts.
    """
    import subprocess as sp

    repo = tmp_path / "repo"
    repo.mkdir()
    sp.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    (repo / "a.txt").write_text("x\n", encoding="utf-8")
    sp.run(["git", "add", "a.txt"], cwd=repo, check=True)
    sp.run(["git", "commit", "-q", "-m", "seed"], cwd=repo, check=True)
    sp.run(["git", "switch", "-qc", "agent6/r-leg2"], cwd=repo, check=True)
    merged_tip = sp.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    (repo / "a.txt").write_text("execution 2 work\n", encoding="utf-8")
    sp.run(["git", "commit", "-qam", "execution 2"], cwd=repo, check=True)
    monkeypatch.chdir(repo)

    layout = _layout(
        tmp_path,
        "r-leg2",
        [
            {"type": "session.start", "session_id": "r-leg2", "user_task": "t"},
            {"type": "session.end", "reason": "finish_session", "all_passed": True},
        ],
    )
    layout.manifest_path.write_text(
        json.dumps(
            {
                "run_branch": "agent6/r-leg2",
                "base_branch": "main",
                "merged": {
                    "into": "main",
                    "sha": "abc123def456",
                    "ts": "2026-01-01T00:00:00Z",
                    "tip": merged_tip,
                },
            }
        ),
        encoding="utf-8",
    )
    result = _snapshot.SessionResult(
        completed=True, reason="finish_session", summary="done", iterations=1, tool_calls=1
    )
    _finalize.print_session_end(
        result,
        layout=layout,
        cwd=repo,
        budget=budget.BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1),
        console_stream=False,
        reporter=reporter.STDIO_REPORTER,
    )
    out = capsys.readouterr().out
    assert "changes merged into" not in out
    assert "merge with:" in out


def test_end_banner_does_not_offer_merge_for_an_auto_merged_branch(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """After auto_merge the footer says it merged, never `sessions merge` on a gone branch."""
    layout = _layout(
        tmp_path,
        "r-merged",
        [
            {"type": "session.start", "session_id": "r-merged", "user_task": "t"},
            {"type": "session.end", "reason": "finish_session", "all_passed": True},
        ],
    )
    layout.manifest_path.write_text(
        json.dumps(
            {
                "run_branch": "agent6/r-merged",
                "base_branch": "main",
                "merged": {"into": "main", "sha": "abc123def456", "ts": "2026-01-01T00:00:00Z"},
            }
        ),
        encoding="utf-8",
    )
    result = _snapshot.SessionResult(
        completed=True, reason="finish_session", summary="done", iterations=1, tool_calls=1
    )
    _finalize.print_session_end(
        result,
        layout=layout,
        cwd=tmp_path,
        budget=budget.BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1),
        console_stream=False,
        reporter=reporter.STDIO_REPORTER,
    )
    out = capsys.readouterr().out
    assert "changes merged into main" in out
    assert "runs merge" not in out


def test_end_banner_does_not_advertise_a_run_branch_that_never_got_a_commit(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run branch that never received a commit is reported as such, not as holding changes."""
    import subprocess as sp

    repo = tmp_path / "repo"
    repo.mkdir()
    sp.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    sp.run(["git", "config", "user.email", "t@t"], cwd=repo, check=True)
    sp.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    (repo / "seed.txt").write_text("seed\n", encoding="utf-8")
    sp.run(["git", "add", "-A"], cwd=repo, check=True)
    sp.run(["git", "commit", "-qm", "seed"], cwd=repo, check=True)
    # The agent's edit is left in the tree; no agent6/miss branch was ever cut.
    (repo / "work.txt").write_text("stranded agent work\n", encoding="utf-8")
    monkeypatch.chdir(repo)

    layout = _layout(
        tmp_path,
        "miss",
        [
            {"type": "session.start", "session_id": "miss", "user_task": "t"},
            {"type": "session.end", "reason": "finish_session", "all_passed": True},
        ],
    )
    layout.manifest_path.write_text(
        json.dumps({"mode": "run", "run_branch": "agent6/miss", "base_branch": "main"}),
        encoding="utf-8",
    )
    result = _snapshot.SessionResult(
        completed=True, reason="finish_session", summary="done", iterations=1, tool_calls=1
    )
    _finalize.print_session_end(
        result,
        layout=layout,
        cwd=repo,
        budget=budget.BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1),
        console_stream=False,
        reporter=reporter.STDIO_REPORTER,
    )
    out = capsys.readouterr().out
    assert "changes are on agent6/miss" not in out
    assert "sessions merge" not in out  # never advertise merge for a missing branch
    assert "no commit on agent6/miss" in out  # the truthful warning
    assert "uncommitted in the working tree" in out


def _seeded_repo(tmp_path: pathlib.Path) -> pathlib.Path:
    import subprocess as sp

    repo = tmp_path / "repo"
    repo.mkdir()
    sp.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    sp.run(["git", "config", "user.email", "t@t"], cwd=repo, check=True)
    sp.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    (repo / "seed.txt").write_text("seed\n", encoding="utf-8")
    sp.run(["git", "add", "-A"], cwd=repo, check=True)
    sp.run(["git", "commit", "-qm", "seed"], cwd=repo, check=True)
    return repo


def _end_footer(
    tmp_path: pathlib.Path,
    repo: pathlib.Path,
    session_id: str,
    manifest: dict[str, object],
    capsys: pytest.CaptureFixture[str],
) -> str:
    layout = _layout(
        tmp_path,
        session_id,
        [
            {"type": "session.start", "session_id": session_id, "user_task": "t"},
            {"type": "session.end", "reason": "finish_session", "all_passed": True},
        ],
    )
    layout.manifest_path.write_text(json.dumps({"mode": "run", **manifest}), encoding="utf-8")
    result = _snapshot.SessionResult(
        completed=True, reason="finish_session", summary="done", iterations=1, tool_calls=1
    )
    _finalize.print_session_end(
        result,
        layout=layout,
        cwd=repo,
        budget=budget.BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1),
        console_stream=False,
        reporter=reporter.STDIO_REPORTER,
    )
    return capsys.readouterr().out


def test_end_banner_of_a_branchless_run_names_its_chain_ref(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `branch_per_run = false` run gets the where-are-my-changes footer, naming its chain ref.

    A branchless run whose commit never landed gets the same WARNING a branchful one does.
    """
    import subprocess as sp

    repo = _seeded_repo(tmp_path)
    monkeypatch.chdir(repo)
    (repo / "work.txt").write_text("agent work\n", encoding="utf-8")
    chain = git_ops.chain_ref_for("nobranch")
    sp.run(["git", "update-ref", chain, "HEAD"], cwd=repo, check=True)
    manifest: dict[str, object] = {
        "session_id": "nobranch",
        "run_branch": None,
        "base_branch": "main",
    }
    out = _end_footer(tmp_path, repo, "nobranch", manifest, capsys)
    assert f"changes are on {chain}" in out
    assert "agent6 sessions merge nobranch" in out and "agent6 sessions diff nobranch" in out
    assert "WARNING" not in out
    # No chain either: the commit never landed, and the footer says so.
    sp.run(["git", "update-ref", "-d", chain], cwd=repo, check=True)
    out = _end_footer(tmp_path, repo, "nobranch2", {**manifest, "session_id": "nobranch2"}, capsys)
    assert "WARNING: the run finished with no commit" in out
    assert "uncommitted in the working tree" in out


def test_end_banner_of_a_merged_branchless_run_says_merged(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A merged branchless run prints the merged footer."""
    import subprocess as sp

    repo = _seeded_repo(tmp_path)
    monkeypatch.chdir(repo)
    chain = git_ops.chain_ref_for("nbmerged")
    sp.run(["git", "update-ref", chain, "HEAD"], cwd=repo, check=True)
    tip = sp.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout
    manifest: dict[str, object] = {
        "session_id": "nbmerged",
        "run_branch": None,
        "base_branch": "main",
        "merged": {"into": "main", "tip": tip.strip(), "sha": tip.strip()},
    }
    out = _end_footer(tmp_path, repo, "nbmerged", manifest, capsys)
    assert "changes merged into main" in out
    assert "sessions merge" not in out


def test_end_banner_of_a_run_that_never_commits_by_design_is_no_warning(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`commit_per_step = false` leaves every edit in the tree by design, not as a failed commit."""
    repo = _seeded_repo(tmp_path)
    monkeypatch.chdir(repo)
    (repo / "work.txt").write_text("agent work\n", encoding="utf-8")
    manifest: dict[str, object] = {
        "session_id": "nocommit",
        "run_branch": "agent6/nocommit",
        "base_branch": "main",
        "policy": {"commit_per_step": False},
    }
    out = _end_footer(tmp_path, repo, "nocommit", manifest, capsys)
    assert "WARNING" not in out and "commit failed" not in out
    assert "commit_per_step" in out and "working tree" in out


def test_end_banner_warns_when_checkout_is_parked_on_the_run_branch(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The checkout is still on the run branch, so the footer names it and how to leave it."""
    import subprocess as sp

    repo = _seeded_repo(tmp_path)
    sp.run(["git", "switch", "-q", "-c", "agent6/r3"], cwd=repo, check=True)
    monkeypatch.chdir(repo)
    manifest: dict[str, object] = {
        "session_id": "r3",
        "run_branch": "agent6/r3",
        "base_branch": "main",
    }
    out = _end_footer(tmp_path, repo, "r3", manifest, capsys)
    assert "changes are on agent6/r3" in out
    assert "you are on agent6/r3" in out
    assert "git switch main" in out


def test_interrupt_end_prints_cost_resume_and_branch_hints(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # A Ctrl-C interrupt printed only "run interrupted": no spend, no resume hint, no branch note.
    layout = _layout(
        tmp_path, "r4", [{"type": "session.start", "session_id": "r4", "user_task": "t"}]
    )
    layout.manifest_path.write_text(
        json.dumps({"run_branch": "agent6/r4", "base_branch": "main"}), encoding="utf-8"
    )

    def _on_run_branch(_p: pathlib.Path) -> git_ops.GitStatus:
        return git_ops.GitStatus(
            branch="agent6/r4", head_sha="x", is_clean=True, untracked_count=0, modified_count=0
        )

    monkeypatch.setattr(git_ops, "status", _on_run_branch)
    _finalize.print_interrupt_end(
        layout=layout,
        cwd=tmp_path,
        budget=budget.BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1),
        reporter=reporter.STDIO_REPORTER,
    )
    out = capsys.readouterr().out
    assert "Token + cost summary" in out  # the budget/cost block
    assert "resume with:  agent6 resume r4" in out
    assert "you are on agent6/r4" in out and "git switch main" in out


def test_provider_error_is_headlined_failed(tmp_path: pathlib.Path, capsys: object) -> None:
    layout = _layout(
        tmp_path,
        "r3",
        [
            {"type": "session.start", "session_id": "r3", "user_task": "t"},
            {"type": "session.end", "reason": "provider_error", "all_passed": False},
        ],
    )
    result = _snapshot.SessionResult(
        completed=False, reason="provider_error", summary="", iterations=1, tool_calls=0
    )
    _finalize.print_session_end(
        result,
        layout=layout,
        cwd=tmp_path,
        budget=budget.BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1),
        console_stream=False,
        reporter=reporter.STDIO_REPORTER,
    )
    out = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "failed" in out and "provider error" in out


def test_end_banner_adds_the_run_total_across_resume_executions(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The tracker's TOTAL line is per-execution; a resumed run's banner states the cumulative spend.
    layout = _layout(
        tmp_path,
        "r7",
        [
            {"type": "session.start", "session_id": "r7", "user_task": "t"},
            {"type": "budget.update", "usd_total": 0.019},
            {"type": "session.end", "reason": "finish_session", "all_passed": True},
            {"type": "loop.resume.start", "iteration": 4},
            {"type": "budget.update", "usd_total": 0.0126},
            {"type": "session.end", "reason": "finish_session", "all_passed": True},
        ],
    )
    result = _snapshot.SessionResult(
        completed=True, reason="finish_session", summary="", iterations=5, tool_calls=2
    )
    _finalize.print_session_end(
        result,
        layout=layout,
        cwd=tmp_path,
        budget=budget.BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1),
        console_stream=False,
        reporter=reporter.STDIO_REPORTER,
    )
    out = capsys.readouterr().out
    assert "RUN TOTAL (all 2 executions): $0.03" in out


def test_end_banner_stays_quiet_on_a_single_execution_run(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    layout = _layout(
        tmp_path,
        "r8",
        [
            {"type": "session.start", "session_id": "r8", "user_task": "t"},
            {"type": "budget.update", "usd_total": 0.01},
            {"type": "session.end", "reason": "finish_session", "all_passed": True},
        ],
    )
    result = _snapshot.SessionResult(
        completed=True, reason="finish_session", summary="", iterations=2, tool_calls=1
    )
    _finalize.print_session_end(
        result,
        layout=layout,
        cwd=tmp_path,
        budget=budget.BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1),
        console_stream=False,
        reporter=reporter.STDIO_REPORTER,
    )
    assert "RUN TOTAL" not in capsys.readouterr().out


def test_finalize_auto_stash_pops_the_run_stash_not_the_latest(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The finalizer restores the stash the run pushed, found by its message, not stash@{0}."""
    import subprocess

    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args: str) -> None:
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "t@t")
    git("config", "user.name", "t")
    (repo / "base.txt").write_text("base\n", encoding="utf-8")
    (repo / "pre.txt").write_text("", encoding="utf-8")
    (repo / "mid.txt").write_text("", encoding="utf-8")
    git("add", "-A")
    git("commit", "-qm", "base")
    (repo / "pre.txt").write_text("pre-run work\n", encoding="utf-8")
    git_ops.stash_tracked_changes(repo, git_ops.auto_stash_message("r1"))
    (repo / "mid.txt").write_text("mid-run work\n", encoding="utf-8")
    git_ops.stash_tracked_changes(repo, "operator stash pushed mid-run")
    base = subprocess.run(
        ["git", "-C", str(repo), "branch", "--show-current"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _finalize.finalize_auto_stash(
        repo,
        base_branch=base,
        run_branch=None,
        auto_pop=True,
        session_id="r1",
        reporter=reporter.STDIO_REPORTER,
    )
    assert "restored your pre-run changes" in capsys.readouterr().err
    assert (repo / "pre.txt").read_text(encoding="utf-8") == "pre-run work\n"
    assert (repo / "mid.txt").read_text(encoding="utf-8") == ""  # the mid-run stash stays a stash


def test_finalize_auto_stash_reports_a_vanished_stash(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A stash the operator popped mid-run is reported; nothing pops what sits at stash@{0}."""
    import subprocess

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
    _finalize.finalize_auto_stash(
        repo,
        base_branch="master",
        run_branch=None,
        auto_pop=True,
        session_id="r1",
        reporter=reporter.STDIO_REPORTER,
    )
    assert "auto-stash not found" in capsys.readouterr().err


def test_finalize_auto_stash_prints_a_failed_bystander_putback(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A restore that raises prints the recovery command and finishes."""
    import subprocess

    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args: str) -> None:
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)

    git("init", "-q", "-b", "main")
    git("config", "user.email", "t@t")
    git("config", "user.name", "t")
    (repo / "base.txt").write_text("base\n", encoding="utf-8")
    git("add", "-A")
    git("commit", "-qm", "base")
    (repo / "base.txt").write_text("pre-run work\n", encoding="utf-8")
    git_ops.stash_tracked_changes(repo, git_ops.auto_stash_message("r1"))

    def raising_restore(cwd: object, entry: object) -> bool:
        raise git_ops.GitError(
            "a stash pushed concurrently ('x') was taken by a raced drop and putting"
            " it back failed; restore it with:\n    git stash store -m 'x' abc123"
        )

    monkeypatch.setattr(git_ops, "restore_stash", raising_restore)
    _finalize.finalize_auto_stash(
        repo,
        base_branch="main",
        run_branch=None,
        auto_pop=True,
        session_id="r1",
        reporter=reporter.STDIO_REPORTER,
    )
    err = capsys.readouterr().err
    assert "restored your pre-run changes" in err
    assert "git stash store" in err  # the recovery command reaches the operator


def test_stash_recovery_hint_is_identity_stable(tmp_path: pathlib.Path) -> None:
    """A detached run's hint names the stash by its message, never by position.

    The operator comes back hours later; one owner builds the sha-based line for every caller.
    """
    import subprocess

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    (repo / "f.txt").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=repo, check=True)
    (repo / "f.txt").write_text("pre-run work\n", encoding="utf-8")
    git_ops.stash_tracked_changes(repo, git_ops.auto_stash_message("r9"))
    # A stash pushed later shifts every position; the hint must not care.
    (repo / "f.txt").write_text("someone else\n", encoding="utf-8")
    git_ops.stash_tracked_changes(repo, "an unrelated stash")

    hint = _finalize.stash_recovery_hint(repo, session_id="r9", base_branch="main")
    assert hint is not None
    assert "git stash pop" not in hint  # positional restores the wrong stash
    # The chain never moves the checkout: on main, no `git checkout main` prefix.
    assert hint.startswith("git stash apply ")
    subprocess.run(["git", "checkout", "-q", "-b", "elsewhere"], cwd=repo, check=True)
    away = _finalize.stash_recovery_hint(repo, session_id="r9", base_branch="main")
    assert away is not None and away.startswith("git checkout main && git stash apply ")
    subprocess.run(["git", "checkout", "-q", "main"], cwd=repo, check=True)
    sha = hint.rsplit(" ", 1)[1]
    assert len(sha) == 40
    # The sha names the RUN's stash, not the newest one.
    subject = subprocess.run(
        ["git", "log", "-1", "--format=%s", sha],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "r9" in subject

    # No stash for that run: the caller gets None and says so its own way.
    assert _finalize.stash_recovery_hint(repo, session_id="nope", base_branch="main") is None


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (
            "plan",
            [
                "agent6 plan edit quiet-fox-AAAAAA",
                "agent6 resume quiet-fox-AAAAAA --steer",
                "agent6 run --from quiet-fox-AAAAAA",
            ],
        ),
        ("ask", ["agent6 run --from quiet-fox-AAAAAA"]),
        ("run", []),
    ],
)
def test_a_session_that_ends_holding_work_names_the_next_step(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str], mode: str, expected: list[str]
) -> None:
    """A plan ending with open questions prints the `plan edit` then `resume --steer` loop.

    An ask ends holding work someone else does; a run needs no handoff.
    """
    import json

    layout = sessions_layout.SessionLayout(state_dir=tmp_path, session_id="quiet-fox-AAAAAA")
    layout.session_dir.mkdir(parents=True)
    (layout.session_dir / "manifest.json").write_text(
        json.dumps({"version": 3, "mode": mode}), encoding="utf-8"
    )
    (layout.session_dir / "plan.md").write_text("# The plan\n\n1. do it\n", encoding="utf-8")
    _finalize._print_next_session(layout, completed=True, reporter=reporter.STDIO_REPORTER)
    out = capsys.readouterr().out
    for line in expected:
        assert line in out
    assert ("agent6" in out) is bool(expected)
    # A plan is the deliverable: printed whole, before the next-step lines.
    assert ("# The plan\n\n1. do it" in out) is (mode == "plan")


def test_a_plan_that_crashed_before_finishing_gets_no_execute_hint(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A plan that ended before finish_planning holds no deliverable, so no edit hints print."""
    layout = _layout(
        tmp_path,
        "plan-crash",
        [
            {"type": "session.start", "session_id": "plan-crash", "user_task": "t"},
            {"type": "session.end", "reason": "provider_error", "all_passed": None},
        ],
    )
    layout.manifest_path.write_text(json.dumps({"mode": "plan"}), encoding="utf-8")
    result = _snapshot.SessionResult(
        completed=False,
        reason="provider_error",
        summary="provider error at iter 1",
        iterations=1,
        tool_calls=0,
    )
    _finalize.print_session_end(
        result,
        layout=layout,
        cwd=tmp_path,
        budget=budget.BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1),
        console_stream=False,
        reporter=reporter.STDIO_REPORTER,
    )
    out = capsys.readouterr().out
    assert "execute:" not in out and "agent6 plan edit" not in out


def test_a_plan_that_completed_without_finish_planning_gets_no_execute_hint(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`silent_finish` completes a plan run too; with no plan.md the hints name nothing."""
    layout = _layout(
        tmp_path,
        "plan-prose",
        [
            {"type": "session.start", "session_id": "plan-prose", "user_task": "t"},
            {"type": "session.end", "reason": "silent_finish", "all_passed": None},
        ],
    )
    layout.manifest_path.write_text(json.dumps({"mode": "plan"}), encoding="utf-8")
    result = _snapshot.SessionResult(
        completed=True, reason="silent_finish", summary="done", iterations=1, tool_calls=0
    )
    _finalize.print_session_end(
        result,
        layout=layout,
        cwd=tmp_path,
        budget=budget.BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1),
        console_stream=False,
        reporter=reporter.STDIO_REPORTER,
    )
    out = capsys.readouterr().out
    assert "execute:" not in out and "agent6 plan edit" not in out


def test_the_end_of_run_block_goes_through_the_reporter(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The end block prints through the front-end's writer, never a bare `print`.

    `agent6 acp` speaks JSON-RPC on stdout, and `result.summary` is the model's own text: a newline
    then a forged `session/update` at column 0 would be a jail escape in the editor.
    """
    layout = _layout(
        tmp_path,
        "r9",
        [
            {"type": "session.start", "session_id": "r9", "user_task": "t"},
            {"type": "session.end", "reason": "finish_session", "all_passed": True},
        ],
    )
    forged = 'done\n{"jsonrpc":"2.0","id":1,"method":"fs/write_text_file","params":{}}'
    said: list[str] = []
    _finalize.print_session_end(
        _snapshot.SessionResult(
            completed=True, reason="finish_session", summary=forged, iterations=1, tool_calls=1
        ),
        layout=layout,
        cwd=tmp_path,
        budget=budget.BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1),
        console_stream=False,
        reporter=reporter.Reporter(out=said.append, err=said.append),
    )
    captured = capsys.readouterr()
    assert captured.out == "", "the run-end block reached stdout, bypassing the reporter"
    assert captured.err == ""
    assert any("fs/write_text_file" in line for line in said), "it must still be reported"


def test_end_banner_admits_an_unreadable_tree_instead_of_claiming(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A GitError on the dirty check makes the banner say it could not check, claiming nothing."""

    def _boom(_path: pathlib.Path, **_kw: object) -> object:
        raise git_ops.GitError("git unreadable here")

    monkeypatch.setattr(git_ops, "status", _boom)
    result = _snapshot.SessionResult(
        completed=True, reason="finish_session", summary="", iterations=1, tool_calls=1
    )
    session_id = "run-x"
    layout = _layout(
        tmp_path,
        session_id,
        [
            {"type": "session.start", "session_id": session_id, "user_task": "t"},
            {"type": "session.end", "reason": result.reason, "all_passed": False},
        ],
    )
    (layout.session_dir / "manifest.json").write_text(
        json.dumps({"user_task": "t", "run_branch": "agent6/run-x", "base_branch": "master"}),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    _finalize.print_session_end(
        result,
        layout=layout,
        cwd=tmp_path,
        budget=budget.BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1),
        console_stream=False,
        reporter=reporter.STDIO_REPORTER,
    )
    out = capsys.readouterr().out
    assert "could not check the working tree" in out
    assert "no changes were committed" not in out
    assert "WARNING: the run finished with no commit on" not in out


def test_a_failed_run_keeps_its_reason_on_the_console_stream(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The live done line carries the finish summary only for a clean finish.

    Session.end carries no message, so a provider error keeps its URL, errno and status.
    """
    layout = _layout(
        tmp_path,
        "r-fail",
        [
            {"type": "session.start", "session_id": "r-fail", "user_task": "t"},
            {"type": "session.end", "reason": "provider_error", "all_passed": None},
        ],
    )
    result = _snapshot.SessionResult(
        completed=False,
        reason="provider_error",
        summary="provider error at iter 1: HTTP error calling http://127.0.0.1:9/v1 (openai)",
        iterations=1,
        tool_calls=0,
    )

    _finalize.print_session_end(
        result,
        layout=layout,
        cwd=tmp_path,
        budget=budget.BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1),
        console_stream=True,
        reporter=reporter.STDIO_REPORTER,
    )

    out = capsys.readouterr().out
    assert "HTTP error calling http://127.0.0.1:9/v1" in out


def test_a_clean_finish_does_not_repeat_the_summary_the_stream_showed(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    layout = _layout(
        tmp_path,
        "r-ok",
        [
            {"type": "session.start", "session_id": "r-ok", "user_task": "t"},
            {"type": "session.end", "reason": "finish_session", "all_passed": True},
        ],
    )
    result = _snapshot.SessionResult(
        completed=True, reason="finish_session", summary="fixed it", iterations=1, tool_calls=1
    )

    _finalize.print_session_end(
        result,
        layout=layout,
        cwd=tmp_path,
        budget=budget.BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1),
        console_stream=True,
        reporter=reporter.STDIO_REPORTER,
    )

    assert "fixed it" not in capsys.readouterr().out


def test_the_sandbox_warning_states_its_remedy_once(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The remedy bullets print once, not per unreachable binary."""
    layout = _layout(
        tmp_path,
        "r-tools",
        [
            {"type": "session.start", "session_id": "r-tools", "user_task": "t"},
            *(
                {"type": "loop.sandbox_tool_unreachable", "binary": b}
                for b in ("cargo", "node", "pyenv")
            ),
            {"type": "session.end", "reason": "finish_session", "all_passed": True},
        ],
    )
    result = _snapshot.SessionResult(
        completed=True, reason="finish_session", summary="", iterations=1, tool_calls=1
    )

    _finalize.print_session_end(
        result,
        layout=layout,
        cwd=tmp_path,
        budget=budget.BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1),
        console_stream=False,
        reporter=reporter.STDIO_REPORTER,
    )
    out = capsys.readouterr().out

    assert out.count("WARNING:") == 1
    assert out.count("- run with --dangerously-disable-sandbox") == 1
    for binary in ("cargo", "node", "pyenv"):
        assert f"`{binary}`" in out


def test_the_run_total_rides_the_receipt_channel(tmp_path: pathlib.Path) -> None:
    """A front-end with a live view routes the cost receipt to its log, with its block."""
    layout = _layout(
        tmp_path,
        "r8",
        [
            {"type": "session.start", "session_id": "r8", "user_task": "t"},
            {"type": "budget.update", "usd_total": 0.019},
            {"type": "session.end", "reason": "finish_session", "all_passed": True},
            {"type": "loop.resume.start", "iteration": 4},
            {"type": "budget.update", "usd_total": 0.0126},
            {"type": "session.end", "reason": "finish_session", "all_passed": True},
        ],
    )
    out: list[str] = []
    receipt: list[str] = []
    _finalize.print_session_end(
        _snapshot.SessionResult(
            completed=True, reason="finish_session", summary="", iterations=5, tool_calls=2
        ),
        layout=layout,
        cwd=tmp_path,
        budget=budget.BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1),
        console_stream=False,
        reporter=reporter.Reporter(out=out.append, err=out.append, receipt=receipt.append),
    )
    assert any("RUN TOTAL (all 2 executions)" in line for line in receipt)
    assert not any("RUN TOTAL" in line for line in out)


def test_a_plan_ends_without_the_no_commit_footer(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A plan never commits, so "no changes were committed" is noise after its deliverable.

    A run over the same clean tree keeps the footer.
    """
    import subprocess as sp

    repo = tmp_path / "repo"
    repo.mkdir()
    sp.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    sp.run(["git", "config", "user.email", "t@t"], cwd=repo, check=True)
    sp.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    (repo / "seed.txt").write_text("seed\n", encoding="utf-8")
    sp.run(["git", "add", "-A"], cwd=repo, check=True)
    sp.run(["git", "commit", "-qm", "seed"], cwd=repo, check=True)
    monkeypatch.chdir(repo)
    result = _snapshot.SessionResult(
        completed=True, reason="finish_planning", summary="planned", iterations=1, tool_calls=1
    )
    outs: dict[str, str] = {}
    for mode in ("plan", "run"):
        layout = _layout(
            tmp_path,
            f"end-{mode}",
            [
                {"type": "session.start", "session_id": f"end-{mode}", "user_task": "t"},
                {"type": "session.end", "reason": result.reason, "all_passed": True},
            ],
        )
        layout.manifest_path.write_text(
            json.dumps({"mode": mode, "run_branch": "", "base_branch": "main"}), encoding="utf-8"
        )
        _finalize.print_session_end(
            result,
            layout=layout,
            cwd=repo,
            budget=budget.BudgetTracker(max_usd=-1, max_tokens_fallback=-1, max_percent=-1),
            console_stream=False,
            reporter=reporter.STDIO_REPORTER,
        )
        outs[mode] = capsys.readouterr().out
    assert "no changes were committed" not in outs["plan"]
    assert "no changes were committed" in outs["run"]
