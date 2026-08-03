# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The end-of-run console headline must agree with `agent6 sessions`.

A finish_run over a red/stale verify emits session.end all_passed=false, so the
listing reads "finished". The console block used to read result.completed
(true for any finish_run) and print "passed" — the exact disagreement
status_word exists to prevent. print_session_end now folds the same session.end.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent6.app import finalize as _finalize
from agent6.app.finalize import print_interrupt_end, print_session_end
from agent6.app.reporter import STDIO_REPORTER, Reporter
from agent6.budget import BudgetTracker
from agent6.git_ops import GitStatus
from agent6.sessions.layout import SessionLayout
from agent6.workflows._run_state import RunResult


def _layout(tmp_path: Path, session_id: str, events: list[dict[str, object]]) -> SessionLayout:
    rd = tmp_path / "sessions" / "runs" / session_id
    rd.mkdir(parents=True)
    (rd / "logs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    return SessionLayout(state_dir=tmp_path, session_id=session_id)


def test_finish_run_over_red_verify_is_not_headlined_passed(tmp_path: Path, capsys: object) -> None:
    layout = _layout(
        tmp_path,
        "r1",
        [
            {"type": "session.start", "session_id": "r1", "user_task": "t"},
            {"type": "session.end", "reason": "finish_run", "all_passed": False},
        ],
    )
    result = RunResult(
        completed=True, reason="finish_run", summary="all tests pass", iterations=3, tool_calls=5
    )
    print_session_end(
        result,
        layout=layout,
        budget=BudgetTracker(max_usd=-1, max_tokens_fallback=-1),
        console_stream=False,
        reporter=STDIO_REPORTER,
    )
    out = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "finished" in out
    assert "passed" not in out.split("\n")[1]  # the headline line, not the agent's summary


def test_all_green_finish_is_headlined_passed(tmp_path: Path, capsys: object) -> None:
    layout = _layout(
        tmp_path,
        "r2",
        [
            {"type": "session.start", "session_id": "r2", "user_task": "t"},
            {"type": "session.end", "reason": "finish_run", "all_passed": True},
        ],
    )
    result = RunResult(
        completed=True, reason="finish_run", summary="done", iterations=2, tool_calls=3
    )
    print_session_end(
        result,
        layout=layout,
        budget=BudgetTracker(max_usd=-1, max_tokens_fallback=-1),
        console_stream=False,
        reporter=STDIO_REPORTER,
    )
    out = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "passed" in out


def test_end_banner_does_not_offer_merge_for_an_auto_merged_branch(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """auto_merge already merged (and auto_prune may have deleted) the run
    branch, so the footer must say it merged, not tell the operator to run
    `agent6 sessions merge` on a branch that is gone."""
    layout = _layout(
        tmp_path,
        "r-merged",
        [
            {"type": "session.start", "session_id": "r-merged", "user_task": "t"},
            {"type": "session.end", "reason": "finish_run", "all_passed": True},
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
    result = RunResult(
        completed=True, reason="finish_run", summary="done", iterations=1, tool_calls=1
    )
    print_session_end(
        result,
        layout=layout,
        budget=BudgetTracker(max_usd=-1, max_tokens_fallback=-1),
        console_stream=False,
        reporter=STDIO_REPORTER,
    )
    out = capsys.readouterr().out
    assert "changes merged into main" in out
    assert "runs merge" not in out


def test_end_banner_warns_when_checkout_is_parked_on_the_run_branch(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = _layout(
        tmp_path,
        "r3",
        [
            {"type": "session.start", "session_id": "r3", "user_task": "t"},
            {"type": "session.end", "reason": "finish_run", "all_passed": True},
        ],
    )
    layout.manifest_path.write_text(
        json.dumps({"run_branch": "agent6/r3", "base_branch": "main"}), encoding="utf-8"
    )

    # The checkout is still on the run branch (branch_per_run never switches back).
    def _on_run_branch(_p: Path) -> GitStatus:
        return GitStatus(
            branch="agent6/r3", head_sha="x", is_clean=True, untracked_count=0, modified_count=0
        )

    monkeypatch.setattr(_finalize, "git_status", _on_run_branch)
    result = RunResult(
        completed=True, reason="finish_run", summary="done", iterations=1, tool_calls=1
    )
    print_session_end(
        result,
        layout=layout,
        budget=BudgetTracker(max_usd=-1, max_tokens_fallback=-1),
        console_stream=False,
        reporter=STDIO_REPORTER,
    )
    out = capsys.readouterr().out
    assert "you are on agent6/r3" in out
    assert "git switch main" in out


def test_interrupt_end_prints_cost_resume_and_branch_hints(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # A Ctrl-C interrupt used to print only "run interrupted": no spend, no resume
    # hint, and no note the user was left on the run branch.
    layout = _layout(
        tmp_path, "r4", [{"type": "session.start", "session_id": "r4", "user_task": "t"}]
    )
    layout.manifest_path.write_text(
        json.dumps({"run_branch": "agent6/r4", "base_branch": "main"}), encoding="utf-8"
    )

    def _on_run_branch(_p: Path) -> GitStatus:
        return GitStatus(
            branch="agent6/r4", head_sha="x", is_clean=True, untracked_count=0, modified_count=0
        )

    monkeypatch.setattr(_finalize, "git_status", _on_run_branch)
    print_interrupt_end(
        layout=layout,
        budget=BudgetTracker(max_usd=-1, max_tokens_fallback=-1),
        reporter=STDIO_REPORTER,
    )
    out = capsys.readouterr().out
    assert "Token + cost summary" in out  # the budget/cost block
    assert "resume with:  agent6 resume r4" in out
    assert "you are on agent6/r4" in out and "git switch main" in out


def test_provider_error_is_headlined_failed(tmp_path: Path, capsys: object) -> None:
    layout = _layout(
        tmp_path,
        "r3",
        [
            {"type": "session.start", "session_id": "r3", "user_task": "t"},
            {"type": "session.end", "reason": "provider_error", "all_passed": False},
        ],
    )
    result = RunResult(
        completed=False, reason="provider_error", summary="", iterations=1, tool_calls=0
    )
    print_session_end(
        result,
        layout=layout,
        budget=BudgetTracker(max_usd=-1, max_tokens_fallback=-1),
        console_stream=False,
        reporter=STDIO_REPORTER,
    )
    out = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "failed" in out and "provider error" in out


def test_end_banner_adds_the_run_total_across_resume_legs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The tracker's "TOTAL" line is per-leg (each resume starts a fresh budget);
    # a resumed run's banner must also state the true cumulative spend.
    layout = _layout(
        tmp_path,
        "r7",
        [
            {"type": "session.start", "session_id": "r7", "user_task": "t"},
            {"type": "budget.update", "usd_total": 0.019},
            {"type": "session.end", "reason": "finish_run", "all_passed": True},
            {"type": "loop.resume.start", "iteration": 4},
            {"type": "budget.update", "usd_total": 0.0126},
            {"type": "session.end", "reason": "finish_run", "all_passed": True},
        ],
    )
    result = RunResult(completed=True, reason="finish_run", summary="", iterations=5, tool_calls=2)
    print_session_end(
        result,
        layout=layout,
        budget=BudgetTracker(max_usd=-1, max_tokens_fallback=-1),
        console_stream=False,
        reporter=STDIO_REPORTER,
    )
    out = capsys.readouterr().out
    assert "RUN TOTAL (all 2 legs): $0.0316" in out


def test_end_banner_stays_quiet_on_a_single_leg_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    layout = _layout(
        tmp_path,
        "r8",
        [
            {"type": "session.start", "session_id": "r8", "user_task": "t"},
            {"type": "budget.update", "usd_total": 0.01},
            {"type": "session.end", "reason": "finish_run", "all_passed": True},
        ],
    )
    result = RunResult(completed=True, reason="finish_run", summary="", iterations=2, tool_calls=1)
    print_session_end(
        result,
        layout=layout,
        budget=BudgetTracker(max_usd=-1, max_tokens_fallback=-1),
        console_stream=False,
        reporter=STDIO_REPORTER,
    )
    assert "RUN TOTAL" not in capsys.readouterr().out


def test_finalize_auto_stash_pops_the_run_stash_not_the_latest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The finalizer restores THE stash the run pushed (found by its run-id
    message), not stash@{0}: a stash pushed during the run otherwise got
    popped as the 'pre-run work' while the real pre-run work stayed hidden."""
    import subprocess

    from agent6.app.finalize import finalize_auto_stash
    from agent6.git_ops import auto_stash_message, stash_all

    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args: str) -> None:
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "t@t")
    git("config", "user.name", "t")
    (repo / "base.txt").write_text("base\n", encoding="utf-8")
    git("add", "-A")
    git("commit", "-qm", "base")
    (repo / "pre.txt").write_text("pre-run work\n", encoding="utf-8")
    stash_all(repo, auto_stash_message("r1"))
    (repo / "mid.txt").write_text("mid-run work\n", encoding="utf-8")
    stash_all(repo, "operator stash pushed mid-run")
    base = subprocess.run(
        ["git", "-C", str(repo), "branch", "--show-current"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    finalize_auto_stash(
        repo,
        base_branch=base,
        run_branch=None,
        auto_pop=True,
        session_id="r1",
        reporter=STDIO_REPORTER,
    )
    assert "restored your pre-run changes" in capsys.readouterr().err
    assert (repo / "pre.txt").is_file()
    assert not (repo / "mid.txt").exists()  # the mid-run stash stays a stash


def test_finalize_auto_stash_reports_a_vanished_stash(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A stash the operator already popped mid-run is reported, not silently
    'restored' (and no longer pops whatever happens to sit at stash@{0})."""
    import subprocess

    from agent6.app.finalize import finalize_auto_stash

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
    finalize_auto_stash(
        repo,
        base_branch="master",
        run_branch=None,
        auto_pop=True,
        session_id="r1",
        reporter=STDIO_REPORTER,
    )
    assert "auto-stash not found" in capsys.readouterr().err


def test_finalize_auto_stash_prints_a_failed_bystander_putback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the restore raises because a raced drop took a concurrent stash and
    putting it back failed, finalization prints the recovery command and
    finishes -- the loss must reach the operator, not crash the finalizer."""
    import subprocess

    from agent6.app import finalize as finalize_mod
    from agent6.app.finalize import finalize_auto_stash
    from agent6.git_ops import GitError, auto_stash_message, stash_all

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
    (repo / "pre.txt").write_text("pre-run work\n", encoding="utf-8")
    stash_all(repo, auto_stash_message("r1"))

    def raising_restore(cwd: object, entry: object) -> bool:
        raise GitError(
            "a stash pushed concurrently ('x') was taken by a raced drop and putting"
            " it back failed; restore it with:\n    git stash store -m 'x' abc123"
        )

    monkeypatch.setattr(finalize_mod, "restore_stash", raising_restore)
    finalize_auto_stash(
        repo,
        base_branch="main",
        run_branch=None,
        auto_pop=True,
        session_id="r1",
        reporter=STDIO_REPORTER,
    )
    err = capsys.readouterr().err
    assert "restored your pre-run changes" in err
    assert "git stash store" in err  # the recovery command reaches the operator


def test_stash_recovery_hint_is_identity_stable(tmp_path: Path) -> None:
    """The hint a DETACHED run prints has the longest window of all -- the
    operator comes back hours later -- and it still named a positional
    `git stash pop`, the exact failure `restore_stash` was changed to avoid.
    One owner builds the sha-based line for every caller."""
    import subprocess

    from agent6.app.finalize import stash_recovery_hint
    from agent6.git_ops import auto_stash_message, stash_all

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    (repo / "f.txt").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=repo, check=True)
    (repo / "f.txt").write_text("pre-run work\n", encoding="utf-8")
    stash_all(repo, auto_stash_message("r9"))
    # A stash pushed later shifts every position; the hint must not care.
    (repo / "f.txt").write_text("someone else\n", encoding="utf-8")
    stash_all(repo, "an unrelated stash")

    hint = stash_recovery_hint(repo, session_id="r9", base_branch="main")
    assert hint is not None
    assert "git stash pop" not in hint  # positional restores the wrong stash
    assert "git stash apply " in hint and "git checkout main" in hint
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
    assert stash_recovery_hint(repo, session_id="nope", base_branch="main") is None


@pytest.mark.parametrize(
    ("mode", "suggested"),
    [("plan", True), ("ask", True), ("run", False)],
)
def test_a_session_that_ends_holding_work_names_the_next_step(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], mode: str, suggested: bool
) -> None:
    """Seeding existed but nothing suggested it, so an operator had to know
    `--from` was there. A plan and an ask end holding work someone else does;
    a run has already done its work and needs no handoff."""
    import json

    from agent6.app.finalize import _print_next_session  # pyright: ignore[reportPrivateUsage]
    from agent6.sessions.layout import SessionLayout

    layout = SessionLayout(state_dir=tmp_path, session_id="quiet-fox-AAAAAA")
    layout.session_dir.mkdir(parents=True)
    (layout.session_dir / "manifest.json").write_text(
        json.dumps({"version": 3, "mode": mode}), encoding="utf-8"
    )
    _print_next_session(layout, reporter=STDIO_REPORTER)
    out = capsys.readouterr().out
    assert ("agent6 run --from quiet-fox-AAAAAA" in out) is suggested


def test_the_end_of_run_block_goes_through_the_reporter(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A front-end that does not own stdout must be able to redirect this.

    `agent6 acp` speaks JSON-RPC on stdout, so a bare `print` here is not a
    cosmetic layering slip: it writes non-JSON lines into the protocol stream,
    and `result.summary` is the model's own `finish_run` text -- unbounded, and
    free to contain newlines. A model could close the prose with a newline and
    emit a forged `session/update` at column 0, which a client that skips
    unparseable lines honours. The editor owns the filesystem and terminal in
    ACP, so that is a jail escape.
    """
    layout = _layout(
        tmp_path,
        "r9",
        [
            {"type": "session.start", "session_id": "r9", "user_task": "t"},
            {"type": "session.end", "reason": "finish_run", "all_passed": True},
        ],
    )
    forged = 'done\n{"jsonrpc":"2.0","id":1,"method":"fs/write_text_file","params":{}}'
    said: list[str] = []
    print_session_end(
        RunResult(completed=True, reason="finish_run", summary=forged, iterations=1, tool_calls=1),
        layout=layout,
        budget=BudgetTracker(max_usd=-1, max_tokens_fallback=-1),
        console_stream=False,
        reporter=Reporter(out=said.append, err=said.append),
    )
    captured = capsys.readouterr()
    assert captured.out == "", "the run-end block reached stdout, bypassing the reporter"
    assert captured.err == ""
    assert any("fs/write_text_file" in line for line in said), "it must still be reported"
