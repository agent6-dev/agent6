# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The end-of-run console headline must agree with `agent6 runs`.

A finish_run over a red/stale verify emits run.end all_passed=false, so the
listing reads "finished". The console block used to read result.completed
(true for any finish_run) and print "passed" — the exact disagreement
status_word exists to prevent. print_run_end now folds the same run.end.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent6.app import finalize as _finalize
from agent6.app.finalize import print_interrupt_end, print_run_end
from agent6.budget import BudgetTracker
from agent6.git_ops import GitStatus
from agent6.runs.layout import RunLayout
from agent6.workflows._run_state import RunResult


def _layout(tmp_path: Path, run_id: str, events: list[dict[str, object]]) -> RunLayout:
    rd = tmp_path / "runs" / run_id
    rd.mkdir(parents=True)
    (rd / "logs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    return RunLayout(state_dir=tmp_path, run_id=run_id)


def test_finish_run_over_red_verify_is_not_headlined_passed(tmp_path: Path, capsys: object) -> None:
    layout = _layout(
        tmp_path,
        "r1",
        [
            {"type": "run.start", "run_id": "r1", "user_task": "t"},
            {"type": "run.end", "reason": "finish_run", "all_passed": False},
        ],
    )
    result = RunResult(
        completed=True, reason="finish_run", summary="all tests pass", iterations=3, tool_calls=5
    )
    print_run_end(
        result,
        layout=layout,
        budget=BudgetTracker(max_input_tokens=1000, max_output_tokens=1000),
        console_stream=False,
    )
    out = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "finished" in out
    assert "passed" not in out.split("\n")[1]  # the headline line, not the agent's summary


def test_all_green_finish_is_headlined_passed(tmp_path: Path, capsys: object) -> None:
    layout = _layout(
        tmp_path,
        "r2",
        [
            {"type": "run.start", "run_id": "r2", "user_task": "t"},
            {"type": "run.end", "reason": "finish_run", "all_passed": True},
        ],
    )
    result = RunResult(
        completed=True, reason="finish_run", summary="done", iterations=2, tool_calls=3
    )
    print_run_end(
        result,
        layout=layout,
        budget=BudgetTracker(max_input_tokens=1000, max_output_tokens=1000),
        console_stream=False,
    )
    out = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "passed" in out


def test_end_banner_warns_when_checkout_is_parked_on_the_run_branch(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = _layout(
        tmp_path,
        "r3",
        [
            {"type": "run.start", "run_id": "r3", "user_task": "t"},
            {"type": "run.end", "reason": "finish_run", "all_passed": True},
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
    print_run_end(
        result,
        layout=layout,
        budget=BudgetTracker(max_input_tokens=1000, max_output_tokens=1000),
        console_stream=False,
    )
    out = capsys.readouterr().out
    assert "you are on agent6/r3" in out
    assert "git switch main" in out


def test_interrupt_end_prints_cost_resume_and_branch_hints(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # A Ctrl-C interrupt used to print only "run interrupted": no spend, no resume
    # hint, and no note the user was left on the run branch.
    layout = _layout(tmp_path, "r4", [{"type": "run.start", "run_id": "r4", "user_task": "t"}])
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
        budget=BudgetTracker(max_input_tokens=1000, max_output_tokens=1000),
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
            {"type": "run.start", "run_id": "r3", "user_task": "t"},
            {"type": "run.end", "reason": "provider_error", "all_passed": False},
        ],
    )
    result = RunResult(
        completed=False, reason="provider_error", summary="", iterations=1, tool_calls=0
    )
    print_run_end(
        result,
        layout=layout,
        budget=BudgetTracker(max_input_tokens=1000, max_output_tokens=1000),
        console_stream=False,
    )
    out = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "failed" in out and "provider error" in out
