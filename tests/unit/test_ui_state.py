# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the pure event-fold in agent6.ui.state."""

from __future__ import annotations

from agent6.ui.state import (
    ApprovalPrompt,
    BudgetView,
    RunState,
    StepView,
    apply_event,
    initial_state,
)


def test_initial_state_is_empty() -> None:
    s = initial_state()
    assert s == RunState()
    assert s.steps == ()
    assert s.budget == BudgetView()


def test_run_start_records_task() -> None:
    s = apply_event(initial_state(), {"type": "run.start", "user_task": "fix bug"})
    assert s.user_task == "fix bug"


def test_plan_ready_populates_steps() -> None:
    s = apply_event(
        initial_state(),
        {"type": "plan.ready", "summary": "do things", "steps": ["a", "b", "c"]},
    )
    assert s.plan_summary == "do things"
    assert len(s.steps) == 3
    assert s.steps[0] == StepView(index=1, title="a")
    assert s.steps[2] == StepView(index=3, title="c")


def test_step_start_then_end_updates_status() -> None:
    s = apply_event(initial_state(), {"type": "plan.ready", "steps": ["a", "b"]})
    s = apply_event(s, {"type": "step.start", "index": 1, "title": "a"})
    assert s.current_step_index == 1
    assert s.steps[0].status == "running"
    s = apply_event(
        s,
        {
            "type": "step.end",
            "index": 1,
            "title": "a",
            "status": "passed",
            "commit_sha": "abc1234",
            "notes": "",
        },
    )
    assert s.steps[0].status == "passed"
    assert s.steps[0].commit_sha == "abc1234"


def test_step_end_unknown_status_defaults_to_failed() -> None:
    s = apply_event(initial_state(), {"type": "plan.ready", "steps": ["a"]})
    s = apply_event(s, {"type": "step.end", "index": 1, "status": "weird"})
    assert s.steps[0].status == "failed"


def test_tool_call_then_result_pairs_up() -> None:
    s = apply_event(
        initial_state(),
        {"type": "tool.call", "name": "read_file", "args": {"path": "foo.py"}},
    )
    assert len(s.tool_calls) == 1
    assert s.tool_calls[0].name == "read_file"
    assert s.tool_calls[0].ok is None
    s = apply_event(
        s, {"type": "tool.result", "name": "read_file", "ok": True, "summary": "100 bytes"}
    )
    assert s.tool_calls[0].ok is True
    assert s.tool_calls[0].result_summary == "100 bytes"


def test_tool_result_with_mismatched_name_does_not_overwrite() -> None:
    s = apply_event(initial_state(), {"type": "tool.call", "name": "read_file", "args": {}})
    s = apply_event(s, {"type": "tool.result", "name": "other", "ok": True, "summary": "x"})
    assert s.tool_calls[0].ok is None


def test_role_call_then_result_clears_in_flight() -> None:
    s = apply_event(
        initial_state(),
        {"type": "role.call", "role": "worker", "model": "gpt-5", "provider": "openai"},
    )
    assert s.last_role is not None
    assert s.last_role.in_flight is True
    s = apply_event(s, {"type": "role.result", "role": "worker", "tokens_in": 10, "tokens_out": 20})
    assert s.last_role is not None
    assert s.last_role.in_flight is False


def test_budget_update_populates_view() -> None:
    s = apply_event(
        initial_state(),
        {
            "type": "budget.update",
            "input_total": 100,
            "output_total": 50,
            "input_cap": 1000,
            "output_cap": 500,
        },
    )
    assert s.budget.input_total == 100
    assert s.budget.input_cap == 1000
    # USD fields default cleanly when the event omits them.
    assert s.budget.usd_total == 0.0
    assert s.budget.usd_partial is False


def test_budget_update_carries_usd_total() -> None:
    s = apply_event(
        initial_state(),
        {
            "type": "budget.update",
            "input_total": 100,
            "output_total": 50,
            "input_cap": 1000,
            "output_cap": 500,
            "usd_total": 0.1234,
            "usd_partial": True,
        },
    )
    assert s.budget.usd_total == 0.1234
    assert s.budget.usd_partial is True


def test_verify_lifecycle() -> None:
    s = apply_event(initial_state(), {"type": "verify.start", "cmd": ["pytest", "-q"]})
    assert s.last_verify is not None
    assert s.last_verify.exit_code is None
    s = apply_event(
        s,
        {
            "type": "verify.end",
            "cmd": ["pytest", "-q"],
            "exit_code": 0,
            "duration_s": 1.5,
            "stdout_tail": "ok",
            "stderr_tail": "",
        },
    )
    assert s.last_verify is not None
    assert s.last_verify.exit_code == 0
    assert s.last_verify.duration_s == 1.5


def test_approval_prompt_then_answer() -> None:
    s = apply_event(
        initial_state(),
        {"type": "approval.prompt", "id": "a001", "prompt": "Allow run_command?"},
    )
    assert len(s.pending_approvals) == 1
    assert s.pending_approvals[0] == ApprovalPrompt(id="a001", prompt="Allow run_command?")
    s = apply_event(s, {"type": "approval.answer", "id": "a001", "approved": True})
    assert s.pending_approvals[0].answered is True
    assert s.pending_approvals[0].approved is True


def test_step_diff_stored_by_index() -> None:
    s = apply_event(
        initial_state(),
        {"type": "step.diff", "index": 2, "commit_sha": "abc", "patch": "diff text"},
    )
    assert s.diffs == {2: "diff text"}


def test_run_end_marks_finished() -> None:
    s = apply_event(initial_state(), {"type": "run.end", "all_passed": True})
    assert s.finished is True
    assert s.all_passed is True


def test_unknown_event_type_still_appends_to_log() -> None:
    s = apply_event(initial_state(), {"type": "totally.new", "x": 1})
    # Unknown events should not change RunState identity-relevant fields
    # but DO go into the log tail.
    assert s.steps == ()
    assert len(s.log_tail) == 1


def test_tool_history_is_bounded() -> None:
    s = initial_state()
    for i in range(200):
        s = apply_event(s, {"type": "tool.call", "name": f"t{i}", "args": {}})
    assert len(s.tool_calls) == 50  # _MAX_TOOL_HISTORY


def test_full_run_trace_replay() -> None:
    """End-to-end: feed a plausible event sequence and assert final state."""
    events = [
        {"type": "run.start", "user_task": "do thing"},
        {"type": "plan.ready", "summary": "a plan", "steps": ["one", "two"]},
        {"type": "step.start", "index": 1, "title": "one"},
        {"type": "role.call", "role": "worker", "model": "gpt-5"},
        {"type": "tool.call", "name": "apply_patch", "args": {"path": "x.py"}},
        {"type": "tool.result", "name": "apply_patch", "ok": True, "summary": "applied=1"},
        {"type": "role.result", "role": "worker", "tokens_in": 50, "tokens_out": 100},
        {
            "type": "budget.update",
            "input_total": 50,
            "output_total": 100,
            "input_cap": 1000,
            "output_cap": 1000,
        },
        {"type": "step.diff", "index": 1, "commit_sha": "abc", "patch": "+ added"},
        {"type": "step.end", "index": 1, "status": "passed", "commit_sha": "abc", "notes": ""},
        {"type": "step.start", "index": 2, "title": "two"},
        {"type": "step.end", "index": 2, "status": "failed", "notes": "boom"},
        {"type": "run.end", "all_passed": False},
    ]
    s = initial_state()
    for e in events:
        s = apply_event(s, e)
    assert s.finished is True
    assert s.all_passed is False
    assert s.steps[0].status == "passed"
    assert s.steps[1].status == "failed"
    assert s.steps[1].notes == "boom"
    assert s.diffs[1] == "+ added"
    assert s.budget.input_total == 50
