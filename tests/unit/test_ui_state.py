# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the pure event-fold in agent6.ui.state."""

from __future__ import annotations

from typing import Any

from agent6.ui.state import (
    ApprovalPrompt,
    BudgetView,
    RunState,
    TaskNodeView,
    apply_event,
    format_log_line,
    initial_state,
)


def test_format_log_line_tool_result_appends_output_tail() -> None:
    """An execution tool's tool.result line shows a one-line stderr/stdout hint
    (full tail is in the event), while a plain result stays summary-only."""
    line = format_log_line(
        {
            "type": "tool.result",
            "name": "run_command",
            "ok": True,
            "summary": "exit=1 in 0.2s",
            "stderr_tail": "boom: file not found\nsecond line",
        }
    )
    assert "run_command" in line and "exit=1" in line and "boom: file not found" in line
    plain = format_log_line(
        {"type": "tool.result", "name": "read_file", "ok": True, "summary": "40 bytes"}
    )
    assert "|" not in plain  # no tail hint for non-execution tools


def _graph_event(nodes: dict[str, Any], cursor: str | None = None) -> dict[str, Any]:
    return {"type": "graph.update", "nodes": nodes, "cursor": cursor}


def test_initial_state_is_empty() -> None:
    s = initial_state()
    assert s == RunState()
    assert s.tasks == ()
    assert s.budget == BudgetView()


def test_run_start_records_task() -> None:
    s = apply_event(initial_state(), {"type": "run.start", "user_task": "fix bug"})
    assert s.user_task == "fix bug"


def test_graph_update_builds_task_tree_dfs_with_depth() -> None:
    # root -> (a -> a1), b  (children order preserved, DFS pre-order, depths)
    nodes = {
        "root": {
            "title": "root",
            "parent_id": None,
            "status": "in_progress",
            "children": ["a", "b"],
        },
        "a": {"title": "task a", "parent_id": "root", "status": "passed", "children": ["a1"]},
        "a1": {"title": "task a1", "parent_id": "a", "status": "pending", "children": []},
        "b": {"title": "task b", "parent_id": "root", "status": "failed", "children": []},
    }
    s = apply_event(initial_state(), _graph_event(nodes, cursor="a1"))
    assert s.cursor_task_id == "a1"
    assert s.tasks == (
        TaskNodeView(id="root", title="root", status="in_progress", depth=0),
        TaskNodeView(id="a", title="task a", status="passed", depth=1),
        TaskNodeView(id="a1", title="task a1", status="pending", depth=2, is_cursor=True),
        TaskNodeView(id="b", title="task b", status="failed", depth=1),
    )


def test_graph_update_latest_snapshot_replaces_prior() -> None:
    s = apply_event(
        initial_state(), _graph_event({"r": {"title": "r", "parent_id": None, "children": []}})
    )
    assert len(s.tasks) == 1
    s = apply_event(s, _graph_event({"r2": {"title": "r2", "parent_id": None, "children": []}}))
    assert tuple(t.id for t in s.tasks) == ("r2",)


def test_graph_update_guards_cycles() -> None:
    # a -> b -> a (cycle) must not infinite-loop; each node appears once.
    nodes = {
        "a": {"title": "a", "parent_id": None, "status": "pending", "children": ["b"]},
        "b": {"title": "b", "parent_id": "a", "status": "pending", "children": ["a"]},
    }
    s = apply_event(initial_state(), _graph_event(nodes))
    assert tuple(t.id for t in s.tasks) == ("a", "b")


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


def test_diff_updated_stores_latest_patch() -> None:
    s = apply_event(initial_state(), {"type": "diff.updated", "sha": "abc", "patch": "diff text"})
    assert s.latest_diff == "diff text"
    # a newer diff replaces the prior one
    s = apply_event(s, {"type": "diff.updated", "sha": "def", "patch": "newer"})
    assert s.latest_diff == "newer"


def test_run_end_marks_finished() -> None:
    s = apply_event(initial_state(), {"type": "run.end", "all_passed": True})
    assert s.finished is True
    assert s.all_passed is True


def test_unknown_event_type_still_appends_to_log() -> None:
    s = apply_event(initial_state(), {"type": "totally.new", "x": 1})
    # Unknown events should not change RunState identity-relevant fields
    # but DO go into the log tail.
    assert s.tasks == ()
    assert len(s.log_tail) == 1


def test_tool_history_is_bounded() -> None:
    s = initial_state()
    for i in range(200):
        s = apply_event(s, {"type": "tool.call", "name": f"t{i}", "args": {}})
    assert len(s.tool_calls) == 50  # _MAX_TOOL_HISTORY


def test_full_run_trace_replay() -> None:
    """End-to-end: feed a plausible event sequence and assert final state."""
    tasks = {
        "one": {"title": "one", "parent_id": None, "status": "passed", "children": []},
        "two": {"title": "two", "parent_id": None, "status": "failed", "children": []},
    }
    events = [
        {"type": "run.start", "user_task": "do thing"},
        {"type": "graph.update", "nodes": tasks, "cursor": "two"},
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
        {"type": "diff.updated", "sha": "abc", "patch": "+ added"},
        {"type": "run.end", "all_passed": False},
    ]
    s = initial_state()
    for e in events:
        s = apply_event(s, e)
    assert s.finished is True
    assert s.all_passed is False
    assert s.tasks[0].status == "passed"
    assert s.tasks[1].status == "failed"
    assert s.cursor_task_id == "two"
    assert s.latest_diff == "+ added"
    assert s.budget.input_total == 50


def test_log_count_is_monotonic_past_window_cap() -> None:
    # log_tail is a sliding window (MAX_LOG_TAIL); log_count must keep growing
    # so a live viewer can diff on it. A length-based diff freezes once the
    # window saturates -- this is the bug log_count fixes.
    from agent6.ui.state import MAX_LOG_TAIL

    s = initial_state()
    n = MAX_LOG_TAIL + 50
    for i in range(n):
        s = apply_event(s, {"type": "loop.note", "msg": f"line {i}"})
    assert len(s.log_tail) == MAX_LOG_TAIL  # window stays capped
    assert s.log_count == n  # but the count keeps climbing
