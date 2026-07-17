# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the pure event-fold in agent6.viewmodel.state."""

from __future__ import annotations

from typing import Any

import pytest

from agent6.viewmodel.state import (
    ApprovalPrompt,
    BudgetView,
    RunState,
    TaskNodeView,
    apply_event,
    fold_run,
    format_log_line,
    initial_state,
    run_state_as_dict,
    run_status_label,
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


def test_run_start_records_run_id() -> None:
    # The loop now stamps the run dir name into run.start; the fold picks it up,
    # so watch --json / the web snapshot report a real id, not "".
    s = apply_event(
        initial_state(),
        {"type": "run.start", "run_id": "deep-granite-CSSYTJ", "user_task": "t"},
    )
    assert s.run_id == "deep-granite-CSSYTJ"


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


def test_apply_edit_args_render_the_edit_kinds_not_a_dict_repr() -> None:
    # The drawer + TUI tool table read args_preview; apply_edit's `edits` list
    # used to leak `[{'kind': 'replace', 'old_string': ...}]` as a Python repr.
    s = apply_event(
        initial_state(),
        {
            "type": "tool.call",
            "name": "apply_edit",
            "args": {
                "path": "calc.py",
                "edits": [
                    {"kind": "replace", "old_string": "a", "new_string": "b"},
                    {"kind": "create", "old_string": "", "new_string": "x"},
                ],
            },
        },
    )
    preview = s.tool_calls[0].args_preview
    assert "old_string" not in preview and "{" not in preview
    assert "path=calc.py" in preview and "edits=replace, create" in preview


def test_dict_valued_tool_summary_renders_as_json_not_python_repr() -> None:
    # A malformed dict summary must not leak `{'unexpected': ...}` (single-quoted
    # Python repr) into the tool table or the log tail.
    s = apply_event(initial_state(), {"type": "tool.call", "name": "weird", "args": {}})
    s = apply_event(
        s, {"type": "tool.result", "name": "weird", "ok": True, "summary": {"unexpected": 1}}
    )
    assert s.tool_calls[0].result_summary == '{"unexpected": 1}'
    line = format_log_line(
        {"type": "tool.result", "name": "weird", "ok": True, "summary": {"unexpected": 1}}
    )
    assert '{"unexpected": 1}' in line and "'unexpected'" not in line


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
    # budget.update never carries a per-model token breakdown (no emitter ever
    # wrote one, and no surface reads it); the fold must not resurrect a
    # permanently-empty field. The per-model breakdown is surfaced from the
    # budget snapshot directly, not the viewmodel.
    assert not hasattr(s.budget, "per_model_tokens")


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


def test_budget_usd_cumulative_across_resume_legs() -> None:
    # Each resume leg's budget.update restarts usd_total from 0; the view banks
    # the finished leg on loop.resume.start so "cost" stays the cumulative
    # spend -- the same rule the hub scanner applies (listing._scan_run_log),
    # keeping the hub row and the run view in agreement.
    def _update(usd: float, *, partial: bool = False) -> dict[str, object]:
        return {"type": "budget.update", "usd_total": usd, "usd_partial": partial}

    s = apply_event(initial_state(), _update(0.02, partial=True))
    s = apply_event(s, {"type": "run.end", "reason": "finish_run", "all_passed": True})
    s = apply_event(s, {"type": "loop.resume.start"})
    # Banked, and the header keeps the old total until the new leg reports.
    assert s.budget.usd_total == 0.02
    s = apply_event(s, _update(0.005))
    assert s.budget.usd_total == pytest.approx(0.025)
    # partial is sticky: leg 1's unpriced spend keeps the total an under-estimate.
    assert s.budget.usd_partial is True
    # A second resume banks the cumulative, not just the last leg.
    s = apply_event(s, {"type": "loop.resume.start"})
    s = apply_event(s, _update(0.001))
    assert s.budget.usd_total == pytest.approx(0.026)


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
    from agent6.viewmodel.state import MAX_LOG_TAIL

    s = initial_state()
    n = MAX_LOG_TAIL + 50
    for i in range(n):
        s = apply_event(s, {"type": "loop.note", "msg": f"line {i}"})
    assert len(s.log_tail) == MAX_LOG_TAIL  # window stays capped
    assert s.log_count == n  # but the count keeps climbing


def test_fold_run_reduces_events_to_a_snapshot() -> None:
    state = fold_run(
        [
            {"type": "run.start", "user_task": "do it"},
            {"type": "role.call", "role": "worker", "model": "m"},
            {"type": "role.text_delta", "text": "hi"},
            {"type": "run.end", "all_passed": True},
        ]
    )
    assert state.user_task == "do it"
    assert state.finished and state.all_passed
    assert state.last_role is not None and state.last_role.streamed_text == "hi"


def test_run_state_as_dict_is_json_serializable() -> None:
    import json

    state = fold_run(
        [
            {"type": "run.start", "user_task": "t"},
            {"type": "tool.call", "name": "grep", "args": {"q": "x"}},
        ]
    )
    d = run_state_as_dict(state)
    assert d["user_task"] == "t"
    assert d["tool_calls"][0]["name"] == "grep"  # tuple -> list, dataclass -> dict
    json.dumps(d)  # the wire form must serialize


def test_run_status_label_distinguishes_stop_finish_error() -> None:
    # All of these set finished=True; the reason is what a user needs to tell them
    # apart. A stopped run must never read as a bare "finished" (looks completed).
    def end(reason: str, all_passed: bool) -> RunState:
        s = apply_event(initial_state(), {"type": "run.start", "user_task": "t"})
        return apply_event(s, {"type": "run.end", "reason": reason, "all_passed": all_passed})

    assert run_status_label(initial_state()) == "running"
    assert run_status_label(end("steer_abort", False)) == "stopped"
    assert run_status_label(end("finish_run", True)) == "passed"
    assert run_status_label(end("finish_run", False)) == "finished"
    assert run_status_label(end("provider_error", False)) == "failed · provider error"
    # and the computed label rides along on the wire dict for the web client
    assert run_state_as_dict(end("steer_abort", False))["status_label"] == "stopped"


def test_resume_start_unfinishes_the_run() -> None:
    # A resume restarts a finished/stopped run in place; the header must show it
    # running again (else steer/stop stay disabled on a live run).
    s = apply_event(initial_state(), {"type": "run.end", "reason": "steer_abort"})
    assert s.finished and s.end_reason == "steer_abort"
    s = apply_event(s, {"type": "loop.resume.start"})
    assert not s.finished and s.end_reason == ""
    assert run_status_label(s) == "running"


def test_role_result_tracks_context_tokens_and_provider() -> None:
    """role.call carries the provider; role.result folds the call's full prompt
    (fresh + cache read + cache write) into ctx_tokens -- the context size the
    ctx% readout is computed from. The value survives the next role.call (no
    per-turn blink) and an error result without usage keeps the last known."""
    from agent6.viewmodel.state import apply_event, initial_state

    s = initial_state()
    s = apply_event(s, {"type": "role.call", "role": "worker", "model": "m", "provider": "p"})
    assert s.last_role is not None and s.last_role.provider == "p"
    assert s.last_role.ctx_tokens == 0  # nothing measured yet
    s = apply_event(
        s,
        {
            "type": "role.result",
            "role": "worker",
            "ok": True,
            "tokens_in": 1_000,
            "cache_read": 40_000,
            "cache_creation": 2_000,
        },
    )
    assert s.last_role is not None and s.last_role.ctx_tokens == 43_000
    s = apply_event(s, {"type": "role.call", "role": "worker", "model": "m", "provider": "p"})
    assert s.last_role is not None and s.last_role.ctx_tokens == 43_000  # carried over
    s = apply_event(s, {"type": "role.result", "role": "worker", "ok": False, "error": "boom"})
    assert s.last_role is not None and s.last_role.ctx_tokens == 43_000  # kept on error
