# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the pure event-fold in agent6.viewmodel.session.state."""

from __future__ import annotations

import json
import pathlib
from typing import Any

import pytest

from agent6.sessions import process
from agent6.viewmodel import (
    ApprovalPrompt,
    BudgetView,
    SessionState,
    TaskNodeView,
    approval_parts,
    context_fill,
    log_line,
    session_state_as_dict,
)
from agent6.viewmodel.session import records as state_types
from agent6.viewmodel.session import state as viewmodel_state


def test_expanded_tool_args_are_uncapped_and_keep_nested_values() -> None:
    """Detail arguments retain long values, every edit and every question option."""
    words = "full argument " * 500 + "LAST ARGUMENT BYTE"
    args = {
        "path": "notes.txt",
        "new_string": words,
        "edits": [{"old_string": "old", "new_string": words, "kind": "replace"}],
        "questions": [{"question": "First?", "options": ["yes", "no"]}, {"question": words}],
    }
    state = viewmodel_state.fold_session(
        [{"type": "tool.call", "name": "apply_edit", "args_preview": args}]
    )
    call = state.tool_calls[0]
    assert words in call.args_full
    assert "old_string" in call.args_full and "options" in call.args_full
    assert "LAST ARGUMENT BYTE" not in call.args_preview
    assert len(log_line.render_args({"new_string": words})) < 100


def test_the_wire_carries_needs_new_work_from_the_one_predicate() -> None:
    """`session_state_as_dict` stamps `needs_new_work` from the fold's end facts.

    A finish over a red gate reads False, a certified finish True.
    """
    start = {"type": "session.start", "user_task": "t"}

    def ended(verification: str | None) -> dict[str, object]:
        end = {"type": "session.stop", "reason": "finish_session", "verification": verification}
        return session_state_as_dict(
            viewmodel_state.apply_event(
                viewmodel_state.apply_event(viewmodel_state.initial_state(), start), end
            )
        )

    assert ended("failed")["needs_new_work"] is False
    assert ended("passed")["needs_new_work"] is True
    assert ended("not_applicable")["needs_new_work"] is True
    assert (
        session_state_as_dict(viewmodel_state.apply_event(viewmodel_state.initial_state(), start))[
            "needs_new_work"
        ]
        is False
    )


def test_end_summary_reaches_the_wire_and_clears_on_resume() -> None:
    """Run headers carry the journaled end sentence without holding it over a resume."""
    summary = "operator stopped the run after turn 4"
    s = viewmodel_state.fold_session(
        [
            {"type": "session.start", "mode": "run"},
            {"type": "session.stop", "reason": "steer_abort", "summary": summary},
        ]
    )
    assert session_state_as_dict(s)["end_summary"] == summary
    s = viewmodel_state.apply_event(s, {"type": "session.resume"})
    assert session_state_as_dict(s)["end_summary"] == ""


def test_a_log_line_is_one_line() -> None:
    """The log-line scrubber keeps newlines, and every log pane paints one row per line.

    A provider error carrying an SSE dump or a traceback painted a dozen rows, every one after
    the first with no timestamp and no event name.
    """
    line = log_line.format_log_line(
        {
            "type": "provider.retry",
            "error": "non-JSON response (status 200):\n: keepalive\n\n: keepalive\n",
        }
    )

    assert "\n" not in line
    assert "keepalive" in line


def test_format_log_line_keeps_the_compaction_reason() -> None:
    """A failed compaction's reason and the operator's /compact focus reach the log view.

    With no case for these types the view printed the bare event name, so a 429'd summariser
    read as nothing having happened.
    """
    failed = log_line.format_log_line(
        {"type": "compact.summarise.failed", "error": "provider 429: rate limited"}
    )
    assert "429" in failed
    gist = log_line.format_log_line({"type": "compact.gist.failed", "error": "boom"})
    assert "boom" in gist
    requested = log_line.format_log_line(
        {"type": "compact.requested", "focus": "keep the auth work"}
    )
    assert "keep the auth work" in requested


def test_a_jail_degraded_row_carries_its_reason() -> None:
    """A degraded sandbox and a stop that left a process behind render their detail in the log."""
    line = log_line.format_log_line(
        {"type": "jail.degraded", "detail": "[agent6-jail] warning: fresh /proc\nmount failed"}
    )
    assert "fresh /proc mount failed" in line
    assert "\n" not in line


def test_context_fill_is_the_one_rule_and_rides_the_wire(monkeypatch: pytest.MonkeyPatch) -> None:
    """One rule gives the `ctx N%` readout on every surface, carried as `context_used_percent`.

    The last completed call's prompt tokens over the model's window; None until both are known.
    """

    def _known(_provider: str, _model: str) -> int | None:
        return 100_000

    def _unknown(_provider: str, _model: str) -> int | None:
        return None

    monkeypatch.setattr("agent6.viewmodel.session.view.cached_context_window", _known)
    s = viewmodel_state.initial_state()
    assert context_fill(s) is None  # no call yet
    for ev in (
        {"type": "session.start", "user_task": "x", "mode": "run"},
        {"type": "model.call", "role": "worker", "model": "m", "provider": "p"},
        {
            "type": "model.result",
            "role": "worker",
            "ok": True,
            "input_tokens": 1_000,
            "cache_read_tokens": 40_000,
        },
    ):
        s = viewmodel_state.apply_event(s, ev)
    assert context_fill(s) == 41
    assert session_state_as_dict(s)["context_used_percent"] == 41
    monkeypatch.setattr("agent6.viewmodel.session.view.cached_context_window", _unknown)
    assert context_fill(s) is None  # an unknown window is not a percent


def test_format_log_line_names_the_pins_in_force() -> None:
    """pin.restored names the pins in force at execution start, for a --pin run or a resume.

    The line said "restored from the snapshot" for both and named none of them.
    """
    line = log_line.format_log_line(
        {"type": "pin.restored", "pins": ["never touch test_calc.py"], "count": 1}
    )
    assert "1 pinned: never touch test_calc.py" in line
    assert "snapshot" not in line
    assert "no pinned instructions" in log_line.format_log_line(
        {"type": "pin.restored", "pins": [], "count": 0}
    )


def test_format_log_line_tool_result_appends_output_tail() -> None:
    """An execution tool's result line shows a one-line output hint; a plain one stays a summary."""
    line = log_line.format_log_line(
        {
            "type": "tool.result",
            "name": "run_command",
            "ok": True,
            "summary": "exit=1 in 0.2s",
            "stderr_tail": "boom: file not found\nsecond line",
        }
    )
    assert "run_command" in line and "exit=1" in line and "boom: file not found" in line
    plain = log_line.format_log_line(
        {"type": "tool.result", "name": "read_file", "ok": True, "summary": "40 bytes"}
    )
    assert "|" not in plain  # no tail hint for non-execution tools


def _graph_event(nodes: dict[str, Any], cursor: str | None = None) -> dict[str, Any]:
    return {"type": "graph.update", "nodes": nodes, "cursor": cursor}


def test_initial_state_is_empty() -> None:
    s = viewmodel_state.initial_state()
    assert s == SessionState()
    assert s.tasks == ()
    assert s.budget == BudgetView()


def test_run_start_records_task() -> None:
    s = viewmodel_state.apply_event(
        viewmodel_state.initial_state(), {"type": "session.start", "user_task": "fix bug"}
    )
    assert s.task_label.task == "fix bug"


def test_run_start_records_run_id() -> None:
    # The loop now stamps the session dir name into session.start; the fold picks it up,
    # so watch --json / the web snapshot report a real id, not "".
    s = viewmodel_state.apply_event(
        viewmodel_state.initial_state(),
        {"type": "session.start", "session_id": "deep-granite-CSSYTJ", "user_task": "t"},
    )
    assert s.session_id == "deep-granite-CSSYTJ"


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
    s = viewmodel_state.apply_event(
        viewmodel_state.initial_state(), _graph_event(nodes, cursor="a1")
    )
    assert s.cursor_task_id == "a1"
    assert s.tasks == (
        TaskNodeView(id="root", title="root", status="in_progress", depth=0, short_id="root"),
        TaskNodeView(id="a", title="task a", status="passed", depth=1, short_id="a"),
        TaskNodeView(
            id="a1", title="task a1", status="pending", depth=2, is_cursor=True, short_id="a1"
        ),
        TaskNodeView(id="b", title="task b", status="failed", depth=1, short_id="b"),
    )


def test_graph_update_orders_multiple_roots_like_tree_order() -> None:
    """Two roots render in id order, the order `tree_order` gives every surface.

    The node map's iteration order is insertion order live and filesystem order after a resume.
    """
    nodes = {
        "01LATER00000000000000000A": {"title": "second", "parent_id": None, "children": []},
        "01EARLY00000000000000000A": {"title": "first", "parent_id": None, "children": []},
    }
    s = viewmodel_state.apply_event(viewmodel_state.initial_state(), _graph_event(nodes))
    assert [t.id for t in s.tasks] == ["01EARLY00000000000000000A", "01LATER00000000000000000A"]


def test_graph_update_shows_an_orphan_after_the_roots_like_tree_order() -> None:
    """A node whose parent is missing is not a root; `tree_order` appends it after the roots."""
    nodes = {
        "01AAAA00000000000000000000": {"title": "orphan", "parent_id": "gone", "children": []},
        "01BBBB00000000000000000000": {"title": "root", "parent_id": None, "children": []},
    }
    s = viewmodel_state.apply_event(viewmodel_state.initial_state(), _graph_event(nodes))
    assert [t.title for t in s.tasks] == ["root", "orphan"]


def test_graph_update_latest_snapshot_replaces_prior() -> None:
    s = viewmodel_state.apply_event(
        viewmodel_state.initial_state(),
        _graph_event({"r": {"title": "r", "parent_id": None, "children": []}}),
    )
    assert len(s.tasks) == 1
    s = viewmodel_state.apply_event(
        s, _graph_event({"r2": {"title": "r2", "parent_id": None, "children": []}})
    )
    assert tuple(t.id for t in s.tasks) == ("r2",)


def test_graph_update_guards_cycles() -> None:
    # a -> b -> a (cycle) must not infinite-loop; each node appears once.
    nodes = {
        "a": {"title": "a", "parent_id": None, "status": "pending", "children": ["b"]},
        "b": {"title": "b", "parent_id": "a", "status": "pending", "children": ["a"]},
    }
    s = viewmodel_state.apply_event(viewmodel_state.initial_state(), _graph_event(nodes))
    assert tuple(t.id for t in s.tasks) == ("a", "b")


def test_tool_call_then_result_pairs_up() -> None:
    s = viewmodel_state.apply_event(
        viewmodel_state.initial_state(),
        {"type": "tool.call", "name": "read_file", "args_preview": {"path": "foo.py"}},
    )
    assert len(s.tool_calls) == 1
    assert s.tool_calls[0].name == "read_file"
    assert s.tool_calls[0].ok is None
    s = viewmodel_state.apply_event(
        s, {"type": "tool.result", "name": "read_file", "ok": True, "summary": "100 bytes"}
    )
    assert s.tool_calls[0].ok is True
    assert s.tool_calls[0].result_summary == "100 bytes"


def test_apply_edit_args_render_the_edit_kinds_not_a_dict_repr() -> None:
    # The drawer + TUI tool table read args_preview; apply_edit's `edits` list
    # used to leak `[{'kind': 'replace', 'old_string': ...}]` as a Python repr.
    s = viewmodel_state.apply_event(
        viewmodel_state.initial_state(),
        {
            "type": "tool.call",
            "name": "apply_edit",
            "args_preview": {
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
    s = viewmodel_state.apply_event(
        viewmodel_state.initial_state(), {"type": "tool.call", "name": "weird", "args_preview": {}}
    )
    s = viewmodel_state.apply_event(
        s, {"type": "tool.result", "name": "weird", "ok": True, "summary": {"unexpected": 1}}
    )
    assert s.tool_calls[0].result_summary == '{"unexpected": 1}'
    line = log_line.format_log_line(
        {"type": "tool.result", "name": "weird", "ok": True, "summary": {"unexpected": 1}}
    )
    assert '{"unexpected": 1}' in line and "'unexpected'" not in line


def test_tool_result_with_mismatched_name_does_not_overwrite() -> None:
    s = viewmodel_state.apply_event(
        viewmodel_state.initial_state(),
        {"type": "tool.call", "name": "read_file", "args_preview": {}},
    )
    s = viewmodel_state.apply_event(
        s, {"type": "tool.result", "name": "other", "ok": True, "summary": "x"}
    )
    assert s.tool_calls[0].ok is None


def test_role_call_then_result_clears_in_flight() -> None:
    s = viewmodel_state.apply_event(
        viewmodel_state.initial_state(),
        {"type": "model.call", "role": "worker", "model": "gpt-5", "provider": "openai"},
    )
    assert s.last_model_call is not None
    assert s.last_model_call.in_flight is True
    s = viewmodel_state.apply_event(
        s, {"type": "model.result", "role": "worker", "input_tokens": 10, "output_tokens": 20}
    )
    assert s.last_model_call is not None
    assert s.last_model_call.in_flight is False


def test_budget_update_populates_view() -> None:
    s = viewmodel_state.apply_event(
        viewmodel_state.initial_state(),
        {
            "type": "budget.update",
            "input_total": 100,
            "output_total": 50,
            "usd_cap": 10.0,
            "tokens_unmetered": 30,
            "tokens_fallback_cap": 1000,
        },
    )
    assert s.budget.execution.input_tokens == 100
    assert s.budget.execution.usd_cap == 10.0
    assert s.budget.execution.unmetered_tokens == 30
    assert s.budget.execution.tokens_fallback_cap == 1000
    # USD fields default cleanly when the event omits them; a missing cap stays unknown.
    assert s.budget.session.usd == 0.0 and s.budget.execution.usd == 0.0
    assert s.budget.session.usd_partial is False
    assert s.budget.execution.plan_cap is None
    # budget.update never carries a per-model token breakdown (no emitter ever
    # wrote one, and no surface reads it); the fold must not resurrect a
    # permanently-empty field. The per-model breakdown is surfaced from the
    # budget snapshot directly, not the viewmodel.
    assert not hasattr(s.budget, "per_model_tokens")


def test_budget_update_carries_usd_total() -> None:
    s = viewmodel_state.apply_event(
        viewmodel_state.initial_state(),
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
    assert s.budget.session.usd == s.budget.execution.usd == 0.1234
    assert s.budget.session.usd_partial is True


def test_budget_usd_cumulative_across_resume_executions() -> None:
    # Each resume execution's budget.update restarts usd_total from 0; the view banks
    # the finished execution on session.resume so "cost" stays the cumulative
    # spend -- the same rule the hub scanner applies (session_log.scan_session_log),
    # keeping the hub row and the run view in agreement.
    def _update(usd: float, *, partial: bool = False) -> dict[str, object]:
        return {"type": "budget.update", "usd_total": usd, "usd_partial": partial}

    s = viewmodel_state.apply_event(viewmodel_state.initial_state(), _update(0.02, partial=True))
    s = viewmodel_state.apply_event(
        s, {"type": "session.stop", "reason": "finish_session", "verification": "passed"}
    )
    s = viewmodel_state.apply_event(s, {"type": "session.resume"})
    # Banked, and the header keeps the old total until the new execution reports.
    assert s.budget.session.usd == 0.02 and s.budget.execution.usd == 0.0
    s = viewmodel_state.apply_event(s, _update(0.005))
    assert s.budget.session.usd == pytest.approx(0.025)
    assert s.budget.execution.usd == pytest.approx(0.005)
    # partial is sticky on the session: execution 1's unpriced spend keeps it an under-estimate.
    assert s.budget.session.usd_partial is True and s.budget.execution.usd_partial is False
    # A second resume banks the cumulative, not just the last execution.
    s = viewmodel_state.apply_event(s, {"type": "session.resume"})
    s = viewmodel_state.apply_event(s, _update(0.001))
    assert s.budget.session.usd == pytest.approx(0.026)


def test_verify_lifecycle() -> None:
    s = viewmodel_state.apply_event(
        viewmodel_state.initial_state(), {"type": "verify.start", "command": ["pytest", "-q"]}
    )
    assert s.last_verify is not None
    assert s.last_verify.exit_code is None
    s = viewmodel_state.apply_event(
        s,
        {
            "type": "verify.end",
            "command": ["pytest", "-q"],
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
    s = viewmodel_state.apply_event(
        viewmodel_state.initial_state(),
        {"type": "approval.prompt", "id": "a001", "prompt": "Allow run_command?"},
    )
    assert len(s.approvals) == 1
    assert s.approvals[0] == ApprovalPrompt(id="a001", prompt="Allow run_command?")
    assert s.approvals[0].head == "Allow run_command?"
    assert s.approvals[0].payload == ""
    s = viewmodel_state.apply_event(s, {"type": "approval.answer", "id": "a001", "approved": True})
    assert s.approvals[0].answered is True
    assert s.approvals[0].approved is True


def test_diff_updated_stores_latest_patch() -> None:
    s = viewmodel_state.apply_event(
        viewmodel_state.initial_state(),
        {"type": "diff.updated", "sha": "abc", "patch": "diff text"},
    )
    assert s.latest_diff == "diff text"
    # a newer diff replaces the prior one
    s = viewmodel_state.apply_event(s, {"type": "diff.updated", "sha": "def", "patch": "newer"})
    assert s.latest_diff == "newer"


def test_run_end_marks_finished() -> None:
    s = viewmodel_state.apply_event(
        viewmodel_state.initial_state(), {"type": "session.stop", "verification": "passed"}
    )
    assert s.finished is True
    assert s.verification == "passed"


def test_unknown_event_type_still_appends_to_log() -> None:
    s = viewmodel_state.apply_event(
        viewmodel_state.initial_state(), {"type": "totally.new", "x": 1}
    )
    # Unknown events should not change SessionState identity-relevant fields
    # but DO go into the log tail.
    assert s.tasks == ()
    assert len(s.log_tail) == 1


def test_tool_history_is_bounded() -> None:
    s = viewmodel_state.initial_state()
    for i in range(200):
        s = viewmodel_state.apply_event(
            s, {"type": "tool.call", "name": f"t{i}", "args_preview": {}}
        )
    assert len(s.tool_calls) == 50  # _MAX_TOOL_HISTORY


def test_full_run_trace_replay() -> None:
    """End-to-end: feed a plausible event sequence and assert final state."""
    tasks = {
        "one": {"title": "one", "parent_id": None, "status": "passed", "children": []},
        "two": {"title": "two", "parent_id": None, "status": "failed", "children": []},
    }
    events = [
        {"type": "session.start", "user_task": "do thing"},
        {"type": "graph.update", "nodes": tasks, "cursor": "two"},
        {"type": "model.call", "role": "worker", "model": "gpt-5"},
        {"type": "tool.call", "name": "apply_patch", "args_preview": {"path": "x.py"}},
        {"type": "tool.result", "name": "apply_patch", "ok": True, "summary": "applied=1"},
        {"type": "model.result", "role": "worker", "input_tokens": 50, "output_tokens": 100},
        {
            "type": "budget.update",
            "input_total": 50,
            "output_total": 100,
            "input_cap": 1000,
            "output_cap": 1000,
        },
        {"type": "diff.updated", "sha": "abc", "patch": "+ added"},
        {"type": "session.stop", "verification": "unverified"},
    ]
    s = viewmodel_state.initial_state()
    for e in events:
        s = viewmodel_state.apply_event(s, e)
    assert s.finished is True
    assert s.verification == "unverified"
    assert s.tasks[0].status == "passed"
    assert s.tasks[1].status == "failed"
    assert s.cursor_task_id == "two"
    assert s.latest_diff == "+ added"
    assert s.budget.execution.input_tokens == 50


def test_log_count_is_monotonic_past_window_cap() -> None:
    # log_tail is a sliding window (MAX_LOG_TAIL); log_count must keep growing
    # so a live viewer can diff on it. A length-based diff freezes once the
    # window saturates -- this is the bug log_count fixes.

    s = viewmodel_state.initial_state()
    n = viewmodel_state.MAX_LOG_TAIL + 50
    for i in range(n):
        s = viewmodel_state.apply_event(s, {"type": "loop.note", "msg": f"line {i}"})
    assert len(s.log_tail) == viewmodel_state.MAX_LOG_TAIL  # window stays capped
    assert s.log_count == n  # but the count keeps climbing


def test_fold_run_reduces_events_to_a_snapshot() -> None:
    state = viewmodel_state.fold_session(
        [
            {"type": "session.start", "user_task": "do it"},
            {"type": "model.call", "role": "worker", "model": "m"},
            {"type": "model.text_delta", "text": "hi"},
            {"type": "session.stop", "verification": "passed"},
        ]
    )
    assert state.task_label.task == "do it"
    assert state.finished and state.verification
    assert state.last_model_call is not None and state.last_model_call.streamed_text == "hi"


def test_run_state_as_dict_is_json_serializable() -> None:
    import json

    state = viewmodel_state.fold_session(
        [
            {"type": "session.start", "user_task": "t"},
            {"type": "tool.call", "name": "grep", "args_preview": {"q": "x"}},
        ]
    )
    d = session_state_as_dict(state)
    assert d["task"] == "t"
    assert d["tool_calls"][0]["name"] == "grep"  # tuple -> list, dataclass -> dict
    json.dumps(d)  # the wire form must serialize


def test_run_state_as_dict_owns_the_dir_backed_identity(tmp_path: pathlib.Path) -> None:
    """The wire owner fills session_id and user_task from the dir when the log has no session.start.

    A resumed or forked execution's log can start at session.resume.
    """
    import json

    session_dir = tmp_path / "sunny-otter-K4Q7B2"
    session_dir.mkdir()
    (session_dir / "manifest.json").write_text(
        json.dumps({"version": 2, "user_task": "queued work"}), encoding="utf-8"
    )
    d = session_state_as_dict(viewmodel_state.fold_session([]), session_dir)
    assert d["session_id"] == "sunny-otter-K4Q7B2"
    assert d["task"] == "queued work"
    assert d["live"] is False  # no worker: the dir word is not live


def test_run_state_as_dict_always_carries_live(tmp_path: pathlib.Path) -> None:
    # `live` is part of the wire shape: None (unknowable) without a dir, a real
    # bool with one -- never an absent key a client must typeof-probe.
    assert session_state_as_dict(viewmodel_state.fold_session([]))["live"] is None
    d = tmp_path / "r"
    d.mkdir()
    assert isinstance(session_state_as_dict(viewmodel_state.fold_session([]), d)["live"], bool)


def test_run_state_as_dict_flags_operator_blocked() -> None:
    """The wire carries operator_blocked from the fold for a dir-less consumer."""
    idle = session_state_as_dict(
        viewmodel_state.fold_session([{"type": "session.start", "user_task": "t"}])
    )
    assert idle["operator_blocked"] is False
    blocked = session_state_as_dict(
        viewmodel_state.fold_session(
            [
                {"type": "session.start", "user_task": "t"},
                {"type": "approval.prompt", "id": "a1", "prompt": "run cmd?"},
            ]
        )
    )
    assert blocked["operator_blocked"] is True


def test_run_status_label_distinguishes_stop_finish_error() -> None:
    # All of these set finished=True; the reason is what a user needs to tell them
    # apart. A stopped run must never read as a bare "finished" (looks completed).
    def end(reason: str, verification: str, *, verify_rc: int | None = None) -> SessionState:
        s = viewmodel_state.apply_event(
            viewmodel_state.initial_state(), {"type": "session.start", "user_task": "t"}
        )
        if verify_rc is not None:
            verify = {"command": ["pytest"], "exit_code": verify_rc, "duration_s": 1.0}
            s = viewmodel_state.apply_event(
                s, {"type": "verify.end", "stdout_tail": "", "stderr_tail": "", **verify}
            )
        return viewmodel_state.apply_event(
            s, {"type": "session.stop", "reason": reason, "verification": verification}
        )

    assert session_state_as_dict(viewmodel_state.initial_state())["status_label"] == "running"
    assert session_state_as_dict(end("steer_abort", "unverified"))["status_label"] == "stopped"
    assert session_state_as_dict(end("finish_session", "passed"))["status_label"] == "passed"
    # verification False is "not certified green": an observed red names the gate,
    # a stale green or a gate nothing ran reads unverified.
    assert session_state_as_dict(end("finish_session", "failed", verify_rc=1))["status_label"] == (
        "finished · gate red"
    )
    assert session_state_as_dict(end("finish_session", "unverified"))["status_label"] == (
        "finished · unverified"
    )
    assert session_state_as_dict(end("finish_session", "unverified", verify_rc=0))[
        "status_label"
    ] == ("finished · unverified")
    assert (
        session_state_as_dict(end("provider_error", "unverified"))["status_label"]
        == "failed · provider error"
    )
    scoped = viewmodel_state.apply_event(
        end("finish_session", "passed"),
        {
            "type": "session.stop",
            "reason": "finish_session",
            "verification": "passed",
            "scoped": True,
        },
    )
    assert session_state_as_dict(scoped)["status_label"] == "passed · scoped gate"
    # the raw status WORD rides along too, so a client can branch on it (the web
    # heartbeat goes quiet on "waiting" instead of painting "working" over a run
    # blocked on the operator).
    assert session_state_as_dict(viewmodel_state.initial_state())["status"] == "running"
    assert session_state_as_dict(end("steer_abort", "unverified"))["status"] == "stopped"
    assert session_state_as_dict(end("finish_session", "passed"))["status"] == "passed"


def test_resume_start_unfinishes_the_run() -> None:
    # A resume restarts a finished/stopped run in place; the header must show it
    # running again (else steer/stop stay disabled on a live run).
    s = viewmodel_state.apply_event(
        viewmodel_state.initial_state(), {"type": "session.stop", "reason": "steer_abort"}
    )
    assert s.finished and s.stop_reason == "steer_abort"
    s = viewmodel_state.apply_event(s, {"type": "session.resume"})
    assert not s.finished and s.stop_reason == ""
    assert session_state_as_dict(s)["status_label"] == "running"


def test_role_result_tracks_context_tokens_and_provider() -> None:
    """model.result folds the call's full prompt into context_tokens, which outlives the next call.

    Fresh, cache read and cache write together; an error result without usage keeps the last known.
    """
    s = viewmodel_state.initial_state()
    s = viewmodel_state.apply_event(
        s, {"type": "model.call", "role": "worker", "model": "m", "provider": "p"}
    )
    assert s.last_model_call is not None and s.last_model_call.provider == "p"
    assert s.last_model_call.context_tokens == 0  # nothing measured yet
    s = viewmodel_state.apply_event(
        s,
        {
            "type": "model.result",
            "role": "worker",
            "ok": True,
            "input_tokens": 1_000,
            "cache_read_tokens": 40_000,
            "cache_write_tokens": 2_000,
        },
    )
    assert s.last_model_call is not None and s.last_model_call.context_tokens == 43_000
    s = viewmodel_state.apply_event(
        s, {"type": "model.call", "role": "worker", "model": "m", "provider": "p"}
    )
    assert (
        s.last_model_call is not None and s.last_model_call.context_tokens == 43_000
    )  # carried over
    s = viewmodel_state.apply_event(
        s, {"type": "model.result", "role": "worker", "ok": False, "error": "boom"}
    )
    assert (
        s.last_model_call is not None and s.last_model_call.context_tokens == 43_000
    )  # kept on error


def test_run_start_after_run_end_unfinishes_without_banking() -> None:
    """A session.start without a resume marker clears the terminal state and banks no usd.

    The ask REPL re-enters run() per follow-up on one BudgetTracker, so usd_total is already
    cumulative and banking would double-count.
    """
    s = viewmodel_state.initial_state()
    s = viewmodel_state.apply_event(s, {"type": "session.start", "user_task": "q"})
    s = viewmodel_state.apply_event(s, {"type": "budget.update", "usd_total": 0.02})
    s = viewmodel_state.apply_event(
        s, {"type": "session.stop", "verification": "unverified", "reason": "answered"}
    )
    s = viewmodel_state.apply_event(s, {"type": "session.start", "user_task": "follow-up"})
    s = viewmodel_state.apply_event(s, {"type": "model.call", "role": "worker", "model": "m"})
    assert s.finished is False
    assert s.stop_reason == ""
    assert session_state_as_dict(s)["status_label"] == "running"
    s = viewmodel_state.apply_event(s, {"type": "budget.update", "usd_total": 0.03})
    assert s.budget.session.usd == pytest.approx(0.03)
    assert s.budget.execution.usd == pytest.approx(0.03)  # nothing banked: one tracker


def test_resume_resets_the_execution_token_counters() -> None:
    """ResumeStart banks usd and drops the dead execution's token counters and caps.

    BudgetView documents them as the current execution's; the header would otherwise render the
    finished execution's ~100%.
    """
    s = viewmodel_state.initial_state()
    s = viewmodel_state.apply_event(s, {"type": "session.start", "user_task": "t"})
    s = viewmodel_state.apply_event(
        s,
        {
            "type": "budget.update",
            "input_total": 9500,
            "output_total": 480,
            "usd_total": 0.2,
            "usd_cap": 10.0,
            "tokens_unmetered": 700,
            "tokens_fallback_cap": 1000,
        },
    )
    s = viewmodel_state.apply_event(
        s, {"type": "session.stop", "verification": "unverified", "reason": "budget_exhausted"}
    )
    s = viewmodel_state.apply_event(s, {"type": "session.resume"})
    assert s.budget.execution == BudgetView().execution  # zeros, the caps unknown again
    assert s.budget.session.usd == pytest.approx(0.2)


def test_resume_resets_the_execution_plan_counters() -> None:
    """A resume drops the plan cap and banks the observed points; the account's reading stays."""
    s = viewmodel_state.initial_state()
    s = viewmodel_state.apply_event(s, {"type": "session.start", "user_task": "t"})
    s = viewmodel_state.apply_event(
        s,
        {
            "type": "budget.update",
            "plan_used_percent": 42.0,
            "plan_observed_points": 3.5,
            "plan_cap": 80.0,
            "plan_resets_at": 1_700_000_000.0,
        },
    )
    s = viewmodel_state.apply_event(
        s, {"type": "session.stop", "verification": "unverified", "reason": "budget_exhausted"}
    )
    s = viewmodel_state.apply_event(s, {"type": "session.resume"})
    assert s.budget.account is not None and s.budget.account.used_percent == 42.0
    assert s.budget.session.plan_observed_points == 3.5
    assert s.budget.execution.plan_observed_points == 0.0
    assert s.budget.execution.plan_cap is None


def test_plan_points_cumulative_across_resume_executions() -> None:
    """The view's plan points sum every execution's, with the banked part beside them.

    The points were the latest execution's, so a resumed run's cost fell to what the resume
    consumed.
    """
    s = viewmodel_state.initial_state()
    s = viewmodel_state.apply_event(s, {"type": "session.start", "user_task": "t"})
    update = {"type": "budget.update", "plan_used_percent": 42.0, "plan_cap": 6.0}
    s = viewmodel_state.apply_event(s, {**update, "plan_observed_points": 2.5})
    s = viewmodel_state.apply_event(s, {"type": "session.resume"})
    s = viewmodel_state.apply_event(s, {**update, "plan_observed_points": 1.0})
    assert s.budget.session.plan_observed_points == 3.5
    assert s.budget.execution.plan_observed_points == 1.0


def test_concurrent_same_name_results_pair_by_call_id() -> None:
    """Concurrent calls pair their results on the stamped call_id.

    Last-entry name pairing cross-stamped two review seats' read_file summaries.
    """
    s = viewmodel_state.initial_state()
    s = viewmodel_state.apply_event(
        s,
        {"type": "tool.call", "name": "read_file", "args_preview": {"path": "a.py"}, "call_id": 1},
    )
    s = viewmodel_state.apply_event(
        s,
        {"type": "tool.call", "name": "read_file", "args_preview": {"path": "b.py"}, "call_id": 2},
    )
    s = viewmodel_state.apply_event(
        s, {"type": "tool.result", "name": "read_file", "ok": True, "summary": "sa", "call_id": 1}
    )
    s = viewmodel_state.apply_event(
        s, {"type": "tool.result", "name": "read_file", "ok": True, "summary": "sb", "call_id": 2}
    )
    a, b = s.tool_calls
    assert (a.result_summary, a.ok) == ("sa", True)
    assert (b.result_summary, b.ok) == ("sb", True)


def test_interleaved_result_for_an_earlier_call_pairs_by_call_id() -> None:
    """Call A, call B, result A lands on A while B stays open."""
    s = viewmodel_state.initial_state()
    s = viewmodel_state.apply_event(
        s,
        {"type": "tool.call", "name": "read_file", "args_preview": {"path": "a.py"}, "call_id": 1},
    )
    s = viewmodel_state.apply_event(
        s, {"type": "tool.call", "name": "grep", "args_preview": {"pattern": "x"}, "call_id": 2}
    )
    s = viewmodel_state.apply_event(
        s, {"type": "tool.result", "name": "read_file", "ok": True, "summary": "sa", "call_id": 1}
    )
    a, b = s.tool_calls
    assert (a.result_summary, a.ok) == ("sa", True)
    assert b.ok is None  # still in flight


def test_tool_count_is_the_executions_dispatched_calls() -> None:
    """`tool_count` exceeds the `tool_calls` window, excludes finish calls, resets on a resume."""
    s = viewmodel_state.apply_event(
        viewmodel_state.initial_state(), {"type": "session.start", "user_task": "t"}
    )
    for i in range(60):
        s = viewmodel_state.apply_event(
            s, {"type": "tool.call", "name": "read_file", "args_preview": {"path": f"{i}.py"}}
        )
    s = viewmodel_state.apply_event(
        s, {"type": "tool.call", "name": "finish_session", "args_preview": {"summary": "done"}}
    )
    assert (s.tool_count, len(s.tool_calls)) == (60, 50)
    s = viewmodel_state.apply_event(s, {"type": "session.resume"})
    s = viewmodel_state.apply_event(s, {"type": "tool.call", "name": "grep", "args_preview": {}})
    assert s.tool_count == 1


def test_compaction_events_fold_into_elision_counters() -> None:
    """/status counts elided markers and live gists in the current context."""
    s = viewmodel_state.initial_state()
    s = viewmodel_state.apply_event(
        s, {"type": "compact.dropped", "count": 3, "calls": ["read_file a.py", "grep 'q'"]}
    )
    s = viewmodel_state.apply_event(
        s,
        {
            "type": "compact.gist.done",
            "gisted": 2,
            "demoted": 1,
            "paths": ["a.py", "b.py"],
            "demoted_paths": ["c.py"],
        },
    )
    s = viewmodel_state.apply_event(
        s, {"type": "compact.dropped", "count": 1, "calls": ["read_file d.py"]}
    )
    assert s.compact_elided == 4
    assert s.compact_gists_live == 1


def test_format_log_line_compaction_identities() -> None:
    dropped = log_line.format_log_line(
        {"type": "compact.dropped", "count": 2, "calls": ["read_file a.py", "grep 'q'"]}
    )
    assert "elided 2" in dropped and "read_file a.py" in dropped and "grep 'q'" in dropped
    gists = log_line.format_log_line(
        {
            "type": "compact.gist.done",
            "gisted": 1,
            "demoted": 1,
            "paths": ["a.py"],
            "demoted_paths": ["b.py"],
        }
    )
    assert "1 distilled (a.py)" in gists and "1 demoted (b.py)" in gists
    done = log_line.format_log_line(
        {"type": "compact.summarise.done", "summary_chars": 2341, "summary": "did things"}
    )
    assert "2341-char progress summary" in done


def test_pin_added_events_accumulate() -> None:
    s = viewmodel_state.initial_state()
    s = viewmodel_state.apply_event(
        s, {"type": "pin.added", "text": "never touch schema", "chars": 18, "count": 1}
    )
    s = viewmodel_state.apply_event(
        s, {"type": "pin.added", "text": "ship X first", "chars": 12, "count": 2}
    )
    assert s.pins == ("never touch schema", "ship X first")


def test_tier2_restart_resets_elision_counters() -> None:
    """A summarise-and-restart resets the /status elision counters with the context it wiped."""
    s = viewmodel_state.initial_state()
    s = viewmodel_state.apply_event(
        s, {"type": "compact.dropped", "count": 9, "calls": ["read_file a.py"]}
    )
    s = viewmodel_state.apply_event(
        s,
        {
            "type": "compact.gist.done",
            "gisted": 3,
            "demoted": 0,
            "paths": ["a.py", "b.py", "c.py"],
            "demoted_paths": [],
        },
    )
    s = viewmodel_state.apply_event(
        s, {"type": "compact.summarise.done", "summary_chars": 900, "summary": "s"}
    )
    assert s.compact_elided == 0
    assert s.compact_gists_live == 0


def test_pins_restored_event_replaces_not_appends() -> None:
    """pin.restored never double-counts a resume's pins and shows a fork's snapshot pins."""
    s = viewmodel_state.initial_state()
    s = viewmodel_state.apply_event(
        s, {"type": "pin.added", "text": "keep A", "chars": 6, "count": 1}
    )
    s = viewmodel_state.apply_event(s, {"type": "pin.restored", "pins": ["keep A"], "count": 1})
    assert s.pins == ("keep A",)  # replace, not append
    fork = viewmodel_state.apply_event(
        viewmodel_state.initial_state(),
        {"type": "pin.restored", "pins": ["keep A", "ship X"], "count": 2},
    )
    assert fork.pins == ("keep A", "ship X")


# One wrong-shaped field per known family, each syntactically valid JSON: the
# containers the fold reads without a type check. None is producible by
# agent6's own writers; a corrupted, hand-edited, or foreign-version log is.
_WRONG_SHAPE_EVENTS: list[dict[str, Any]] = [
    {"type": "graph.update", "nodes": [1]},
    {"type": "graph.update", "nodes": 7},
    {"type": "tool.call", "name": "x", "args_preview": [1]},
    {"type": "tool.call", "name": "x", "args_preview": "not-a-dict"},
    {"type": "question.prompt", "questions": {"question": "q"}},
    {"type": "question.answer", "id": "q1", "answers": 5},
    {"type": "compact.dropped", "count": 1, "calls": 5},
    {"type": "compact.gist.done", "gisted": 1, "paths": True, "demoted": 1, "demoted_paths": 1},
    {"type": "graph.update", "nodes": {"a": {"title": "a", "children": 5}}},
]


@pytest.mark.parametrize(
    "bad", _WRONG_SHAPE_EVENTS, ids=lambda e: f"{e['type']}-{type(sorted(e)[0]).__name__}"
)
def test_fold_is_total_for_wrong_shaped_containers(bad: dict[str, Any]) -> None:
    """Any syntactically valid JSON object folds without raising.

    The fold runs unwrapped inside live tails, so one wrong-shaped field crashed every viewer
    and turned the web run endpoint into a 500.
    """
    state = viewmodel_state.fold_session(
        [
            {"type": "session.start", "user_task": "t"},
            bad,
            {"type": "session.stop", "reason": "finish_session", "verification": "passed"},
        ]
    )
    assert state.finished  # the fold survived the bad line and kept folding


def test_a_question_asked_while_no_model_runs_is_the_harness_s() -> None:
    """A question before session.start or a resumed execution's start is agent6's, not the model's.

    The TUI modal and the web prompt box name the asker from this flag.
    """
    q = {"type": "question.prompt", "id": "question-1", "questions": [{"question": "stash?"}]}
    before = viewmodel_state.fold_session([q])
    assert before.questions[0].from_harness is True
    during = viewmodel_state.fold_session([{"type": "session.start", "user_task": "t"}, q])
    assert during.questions[0].from_harness is False
    after = viewmodel_state.fold_session(
        [
            {"type": "session.start", "user_task": "t"},
            {"type": "session.stop", "reason": "finish_session", "verification": "passed"},
            {**q, "id": "question-2"},
        ]
    )
    assert after.questions[-1].from_harness is True


def test_run_state_as_dict_carries_what_the_run_serves(tmp_path: pathlib.Path) -> None:
    """`ports` is the wire form of `listening_ports`, so a header can name the agent's server."""
    import os
    import socket

    d = tmp_path / "r"
    d.mkdir()
    assert session_state_as_dict(viewmodel_state.fold_session([]), d)["ports"] == []
    with socket.socket() as srv:
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port = srv.getsockname()[1]
        # this process stands in for the holder
        process.write_session_netns_pid(d, os.getpid())
        assert port in session_state_as_dict(viewmodel_state.fold_session([]), d)["ports"]


def test_approval_parts_is_the_one_shape_every_surface_renders() -> None:
    """A dispatch prompt splits into the "Allow <tool>" head and the payload; JSON carries both."""
    assert approval_parts("Allow run_command: pytest -q tests") == (
        "Allow run_command",
        "pytest -q tests",
    )
    assert approval_parts("Allow run_command: sh -c 'a\nb'") == (
        "Allow run_command",
        "sh -c 'a\nb'",
    )
    assert approval_parts("Allow fetch?") == ("Allow fetch?", "")


def test_auto_commits_fold_into_the_commit_list() -> None:
    """Commits stay oldest first; a sha-less auto_commit adds no commit."""
    s = viewmodel_state.apply_event(
        viewmodel_state.initial_state(),
        {"type": "checkpoint.committed", "turn": 3, "sha": "a" * 40, "subject": "fix parser"},
    )
    s = viewmodel_state.apply_event(s, {"type": "checkpoint.committed", "turn": 4, "sha": ""})
    s = viewmodel_state.apply_event(
        s, {"type": "checkpoint.committed", "turn": 5, "sha": "b" * 40, "subject": "add test"}
    )
    assert [(st.turn, st.sha[:1], st.subject) for st in s.commits] == [
        (3, "a", "fix parser"),
        (5, "b", "add test"),
    ]


def test_run_history_is_named_commits_on_the_wire() -> None:
    """The snapshot exposes commits, never the removed steps field."""
    from agent6.viewmodel.session import view as state_view

    state = viewmodel_state.fold_session(
        [{"type": "checkpoint.committed", "turn": 3, "sha": "a" * 40, "subject": "fix parser"}]
    )
    payload = json.loads(json.dumps(state_view.session_state_as_dict(state)))
    assert "steps" not in payload
    assert payload["commits"] == [
        {
            "turn": 3,
            "sha": "a" * 40,
            "subject": "fix parser",
            "label": "turn 3 · aaaaaaa · fix parser",
        }
    ]


def test_fold_until_commit_stops_at_that_commit() -> None:
    """The fold includes the selected commit and refuses a foreign sha."""
    events = [
        {"type": "session.start", "session_id": "s", "mode": "run", "user_task": "t"},
        {"type": "checkpoint.committed", "turn": 1, "sha": "a" * 40, "subject": "one"},
        {"type": "checkpoint.committed", "turn": 2, "sha": "b" * 40, "subject": "two"},
    ]
    at = viewmodel_state.fold_until_commit(events, "a" * 40)
    assert at is not None and [s.turn for s in at.commits] == [1]
    assert [s.turn for s in viewmodel_state.fold_session(events).commits] == [1, 2]
    assert viewmodel_state.fold_until_commit(events, "c" * 40) is None
    # A prefix of at least seven hex digits names the commit, like `/diff?sha=`.
    by_prefix = viewmodel_state.fold_until_commit(events, "b" * 7)
    assert by_prefix is not None and [s.turn for s in by_prefix.commits] == [1, 2]
    assert viewmodel_state.fold_until_commit(events, "b" * 6) is None


def test_a_cut_log_line_says_it_was_cut() -> None:
    """A clipped log-line field is marked, so a provider error never reads as ending there."""
    error = "E" * 150 + "-TRUNCATION-BOUNDARY-" + "Z" * 60
    line = log_line.format_log_line(
        {"ts": "2026-09-01T22:00:00.000Z", "type": "model.result", "role": "worker", "error": error}
    )
    assert "…" in line and "BOUNDARY" not in line
    prompt = log_line.format_log_line(
        {"ts": "t", "type": "approval.prompt", "prompt": "run " + "x" * 200}
    )
    assert "…" in prompt
    short = log_line.format_log_line({"ts": "t", "type": "approval.prompt", "prompt": "run ls"})
    assert "…" not in short


def test_one_owner_for_the_open_approval() -> None:
    """One rule picks which pending approval an answer goes to.

    The server took the oldest, the conversation view offered the newest and the dashboard the
    oldest with no regard to a modal already up.
    """
    s = viewmodel_state.initial_state()
    for aid in ("a1", "a2", "a3"):
        s = viewmodel_state.apply_event(
            s, {"type": "approval.prompt", "id": aid, "prompt": f"run {aid}", "standing": True}
        )
    s = viewmodel_state.apply_event(s, {"type": "approval.answer", "id": "a1", "answer": "yes"})
    current = viewmodel_state.open_approval_of(s)
    assert current is not None and current.id == "a2"
    taken = viewmodel_state.open_approval_of(s, taken=lambda aid: aid == "a2")
    assert taken is not None and taken.id == "a3"
    assert viewmodel_state.open_approval_of(s, taken=lambda _aid: True) is None


def test_a_commit_label_is_one_line_every_surface_shows() -> None:
    """The shared commit label has no dangling separator when the subject is empty."""
    assert (
        state_types.RunCommit(3, "abcdef1234", "fix: parse").label
        == "turn 3 · abcdef1 · fix: parse"
    )
    assert state_types.RunCommit(3, "abcdef1234", "").label == "turn 3 · abcdef1"


def test_the_end_takes_the_cursor_off_the_task_in_progress() -> None:
    """Once the run ended no task is in progress: each draws its own status."""
    nodes = {
        "root": {"title": "root", "parent_id": None, "status": "passed", "children": ["a"]},
        "a": {"title": "task a", "parent_id": "root", "status": "passed", "children": []},
    }
    s = viewmodel_state.apply_event(
        viewmodel_state.initial_state(), _graph_event(nodes, cursor="a")
    )
    assert s.tasks[1].is_cursor and s.tasks[1].glyph == "▸"
    s = viewmodel_state.apply_event(
        s, {"type": "session.stop", "reason": "finish_session", "verification": "passed"}
    )
    assert not s.tasks[1].is_cursor and s.tasks[1].glyph == "✓"


def test_the_snapshot_keeps_the_folded_original_task(tmp_path: pathlib.Path) -> None:
    """The snapshot retains the folded label even before its journal reaches disk."""
    from agent6.app import stamps
    from agent6.sessions import manifest as sessions_manifest

    session_dir = tmp_path / "state" / "sessions" / "runs" / "run-AAAA11"
    session_dir.mkdir(parents=True)
    (session_dir / "logs.jsonl").write_text("", encoding="utf-8")
    stamps.write_manifest(
        session_dir / "manifest.json",
        sessions_manifest.SessionManifest(
            session_id="run-AAAA11", mode="run", user_task="the follow-up"
        ),
    )
    s = viewmodel_state.apply_event(
        viewmodel_state.initial_state(), {"type": "session.start", "user_task": "the first task"}
    )
    d = session_state_as_dict(s, session_dir)
    assert d["task"] == "the first task"
    assert d["task_line"] == "the first task"


@pytest.mark.parametrize("manifest_text", ["", "not JSON"])
def test_task_identity_survives_missing_or_invalid_manifest(
    tmp_path: pathlib.Path,
    manifest_text: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Snapshot task fields agree without a readable manifest or a journal scan."""
    from agent6.viewmodel import listing

    def no_scan(*_args: object, **_kwargs: object) -> None:
        pytest.fail("a snapshot must not scan the journal")

    monkeypatch.setattr(listing, "summarize_session_dir", no_scan)
    if manifest_text:
        (tmp_path / "manifest.json").write_text(manifest_text)
    folded = viewmodel_state.fold_session(
        [
            {"type": "session.start", "user_task": "original task"},
            {"type": "session.stop"},
            {"type": "session.resume"},
        ]
    )
    payload = session_state_as_dict(folded, tmp_path)
    assert payload["task"] == payload["task_line"] == "original task"
    assert payload["continued_task"] == ""


def test_snapshot_reads_manifest_once(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Task, mode and policy share the snapshot's manifest read."""
    from agent6.sessions import manifest

    original = manifest.read_manifest
    calls = 0

    def counted(path: pathlib.Path) -> manifest.SessionManifest:
        nonlocal calls
        calls += 1
        return original(path)

    (tmp_path / "manifest.json").write_text(json.dumps({"user_task": "original task"}))
    monkeypatch.setattr(manifest, "read_manifest", counted)
    folded = viewmodel_state.fold_session([{"type": "session.start", "user_task": "original task"}])
    session_state_as_dict(folded, tmp_path)
    assert calls == 1


def test_a_journal_written_before_the_turn_rename_keeps_its_commit_turns() -> None:
    """An `auto_commit` event stamped `iteration` folds to a commit with that turn number."""
    s = viewmodel_state.apply_event(
        viewmodel_state.initial_state(),
        {"type": "checkpoint.committed", "iteration": 3, "sha": "a" * 40, "subject": "fix parser"},
    )
    assert [st.turn for st in s.commits] == [3]


def test_a_queued_question_carries_across_executions_and_blocks_nothing() -> None:
    """A queued question stays pending past a resume; only a waiting question blocks the run."""
    from agent6.viewmodel.session import view as state_view

    queued = {
        "type": "question.prompt",
        "id": "question-1",
        "questions": [{"question": "a?", "options": []}],
        "call_id": 1,
        "wait": False,
    }
    waiting = {**queued, "id": "question-2", "call_id": 2, "wait": True}
    rows = [
        {"type": "session.start", "user_task": "t"},
        queued,
        {"type": "session.resume", "user_task": "t", "steer": ""},
    ]
    state = viewmodel_state.fold_session(rows)
    (first,) = state.questions
    assert first.id == "question-1" and not first.answered and first.wait is False
    assert state_view.status_facts(state).operator_blocked is False
    facts = state_view.status_facts(viewmodel_state.fold_session([*rows, waiting]))
    assert facts.operator_blocked and facts.blocked_kind == "question"
    late = {
        "type": "question.answer",
        "id": "question-1",
        "answers": ["x"],
        "source": "late",
        "unseen": False,
    }
    state = viewmodel_state.fold_session([*rows, waiting, late])
    assert [(q.id, q.answered, q.source) for q in state.questions] == [
        ("question-1", True, "late"),
        ("question-2", False, ""),
    ]
    assert state_view.status_facts(state).operator_blocked


def test_a_questions_kind_is_one_word_on_every_surface() -> None:
    """Waiting while its execution lives, asked by an earlier execution after, queued, answered."""
    rows = [
        {"type": "session.start", "user_task": "t"},
        {"type": "question.prompt", "id": "question-1", "questions": [{"question": "a?"}]},
        {
            "type": "question.prompt",
            "id": "question-2",
            "questions": [{"question": "b?"}],
            "wait": False,
        },
    ]
    before = viewmodel_state.fold_session(rows)
    assert [q.kind() for q in before.questions] == ["waiting", "queued"]
    after = viewmodel_state.fold_session([*rows, {"type": "session.resume", "user_task": "t"}])
    assert [q.kind() for q in after.questions] == [
        "asked by an earlier execution",
        "queued",
    ]
    answered = viewmodel_state.fold_session(
        [*rows, {"type": "question.answer", "id": "question-2", "answers": ["x"], "source": "late"}]
    )
    assert answered.questions[1].kind() == "answered"


def test_a_question_outside_an_execution_is_nobodys_to_answer_later(tmp_path: pathlib.Path) -> None:
    """agent6's own start gate is pending while its asker lives and nobody's once it is gone."""
    import os

    from agent6.app import resume_questions
    from agent6.sessions import process
    from agent6.viewmodel.session import log as session_log
    from agent6.viewmodel.session import view as state_view

    gate = {"type": "question.prompt", "id": "question-1", "questions": [{"question": "stash?"}]}
    log = tmp_path / "logs.jsonl"
    log.write_text(json.dumps(gate) + "\n", encoding="utf-8")
    state = viewmodel_state.fold_session([gate])
    assert state.questions[0].from_harness
    assert state_view.status_facts(state).pending_questions == 0
    assert session_log.scan_session_log(log).pending_questions == 0
    assert viewmodel_state.pending_questions(tmp_path) == ()
    carried = resume_questions.carried_questions(log)
    assert (carried.pending, carried.next_number) == ((), 1)
    process.write_worker_pid(tmp_path, os.getpid())
    assert [q.id for q in viewmodel_state.pending_questions(tmp_path)] == ["question-1"]
    assert resume_questions.carried_questions(log).pending == ()


def test_a_reused_question_id_names_the_latest_question(tmp_path: pathlib.Path) -> None:
    """A journal from before ids ran across executions holds one pending question per id."""
    from agent6.app import resume_questions

    rows = [
        {"type": "session.start", "mode": "run", "user_task": "t"},
        {"type": "question.prompt", "id": "question-1", "questions": [{"question": "old?"}]},
        {"type": "session.resume"},
        {"type": "question.prompt", "id": "question-1", "questions": [{"question": "new?"}]},
    ]
    log = tmp_path / "logs.jsonl"
    log.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    state = viewmodel_state.fold_session(rows)
    assert [(q.id, q.questions[0].question) for q in state.questions] == [("question-1", "new?")]
    carried = resume_questions.carried_questions(log)
    assert [(q.id, q.questions[0].question) for q in carried.pending] == [("question-1", "new?")]
    assert carried.next_number == 2


def test_the_state_names_its_execution_from_the_start_rows() -> None:
    """The live state carries the current execution's id, and a resume replaces it."""
    s = viewmodel_state.apply_event(
        viewmodel_state.initial_state(),
        {"type": "session.start", "mode": "run", "user_task": "t", "execution_id": "first"},
    )
    assert s.execution_id == "first"
    s = viewmodel_state.apply_event(s, {"type": "session.resume", "execution_id": "second"})
    assert s.execution_id == "second"
    assert session_state_as_dict(s)["execution_id"] == "second"


def test_the_account_reading_names_its_provider_and_survives_a_resume() -> None:
    """The view keeps the latest account reading, with the provider and when it was read."""
    s = viewmodel_state.apply_event(viewmodel_state.initial_state(), {"type": "session.start"})
    s = viewmodel_state.apply_event(
        s,
        {
            "type": "budget.update",
            "ts": "2026-07-14T10:00:30+00:00",
            "usd_total": 0.0,
            "plan_observed_points": 1.0,
            "plan_cap": 5.0,
            "account": {
                "provider": "chatgpt",
                "used_percent": 42.0,
                "resets_at": 2e9,
                "windows": [
                    {
                        "name": "primary",
                        "used_percent": 42.0,
                        "window_minutes": 300,
                        "resets_at": 2e9,
                    }
                ],
                "credits_has": False,
                "credits_unlimited": False,
                "credits_balance": "",
            },
        },
    )
    account = s.budget.account
    assert account is not None and account.provider == "chatgpt" and account.observed_at > 0
    assert [w.name for w in account.windows] == ["primary"]
    s = viewmodel_state.apply_event(s, {"type": "session.resume"})
    assert s.budget.account == account
    assert s.budget.execution.plan_cap is None and s.budget.session.plan_observed_points == 1.0
