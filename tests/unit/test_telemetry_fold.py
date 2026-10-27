# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The telemetry projection: spans, metrics and logs from one journal, metadata-only by default."""

from __future__ import annotations

import datetime
import json
import pathlib
from typing import Any

from agent6.viewmodel import telemetry, telemetry_metrics, telemetry_records
from tests.unit import test_review_queue as queue_tests

_T0 = datetime.datetime(2026, 10, 25, 10, 0, tzinfo=datetime.UTC)


def _ts(seconds: float) -> str:
    return (_T0 + datetime.timedelta(seconds=seconds)).isoformat()


def _journal() -> list[dict[str, Any]]:
    """One execution: a model call, a gated tool, a red gate, a blocking panel, an end."""
    start, panel = queue_tests._panel_rows(block=1)  # pyright: ignore[reportPrivateUsage]
    return [
        {
            "type": "session.start",
            "ts": _ts(0),
            "mode": "run",
            "user_task": "SECRET TASK",
            "execution_id": "e1",
        },
        {"type": "loop.budget", "ts": _ts(1), "turn": 1},
        {
            "type": "role.call",
            "ts": _ts(1),
            "role": "worker",
            "model": "m",
            "provider": "p",
            "model_call_id": "c1",
        },
        {
            "type": "role.result",
            "ts": _ts(3),
            "role": "worker",
            "model": "m",
            "provider": "p",
            "model_call_id": "c1",
            "ok": True,
            "tokens_in": 10,
            "tokens_out": 5,
            "cache_read": 2,
            "cache_creation": 0,
            "stop_reason": "tool_use",
        },
        {
            "type": "tool.call",
            "ts": _ts(3),
            "name": "run_command",
            "args": {"argv": ["rm", "-rf", "build"]},
            "call_id": 1,
        },
        {"type": "approval.prompt", "ts": _ts(3), "id": "a1", "prompt": "Run rm?", "call_id": 1},
        {"type": "approval.answer", "ts": _ts(9), "id": "a1", "approved": True, "source": "tui"},
        {
            "type": "tool.result",
            "ts": _ts(10),
            "name": "run_command",
            "ok": True,
            "summary": "rc=0",
            "call_id": 1,
        },
        {"type": "verify.start", "ts": _ts(10), "verify_id": "v1", "cmd": ["pytest"]},
        {
            "type": "verify.end",
            "ts": _ts(40),
            "verify_id": "v1",
            "cmd": ["pytest"],
            "exit_code": 1,
            "duration_s": 30.0,
            "tree_before": "t",
            "tree_after": "t",
            "scoped": False,
        },
        {"type": "loop.budget", "ts": _ts(41), "turn": 2},
        {**start, "ts": _ts(42)},
        {**panel, "ts": _ts(45)},
        {
            "type": "budget.update",
            "ts": _ts(50),
            "usd_total": 0.25,
            "usd_partial": False,
            "plan_consumed": 3,
            "plan_used_percent": 12,
        },
        {
            "type": "session.end",
            "ts": _ts(60),
            "reason": "finish_session",
            "turns": 2,
            "all_passed": False,
            "verified": "failed",
            "tree": "t",
        },
    ]


def _run(
    rows: list[dict[str, Any]], *, capture: telemetry.Capture = "metadata"
) -> tuple[
    telemetry.TelemetryFold, list[telemetry_records.Span], list[telemetry_records.LogRecord]
]:
    fold = telemetry.TelemetryFold("run-AAAAAA", capture=capture)
    spans: list[telemetry_records.Span] = []
    logs: list[telemetry_records.LogRecord] = []
    for row in rows:
        emitted = fold.feed(row)
        spans.extend(emitted.spans)
        logs.extend(emitted.logs)
    closing = fold.close()
    spans.extend(closing.spans)
    return fold, spans, logs


def _review_version(rows: list[dict[str, Any]]) -> int:
    from agent6.viewmodel import review

    view = review.ReviewView()
    for row in rows:
        view = review.apply_event(view, row)
    return view.version


def _by_name(spans: list[telemetry_records.Span]) -> dict[str, telemetry_records.Span]:
    return {span.name: span for span in spans}


def _attr(span: telemetry_records.Span | telemetry_records.LogRecord, key: str) -> object:
    return dict(span.attributes).get(key)


def _counter(fold: telemetry.TelemetryFold, name: str, **labels: object) -> float:
    key = (name, tuple(sorted(labels.items())))
    return fold.metrics().counters.get(key, 0.0)  # type: ignore[arg-type]


def test_a_journal_folds_into_a_deterministic_span_tree() -> None:
    """Each unit of work is a span under its turn; the ids derive from the journal."""
    _fold1, spans, _logs = _run(_journal())
    _fold2, again, _ = _run(_journal())
    assert spans == again
    turns = [s for s in spans if s.name == telemetry_records.SPAN_TURN]
    assert [_attr(t, "agent6.turn") for t in turns] == [1, 2]
    named = _by_name(spans)
    execution = named[telemetry_records.SPAN_EXECUTION]
    assert len(execution.trace_id) == 32 and len(execution.span_id) == 16
    assert execution.parent_span_id == "" and {s.trace_id for s in spans} == {execution.trace_id}
    assert all(t.parent_span_id == execution.span_id for t in turns)
    turn_one, turn_two = (t.span_id for t in turns)
    assert named[telemetry_records.SPAN_MODEL_CALL].parent_span_id == turn_one
    assert named[telemetry_records.SPAN_TOOL].parent_span_id == turn_one
    assert (
        named[telemetry_records.SPAN_WAIT].parent_span_id
        == named[telemetry_records.SPAN_TOOL].span_id
    )
    assert named[telemetry_records.SPAN_VERIFY].parent_span_id == turn_one
    assert named[telemetry_records.SPAN_REVIEW].parent_span_id == turn_two
    assert named[telemetry_records.SPAN_MODEL_CALL].kind == "client"
    assert (
        named[telemetry_records.SPAN_MODEL_CALL].status,
        named[telemetry_records.SPAN_TOOL].status,
    ) == (
        "ok",
        "ok",
    )
    assert (
        named[telemetry_records.SPAN_VERIFY].status,
        named[telemetry_records.SPAN_REVIEW].status,
    ) == (
        "error",
        "error",
    )
    assert execution.status == "error" and _attr(execution, "agent6.verified") == "failed"
    assert execution.start_ns == telemetry_records.event_ns(_ts(0))
    assert execution.end_ns == telemetry_records.event_ns(_ts(60))
    assert (
        named[telemetry_records.SPAN_WAIT].end_ns - named[telemetry_records.SPAN_WAIT].start_ns
        == 6_000_000_000
    )
    assert _attr(named[telemetry_records.SPAN_MODEL_CALL], "agent6.tokens_out") == 5
    assert _attr(named[telemetry_records.SPAN_VERIFY], "agent6.verify.outcome") == "failed"
    assert _attr(named[telemetry_records.SPAN_REVIEW], "agent6.panel.blocked") is True
    assert _fold1.first_ns == telemetry_records.event_ns(_ts(0))


def test_metadata_capture_carries_no_task_arguments_or_commands() -> None:
    """The default export names outcomes and ids, never the task, the argv or the gate command."""
    _fold, spans, logs = _run(_journal())
    text = json.dumps([s.attributes for s in spans] + [(log.body, log.attributes) for log in logs])
    assert "SECRET" not in text and '"rm"' not in text and "pytest" not in text
    assert "agent6.task" not in text and "agent6.tool.args" not in text


def test_content_capture_adds_only_the_capped_excerpts() -> None:
    long_task = "x" * (telemetry_records.CONTENT_CHARS + 50)
    rows = _journal()
    rows[0] = {**rows[0], "user_task": long_task}
    _fold, spans, _logs = _run(rows, capture="content")
    named = _by_name(spans)
    task = _attr(named[telemetry_records.SPAN_EXECUTION], "agent6.task")
    assert (
        isinstance(task, str)
        and len(task) == telemetry_records.CONTENT_CHARS
        and task.endswith("…")
    )
    assert "rm" in str(_attr(named[telemetry_records.SPAN_TOOL], "agent6.tool.args"))
    assert _attr(named[telemetry_records.SPAN_VERIFY], "agent6.verify.cmd") == "pytest"


def test_the_metrics_count_what_the_journal_records() -> None:
    fold, _spans, _logs = _run(_journal())
    assert _counter(fold, "agent6_sessions_total") == 1
    assert _counter(fold, "agent6_executions_total") == 1
    assert _counter(fold, "agent6_tokens_total", model="m", provider="p", type="input") == 10
    assert _counter(fold, "agent6_tokens_total", model="m", provider="p", type="output") == 5
    assert _counter(fold, "agent6_tokens_total", model="m", provider="p", type="cache_read") == 2
    assert (
        _counter(fold, "agent6_tokens_total", model="m", provider="p", type="cache_creation") == 0
    )
    assert _counter(fold, "agent6_model_calls_total", model="m", outcome="ok", provider="p") == 1
    assert _counter(fold, "agent6_tool_calls_total", outcome="ok", tool="run_command") == 1
    assert _counter(fold, "agent6_approvals_total", decision="approved", source="tui") == 1
    assert _counter(fold, "agent6_verify_attempts_total", outcome="failed") == 1
    assert _counter(fold, "agent6_findings_total", severity="block") == 1
    assert _counter(fold, "agent6_cost_usd_total") == 0.25
    assert _counter(fold, "agent6_plan_observed_points_total") == 3
    assert (
        _counter(fold, "agent6_execution_ends_total", reason="finish_session", verified="failed")
        == 1
    )
    snapshot = fold.metrics()
    assert snapshot.gauges[("agent6_plan_used_ratio", ())] == 0.12
    assert snapshot.gauges[("agent6_cost_partial", ())] == 0.0
    assert snapshot.gauges[("agent6_open_findings", (("severity", "block"),))] == 1
    verify = snapshot.histograms[("agent6_verify_duration_seconds", ())]
    assert verify.count == 1 and verify.total == 30.0
    assert verify.counts[verify.bounds.index(30.0)] == 1 and sum(verify.counts) == 1
    wait = snapshot.histograms[("agent6_operator_wait_duration_seconds", (("kind", "approval"),))]
    assert wait.count == 1 and wait.total == 6.0


def test_significant_events_become_log_records_on_their_spans() -> None:
    _fold, spans, logs = _run(_journal())
    named = _by_name(spans)
    bodies = {log.body: log for log in logs}
    assert bodies["verify failed"].severity == "ERROR"
    assert bodies["verify failed"].span_id == named[telemetry_records.SPAN_VERIFY].span_id
    assert bodies["review panel blocked"].severity == "WARN"
    assert bodies["approval approved"].span_id == named[telemetry_records.SPAN_WAIT].span_id
    ended = bodies["execution ended: finish_session, gate failed"]
    assert (
        ended.severity == "ERROR"
        and ended.span_id == named[telemetry_records.SPAN_EXECUTION].span_id
    )
    assert all(log.trace_id == named[telemetry_records.SPAN_EXECUTION].trace_id for log in logs)


def test_what_follows_an_end_still_folds_against_it() -> None:
    """The settled budget reading, the stranded notice and a disposition land after session.end."""
    start, panel = queue_tests._panel_rows(block=1)  # pyright: ignore[reportPrivateUsage]
    rows = [
        *_journal(),
        {"type": "budget.update", "ts": _ts(61), "usd_total": 0.40, "plan_consumed": 5},
        {"type": "session.stranded", "ts": _ts(62)},
        {
            "type": "loop.review.finding.resolved",
            "ts": _ts(300),
            "finding_id": "f0",
            "request_id": "r1",
            "source": "cli",
            "version": _review_version(_journal()) + 1,
            "resolution": {"disposition": "wont_fix", "reason": "known"},
        },
        {"type": "loop.budget", "ts": _ts(301), "turn": 3},  # nothing after an end opens work
    ]
    fold, spans, logs = _run(rows)
    assert _counter(fold, "agent6_cost_usd_total") == 0.40
    assert _counter(fold, "agent6_plan_observed_points_total") == 5
    root = _by_name(spans)[telemetry_records.SPAN_EXECUTION]
    bodies = {log.body: log for log in logs}
    assert bodies["edits left uncommitted"].span_id == root.span_id
    resolved = bodies["finding resolved"]
    assert resolved.span_id == root.span_id and _attr(resolved, "agent6.disposition") == "wont_fix"
    assert fold.metrics().gauges[("agent6_open_findings", (("severity", "block"),))] == 0
    assert [s.name for s in spans].count(telemetry_records.SPAN_TURN) == 2
    assert start["type"] == "loop.review.start" and panel["type"] == "loop.review.panel"


def test_the_ask_repls_repeated_starts_share_one_cumulative_tracker() -> None:
    """A second `session.start` on one journal restarts nothing: only a resume banks the spend."""
    rows = [
        {"type": "session.start", "ts": _ts(0), "mode": "ask", "execution_id": "q1"},
        {"type": "budget.update", "ts": _ts(1), "usd_total": 0.10},
        {"type": "session.end", "ts": _ts(2), "reason": "answered"},
        {"type": "session.start", "ts": _ts(3), "mode": "ask", "execution_id": "q2"},
        {"type": "budget.update", "ts": _ts(4), "usd_total": 0.15},
        {"type": "session.end", "ts": _ts(5), "reason": "answered"},
    ]
    fold, spans, _logs = _run(rows)
    assert _counter(fold, "agent6_cost_usd_total") == 0.15
    assert len({s.trace_id for s in spans}) == 2


def test_a_resume_opens_a_second_trace_and_restarts_the_spend_baseline() -> None:
    rows = [
        *_journal(),
        {"type": "loop.resume.start", "ts": _ts(100), "execution_id": "e2", "turn": 2},
        {"type": "budget.update", "ts": _ts(110), "usd_total": 0.10, "usd_partial": True},
        {"type": "session.end", "ts": _ts(120), "reason": "finish_session", "verified": "passed"},
    ]
    fold, spans, _logs = _run(rows)
    executions = [s for s in spans if s.name == telemetry_records.SPAN_EXECUTION]
    assert len({s.trace_id for s in executions}) == 2
    assert [s.status for s in executions] == ["error", "ok"]
    assert _attr(executions[1], "agent6.execution.kind") == "resume"
    assert _attr(executions[1], "agent6.mode") == "run"
    assert _counter(fold, "agent6_cost_usd_total") == 0.35
    assert fold.metrics().gauges[("agent6_cost_partial", ())] == 1.0
    assert _counter(fold, "agent6_executions_total") == 2
    assert _counter(fold, "agent6_sessions_total") == 1


def test_work_still_open_at_the_end_or_the_close_reads_incomplete() -> None:
    """An unanswered tool at the end, and a journal that stops mid-turn, keep real bounds."""
    rows = _journal()
    ended = [row for row in rows if row["type"] not in ("tool.result", "approval.answer")]
    _fold, spans, _logs = _run(ended)
    named = _by_name(spans)
    assert _attr(named[telemetry_records.SPAN_TOOL], "agent6.incomplete") is True
    assert named[telemetry_records.SPAN_TOOL].status == "unset"
    assert named[telemetry_records.SPAN_TOOL].end_ns == telemetry_records.event_ns(_ts(60))
    assert _attr(named[telemetry_records.SPAN_WAIT], "agent6.incomplete") is True
    crashed = rows[:5]  # ends after the tool call
    _fold, spans, _logs = _run(crashed)
    named = _by_name(spans)
    assert _attr(named[telemetry_records.SPAN_EXECUTION], "agent6.incomplete") is True
    assert named[telemetry_records.SPAN_EXECUTION].status == "unset"
    assert named[telemetry_records.SPAN_EXECUTION].end_ns == telemetry_records.event_ns(_ts(3))
    assert _attr(named[telemetry_records.SPAN_TURN], "agent6.incomplete") is True
    fold, spans, _logs = _run(_journal())
    assert (
        fold.close() == telemetry_records.Emitted()
    )  # an ended execution has nothing to close twice


def test_turn_zero_and_a_failed_tool_read_as_the_journal_means_them() -> None:
    """The harness's pre-turn work opens no turn span; a failed dispatch counts as an error."""
    rows = [
        {"type": "session.start", "ts": _ts(0), "execution_id": "e"},
        {"type": "loop.verify_harness", "ts": _ts(1), "turn": 0},
        {"type": "tool.call", "ts": _ts(2), "name": "read_file", "call_id": 1, "turn": 1},
        {"type": "tool.result", "ts": _ts(3), "name": "read_file", "ok": False, "call_id": 1},
        {"type": "session.end", "ts": _ts(4), "reason": "finish_session", "verified": "passed"},
    ]
    fold, spans, _logs = _run(rows)
    assert [_attr(s, "agent6.turn") for s in spans if s.name == telemetry_records.SPAN_TURN] == [1]
    assert _counter(fold, "agent6_tool_calls_total", outcome="error", tool="read_file") == 1


def test_a_journal_without_pairing_ids_pairs_only_what_is_certain() -> None:
    """Without `model_call_id` and `call_id`, one open call pairs with a result; two do not."""
    golden = pathlib.Path(__file__).parent / "data" / "golden_session_logs.jsonl"
    rows: list[dict[str, Any]] = []
    for line in golden.read_text(encoding="utf-8").splitlines():
        try:
            parsed = json.loads(line)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            rows.append(parsed)
    fold, spans, _logs = _run(rows)
    calls = [s for s in spans if s.name == telemetry_records.SPAN_MODEL_CALL]
    assert len(calls) == 2 and all(_attr(s, "agent6.ok") is not None for s in calls)
    tools = [s for s in spans if s.name == telemetry_records.SPAN_TOOL]
    assert len(tools) == 4 and all(_attr(s, "agent6.incomplete") is None for s in tools)
    concurrent = [
        {"type": "session.start", "ts": _ts(0), "execution_id": "e"},
        {"type": "role.call", "ts": _ts(1), "role": "review:a", "model": "m", "provider": "p"},
        {"type": "role.call", "ts": _ts(1), "role": "review:b", "model": "m", "provider": "p"},
        {"type": "role.result", "ts": _ts(2), "role": "review:b", "ok": True, "tokens_in": 7},
        {"type": "session.end", "ts": _ts(3), "reason": "finish_session", "verified": "passed"},
    ]
    fold, spans, _logs = _run(concurrent)
    calls = [s for s in spans if s.name == telemetry_records.SPAN_MODEL_CALL]
    assert all(_attr(s, "agent6.incomplete") is True for s in calls)
    unattributed = _counter(
        fold, "agent6_tokens_total", model="unattributed", provider="unattributed", type="input"
    )
    assert unattributed == 7


def test_free_labels_stop_at_the_bound() -> None:
    rows: list[dict[str, Any]] = [{"type": "session.start", "ts": _ts(0), "execution_id": "e"}]
    for i in range(telemetry_metrics.LABEL_VALUES + 5):
        rows.append(
            {
                "type": "role.call",
                "ts": _ts(i),
                "role": "worker",
                "model": f"m{i}",
                "provider": "p",
                "model_call_id": f"c{i}",
            }
        )
        rows.append(
            {
                "type": "role.result",
                "ts": _ts(i),
                "model_call_id": f"c{i}",
                "ok": True,
                "tokens_in": 1,
            }
        )
    fold, _spans, _logs = _run(rows)
    models = {
        dict(key[1])["model"]
        for key in fold.metrics().counters
        if key[0] == "agent6_model_calls_total"
    }
    assert len(models) == telemetry_metrics.LABEL_VALUES + 1 and "other" in models
    assert (
        _counter(fold, "agent6_model_calls_total", model="other", outcome="ok", provider="p") == 5
    )


def test_merged_snapshots_add_counts_and_keep_the_latest_reading() -> None:
    one = telemetry_metrics.MetricsSnapshot(
        {("agent6_sessions_total", ()): 1.0},
        {
            ("agent6_plan_used_ratio", ()): 0.1,
            ("agent6_open_findings", (("severity", "block"),)): 1,
        },
        {
            ("agent6_verify_duration_seconds", ()): telemetry_metrics.Histogram(
                (1.0,), (1, 0), 0.5, 1
            )
        },
    )
    two = telemetry_metrics.MetricsSnapshot(
        {("agent6_sessions_total", ()): 1.0},
        {
            ("agent6_plan_used_ratio", ()): 0.4,
            ("agent6_open_findings", (("severity", "block"),)): 2,
        },
        {
            ("agent6_verify_duration_seconds", ()): telemetry_metrics.Histogram(
                (1.0,), (0, 1), 9.0, 1
            )
        },
    )
    merged = telemetry_metrics.merge_snapshots([one, two])
    assert merged.counters[("agent6_sessions_total", ())] == 2.0
    assert merged.gauges[("agent6_plan_used_ratio", ())] == 0.4
    assert merged.gauges[("agent6_open_findings", (("severity", "block"),))] == 3
    histogram = merged.histograms[("agent6_verify_duration_seconds", ())]
    assert (histogram.counts, histogram.total, histogram.count) == ((1, 1), 9.5, 2)


def test_event_ns_keeps_microseconds_exactly() -> None:
    assert (
        telemetry_records.event_ns("2026-10-25T10:00:00.123456+00:00") == 1_792_922_400_123_456_000
    )
    assert telemetry_records.event_ns("2026-10-25T10:00:00") == telemetry_records.event_ns(
        "2026-10-25T10:00:00+00:00"
    )
    assert telemetry_records.event_ns(1.5) == 1_500_000_000
    assert (
        telemetry_records.event_ns("yesterday") is None and telemetry_records.event_ns(True) is None
    )
