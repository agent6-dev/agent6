# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Machine journals and their retained worker logs share a telemetry trace."""

from __future__ import annotations

import json
import pathlib
from typing import Any

from agent6.viewmodel import telemetry, telemetry_records


def journal() -> list[dict[str, Any]]:
    path = pathlib.Path(__file__).parent / "data" / "golden_journal.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def worker() -> list[dict[str, Any]]:
    return [
        {"type": "session.start", "ts": "2026-07-16T00:00:03+00:00", "execution_id": "worker"},
        {
            "type": "role.call",
            "ts": "2026-07-16T00:00:03+00:00",
            "model_call_id": "c",
            "model": "m",
            "provider": "p",
        },
        {
            "type": "role.result",
            "ts": "2026-07-16T00:00:04+00:00",
            "model_call_id": "c",
            "tokens_in": 1000,
            "tokens_out": 200,
        },
        {"type": "budget.update", "ts": "2026-07-16T00:00:04+00:00", "usd_total": 0.25},
        {
            "type": "session.end",
            "ts": "2026-07-16T00:00:04+00:00",
            "reason": "finish_session",
            "verified": "passed",
        },
    ]


def folded() -> tuple[
    telemetry.TelemetryFold, list[telemetry_records.Span], list[telemetry_records.LogRecord]
]:
    fold = telemetry.TelemetryFold("demo")
    spans: list[telemetry_records.Span] = []
    logs: list[telemetry_records.LogRecord] = []
    for event in journal():
        emitted = fold.feed(event)
        spans.extend(emitted.spans)
        logs.extend(emitted.logs)
    spans.extend(fold.close().spans)
    return fold, spans, logs


def test_machine_states_waits_and_spend_survive_without_worker_logs() -> None:
    """A machine journal alone preserves every state visit, wait and booked spend."""
    fold, spans, logs = folded()
    assert [dict(s.attributes)["agent6.machine.state"] for s in spans] == [
        "scan",
        "branch",
        "poll",
        "review",
        "stop_ok",
    ]
    assert len({s.span_id for s in spans}) == 5
    assert len({s.trace_id for s in spans}) == 1
    assert all(s.parent_span_id == "" for s in spans)
    assert spans == folded()[1]
    assert [(s.end_ns - s.start_ns) / 1e9 for s in spans] == [1, 1, 1, 1, 2]
    (wait,) = logs
    assert wait.body == "machine wait signal" and wait.span_id == spans[2].span_id
    assert "operator" not in json.dumps(wait.attributes)
    counts = fold.metrics().counters
    assert counts[("agent6_sessions_total", ())] == 1
    assert counts[("agent6_cost_usd_total", ())] == 0.25
    assert sum(n for (name, _), n in counts.items() if name == "agent6_tokens_total") == 1200
    assert counts[("agent6_tool_calls_total", (("outcome", "ok"), ("tool", "run_command")))] == 1
