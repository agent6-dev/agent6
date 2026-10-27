# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Project machine transitions and booked spend without reading files."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from agent6.viewmodel import telemetry_metrics, telemetry_records


def state_parent(session_id: str, seq: int) -> telemetry_records.SpanContext:
    """Return the trace and state span ids shared by a machine and its worker journals."""
    trace = telemetry_records.hex_id(f"agent6:machine:{session_id}", 32)
    return telemetry_records.SpanContext(
        trace, telemetry_records.hex_id(f"{trace}:state:{seq}", 16)
    )


class MachineFold:
    """Fold completed state visits; a parked state waits for its journaled transition."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.meter = telemetry_metrics.Meter()
        self._start_ns = 0
        self._spend: dict[int, telemetry_metrics.Meter] = {}

    def feed(self, event: Mapping[str, Any], now: int) -> telemetry_records.Emitted:
        """Return a completed state span and any wait record, recording applicable metrics."""
        kind = event.get("type")
        if kind == "machine.begin":
            self._start_ns = now
            self.meter.count("agent6_sessions_total")
            return telemetry_records.Emitted()
        seq = telemetry_records.as_int(event.get("seq", event.get("transitions")))
        if seq is None:
            return telemetry_records.Emitted()
        fact = event.get("fact", {}) if kind == "step" else event
        if not isinstance(fact, dict):
            return telemetry_records.Emitted()
        if kind in ("step", "attempt.spend", "machine.end"):
            self._book(seq, fact)
        if kind not in ("step", "machine.end"):
            return telemetry_records.Emitted()
        context = state_parent(self.session_id, seq)
        fields: dict[str, telemetry_records.AttributeValue] = {
            "agent6.session.id": self.session_id,
            "agent6.machine.state": str(event.get("state", "")),
            "agent6.machine.seq": seq,
        }
        status: telemetry_records.Status = "unset"
        logs: tuple[telemetry_records.LogRecord, ...] = ()
        fact_kind = fact.get("kind")
        if kind == "machine.end":
            status = "ok" if event.get("status") == "ok" else "error"
            fields["agent6.machine.status"] = str(event.get("status", ""))
        else:
            fields["agent6.machine.kind"] = str(fact_kind or "")
            fields["agent6.machine.label"] = str(event.get("label", ""))
            fields["agent6.machine.goto"] = str(event.get("goto", ""))
        if fact_kind == "tool":
            ok = fact.get("exit_code") == 0 and fact.get("timed_out") is not True
            status = "ok" if ok else "error"
            self.meter.count(
                "agent6_tool_calls_total", tool="run_command", outcome="ok" if ok else "error"
            )
        elif fact_kind == "agent":
            status = "ok" if fact.get("outcome") == "ok" else "error"
        elif fact_kind == "wait":
            woke_by = str(fact.get("woke_by", ""))
            fields["agent6.wait.woke_by"] = woke_by
            logs = (
                telemetry_records.LogRecord(
                    now,
                    "INFO",
                    f"machine wait {woke_by}",
                    telemetry_records.attributes(fields),
                    context.trace_id,
                    context.span_id,
                ),
            )
            if woke_by == "signal":
                self.meter.observe(
                    "agent6_operator_wait_duration_seconds",
                    max(0, now - self._start_ns) / 1e9,
                    kind="machine",
                )
        span = telemetry_records.Span(
            context.trace_id,
            context.span_id,
            "",
            telemetry_records.SPAN_MACHINE_STATE,
            "internal",
            self._start_ns,
            max(self._start_ns, now),
            status,
            telemetry_records.attributes(fields),
        )
        self._start_ns = now
        return telemetry_records.Emitted((span,), logs)

    def _book(self, seq: int, fact: Mapping[str, Any]) -> None:
        meter = self._spend.setdefault(seq, telemetry_metrics.Meter())
        for name, field, labels in (
            ("agent6_cost_usd_total", "usd", {}),
            ("agent6_tokens_total", "input_tokens", {"type": "input"}),
            ("agent6_tokens_total", "output_tokens", {"type": "output"}),
        ):
            amount = telemetry_records.as_number(fact.get(field)) or 0.0
            if amount > 0:
                meter.count(name, amount, **labels)
        if fact.get("usd_partial") is True:
            meter.gauge("agent6_cost_partial", 1.0)

    def metrics(
        self, workers: Mapping[int, telemetry_metrics.MetricsSnapshot]
    ) -> telemetry_metrics.MetricsSnapshot:
        """Return worker readings plus spend missing from pruned or interrupted logs."""
        missing = telemetry_metrics.Meter()
        for seq, meter in self._spend.items():
            booked = meter.snapshot()
            worker = workers.get(seq)
            for (name, labels), amount in booked.counters.items():
                seen = (
                    sum(
                        value
                        for (metric, dimensions), value in worker.counters.items()
                        if metric == name and all(pair in dimensions for pair in labels)
                    )
                    if worker is not None
                    else 0.0
                )
                if amount > seen:
                    fields = dict(labels)
                    if name == "agent6_tokens_total":
                        fields.update(model="unattributed", provider="unattributed")
                    missing.count(name, amount - seen, **fields)
            if booked.gauges:
                missing.gauge("agent6_cost_partial", 1.0)
        return telemetry_metrics.merge_snapshots(
            [self.meter.snapshot(), *workers.values(), missing.snapshot()]
        )
