# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Project a session journal onto telemetry records: spans, metrics and log records.

The projection is pure and metadata-only by default: outcomes, counts, timings and the
journal's own ids, never task text, arguments, outputs or diffs. `capture = "content"` adds
the capped excerpts `TelemetryConfig.capture` lists and nothing else. The records and the
open work live in `telemetry_records`, the metrics in `telemetry_metrics`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, ClassVar, Literal

from agent6 import evidence
from agent6.viewmodel import review, telemetry_machine, telemetry_metrics, telemetry_records

Capture = Literal["metadata", "content"]
_TOKEN_TYPES = (
    ("input", "tokens_in"),
    ("output", "tokens_out"),
    ("cache_read", "cache_read"),
    ("cache_creation", "cache_creation"),
)
_FINDING_ACTIONS = ("loop.review.finding.resolved", "loop.review.finding.reopened")
# What still folds once an execution has ended: the settled budget, the stranded notice and
# the review dispositions; everything else belongs to the next execution.
_AFTER_END = frozenset({"budget.update", "session.stranded"})


def tool_label(name: str) -> str:
    """Return the tool label: the built-in name, or `mcp` for any MCP server's tool."""
    return "mcp" if name.startswith("mcp__") else name


class TelemetryFold:
    """Fold one session's journal, in order, into telemetry records.

    `feed` takes each event as it lands and returns the spans and log records it completed;
    `metrics` returns the cumulative snapshot; `close` ends whatever is still open, marked
    incomplete, so a crashed worker's work keeps its real duration bound. An ended execution
    stays current for the settled budget reading, the stranded notice and the review
    dispositions that follow its end.

    Args:
        session_id: The session the journal belongs to.
        capture: `metadata`, or `content` for the capped excerpts.
        parent: The machine state a worker belongs to, None for a standalone run.
    """

    def __init__(
        self,
        session_id: str,
        *,
        capture: Capture = "metadata",
        parent: telemetry_records.SpanContext | None = None,
    ) -> None:
        self.session_id = session_id
        self.capture = capture
        self._parent = parent
        self._machine: telemetry_machine.MachineFold | None = None
        self.meter = telemetry_metrics.Meter()
        self.first_ns: int | None = None
        self._execution: telemetry_records.Execution | None = None
        self._executions = 0
        self._mode = ""
        self._last_ns = 0
        self._review = review.ReviewView()
        self._findings: set[str] = set()
        # The budget readings are cumulative per tracker: a resume banks the prior spend and
        # restarts them; the ask REPL's repeated starts share one tracker.
        self._usd_seen = 0.0
        self._plan_seen = 0.0

    # --- the public surface ---------------------------------------------------

    def feed(self, event: Mapping[str, Any]) -> telemetry_records.Emitted:
        """Fold one journal event.

        Args:
            event: The event, envelope included.

        Returns:
            The spans and log records the event completed.
        """
        kind = event.get("type")
        if not isinstance(kind, str):
            return telemetry_records.Emitted()
        ns = telemetry_records.event_ns(event.get("ts"))
        if ns is not None:
            self._last_ns = ns
            if self.first_ns is None or ns < self.first_ns:
                self.first_ns = ns
        now = self._last_ns
        if kind == "machine.begin":
            self._machine = telemetry_machine.MachineFold(self.session_id)
        if self._machine is not None:
            return self._machine.feed(event, now)
        spans: list[telemetry_records.Span] = []
        logs: list[telemetry_records.LogRecord] = []
        if kind in ("session.start", "loop.resume.start"):
            spans.extend(self._start_execution(event, now))
        if kind.startswith("loop.review."):
            logs.extend(self._findings_update(event, now))
        execution = self._execution
        if execution is None or (execution.ended and kind not in _AFTER_END):
            return telemetry_records.Emitted(tuple(spans), tuple(logs)).under(self._parent)
        turn = telemetry_records.as_int(event.get("turn", event.get("iteration")))
        # A turn only advances; turn 0 is the harness's own work before the first turn.
        if turn is not None and turn >= 1 and turn > (execution.turn_number or 0):
            spans.extend(self._turn(execution, turn, now))
        handler = self._HANDLERS.get(kind)
        if handler is not None:
            handler(self, telemetry_records.Step(execution, event, now, spans, logs))
        return telemetry_records.Emitted(tuple(spans), tuple(logs)).under(self._parent)

    def close(self, at_ns: int | None = None) -> telemetry_records.Emitted:
        """End every open span as incomplete, at the given time or the last event's.

        Args:
            at_ns: The end time; None takes the last event's.

        Returns:
            The spans closed; nothing for an execution that ended.
        """
        execution = self._execution
        self._execution = None
        if execution is None or execution.ended:
            return telemetry_records.Emitted()
        now = at_ns if at_ns is not None else self._last_ns
        spans = [execution.end(s, now, "unset", incomplete=True) for s in execution.open_spans()]
        return telemetry_records.Emitted(tuple(spans), ()).under(self._parent)

    def metrics(
        self, *, workers: Mapping[int, telemetry_metrics.MetricsSnapshot] | None = None
    ) -> telemetry_metrics.MetricsSnapshot:
        """Return cumulative metrics, including a machine's retained worker readings."""
        if self._machine is not None:
            return self._machine.metrics(workers or {})
        return self.meter.snapshot()

    # --- executions and turns -------------------------------------------------

    def _start_execution(self, event: Mapping[str, Any], now: int) -> list[telemetry_records.Span]:
        spans = list(self.close(now).spans)
        if self._executions == 0:
            self.meter.count("agent6_sessions_total")
        self._executions += 1
        execution_id = str(event.get("execution_id") or f"execution:{self._executions}")
        trace_id = telemetry_records.hex_id(f"agent6:{self.session_id}:{execution_id}", 32)
        mode = event.get("mode")
        if isinstance(mode, str) and mode:
            self._mode = mode
        kind = "start" if event["type"] == "session.start" else "resume"
        if kind == "resume":
            self._usd_seen = 0.0
            self._plan_seen = 0.0
        fields: dict[str, telemetry_records.AttributeValue] = {
            "agent6.session.id": self.session_id,
            "agent6.execution.id": execution_id,
            "agent6.execution.kind": kind,
        }
        if self._mode:
            fields["agent6.mode"] = self._mode
        if self.capture == "content":
            if event.get("user_task"):
                fields["agent6.task"] = telemetry_records.excerpt(event["user_task"])
            if event.get("steer"):
                fields["agent6.steer"] = telemetry_records.excerpt(event["steer"])
        root = telemetry_records.OpenSpan(
            telemetry_records.hex_id(f"{trace_id}:execution", 16),
            "",
            telemetry_records.SPAN_EXECUTION,
            now,
            "internal",
            fields,
        )
        self._execution = telemetry_records.Execution(trace_id, root)
        self.meter.count("agent6_executions_total")
        return spans

    def _turn(
        self, execution: telemetry_records.Execution, turn: int, now: int
    ) -> list[telemetry_records.Span]:
        spans: list[telemetry_records.Span] = []
        if execution.turn is not None:
            spans.append(execution.end(execution.turn, now, "unset"))
        execution.turn = telemetry_records.OpenSpan(
            telemetry_records.hex_id(f"{execution.trace_id}:turn:{turn}", 16),
            execution.span.span_id,
            telemetry_records.SPAN_TURN,
            now,
            attributes={"agent6.turn": turn},
        )
        execution.turn_number = turn
        return spans

    # --- model calls ------------------------------------------------------------

    def _role_call(self, step: telemetry_records.Step) -> None:
        execution, event = step.execution, step.event
        execution.sequence += 1
        key = str(event.get("model_call_id") or f"seq:{execution.sequence}")
        seat = telemetry_records.as_int(event.get("seat_index"))
        parent = execution.parent()
        if seat is not None and execution.panels:
            parent = next(reversed(execution.panels.values())).span_id
        fields: dict[str, telemetry_records.AttributeValue] = {
            "agent6.role": str(event.get("role", "")),
            "agent6.model": str(event.get("model", "")),
            "agent6.provider": str(event.get("provider", "")),
        }
        if seat is not None:
            fields["agent6.seat"] = seat
        span_id = telemetry_records.hex_id(f"{execution.trace_id}:call:{key}", 16)
        execution.calls[key] = telemetry_records.OpenSpan(
            span_id, parent, telemetry_records.SPAN_MODEL_CALL, step.now, "client", fields
        )

    def _role_result(self, step: telemetry_records.Step) -> None:
        execution, event = step.execution, step.event
        key = str(event.get("model_call_id") or "")
        span = execution.calls.get(key)
        if span is None and not key and len(execution.calls) == 1:
            span = next(iter(execution.calls.values()))  # one open call: the pairing is certain
        ok = event.get("ok") is not False
        known_model = span.attributes["agent6.model"] if span is not None else "unattributed"
        known_provider = span.attributes["agent6.provider"] if span is not None else "unattributed"
        model = self.meter.label("model", event.get("model") or known_model)
        provider = self.meter.label("provider", event.get("provider") or known_provider)
        outcome = "ok" if ok else "error"
        self.meter.count(
            "agent6_model_calls_total", outcome=outcome, model=model, provider=provider
        )
        for kind, field in _TOKEN_TYPES:
            tokens = telemetry_records.as_int(event.get(field)) or 0
            if tokens:
                self.meter.count(
                    "agent6_tokens_total", tokens, type=kind, model=model, provider=provider
                )
        if span is None:
            return
        span.attributes["agent6.ok"] = ok
        for _kind, field in _TOKEN_TYPES:
            if (value := telemetry_records.as_int(event.get(field))) is not None:
                span.attributes[f"agent6.{field}"] = value
        if event.get("stop_reason"):
            span.attributes["agent6.stop_reason"] = str(event["stop_reason"])
        if not ok and self.capture == "content" and event.get("error"):
            span.attributes["agent6.model.error"] = telemetry_records.excerpt(event["error"])
        step.spans.append(execution.end(span, step.now, "ok" if ok else "error"))
        if not ok:
            fields = {"agent6.model": model, "agent6.provider": provider}
            step.logs.append(
                execution.log(step.now, "ERROR", "model call failed", fields, span.span_id)
            )

    # --- tools -----------------------------------------------------------------------

    def _tool_call(self, step: telemetry_records.Step) -> None:
        execution, event = step.execution, step.event
        execution.sequence += 1
        call_id = event.get("call_id")
        key = str(call_id) if call_id is not None else f"seq:{execution.sequence}"
        fields: dict[str, telemetry_records.AttributeValue] = {
            "agent6.tool": str(event.get("name", ""))
        }
        if self.capture == "content" and "args" in event:
            fields["agent6.tool.args"] = telemetry_records.excerpt(event["args"])
        span_id = telemetry_records.hex_id(f"{execution.trace_id}:tool:{key}", 16)
        execution.tools[key] = telemetry_records.OpenSpan(
            span_id, execution.parent(), telemetry_records.SPAN_TOOL, step.now, attributes=fields
        )

    def _tool_result(self, step: telemetry_records.Step) -> None:
        execution, event = step.execution, step.event
        call_id = event.get("call_id")
        key = str(call_id) if call_id is not None else ""
        span = execution.tools.get(key)
        if span is None and not key and len(execution.tools) == 1:
            span = next(iter(execution.tools.values()))
        ok = event.get("ok") is not False
        tool = self.meter.label("tool", tool_label(str(event.get("name", ""))))
        self.meter.count("agent6_tool_calls_total", tool=tool, outcome="ok" if ok else "error")
        if span is None:
            return
        span.attributes["agent6.ok"] = ok
        step.spans.append(execution.end(span, step.now, "ok" if ok else "error"))

    # --- verification ------------------------------------------------------------

    def _verify_start(self, step: telemetry_records.Step) -> None:
        execution, event = step.execution, step.event
        execution.sequence += 1
        key = str(event.get("verify_id") or f"seq:{execution.sequence}")
        fields: dict[str, telemetry_records.AttributeValue] = {
            "agent6.verify.id": key,
            "agent6.verify.scoped": event.get("scoped") is True,
        }
        if self.capture == "content" and event.get("cmd"):
            fields["agent6.verify.cmd"] = telemetry_records.excerpt(
                " ".join(str(p) for p in event["cmd"])
            )
        span_id = telemetry_records.hex_id(f"{execution.trace_id}:verify:{key}", 16)
        execution.verifies[key] = telemetry_records.OpenSpan(
            span_id, execution.parent(), telemetry_records.SPAN_VERIFY, step.now, attributes=fields
        )

    def _verify_end(self, step: telemetry_records.Step) -> None:
        execution, event, now = step.execution, step.event, step.now
        key = str(event.get("verify_id") or "")
        span = execution.verifies.get(key)
        if span is None and not key and len(execution.verifies) == 1:
            span = next(iter(execution.verifies.values()))
        outcome = evidence.outcome(event)
        self.meter.count("agent6_verify_attempts_total", outcome=outcome)
        if (seconds := telemetry_records.as_number(event.get("duration_s"))) is not None:
            self.meter.observe("agent6_verify_duration_seconds", seconds)
        fields: dict[str, telemetry_records.AttributeValue | None] = {
            "agent6.verify.outcome": outcome,
            "agent6.verify.exit_code": telemetry_records.as_int(event.get("exit_code")),
        }
        severity: telemetry_records.Severity = "INFO" if outcome == "passed" else "ERROR"
        span_id = span.span_id if span is not None else ""
        step.logs.append(execution.log(now, severity, f"verify {outcome}", fields, span_id))
        if span is None:
            return
        span.attributes["agent6.verify.outcome"] = outcome
        if (code := telemetry_records.as_int(event.get("exit_code"))) is not None:
            span.attributes["agent6.verify.exit_code"] = code
        step.spans.append(execution.end(span, now, "ok" if outcome == "passed" else "error"))

    # --- review ----------------------------------------------------------------------

    def _review_start(self, step: telemetry_records.Step) -> None:
        execution, event = step.execution, step.event
        key = str(event.get("panel_id") or f"seq:{execution.sequence}")
        seats = event.get("seats")
        fields: dict[str, telemetry_records.AttributeValue] = {
            "agent6.panel.id": key,
            "agent6.panel.trigger": str(event.get("trigger", "")),
            "agent6.panel.seats": len(seats) if isinstance(seats, list) else 0,
        }
        span_id = telemetry_records.hex_id(f"{execution.trace_id}:panel:{key}", 16)
        execution.panels[key] = telemetry_records.OpenSpan(
            span_id, execution.parent(), telemetry_records.SPAN_REVIEW, step.now, attributes=fields
        )

    def _review_seat(self, step: telemetry_records.Step) -> None:
        execution, event = step.execution, step.event
        panel = execution.panels.get(str(event.get("panel_id") or ""))
        verdict = str(event.get("verdict", ""))
        findings = event.get("findings")
        fields: dict[str, telemetry_records.AttributeValue | None] = {
            "agent6.seat": telemetry_records.as_int(event.get("seat_index")),
            "agent6.verdict": verdict or None,
            "agent6.findings": len(findings) if isinstance(findings, list) else 0,
        }
        severity: telemetry_records.Severity = "WARN" if verdict == "block" else "INFO"
        body = f"review seat {verdict or 'abstained'}"
        span_id = panel.span_id if panel is not None else ""
        step.logs.append(execution.log(step.now, severity, body, fields, span_id))

    def _review_panel(self, step: telemetry_records.Step) -> None:
        execution, event, now = step.execution, step.event, step.now
        key = str(event.get("panel_id") or "")
        panel = execution.panels.get(key)
        if panel is None:
            return
        blocked = event.get("blocked") is True
        completion = str(event.get("completion", "complete"))
        panel.attributes["agent6.panel.blocked"] = blocked
        panel.attributes["agent6.panel.completion"] = completion
        if (n_block := telemetry_records.as_int(event.get("n_block"))) is not None:
            panel.attributes["agent6.panel.block_findings"] = n_block
        step.spans.append(execution.end(panel, now, "error" if blocked else "ok"))
        body = "review panel blocked" if blocked else f"review panel {completion}"
        severity: telemetry_records.Severity = "WARN" if blocked else "INFO"
        step.logs.append(
            execution.log(now, severity, body, {"agent6.panel.id": key}, panel.span_id)
        )

    def _review_skipped(self, step: telemetry_records.Step) -> None:
        execution, event, now = step.execution, step.event, step.now
        key = str(event.get("panel_id") or "")
        panel = execution.panels.get(key)
        fields: dict[str, telemetry_records.AttributeValue | None] = {
            "agent6.panel.id": key or None
        }
        if self.capture == "content" and event.get("reason"):
            fields["agent6.detail"] = telemetry_records.excerpt(event["reason"])
        span_id = panel.span_id if panel is not None else ""
        step.logs.append(execution.log(now, "WARN", "review skipped", fields, span_id))
        if panel is not None:
            panel.attributes["agent6.panel.completion"] = "skipped"
            step.spans.append(execution.end(panel, now, "unset"))

    def _findings_update(
        self, event: Mapping[str, Any], now: int
    ) -> list[telemetry_records.LogRecord]:
        """Fold a review event into the finding counts; an applied disposition becomes a log record.

        Dispositions arrive from the operator's surfaces after the run ends, so this runs
        with or without a live execution and hangs the record on the last execution's root.

        Args:
            event: The review event.
            now: Its time.

        Returns:
            The log record of a disposition the review fold took; nothing otherwise.
        """
        before = self._review
        self._review = review.apply_event(before, event)
        view = self._review
        for finding in view.findings:
            if finding.finding_id not in self._findings:
                self._findings.add(finding.finding_id)
                severity = finding.observations[-1].finding.severity
                self.meter.count("agent6_findings_total", severity=severity)
        for severity in ("block", "warn", "nit"):
            open_count = float(getattr(view.open_counts, severity))
            self.meter.gauge("agent6_open_findings", open_count, severity=severity)
        execution = self._execution
        kind = event.get("type")
        refused = view.read_failure is not None and view.read_failure != before.read_failure
        if execution is None or kind not in _FINDING_ACTIONS or refused:
            return []
        finding_id = str(event.get("finding_id", ""))
        fields: dict[str, telemetry_records.AttributeValue | None] = {
            "agent6.finding.id": finding_id,
            "agent6.source": str(event.get("source", "")) or None,
        }
        resolution = event.get("resolution")
        if kind == _FINDING_ACTIONS[0] and isinstance(resolution, dict):
            fields["agent6.disposition"] = str(resolution.get("disposition", ""))
        if self.capture == "content":
            match = next((f for f in view.findings if f.finding_id == finding_id), None)
            if match is not None:
                report = match.observations[-1].finding
                fields["agent6.finding"] = telemetry_records.excerpt(
                    f"{report.location}: {report.title}"
                )
        body = "finding resolved" if kind == _FINDING_ACTIONS[0] else "finding reopened"
        span_id = execution.span.span_id if execution.ended else ""
        return [execution.log(now, "INFO", body, fields, span_id)]

    # --- operator waits -------------------------------------------------------------

    def _prompt(self, step: telemetry_records.Step) -> None:
        execution, event = step.execution, step.event
        kind = "approval" if event["type"] == "approval.prompt" else "question"
        key = f"{kind}:{event.get('id', '')}"
        parent = execution.parent()
        call_id = event.get("call_id")
        if call_id is not None and str(call_id) in execution.tools:
            parent = execution.tools[str(call_id)].span_id
        span_id = telemetry_records.hex_id(f"{execution.trace_id}:wait:{key}", 16)
        execution.waits[key] = telemetry_records.OpenSpan(
            span_id,
            parent,
            telemetry_records.SPAN_WAIT,
            step.now,
            attributes={"agent6.wait.kind": kind},
        )

    def _answer(self, step: telemetry_records.Step) -> None:
        execution, event, now = step.execution, step.event, step.now
        kind = "approval" if event["type"] == "approval.answer" else "question"
        span = execution.waits.get(f"{kind}:{event.get('id', '')}")
        source = self.meter.label("source", event.get("source"))
        span_id = span.span_id if span is not None else ""
        if kind == "approval":
            decision = "approved" if event.get("approved") is True else "denied"
            self.meter.count("agent6_approvals_total", decision=decision, source=source)
            fields = {"agent6.source": source}
            step.logs.append(execution.log(now, "INFO", f"approval {decision}", fields, span_id))
        if span is None:
            return
        span.attributes["agent6.source"] = source
        if kind == "approval":
            span.attributes["agent6.approved"] = event.get("approved") is True
        else:
            span.attributes["agent6.unseen"] = event.get("unseen") is True
        waited = max(0, now - span.start_ns) / 1_000_000_000
        self.meter.observe("agent6_operator_wait_duration_seconds", waited, kind=kind)
        step.spans.append(execution.end(span, now, "ok"))

    # --- budget, degradation and the end ----------------------------------------------

    def _budget_update(self, step: telemetry_records.Step) -> None:
        event = step.event
        usd = telemetry_records.as_number(event.get("usd_total"))
        if usd is not None and usd > self._usd_seen:
            self.meter.count("agent6_cost_usd_total", usd - self._usd_seen)
            self._usd_seen = usd
        self.meter.gauge("agent6_cost_partial", 1.0 if event.get("usd_partial") is True else 0.0)
        plan = telemetry_records.as_number(event.get("plan_consumed"))
        if plan is not None and plan > self._plan_seen:
            self.meter.count("agent6_plan_observed_points_total", plan - self._plan_seen)
            self._plan_seen = plan
        if (used := telemetry_records.as_number(event.get("plan_used_percent"))) is not None:
            self.meter.gauge("agent6_plan_used_ratio", used / 100.0)

    def _jail_degraded(self, step: telemetry_records.Step) -> None:
        fields: dict[str, telemetry_records.AttributeValue | None] = {}
        if self.capture == "content" and step.event.get("detail"):
            fields["agent6.detail"] = telemetry_records.excerpt(step.event["detail"])
        step.logs.append(step.execution.log(step.now, "WARN", "jail degraded", fields))

    def _session_end(self, step: telemetry_records.Step) -> None:
        execution, event, now = step.execution, step.event, step.now
        if execution.ended:
            return
        reason = self.meter.label("reason", event.get("reason"))
        verified = str(event.get("verified") or "unverified")
        for span in execution.open_spans():
            if span is not execution.span and span is not execution.turn:
                step.spans.append(execution.end(span, now, "unset", incomplete=True))
        if execution.turn is not None:
            step.spans.append(execution.end(execution.turn, now, "unset"))
        root = execution.span
        root.attributes["agent6.end.reason"] = reason
        root.attributes["agent6.verified"] = verified
        if (turns := telemetry_records.as_int(event.get("turns"))) is not None:
            root.attributes["agent6.turns"] = turns
        if event.get("scoped") is True:
            root.attributes["agent6.verify.scoped"] = True
        status: telemetry_records.Status = "unset"
        if verified == "passed":
            status = "ok"
        elif verified in ("failed", "timed_out"):
            status = "error"
        self.meter.count("agent6_execution_ends_total", reason=reason, verified=verified)
        severity: telemetry_records.Severity = "WARN"
        if status == "ok":
            severity = "INFO"
        elif status == "error" or reason == "budget_exhausted":
            severity = "ERROR"
        fields = {"agent6.end.reason": reason, "agent6.verified": verified}
        body = f"execution ended: {reason}, gate {verified}"
        step.logs.append(execution.log(now, severity, body, fields, root.span_id))
        step.spans.append(execution.end(root, now, status))
        execution.ended = True

    def _stranded(self, step: telemetry_records.Step) -> None:
        execution = step.execution
        log = execution.log(step.now, "WARN", "edits left uncommitted", {}, execution.span.span_id)
        step.logs.append(log)

    _HANDLERS: ClassVar[dict[str, Callable[[TelemetryFold, telemetry_records.Step], None]]] = {
        "role.call": _role_call,
        "role.result": _role_result,
        "tool.call": _tool_call,
        "tool.result": _tool_result,
        "verify.start": _verify_start,
        "verify.end": _verify_end,
        "loop.review.start": _review_start,
        "loop.review.seat": _review_seat,
        "loop.review.panel": _review_panel,
        "loop.review.skipped": _review_skipped,
        "approval.prompt": _prompt,
        "question.prompt": _prompt,
        "approval.answer": _answer,
        "question.answer": _answer,
        "budget.update": _budget_update,
        "jail.degraded": _jail_degraded,
        "session.end": _session_end,
        "session.stranded": _stranded,
    }
