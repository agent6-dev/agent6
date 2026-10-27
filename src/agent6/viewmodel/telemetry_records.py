# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The telemetry records the fold emits, and the open work it keeps while folding.

`Span`, `LogRecord` and `Emitted` are what `telemetry.TelemetryFold` returns and the encoders
read; `OpenSpan`, `Execution` and `Step` are the fold's working state. Every id derives from
the journal's identities through `hex_id`, so a replay yields the same records.
"""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
from collections.abc import Mapping
from typing import Any, Literal

from agent6.viewmodel import telemetry_metrics

AttributeValue = telemetry_metrics.AttributeValue
Attributes = telemetry_metrics.Attributes
Severity = Literal["INFO", "WARN", "ERROR"]
Status = Literal["unset", "ok", "error"]
SpanKind = Literal["internal", "client"]

# The span names, one per unit of work the journal records.
SPAN_EXECUTION = "agent6.execution"
SPAN_TURN = "agent6.turn"
SPAN_MODEL_CALL = "agent6.model_call"
SPAN_TOOL = "agent6.tool"
SPAN_VERIFY = "agent6.verify"
SPAN_REVIEW = "agent6.review_panel"
SPAN_WAIT = "agent6.wait"
# Every content excerpt is cut here.
CONTENT_CHARS = 1024
_EPOCH = datetime.datetime(1970, 1, 1, tzinfo=datetime.UTC)


@dataclasses.dataclass(frozen=True, slots=True)
class Span:
    """One finished span.

    Attributes:
        trace_id: 32 hex characters, one trace per execution.
        span_id: 16 hex characters, derived from the journal id of the work.
        parent_span_id: The parent's span id; "" for a root.
        name: One of the `SPAN_*` names.
        kind: `client` for a model call, `internal` otherwise.
        start_ns: The start, in nanoseconds since the epoch.
        end_ns: The end, in nanoseconds since the epoch.
        status: The outcome; `unset` when the journal did not decide one.
        attributes: Metadata, and content excerpts when captured.
    """

    trace_id: str
    span_id: str
    parent_span_id: str
    name: str
    kind: SpanKind
    start_ns: int
    end_ns: int
    status: Status
    attributes: Attributes


@dataclasses.dataclass(frozen=True, slots=True)
class LogRecord:
    """One significant event, tied to the span it happened in."""

    time_ns: int
    severity: Severity
    body: str
    attributes: Attributes
    trace_id: str
    span_id: str


@dataclasses.dataclass(frozen=True, slots=True)
class Emitted:
    """What one event, or the close, completed."""

    spans: tuple[Span, ...] = ()
    logs: tuple[LogRecord, ...] = ()


def event_ns(value: object) -> int | None:
    """Return an event `ts` as whole nanoseconds since the epoch, None when unparseable.

    Args:
        value: The raw `ts` field: the ISO-8601 string `EventSink` writes, or epoch seconds.

    Returns:
        The nanoseconds, exact to the microsecond the stamp carries; None for anything else.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value * 1_000_000_000)
    if not isinstance(value, str):
        return None
    try:
        stamp = datetime.datetime.fromisoformat(value)
    except ValueError:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=datetime.UTC)
    delta = stamp - _EPOCH
    return (delta.days * 86_400 + delta.seconds) * 1_000_000_000 + delta.microseconds * 1_000


def hex_id(seed: str, length: int) -> str:
    """Return the first `length` hex characters of the seed's SHA-256: a span or trace id."""
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:length]


def attributes(pairs: Mapping[str, AttributeValue | None]) -> Attributes:
    """Return the pairs as record attributes, the None values left out."""
    return tuple((key, value) for key, value in pairs.items() if value is not None)


def excerpt(value: object) -> str:
    """Return a content excerpt: the text, or its JSON, cut at `CONTENT_CHARS`."""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= CONTENT_CHARS else text[: CONTENT_CHARS - 1] + "…"


def as_int(value: object) -> int | None:
    """Return an event field as an int; None for anything else, a bool included."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def as_number(value: object) -> float | None:
    """Return an event field as a float; None for anything else, a bool included."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


@dataclasses.dataclass(slots=True)
class OpenSpan:
    """A span that has started and not ended."""

    span_id: str
    parent_span_id: str
    name: str
    start_ns: int
    kind: SpanKind = "internal"
    attributes: dict[str, AttributeValue] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(slots=True)
class Execution:
    """The execution whose events are being folded; it stays after its end for what follows."""

    trace_id: str
    span: OpenSpan
    ended: bool = False
    turn: OpenSpan | None = None
    turn_number: int | None = None
    calls: dict[str, OpenSpan] = dataclasses.field(default_factory=dict)
    tools: dict[str, OpenSpan] = dataclasses.field(default_factory=dict)
    verifies: dict[str, OpenSpan] = dataclasses.field(default_factory=dict)
    panels: dict[str, OpenSpan] = dataclasses.field(default_factory=dict)
    waits: dict[str, OpenSpan] = dataclasses.field(default_factory=dict)
    sequence: int = 0

    def groups(self) -> tuple[dict[str, OpenSpan], ...]:
        """Return the open spans by kind, innermost kind first."""
        return (self.waits, self.verifies, self.tools, self.calls, self.panels)

    def open_spans(self) -> list[OpenSpan]:
        """Return every open span, innermost first, the root last."""
        spans = [span for group in self.groups() for span in group.values()]
        if self.turn is not None:
            spans.append(self.turn)
        spans.append(self.span)
        return spans

    def parent(self) -> str:
        """Return the span id new work hangs from: the open turn, else the root."""
        return self.turn.span_id if self.turn is not None else self.span.span_id

    def end(self, span: OpenSpan, now: int, status: Status, *, incomplete: bool = False) -> Span:
        """Close an open span and return the record.

        Args:
            span: The open span.
            now: The end time; a span never ends before it starts.
            status: The outcome.
            incomplete: The journal never ended the work; the record says so.

        Returns:
            The finished span.
        """
        if incomplete:
            span.attributes["agent6.incomplete"] = True
        if span is self.turn:
            self.turn = None
        for group in self.groups():
            for key, open_span in list(group.items()):
                if open_span is span:
                    del group[key]
        return Span(
            self.trace_id,
            span.span_id,
            span.parent_span_id,
            span.name,
            span.kind,
            span.start_ns,
            max(now, span.start_ns),
            status,
            attributes(span.attributes),
        )

    def log(
        self,
        now: int,
        severity: Severity,
        body: str,
        fields: Mapping[str, AttributeValue | None],
        span_id: str = "",
    ) -> LogRecord:
        """Return a log record on the given span, else on the open turn or the root."""
        span = span_id or self.parent()
        return LogRecord(now, severity, body, attributes(fields), self.trace_id, span)


@dataclasses.dataclass(slots=True)
class Step:
    """One event being folded, and where its spans and log records go."""

    execution: Execution
    event: Mapping[str, Any]
    now: int
    spans: list[Span]
    logs: list[LogRecord]
