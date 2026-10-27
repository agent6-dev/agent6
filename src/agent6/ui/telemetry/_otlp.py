# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Encode telemetry records as OTLP/HTTP JSON request bodies.

OTLP's JSON is the protobuf JSON mapping with its own rules: trace and span ids are hex
strings, 64-bit integers and nanosecond timestamps are decimal strings, enums are numbers.
One function per signal returns the body a collector accepts at `/v1/traces`, `/v1/metrics`
and `/v1/logs`.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

from agent6 import __version__
from agent6.viewmodel import telemetry_metrics, telemetry_records

SCOPE = {"name": "agent6.telemetry", "version": __version__}
TRACES_PATH = "/v1/traces"
METRICS_PATH = "/v1/metrics"
LOGS_PATH = "/v1/logs"
_SPAN_KIND = {"internal": 1, "client": 3}
_STATUS = {"unset": 0, "ok": 1, "error": 2}
_SEVERITY = {"INFO": 9, "WARN": 13, "ERROR": 17}
_CUMULATIVE = 2
_PATHS = {"traces": TRACES_PATH, "metrics": METRICS_PATH, "logs": LOGS_PATH}


def path_for(signal: str) -> str:
    """Return the request path of a signal: traces, metrics or logs."""
    return _PATHS[signal]


def _value(value: telemetry_records.AttributeValue) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"a telemetry value must be finite, not {value!r}")
        return {"doubleValue": value}
    return {"stringValue": value}


def _attributes(
    pairs: Iterable[tuple[str, telemetry_records.AttributeValue]],
) -> list[dict[str, Any]]:
    return [{"key": key, "value": _value(value)} for key, value in pairs]


def _double(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError(f"a telemetry value must be finite, not {value!r}")
    return float(value)


def _span(span: telemetry_records.Span) -> dict[str, Any]:
    body: dict[str, Any] = {
        "traceId": span.trace_id,
        "spanId": span.span_id,
        "name": span.name,
        "kind": _SPAN_KIND[span.kind],
        "startTimeUnixNano": str(span.start_ns),
        "endTimeUnixNano": str(span.end_ns),
        "attributes": _attributes(span.attributes),
        "status": {"code": _STATUS[span.status]},
    }
    if span.parent_span_id:
        body["parentSpanId"] = span.parent_span_id
    return body


def traces_body(
    spans: Iterable[telemetry_records.Span], resource: telemetry_records.Attributes
) -> dict[str, Any]:
    """Return the `/v1/traces` request body for the spans under one resource."""
    return {
        "resourceSpans": [
            {
                "resource": {"attributes": _attributes(resource)},
                "scopeSpans": [{"scope": SCOPE, "spans": [_span(s) for s in spans]}],
            }
        ]
    }


def logs_body(
    records: Iterable[telemetry_records.LogRecord], resource: telemetry_records.Attributes
) -> dict[str, Any]:
    """Return the `/v1/logs` request body for the records under one resource."""
    rows = [
        {
            "timeUnixNano": str(record.time_ns),
            "severityNumber": _SEVERITY[record.severity],
            "severityText": record.severity,
            "body": {"stringValue": record.body},
            "attributes": _attributes(record.attributes),
            "traceId": record.trace_id,
            "spanId": record.span_id,
        }
        for record in records
    ]
    return {
        "resourceLogs": [
            {
                "resource": {"attributes": _attributes(resource)},
                "scopeLogs": [{"scope": SCOPE, "logRecords": rows}],
            }
        ]
    }


def _point(labels: telemetry_records.Attributes, start_ns: int, now_ns: int) -> dict[str, Any]:
    return {
        "attributes": _attributes(labels),
        "startTimeUnixNano": str(start_ns),
        "timeUnixNano": str(now_ns),
    }


def metrics_body(
    snapshot: telemetry_metrics.MetricsSnapshot,
    resource: telemetry_records.Attributes,
    *,
    start_ns: int,
    now_ns: int,
) -> dict[str, Any]:
    """Return the `/v1/metrics` request body: every metric with data, cumulative since `start_ns`.

    Args:
        snapshot: The fold's cumulative metrics.
        resource: The resource attributes.
        start_ns: When the cumulative window began.
        now_ns: The reading's time.

    Returns:
        The request body.
    """
    metrics: list[dict[str, Any]] = []
    for definition in telemetry_metrics.METRICS:
        entry: dict[str, Any] = {
            "name": definition.name,
            "description": definition.help,
            "unit": definition.unit,
        }
        if definition.kind == "histogram":
            points = [
                {
                    **_point(labels, start_ns, now_ns),
                    "count": str(histogram.count),
                    "sum": _double(histogram.total),
                    "bucketCounts": [str(count) for count in histogram.counts],
                    "explicitBounds": list(histogram.bounds),
                }
                for (name, labels), histogram in sorted(snapshot.histograms.items())
                if name == definition.name
            ]
            if points:
                entry["histogram"] = {"dataPoints": points, "aggregationTemporality": _CUMULATIVE}
                metrics.append(entry)
            continue
        source = snapshot.counters if definition.kind == "counter" else snapshot.gauges
        points = [
            {**_point(labels, start_ns, now_ns), "asDouble": _double(value)}
            for (name, labels), value in sorted(source.items())
            if name == definition.name
        ]
        if not points:
            continue
        if definition.kind == "counter":
            entry["sum"] = {
                "dataPoints": points,
                "aggregationTemporality": _CUMULATIVE,
                "isMonotonic": True,
            }
        else:
            entry["gauge"] = {"dataPoints": points}
        metrics.append(entry)
    return {
        "resourceMetrics": [
            {
                "resource": {"attributes": _attributes(resource)},
                "scopeMetrics": [{"scope": SCOPE, "metrics": metrics}],
            }
        ]
    }
