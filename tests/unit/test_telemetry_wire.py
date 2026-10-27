# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The OTLP/HTTP JSON and Prometheus text encodings of the telemetry records.

The wire golden pins the three OTLP bodies for one journal, so a drift in the encoding or
the projection is a deliberate edit here; the scope version is normalized, so a release
bump is not.
"""

from __future__ import annotations

import json
import math
import pathlib

import pytest

from agent6 import __version__
from agent6.ui.telemetry import _otlp, _prometheus  # pyright: ignore[reportPrivateUsage]
from agent6.viewmodel import telemetry_metrics, telemetry_records
from tests.unit import test_telemetry_fold as fold_tests

_GOLDEN = pathlib.Path(__file__).parent / "data" / "golden_telemetry_wire.json"
_RESOURCE: telemetry_records.Attributes = (("service.name", "agent6"), ("agent6.repo", "0123abcd"))
_START = telemetry_records.event_ns(fold_tests._ts(0))  # pyright: ignore[reportPrivateUsage]
_NOW = telemetry_records.event_ns(fold_tests._ts(60))  # pyright: ignore[reportPrivateUsage]


def _bodies() -> dict[str, object]:
    fold, spans, logs = fold_tests._run(fold_tests._journal())  # pyright: ignore[reportPrivateUsage]
    assert _START is not None and _NOW is not None
    return {
        "traces": _otlp.traces_body(spans, _RESOURCE),
        "metrics": _otlp.metrics_body(fold.metrics(), _RESOURCE, start_ns=_START, now_ns=_NOW),
        "logs": _otlp.logs_body(logs, _RESOURCE),
    }


def _golden_text() -> str:
    text = json.dumps(_bodies(), indent=2, sort_keys=True) + "\n"
    return text.replace(f'"version": "{__version__}"', '"version": "<agent6 version>"')


def test_the_otlp_bodies_match_the_golden() -> None:
    """The three request bodies for one journal are pinned byte for byte, the version aside."""
    assert _golden_text() == _GOLDEN.read_text(encoding="utf-8"), (
        "the OTLP wire drifted; if intended, regenerate: python -c "
        '"import tests.unit.test_telemetry_wire as t; '
        "open(t._GOLDEN,'w').write(t._golden_text())\""
    )


def test_otlp_json_follows_the_protobuf_mapping_with_its_otlp_exceptions() -> None:
    """Ids are hex strings, 64-bit numbers are decimal strings, enums are numbers."""
    bodies = _bodies()
    traces = bodies["traces"]
    assert isinstance(traces, dict)
    spans = traces["resourceSpans"][0]["scopeSpans"][0]["spans"]
    root = next(s for s in spans if s["name"] == telemetry_records.SPAN_EXECUTION)
    call = next(s for s in spans if s["name"] == telemetry_records.SPAN_MODEL_CALL)
    assert len(root["traceId"]) == 32 and len(root["spanId"]) == 16
    assert "parentSpanId" not in root and call["parentSpanId"] != root["spanId"]
    assert root["startTimeUnixNano"] == str(_START) and root["endTimeUnixNano"] == str(_NOW)
    assert (root["kind"], call["kind"]) == (1, 3)
    assert root["status"] == {"code": 2} and call["status"] == {"code": 1}
    by_key = {a["key"]: a["value"] for a in call["attributes"]}
    assert by_key["agent6.tokens_in"] == {"intValue": "10"}
    assert by_key["agent6.ok"] == {"boolValue": True}
    assert by_key["agent6.model"] == {"stringValue": "m"}
    scope = traces["resourceSpans"][0]["scopeSpans"][0]["scope"]
    assert scope["name"] == "agent6.telemetry" and scope["version"] == __version__
    assert traces["resourceSpans"][0]["resource"]["attributes"][0] == {
        "key": "service.name",
        "value": {"stringValue": "agent6"},
    }
    logs = bodies["logs"]
    assert isinstance(logs, dict)
    records = logs["resourceLogs"][0]["scopeLogs"][0]["logRecords"]
    failed = next(r for r in records if r["body"] == {"stringValue": "verify failed"})
    assert (failed["severityNumber"], failed["severityText"]) == (17, "ERROR")
    assert failed["traceId"] == root["traceId"] and len(failed["spanId"]) == 16
    assert _otlp.path_for("logs") == "/v1/logs"


def test_otlp_metrics_are_cumulative_sums_gauges_and_histograms() -> None:
    """Counters are monotonic cumulative sums; a histogram carries string bucket counts."""
    bodies = _bodies()
    metrics = bodies["metrics"]
    assert isinstance(metrics, dict)
    rows = {m["name"]: m for m in metrics["resourceMetrics"][0]["scopeMetrics"][0]["metrics"]}
    tokens = rows["agent6_tokens_total"]
    assert tokens["unit"] == "{token}" and tokens["sum"]["isMonotonic"] is True
    assert tokens["sum"]["aggregationTemporality"] == 2
    point = next(
        p
        for p in tokens["sum"]["dataPoints"]
        if {"key": "type", "value": {"stringValue": "input"}} in p["attributes"]
    )
    assert point["asDouble"] == 10.0
    assert point["startTimeUnixNano"] == str(_START) and point["timeUnixNano"] == str(_NOW)
    ratio = rows["agent6_plan_used_ratio"]["gauge"]["dataPoints"][0]
    assert ratio["asDouble"] == pytest.approx(0.12)
    verify = rows["agent6_verify_duration_seconds"]["histogram"]
    (point,) = verify["dataPoints"]
    assert point["count"] == "1" and point["sum"] == 30.0
    bounds = telemetry_metrics.METRIC_BY_NAME["agent6_verify_duration_seconds"].bounds
    assert point["explicitBounds"] == list(bounds)
    assert len(point["bucketCounts"]) == len(point["explicitBounds"]) + 1
    assert point["bucketCounts"][point["explicitBounds"].index(30.0)] == "1"
    assert "agent6_sessions_total" in rows
    # A metric nothing observed is left out rather than sent empty.
    for metric in rows.values():
        data = metric.get("sum") or metric.get("gauge") or metric.get("histogram")
        assert data["dataPoints"]


def test_a_non_finite_value_is_refused_not_serialized() -> None:
    """Python would write `NaN`, which OTLP JSON does not accept."""
    snapshot = telemetry_metrics.MetricsSnapshot({("agent6_cost_usd_total", ()): math.nan}, {}, {})
    with pytest.raises(ValueError, match="finite"):
        _otlp.metrics_body(snapshot, _RESOURCE, start_ns=1, now_ns=2)
    with pytest.raises(ValueError, match="finite"):
        _prometheus.exposition(snapshot)


def test_the_prometheus_exposition_follows_the_text_format() -> None:
    """HELP and TYPE per metric, escaped labels, cumulative buckets up to +Inf, no timestamps."""
    fold, _spans, _logs = fold_tests._run(fold_tests._journal())  # pyright: ignore[reportPrivateUsage]
    text = _prometheus.exposition(fold.metrics())
    lines = text.splitlines()
    assert lines[0] == "# HELP agent6_sessions_total Sessions whose journal was read."
    assert lines[1] == "# TYPE agent6_sessions_total counter"
    assert "agent6_sessions_total 1" in lines
    assert 'agent6_tokens_total{model="m",provider="p",type="input"} 10' in lines
    assert "# TYPE agent6_verify_duration_seconds histogram" in lines
    buckets = [line for line in lines if line.startswith("agent6_verify_duration_seconds_bucket")]
    assert buckets[0] == 'agent6_verify_duration_seconds_bucket{le="1"} 0'
    assert 'agent6_verify_duration_seconds_bucket{le="30"} 1' in buckets
    assert buckets[-1] == 'agent6_verify_duration_seconds_bucket{le="+Inf"} 1'
    assert "agent6_verify_duration_seconds_sum 30" in lines
    assert "agent6_verify_duration_seconds_count 1" in lines
    assert "agent6_plan_used_ratio 0.12" in lines
    assert text.endswith("\n") and not any(" 1793" in line for line in lines)
    odd = telemetry_metrics.MetricsSnapshot(
        {("agent6_tool_calls_total", (("outcome", "ok"), ("tool", 'a"b\\c\nd'))): 2.0}, {}, {}
    )
    assert (
        'agent6_tool_calls_total{outcome="ok",tool="a\\"b\\\\c\\nd"} 2'
        in _prometheus.exposition(odd)
    )
    assert _prometheus.exposition(telemetry_metrics.MetricsSnapshot({}, {}, {})) == ""
