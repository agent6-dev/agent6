# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The telemetry metrics: their definitions, the meter that accumulates them, and the merge.

Every exposition (OTLP, Prometheus text) reads `METRICS` for names, kinds, meanings and
units, so one table owns them. Free labels (model, provider, tool, source, reason) stop at
`LABEL_VALUES` distinct values and read `other` past it.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping
from typing import Literal

AttributeValue = str | int | float | bool
Attributes = tuple[tuple[str, AttributeValue], ...]
# Distinct values per free label before "other".
LABEL_VALUES = 64
_VERIFY_BOUNDS = (1.0, 5.0, 15.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1800.0)
_WAIT_BOUNDS = (1.0, 5.0, 15.0, 60.0, 300.0, 900.0, 3600.0)


@dataclasses.dataclass(frozen=True, slots=True)
class MetricDef:
    """One metric's name, kind, meaning and unit, shared by every exposition.

    Attributes:
        name: The Prometheus name; the OTLP name is the same.
        kind: counter, gauge or histogram.
        help: The meaning, one sentence.
        unit: The unit in UCUM terms (`1`, `s`, `USD`, `{token}`).
        bounds: A histogram's bucket bounds.
        merge: How sessions' readings of a gauge combine: a count adds up, a reading keeps
            the latest session's value.
    """

    name: str
    kind: Literal["counter", "gauge", "histogram"]
    help: str
    unit: str
    bounds: tuple[float, ...] = ()
    merge: Literal["sum", "last"] = "sum"


METRICS: tuple[MetricDef, ...] = (
    MetricDef("agent6_sessions_total", "counter", "Sessions whose journal was read.", "1"),
    MetricDef("agent6_executions_total", "counter", "Execution starts: runs and resumes.", "1"),
    MetricDef(
        "agent6_execution_ends_total",
        "counter",
        "Execution ends by end reason and the gate's word for the final tree.",
        "1",
    ),
    MetricDef(
        "agent6_model_calls_total", "counter", "Model calls by model, provider and outcome.", "1"
    ),
    MetricDef(
        "agent6_tokens_total",
        "counter",
        "Tokens by type (input, output, cache_read, cache_creation), model and provider.",
        "{token}",
    ),
    MetricDef(
        "agent6_cost_usd_total",
        "counter",
        "Priced spend from the journal's cumulative budget readings.",
        "USD",
    ),
    MetricDef(
        "agent6_cost_partial",
        "gauge",
        "1 when the latest budget reading says some spend is unpriced, so the cost is a floor.",
        "1",
        merge="last",
    ),
    MetricDef(
        "agent6_plan_observed_points_total",
        "counter",
        "Plan-window movement the run observed; the account's whole window, not this run alone.",
        "1",
    ),
    MetricDef(
        "agent6_plan_used_ratio",
        "gauge",
        "The latest plan-window reading, 0 to 1.",
        "1",
        merge="last",
    ),
    MetricDef("agent6_tool_calls_total", "counter", "Tool dispatches by tool and outcome.", "1"),
    MetricDef("agent6_verify_attempts_total", "counter", "Verification attempts by outcome.", "1"),
    MetricDef(
        "agent6_verify_duration_seconds",
        "histogram",
        "Verification attempt durations.",
        "s",
        _VERIFY_BOUNDS,
    ),
    MetricDef("agent6_findings_total", "counter", "Distinct review findings by severity.", "1"),
    MetricDef("agent6_open_findings", "gauge", "Open review findings by severity.", "1"),
    MetricDef("agent6_approvals_total", "counter", "Approval answers by decision and source.", "1"),
    MetricDef(
        "agent6_operator_wait_duration_seconds",
        "histogram",
        "Time a run waited on an operator, by prompt kind.",
        "s",
        _WAIT_BOUNDS,
    ),
)
METRIC_BY_NAME = {m.name: m for m in METRICS}


@dataclasses.dataclass(frozen=True, slots=True)
class Histogram:
    """A fixed-bucket histogram; `counts` has one entry per bound plus the +Inf bucket."""

    bounds: tuple[float, ...]
    counts: tuple[int, ...]
    total: float
    count: int


@dataclasses.dataclass(frozen=True, slots=True)
class MetricsSnapshot:
    """The cumulative metrics, keyed by metric name and sorted label pairs."""

    counters: Mapping[tuple[str, Attributes], float]
    gauges: Mapping[tuple[str, Attributes], float]
    histograms: Mapping[tuple[str, Attributes], Histogram]


class Labels:
    """Bound the distinct values of one free label; the overflow reads `other`."""

    def __init__(self) -> None:
        self._seen: set[str] = set()

    def value(self, raw: object) -> str:
        """Return the label value for a raw field, bounded."""
        text = str(raw) if raw not in (None, "") else "unknown"
        if text in self._seen or len(self._seen) < LABEL_VALUES:
            self._seen.add(text)
            return text
        return "other"


class Meter:
    """Accumulate the counters, gauges and histograms `METRICS` defines."""

    def __init__(self) -> None:
        self._counters: dict[tuple[str, Attributes], float] = {}
        self._gauges: dict[tuple[str, Attributes], float] = {}
        self._histograms: dict[tuple[str, Attributes], tuple[list[int], float, int]] = {}
        self._labels: dict[str, Labels] = {}

    def label(self, dimension: str, raw: object) -> str:
        """Return a bounded value for a free label dimension."""
        return self._labels.setdefault(dimension, Labels()).value(raw)

    def count(self, name: str, amount: float = 1.0, **labels: AttributeValue) -> None:
        """Add to a counter."""
        key = (name, tuple(sorted(labels.items())))
        self._counters[key] = self._counters.get(key, 0.0) + amount

    def gauge(self, name: str, value: float, **labels: AttributeValue) -> None:
        """Set a gauge."""
        self._gauges[(name, tuple(sorted(labels.items())))] = value

    def observe(self, name: str, value: float, **labels: AttributeValue) -> None:
        """Record one histogram observation."""
        bounds = METRIC_BY_NAME[name].bounds
        key = (name, tuple(sorted(labels.items())))
        counts, total, count = self._histograms.get(key, ([0] * (len(bounds) + 1), 0.0, 0))
        index = next((i for i, bound in enumerate(bounds) if value <= bound), len(bounds))
        counts[index] += 1
        self._histograms[key] = (counts, total + value, count + 1)

    def snapshot(self) -> MetricsSnapshot:
        """Return the cumulative reading."""
        histograms = {
            key: Histogram(METRIC_BY_NAME[key[0]].bounds, tuple(counts), total, count)
            for key, (counts, total, count) in self._histograms.items()
        }
        return MetricsSnapshot(dict(self._counters), dict(self._gauges), histograms)


def merge_snapshots(snapshots: Iterable[MetricsSnapshot]) -> MetricsSnapshot:
    """Return one snapshot for several sessions, in time order.

    Counters and histograms add up; a gauge follows its definition's `merge`: a count
    (open findings) adds up, a reading keeps the latest session's value.

    Args:
        snapshots: The sessions' snapshots, oldest first.

    Returns:
        The merged snapshot.
    """
    counters: dict[tuple[str, Attributes], float] = {}
    gauges: dict[tuple[str, Attributes], float] = {}
    histograms: dict[tuple[str, Attributes], Histogram] = {}
    for snapshot in snapshots:
        for key, value in snapshot.counters.items():
            counters[key] = counters.get(key, 0.0) + value
        for key, value in snapshot.gauges.items():
            adds = METRIC_BY_NAME[key[0]].merge == "sum"
            gauges[key] = gauges.get(key, 0.0) + value if adds else value
        for key, histogram in snapshot.histograms.items():
            seen = histograms.get(key)
            if seen is None:
                histograms[key] = histogram
                continue
            counts = tuple(a + b for a, b in zip(seen.counts, histogram.counts, strict=True))
            histograms[key] = Histogram(
                seen.bounds, counts, seen.total + histogram.total, seen.count + histogram.count
            )
    return MetricsSnapshot(counters, gauges, histograms)
