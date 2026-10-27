# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Render a metrics snapshot in the Prometheus text exposition format (0.0.4).

The textfile collector of node_exporter reads the result; counters end in `_total`, every
metric carries `HELP` and `TYPE`, histograms expose cumulative `_bucket` rows up to `+Inf`,
and no sample carries a timestamp.
"""

from __future__ import annotations

import math

from agent6.viewmodel import telemetry_metrics


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _escape_help(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n")


def _labels(pairs: telemetry_metrics.Attributes, extra: tuple[tuple[str, str], ...] = ()) -> str:
    items = [(key, str(value)) for key, value in pairs] + list(extra)
    if not items:
        return ""
    return "{" + ",".join(f'{key}="{_escape(value)}"' for key, value in items) + "}"


def _number(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError(f"a metric value must be finite, not {value!r}")
    text = repr(float(value))
    return text[:-2] if text.endswith(".0") else text


def _bound(bound: float) -> str:
    return "+Inf" if math.isinf(bound) else _number(bound)


def exposition(snapshot: telemetry_metrics.MetricsSnapshot) -> str:
    """Return the exposition text: each defined metric with data, in definition order."""
    lines: list[str] = []
    for definition in telemetry_metrics.METRICS:
        name = definition.name
        if definition.kind == "histogram":
            rows = sorted(
                (labels, h) for (metric, labels), h in snapshot.histograms.items() if metric == name
            )
            if not rows:
                continue
            lines += [f"# HELP {name} {_escape_help(definition.help)}", f"# TYPE {name} histogram"]
            for labels, histogram in rows:
                running = 0
                for bound, count in zip(
                    (*histogram.bounds, math.inf), histogram.counts, strict=True
                ):
                    running += count
                    lines.append(
                        f"{name}_bucket{_labels(labels, (('le', _bound(bound)),))} {running}"
                    )
                lines.append(f"{name}_sum{_labels(labels)} {_number(histogram.total)}")
                lines.append(f"{name}_count{_labels(labels)} {histogram.count}")
            continue
        source = snapshot.counters if definition.kind == "counter" else snapshot.gauges
        rows = sorted((labels, v) for (metric, labels), v in source.items() if metric == name)
        if not rows:
            continue
        lines += [
            f"# HELP {name} {_escape_help(definition.help)}",
            f"# TYPE {name} {definition.kind}",
        ]
        lines += [f"{name}{_labels(labels)} {_number(value)}" for labels, value in rows]
    return "\n".join(lines) + "\n" if lines else ""
