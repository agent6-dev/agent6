# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Real collectors accept the telemetry export of the wire golden's journals.

Run with `uv run pytest -m collector` and otelcol-contrib and promtool on PATH.
AGENT6_OTELCOL is a test-only path to otelcol-contrib, checked before PATH.
Each half skips with the missing binary's name when its tool is absent.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import socket
import subprocess
import time
from typing import Any

import pytest

from agent6 import paths
from agent6.config import TelemetryConfig
from agent6.ui.telemetry import export
from tests import collector_env
from tests.unit import test_telemetry_fold as fold_tests
from tests.unit import test_telemetry_machine as machine_tests

pytestmark = pytest.mark.collector

_DATA = pathlib.Path(__file__).parents[1] / "unit" / "data"
_SIGNALS = ("logs", "metrics", "traces")


@dataclasses.dataclass(frozen=True, slots=True)
class _Golden:
    repo: pathlib.Path
    bodies: dict[str, Any]


@pytest.fixture(params=("run", "machine"))
def golden(request: pytest.FixtureRequest, tmp_path: pathlib.Path) -> _Golden:
    """Seed the real journal layout from the same inputs as the wire golden."""
    repo = tmp_path / "repo"
    repo.mkdir()
    state = paths.state_dir(repo)
    bodies = json.loads((_DATA / "golden_telemetry_wire.json").read_text(encoding="utf-8"))
    if request.param == "run":
        journals = {
            state / "sessions" / "runs" / "run-AAAAAA" / "logs.jsonl": fold_tests._journal()
        }
    else:
        root = state / "machines" / "demo"
        journals = {
            root / "journal.jsonl": machine_tests.journal(),
            root / "states" / "0003-review" / "logs.jsonl": machine_tests.worker(),
        }
        bodies = bodies["machine"]
    for path, rows in journals.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return _Golden(repo, bodies)


def _counts(bodies: list[dict[str, Any]]) -> dict[str, int]:
    """Count received records across OTLP batches, including distinct traces."""
    spans = [
        span
        for body in bodies
        for resource in body.get("resourceSpans", [])
        for scope in resource.get("scopeSpans", [])
        for span in scope.get("spans", [])
    ]
    return {
        "spans": len(spans),
        "traces": len({span["traceId"] for span in spans}),
        "logs": sum(
            len(scope.get("logRecords", []))
            for body in bodies
            for resource in body.get("resourceLogs", [])
            for scope in resource.get("scopeLogs", [])
        ),
        "metrics": sum(
            len(scope.get("metrics", []))
            for body in bodies
            for resource in body.get("resourceMetrics", [])
            for scope in resource.get("scopeMetrics", [])
        ),
    }


def test_otlp_collector_receives_the_golden_counts(golden: _Golden, tmp_path: pathlib.Path) -> None:
    """Collector output retains every golden span, trace, metric and log record."""
    binary = collector_env.require_collector()
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    outputs = {signal: tmp_path / f"{signal}.jsonl" for signal in _SIGNALS}
    config = tmp_path / "collector.yaml"
    # JSON is YAML; paths stay quoted without a YAML dependency.
    config.write_text(
        json.dumps(
            {
                "receivers": {"otlp": {"protocols": {"http": {"endpoint": f"127.0.0.1:{port}"}}}},
                "exporters": {
                    f"file/{signal}": {"path": str(path), "format": "json"}
                    for signal, path in outputs.items()
                },
                "service": {
                    "telemetry": {"metrics": {"level": "none"}},
                    "pipelines": {
                        signal: {"receivers": ["otlp"], "exporters": [f"file/{signal}"]}
                        for signal in _SIGNALS
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    log = tmp_path / "collector.log"
    with log.open("w", encoding="utf-8") as output:
        process = subprocess.Popen(
            [binary, "--config", str(config)],
            cwd=tmp_path,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline and process.poll() is None:
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                        break
                except OSError:
                    time.sleep(0.05)
            else:
                pytest.fail(
                    f"otelcol-contrib did not listen on 127.0.0.1:{port} "
                    f"(exit {process.poll()}):\n" + log.read_text(encoding="utf-8")
                )
            report = export.export(
                golden.repo,
                TelemetryConfig(
                    traces="otlp",
                    metrics="otlp",
                    logs="otlp",
                    endpoint=f"http://127.0.0.1:{port}",
                ),
            )
            assert report.notes == (), log.read_text(encoding="utf-8")
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
    diagnostics = log.read_text(encoding="utf-8")
    assert process.returncode == 0, diagnostics
    expected = _counts([golden.bodies[signal] for signal in _SIGNALS])
    received: list[dict[str, Any]] = []
    for path in outputs.values():
        assert path.is_file(), diagnostics
        received.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
    assert _counts(received) == expected, diagnostics
    assert (report.sessions, report.spans, report.logs, report.metric_posts) == (
        1,
        expected["spans"],
        expected["logs"],
        1,
    )


@pytest.mark.collector(binary="promtool")
def test_promtool_accepts_the_exported_textfile(golden: _Golden, tmp_path: pathlib.Path) -> None:
    """Promtool accepts the real exporter's Prometheus textfile."""
    binary = collector_env.require_collector("promtool")
    target = tmp_path / "agent6.prom"
    report = export.export(
        golden.repo, TelemetryConfig(metrics="prometheus", prometheus_file=str(target))
    )
    assert report.sessions == 1 and report.prometheus_file == str(target) and report.notes == ()
    assert target.stat().st_size > 0
    with target.open(encoding="utf-8") as metrics:
        checked = subprocess.run(
            [binary, "check", "metrics"],
            stdin=metrics,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    assert checked.returncode == 0, checked.stdout + checked.stderr
