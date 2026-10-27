# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 telemetry export`: send this repository's journals where `[telemetry]` points."""

from __future__ import annotations

import pathlib
import signal
import threading

from agent6 import secret_store
from agent6.config import ConfigError, layer
from agent6.ui.cli import _common
from agent6.ui.telemetry import export


def cmd_export(config_paths: tuple[pathlib.Path, ...], session_id: str, *, follow: bool) -> int:
    """Export the journals, print what was delivered, and fail loudly on a refused endpoint.

    Args:
        config_paths: The `--config` files in command-line order.
        session_id: One session, or "" for every session with a journal.
        follow: Keep reading the session's journal as it grows.

    Returns:
        0 when every selected destination took its records; 1 when one refused or rejected
        records; 2 when nothing is selected, a secret is missing or the session does not exist.
    """
    try:
        cfg = layer.load_effective(pathlib.Path.cwd(), config_paths).config
    except ConfigError as exc:
        _common.error(str(exc))
        return 2
    telemetry = cfg.telemetry
    if (telemetry.traces, telemetry.metrics, telemetry.logs) == ("none", "none", "none"):
        _common.error(
            "nothing is selected: set telemetry.traces, telemetry.metrics or telemetry.logs"
            " (`agent6 config show telemetry`)"
        )
        return 2
    try:
        headers = secret_store.telemetry_headers(telemetry.headers) if telemetry.otlp else {}
    except secret_store.SecretsError as exc:
        _common.error(str(exc))
        return 2
    stop = threading.Event()
    if follow:
        # Ctrl-C ends the follow through the same drain a stopped tail takes: what is queued
        # posts, the cursors advance, the metrics publish once more.
        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, lambda _sig, _frame: stop.set())
    try:
        report = export.export(
            pathlib.Path.cwd(),
            telemetry,
            session_id=session_id,
            follow=follow,
            headers=headers,
            should_stop=stop.is_set,
        )
    except export.ExportError as exc:
        _common.error(str(exc))
        return 2
    delivered = [
        f"{report.sessions} session{'' if report.sessions == 1 else 's'}",
        f"{report.spans} spans",
        f"{report.logs} log records",
        f"{report.metric_posts} metrics posts",
    ]
    if report.prometheus_file:
        delivered.append(f"metrics written to {report.prometheus_file}")
    print("[telemetry] " + ", ".join(delivered))
    for note in report.notes:
        print(f"[telemetry] {note}")
    return 1 if report.notes else 0
