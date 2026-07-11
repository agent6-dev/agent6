# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the plain logs.jsonl formatter."""

from __future__ import annotations

import json

from agent6.ui.cli.plan_watch import (
    event_epoch,  # pyright: ignore[reportPrivateUsage]
    format_plain_event,  # pyright: ignore[reportPrivateUsage]
)


def test_format_plain_event_renders_known_fields() -> None:
    raw = json.dumps(
        {
            "ts": 1100.0,
            "event": "loop.auto_commit",
            "run_id": "ignored-by-formatter",
            "iteration": 3,
            "sha": "abc123def456",
        }
    )
    out = format_plain_event(raw, run_start_ts=1000.0)
    assert "+  100.0s" in out
    assert "loop.auto_commit" in out
    assert "iteration=3" in out
    assert "sha='abc123def456'" in out
    assert "run_id" not in out  # filtered


def test_format_plain_event_handles_garbage_line() -> None:
    out = format_plain_event("not-json-at-all\n", run_start_ts=0.0)
    assert out == "not-json-at-all"


def test_format_plain_event_no_ts_anchor() -> None:
    raw = json.dumps({"event": "ping"})
    out = format_plain_event(raw, run_start_ts=None)
    assert "ping" in out


def test_event_epoch_parses_iso_and_numbers() -> None:
    # EventSink writes ISO-8601 strings; the anchor must parse those.
    assert event_epoch("2026-06-08T05:41:39.762404+00:00") is not None
    assert event_epoch(1100.0) == 1100.0
    assert event_epoch("not-a-timestamp") is None
    assert event_epoch(None) is None
    assert event_epoch(True) is None  # bool is not a usable epoch


def test_format_plain_event_renders_elapsed_for_iso_ts() -> None:
    # Regression: ts is an ISO string (events.py), not a number; the elapsed
    # column must still render rather than always blanking.
    start = "2026-06-08T05:41:39+00:00"
    later = "2026-06-08T05:42:39+00:00"  # +60s
    anchor = event_epoch(start)
    out = format_plain_event(
        json.dumps({"ts": later, "type": "loop.auto_commit"}), run_start_ts=anchor
    )
    assert "+   60.0s" in out
    assert "loop.auto_commit" in out
