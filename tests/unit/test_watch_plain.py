# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the plain events.jsonl formatter."""

from __future__ import annotations

import json

from agent6.cli.plan_watch import _format_plain_event  # pyright: ignore[reportPrivateUsage]


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
    out = _format_plain_event(raw, run_start_ts=1000.0)
    assert "+  100.0s" in out
    assert "loop.auto_commit" in out
    assert "iteration=3" in out
    assert "sha='abc123def456'" in out
    assert "run_id" not in out  # filtered


def test_format_plain_event_handles_garbage_line() -> None:
    out = _format_plain_event("not-json-at-all\n", run_start_ts=0.0)
    assert out == "not-json-at-all"


def test_format_plain_event_no_ts_anchor() -> None:
    raw = json.dumps({"event": "ping"})
    out = _format_plain_event(raw, run_start_ts=None)
    assert "ping" in out
