# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the stdlib JSONL tail-follower."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from agent6.ui.tail import tail_events


def test_tail_yields_existing_lines_in_non_follow_mode(tmp_path: Path) -> None:
    p = tmp_path / "logs.jsonl"
    p.write_text(
        json.dumps({"type": "run.start"}) + "\n" + json.dumps({"type": "run.end"}) + "\n",
        encoding="utf-8",
    )
    out = list(tail_events(p, follow=False))
    assert [e["type"] for e in out] == ["run.start", "run.end"]


def test_tail_skips_malformed_lines(tmp_path: Path) -> None:
    p = tmp_path / "logs.jsonl"
    p.write_text(
        json.dumps({"type": "a"}) + "\n" + "{not json\n" + json.dumps({"type": "b"}) + "\n",
        encoding="utf-8",
    )
    out = list(tail_events(p, follow=False))
    assert [e["type"] for e in out] == ["a", "b"]


def test_tail_skips_non_dict_json(tmp_path: Path) -> None:
    p = tmp_path / "logs.jsonl"
    p.write_text("[]\n" + json.dumps({"type": "x"}) + "\n", encoding="utf-8")
    out = list(tail_events(p, follow=False))
    assert [e["type"] for e in out] == ["x"]


def test_tail_returns_when_file_missing_and_not_follow(tmp_path: Path) -> None:
    out = list(tail_events(tmp_path / "missing", follow=False))
    assert out == []


def test_tail_stops_at_run_end_when_requested(tmp_path: Path) -> None:
    p = tmp_path / "logs.jsonl"
    p.write_text(
        json.dumps({"type": "a"})
        + "\n"
        + json.dumps({"type": "run.end"})
        + "\n"
        + json.dumps({"type": "after"})
        + "\n",
        encoding="utf-8",
    )
    out = list(tail_events(p, follow=False, stop_when_finished=True))
    assert [e["type"] for e in out] == ["a", "run.end"]


def test_tail_follows_appended_lines(tmp_path: Path) -> None:
    p = tmp_path / "logs.jsonl"
    p.write_text(json.dumps({"type": "first"}) + "\n", encoding="utf-8")

    def writer() -> None:
        time.sleep(0.3)
        with p.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"type": "second"}) + "\n")
            fh.flush()
        time.sleep(0.2)
        with p.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"type": "run.end"}) + "\n")
            fh.flush()

    t = threading.Thread(target=writer, daemon=True)
    t.start()
    out = list(tail_events(p, follow=True, poll_s=0.05, stop_when_finished=True))
    t.join(timeout=2)
    assert [e["type"] for e in out] == ["first", "second", "run.end"]
