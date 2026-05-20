# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for `agent6.events.EventSink`."""

from __future__ import annotations

import json
from pathlib import Path

from agent6.events import EventSink


def _read_lines(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_emit_appends_json_lines(tmp_path: Path) -> None:
    sink = EventSink(tmp_path / "logs.jsonl")
    sink.emit("run.start", task="do a thing")
    sink.emit("step.start", index=1, title="hello")
    lines = _read_lines(tmp_path / "logs.jsonl")
    assert len(lines) == 2
    assert lines[0]["type"] == "run.start"
    assert lines[0]["task"] == "do a thing"
    assert "ts" in lines[0]
    assert lines[1]["type"] == "step.start"
    assert lines[1]["index"] == 1


def test_emit_creates_parent_dir(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "deeper" / "logs.jsonl"
    sink = EventSink(target)
    sink.emit("hello")
    assert target.is_file()


def test_emit_drops_non_serializable_fields(tmp_path: Path) -> None:
    sink = EventSink(tmp_path / "logs.jsonl")

    # Path is handled via default; an object with a circular ref should be skipped.
    class Bad:
        pass

    bad = Bad()
    bad.self_ref = bad  # type: ignore[attr-defined]
    # repr() handles it via _json_default fallback; ensure no exception escapes.
    sink.emit("ok", x=1, p=tmp_path / "a", weird=bad)
    lines = _read_lines(tmp_path / "logs.jsonl")
    assert len(lines) == 1
    assert lines[0]["x"] == 1
    p_value = lines[0]["p"]
    assert isinstance(p_value, str)
    assert p_value.endswith("/a")


def test_emit_swallows_oserror(tmp_path: Path) -> None:
    # Point at a path under a regular file → mkdir will fail.
    blocker = tmp_path / "blocker"
    blocker.write_text("", encoding="utf-8")
    sink = EventSink(blocker / "subdir" / "logs.jsonl")
    # Must not raise.
    sink.emit("anything")
