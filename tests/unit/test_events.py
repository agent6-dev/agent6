# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for `agent6.events.EventSink`."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

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


def test_delta_events_flush_but_do_not_fsync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ephemeral streaming deltas skip fsync (a reasoning model emits tens of
    thousands; an fsync each throttles the SSE read). Durable events still fsync.
    They are still written + flushed so tailers see them live."""
    synced: list[int] = []

    def _fake_fsync(fd: int) -> None:
        synced.append(fd)

    monkeypatch.setattr(os, "fsync", _fake_fsync)
    sink = EventSink(tmp_path / "logs.jsonl")

    sink.emit("role.thinking_delta", text="reasoning")
    sink.emit("role.text_delta", text="answer")
    assert synced == []  # no fsync for the deltas

    sink.emit("tool.call", name="read_file")
    assert len(synced) == 1  # a durable event fsyncs

    # all three are on disk regardless (flush, not fsync, makes them readable)
    types = [
        json.loads(line)["type"] for line in (tmp_path / "logs.jsonl").read_text().splitlines()
    ]
    assert types == ["role.thinking_delta", "role.text_delta", "tool.call"]
