# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for `agent6.events.EventSink`."""

from __future__ import annotations

import json
import os
import pathlib

import pytest

from agent6 import events
from agent6 import paths as agent6_paths


def _read_lines(path: pathlib.Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_emit_appends_json_lines(tmp_path: pathlib.Path) -> None:
    sink = events.EventSink(tmp_path / "logs.jsonl")
    sink.emit("session.start", task="do a thing")
    sink.emit("step.start", index=1, title="hello")
    lines = _read_lines(tmp_path / "logs.jsonl")
    assert len(lines) == 2
    assert lines[0]["type"] == "session.start"
    assert lines[0]["task"] == "do a thing"
    assert "ts" in lines[0]
    assert lines[1]["type"] == "step.start"
    assert lines[1]["index"] == 1


def test_emit_creates_parent_dir(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "nested" / "deeper" / "logs.jsonl"
    sink = events.EventSink(target)
    sink.emit("hello")
    assert target.is_file()


def test_emit_reprs_non_serializable_fields(tmp_path: pathlib.Path) -> None:
    """The sink never drops a field: an unknown object, circular refs included, lands as a repr."""
    sink = events.EventSink(tmp_path / "logs.jsonl")

    class Bad:
        pass

    bad = Bad()
    bad.self_ref = bad  # type: ignore[attr-defined]
    sink.emit("ok", x=1, p=tmp_path / "a", weird=bad)
    lines = _read_lines(tmp_path / "logs.jsonl")
    assert len(lines) == 1
    assert lines[0]["x"] == 1
    p_value = lines[0]["p"]
    assert isinstance(p_value, str)
    assert p_value.endswith("/a")
    weird = lines[0]["weird"]
    assert isinstance(weird, str) and "Bad" in weird  # repr'd, not dropped


def test_durable_emit_raises_on_unwritable_journal(tmp_path: pathlib.Path) -> None:
    """A durable event that cannot land raises, and the in-process listener is not notified.

    The journal is the read model every surface trusts; a lost session.end renders "running"
    forever. Deltas stay best-effort and still render live, since the lossless transcripts keep
    their copy.
    """
    # Point at a path under a regular file -> mkdir will fail.
    blocker = tmp_path / "blocker"
    blocker.write_text("", encoding="utf-8")
    sink = events.EventSink(blocker / "subdir" / "logs.jsonl")
    seen: list[dict[str, object]] = []
    sink.subscribe(seen.append)
    with pytest.raises(events.EventWriteError, match="unwritable"):
        sink.emit("session.end", reason="finish_session", all_passed=True)
    assert seen == []
    sink.emit("role.text_delta", text="still live")  # ephemeral: must not raise
    assert [e["type"] for e in seen] == ["role.text_delta"]


def test_delta_events_flush_but_do_not_fsync(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Streaming deltas are flushed but not fsynced; durable events still fsync.

    A reasoning model emits tens of thousands of deltas, and an fsync each throttles the SSE read;
    the flush keeps them live for tailers.
    """
    synced: list[int] = []

    def _fake_fsync(fd: int) -> None:
        synced.append(fd)

    monkeypatch.setattr(os, "fsync", _fake_fsync)
    sink = events.EventSink(tmp_path / "logs.jsonl")

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


def test_emit_survives_lone_surrogate(tmp_path: pathlib.Path) -> None:
    """A lone surrogate is recorded lossily and the file stays strictly valid UTF-8.

    `json.dumps(ensure_ascii=False)` passes it through, and a text-mode write raises
    UnicodeEncodeError, a ValueError the OSError guard does not catch.
    """
    import json

    sink = events.EventSink(tmp_path / "logs.jsonl")
    sink.emit("session.start", user_task="caf\udce9")
    sink.emit("tool.call", args={"summary": "done \ud83d"})
    lines = [
        json.loads(line)
        for line in (tmp_path / "logs.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert [e["type"] for e in lines] == ["session.start", "tool.call"]
    assert "?" in lines[0]["user_task"]  # the surrogate was replaced, not dropped


def test_a_value_that_merely_answers_isoformat_encodes_as_its_repr(tmp_path: pathlib.Path) -> None:
    """The encoder's date branch keys on the datetime types, not on an `isoformat` attribute.

    A mock, whose every attribute is another mock, recursed without end and hung the journal write.
    """
    import datetime
    from unittest import mock as unittest_mock

    assert (
        events._json_default(datetime.datetime(2026, 1, 2, tzinfo=datetime.UTC))
        == "2026-01-02T00:00:00+00:00"
    )
    mock = unittest_mock.MagicMock()
    assert events._json_default(mock) == repr(mock)


def test_the_log_dir_is_created_once_not_per_event(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sink creates its directory through the state tree's one creator, once, not per emit.

    The creator's handback walks the whole dir under sudo.
    """
    calls: list[pathlib.Path] = []
    real = agent6_paths.mkdir_for_real_user

    def counting(path: pathlib.Path) -> None:
        calls.append(path)
        real(path)

    monkeypatch.setattr(agent6_paths, "mkdir_for_real_user", counting)
    sink = events.EventSink(tmp_path / "run" / "logs.jsonl")
    sink.emit("session.start")
    sink.emit("loop.tool.call", name="read_file")
    assert calls == [tmp_path / "run"]
    assert (tmp_path / "run" / "logs.jsonl").read_text(encoding="utf-8").count("\n") == 2
