# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for `agent6.sessions.event_log.EventSink`."""

from __future__ import annotations

import json
import os
import pathlib

import pytest

from agent6 import paths as agent6_paths
from agent6.sessions import event_log, journal
from agent6.viewmodel import tail


def _read_lines(path: pathlib.Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_emit_appends_json_lines(tmp_path: pathlib.Path) -> None:
    sink = event_log.EventSink(tmp_path / "logs.jsonl")
    sink.emit(journal.LoopNote(text="do a thing"))
    sink.emit(journal.LoopToolCall(turn=1, name="read_file"))
    lines = _read_lines(tmp_path / "logs.jsonl")
    assert len(lines) == 2
    assert lines[0]["type"] == "loop.note"
    assert lines[0]["text"] == "do a thing"
    assert "ts" in lines[0]
    assert lines[1]["type"] == "loop.tool.call"
    assert lines[1]["turn"] == 1


def test_reopen_drops_torn_tail_before_resumed_events(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A torn delta cannot swallow the next execution's resume start."""
    path = tmp_path / "logs.jsonl"
    first = event_log.EventSink(path)
    first.emit(journal.SessionStart(user_task="task", mode="execute", execution_id="first"))
    committed = path.read_bytes()
    fragment = b'{"type":"role.text_delta","text":"caf\xc3'
    with path.open("ab") as fh:
        fh.write(fragment)

    resumed = event_log.EventSink(path)
    resumed.emit(
        journal.SessionResume(
            session_id="session",
            user_task="task",
            steer="continue",
            mode="execute",
            turn=1,
            messages=2,
            execution_id="resumed",
        )
    )
    resumed.emit(journal.RoleTextDelta(role="worker", text="done"))
    resumed.emit(
        journal.SessionStop(
            execution_id="execution",
            reason="finish_session",
            turns=1,
            verification="passed",
        )
    )

    expected = ["session.start", "session.resume", "role.text_delta", "session.stop"]
    assert [event["type"] for event in tail.tail_events(path, follow=False)] == expected
    assert [event["type"] for event in tail.LogTail(path).read()] == expected
    assert path.read_bytes().startswith(committed)
    assert len(caplog.records) == 1
    assert str(len(fragment)) in caplog.records[0].message
    assert str(path) in caplog.records[0].message


def test_emit_creates_parent_dir(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "nested" / "deeper" / "logs.jsonl"
    sink = event_log.EventSink(target)
    sink.emit(journal.SessionStranded())
    assert target.is_file()


def test_durable_events_have_unique_ids(tmp_path: pathlib.Path) -> None:
    """Give identical durable payloads distinct identities without timestamp ordering."""
    sink = event_log.EventSink(tmp_path / "logs.jsonl")
    sink.emit(journal.LoopNote(text="same"))
    sink.emit(journal.LoopNote(text="same"))
    rows = _read_lines(sink.path)
    assert all(row.get("event_id") for row in rows)
    assert rows[0]["event_id"] != rows[1]["event_id"]


def test_emit_reprs_non_serializable_fields(tmp_path: pathlib.Path) -> None:
    """The sink never drops a field: an unknown object, circular refs included, lands as a repr."""
    sink = event_log.EventSink(tmp_path / "logs.jsonl")

    class Bad:
        pass

    bad = Bad()
    bad.self_ref = bad  # type: ignore[attr-defined]
    sink.emit(
        journal.ToolCall(
            name="example", args_preview={"x": 1, "p": tmp_path / "a", "weird": bad}, call_id=1
        )
    )
    lines = _read_lines(tmp_path / "logs.jsonl")
    assert len(lines) == 1
    args = lines[0]["args_preview"]
    assert isinstance(args, dict)
    assert args["x"] == 1
    p_value = args["p"]
    assert isinstance(p_value, str)
    assert p_value.endswith("/a")
    weird = args["weird"]
    assert isinstance(weird, str) and "Bad" in weird  # repr'd, not dropped


def test_durable_emit_raises_on_unwritable_journal(tmp_path: pathlib.Path) -> None:
    """A durable event that cannot land raises, and the in-process listener is not notified.

    The journal is the read model every surface trusts; a lost session.stop renders "running"
    forever. Deltas stay best-effort and still render live, since the lossless transcripts keep
    their copy.
    """
    # Point at a path under a regular file -> mkdir will fail.
    blocker = tmp_path / "blocker"
    blocker.write_text("", encoding="utf-8")
    sink = event_log.EventSink(blocker / "subdir" / "logs.jsonl")
    seen: list[dict[str, object]] = []
    sink.subscribe(seen.append)
    with pytest.raises(event_log.EventWriteError, match="unwritable"):
        sink.emit(
            journal.SessionStop(
                execution_id="execution",
                reason="finish_session",
                turns=1,
                verification="passed",
            )
        )
    assert seen == []
    sink.emit(journal.RoleTextDelta(role="worker", text="still live"))  # ephemeral: must not raise
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
    sink = event_log.EventSink(tmp_path / "logs.jsonl")

    sink.emit(journal.RoleThinkingDelta(role="worker", text="reasoning"))
    sink.emit(journal.RoleTextDelta(role="worker", text="answer"))
    assert synced == []  # no fsync for the deltas

    sink.emit(journal.ToolCall(name="read_file", args_preview={}, call_id=1))
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

    sink = event_log.EventSink(tmp_path / "logs.jsonl")
    sink.emit(journal.LoopNote(text="caf\udce9"))
    sink.emit(
        journal.ToolCall(name="finish_session", args_preview={"summary": "done \ud83d"}, call_id=1)
    )
    lines = [
        json.loads(line)
        for line in (tmp_path / "logs.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert [e["type"] for e in lines] == ["loop.note", "tool.call"]
    assert "?" in lines[0]["text"]  # the surrogate was replaced, not dropped


def test_a_value_that_merely_answers_isoformat_encodes_as_its_repr(tmp_path: pathlib.Path) -> None:
    """The encoder's date branch keys on the datetime types, not on an `isoformat` attribute.

    A mock, whose every attribute is another mock, recursed without end and hung the journal write.
    """
    import datetime
    import unittest.mock

    assert (
        event_log._json_default(datetime.datetime(2026, 1, 2, tzinfo=datetime.UTC))
        == "2026-01-02T00:00:00+00:00"
    )
    mock = unittest.mock.MagicMock()
    assert event_log._json_default(mock) == repr(mock)


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
    sink = event_log.EventSink(tmp_path / "run" / "logs.jsonl")
    sink.emit(journal.LoopNote(text="ready"))
    sink.emit(journal.LoopToolCall(name="read_file", turn=1))
    assert calls == [tmp_path / "run"]
    assert (tmp_path / "run" / "logs.jsonl").read_text(encoding="utf-8").count("\n") == 2
