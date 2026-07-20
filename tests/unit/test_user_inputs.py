# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the interactive user-input audit sink."""

from __future__ import annotations

import json
from pathlib import Path

from agent6.events import UserInputSink


def _read_all(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_user_input_sink_writes_strict_schema(tmp_path: Path) -> None:
    sink = UserInputSink(tmp_path / "user_inputs.jsonl")
    sink.record(kind="plan_approval", prompt="Proceed with plan?", answer="yes")
    sink.record(
        kind="plan_qa_answer",
        prompt="Which DB?",
        answer="postgres",
        source="answers_file",
        question_index=2,
    )
    rows = _read_all(tmp_path / "user_inputs.jsonl")
    assert len(rows) == 2
    assert rows[0]["kind"] == "plan_approval"
    assert rows[0]["answer"] == "yes"
    assert rows[0]["source"] == "stdin"  # default
    assert "ts" in rows[0]
    assert rows[1]["source"] == "answers_file"
    assert rows[1]["question_index"] == 2


def test_user_input_sink_reserved_keys_cannot_be_shadowed(tmp_path: Path) -> None:
    sink = UserInputSink(tmp_path / "u.jsonl")
    # Attempt to override schema fields via **extra; reserved keys must win.
    sink.record(
        kind="x",
        prompt="p",
        answer="a",
        ts="forged",  # would shadow timestamp
        kind_extra="ok",
    )
    row = _read_all(tmp_path / "u.jsonl")[0]
    assert row["ts"] != "forged"
    assert row["kind"] == "x"
    assert row["kind_extra"] == "ok"


def test_user_input_sink_swallows_unserializable(tmp_path: Path) -> None:
    sink = UserInputSink(tmp_path / "u.jsonl")

    class Bad:
        pass

    # default=repr should keep this serializable, so the row should land.
    sink.record(kind="x", prompt="p", answer="a", thing=Bad())
    rows = _read_all(tmp_path / "u.jsonl")
    assert len(rows) == 1
    rendered = str(rows[0]["thing"])
    assert "Bad" in rendered


def test_user_input_sink_creates_parent_dir(tmp_path: Path) -> None:
    target = tmp_path / "deep" / "nested" / "u.jsonl"
    sink = UserInputSink(target)
    sink.record(kind="x", prompt="p", answer="a")
    assert target.exists()


def test_record_survives_lone_surrogate(tmp_path: Path) -> None:
    """Same lossy-encode contract as EventSink.emit: a lone surrogate in a
    prompt/answer must not crash the audit trail, and the file stays strict
    UTF-8."""
    import json

    sink = UserInputSink(tmp_path / "user_inputs.jsonl")
    sink.record(kind="k", prompt="caf\udce9", answer="ok")
    rows = [
        json.loads(line)
        for line in (tmp_path / "user_inputs.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert rows and rows[0]["answer"] == "ok"
