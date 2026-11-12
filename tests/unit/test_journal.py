# SPDX-License-Identifier: Apache-2.0
"""The journal's vocabulary: every record has a writer, every writer passes a record."""

from __future__ import annotations

import ast
import json
import pathlib

import pydantic
import pytest

from agent6.sessions import event_log, journal

_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _record_references(source: str) -> set[str]:
    """Collect the `journal.<Record>` names a module refers to."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "journal"
            and node.attr[:1].isupper()
        ):
            found.add(node.attr)
    return found


def test_every_record_has_a_writer_and_the_map_is_in_type_order() -> None:
    """A record nothing writes, or a writer with no record, needs an explicit vocabulary edit."""
    referenced: set[str] = set()
    for path in (_ROOT / "src" / "agent6").rglob("*.py"):
        if path.name == "journal.py":
            continue
        referenced |= _record_references(path.read_text(encoding="utf-8"))
    records = {cls.__name__: event_type for event_type, cls in journal.EVENT_TYPES.items()}
    unwritten = {name for name in records if name not in referenced}
    assert not unwritten, f"records nothing writes: {sorted(unwritten)}"
    assert list(journal.EVENT_TYPES) == sorted(journal.EVENT_TYPES)
    for event_type, cls in journal.EVENT_TYPES.items():
        assert event_type == cls.TYPE
        assert (cls.__doc__ or "").strip(), event_type
        for name, info in cls.model_fields.items():
            assert info.description, f"{event_type}.{name} has no meaning"


def test_a_record_refuses_an_unknown_field_and_a_wrong_type() -> None:
    """The record is the schema: a writer cannot invent a field or mistype one."""
    with pytest.raises(pydantic.ValidationError, match="typo"):
        journal.LoopNote(text="ok", typo=1)  # type: ignore[call-arg]
    with pytest.raises(pydantic.ValidationError, match="text"):
        journal.LoopNote(text=3)  # type: ignore[arg-type]
    with pytest.raises(pydantic.ValidationError, match="text"):
        journal.LoopNote()  # type: ignore[call-arg]
    with pytest.raises(pydantic.ValidationError, match="nodes"):
        journal.GraphUpdate(nodes=[], cursor=None)  # type: ignore[arg-type]


def test_the_sink_writes_the_envelope_and_leaves_unset_fields_out(tmp_path: pathlib.Path) -> None:
    sink = event_log.EventSink(tmp_path / "logs.jsonl")
    sink.emit(journal.LoopNote(text="ready"))
    sink.emit(journal.GraphUpdate(nodes={}, cursor=None))
    sink.emit(journal.LoopPinRestored(pins=("keep tests",), count=1))
    sink.emit(journal.BudgetUpdate(usd_total=0))
    rows = [json.loads(line) for line in sink.path.read_text().splitlines()]
    assert [row["type"] for row in rows] == [
        "loop.note",
        "graph.update",
        "loop.pin.restored",
        "budget.update",
    ]
    assert all(row["event_id"] and row["ts"] for row in rows)
    assert rows[0]["text"] == "ready"
    assert rows[1]["cursor"] is None  # set to None: written as null
    assert rows[2]["pins"] == ["keep tests"]
    assert rows[3] == {
        "ts": rows[3]["ts"],
        "type": "budget.update",
        "event_id": rows[3]["event_id"],
        "usd_total": 0,
    }


def test_provider_request_documents_call_and_seat_identity() -> None:
    """The request record includes its parent call and optional panel seat identity."""
    fields = journal.ProviderRequest.model_fields
    assert {"model_call_id", "panel_id", "seat_index"} <= fields.keys()
    assert fields["model_call_id"].is_required()
    assert not fields["panel_id"].is_required() and not fields["seat_index"].is_required()


def test_session_lifecycle_names_keep_creation_resume_and_end_distinct() -> None:
    """The vocabulary names session creation, resumed executions and execution ends apart."""
    for name in ("session.start", "session.resume", "session.stop"):
        assert name in journal.EVENT_TYPES, name
    assert "loop.resume.start" not in journal.EVENT_TYPES
    assert frozenset({"session.resume", "session.start"}) == journal.SESSION_START_EVENTS
