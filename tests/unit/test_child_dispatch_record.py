# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The dispatch record: written before a launch, advanced at each boundary, read by a resume."""

from __future__ import annotations

import json
import pathlib

import pytest

from agent6 import child_records
from agent6.sessions import child_dispatch


def _record(identity: str = "run-p1") -> child_dispatch.DispatchRecord:
    return child_dispatch.DispatchRecord(
        identity=identity,
        source="model",
        owner="exec-1",
        turn=3,
        base="a" * 40,
        segments=(child_dispatch.SegmentRecord(task="audit", spec="2"),),
        children=(
            child_dispatch.ChildRecord(
                session_id=f"{identity}-l1",
                number=1,
                task="audit",
                limits=child_records.ChildLimits(max_turns=200, deadline_at=1.0e9),
            ),
            child_dispatch.ChildRecord(session_id=f"{identity}-l2", number=2, task="audit"),
        ),
    )


def test_a_record_round_trips_with_every_nested_shape(tmp_path: pathlib.Path) -> None:
    record = _record()
    child_dispatch.write_record(tmp_path, record, now=123.5)
    (read,) = child_dispatch.read_records(tmp_path)
    assert read == record.model_copy(update={"written_at": 123.5})
    assert read.child("run-p1-l1").limits == child_records.ChildLimits(200, 1.0e9)
    assert child_dispatch.record_path(tmp_path, "run-p1").is_file()
    assert not list(child_dispatch.records_dir(tmp_path).glob("*.tmp*"))


def test_a_session_without_records_reads_as_none(tmp_path: pathlib.Path) -> None:
    """An older session never wrote a records dir; a resume sees nothing to reconcile."""
    assert child_dispatch.read_records(tmp_path) == ()


def test_records_read_in_dispatch_order(tmp_path: pathlib.Path) -> None:
    for identity in ("run-p10", "run-p2", "run-p1"):
        child_dispatch.write_record(tmp_path, _record(identity), now=1.0)
    assert [r.identity for r in child_dispatch.read_records(tmp_path)] == [
        "run-p1",
        "run-p2",
        "run-p10",
    ]


def test_a_newer_or_malformed_record_refuses(tmp_path: pathlib.Path) -> None:
    child_dispatch.write_record(tmp_path, _record(), now=1.0)
    path = child_dispatch.record_path(tmp_path, "run-p1")
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["version"] = child_dispatch.RECORD_VERSION + 1
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(child_dispatch.DispatchRecordError, match="newer"):
        child_dispatch.read_records(tmp_path)
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(child_dispatch.DispatchRecordError, match="invalid dispatch record"):
        child_dispatch.read_records(tmp_path)


def test_advancing_returns_new_records_and_keeps_the_rest(tmp_path: pathlib.Path) -> None:
    record = _record()
    joined = record.with_child(
        record.child("run-p1-l1").model_copy(
            update={
                "join": child_dispatch.JoinRecord(
                    status="joined", sha="b" * 40, target_before="a" * 40, target_after="b" * 40
                ),
                "usage": child_dispatch.UsageRecord(usd=1.25),
                "settled": True,
            }
        )
    )
    assert record.child("run-p1-l1").join is None
    assert joined.child("run-p1-l1").join is not None
    assert joined.child("run-p1-l2") == record.child("run-p1-l2")
    done = joined.advanced("complete", message="[parallel] group run-p1 complete")
    assert (done.state, done.message) == ("complete", "[parallel] group run-p1 complete")
    assert joined.state == "prepared"
    with pytest.raises(KeyError):
        record.child("nope")
