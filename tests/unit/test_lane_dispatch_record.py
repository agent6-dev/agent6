# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The dispatch record: written before a launch, advanced at each boundary, read by a resume."""

from __future__ import annotations

import json
import pathlib

import pytest

from agent6 import lane_records
from agent6.sessions import lane_dispatch


def _record(identity: str = "run-p1") -> lane_dispatch.DispatchRecord:
    return lane_dispatch.DispatchRecord(
        identity=identity,
        source="model",
        owner="exec-1",
        turn=3,
        base="a" * 40,
        segments=(lane_dispatch.SegmentRecord(task="audit", spec="2"),),
        children=(
            lane_dispatch.ChildRecord(
                session_id=f"{identity}-l1",
                lane=1,
                task="audit",
                limits=lane_records.LaneLimits(max_turns=200, deadline_at=1.0e9),
            ),
            lane_dispatch.ChildRecord(session_id=f"{identity}-l2", lane=2, task="audit"),
        ),
    )


def test_a_record_round_trips_with_every_nested_shape(tmp_path: pathlib.Path) -> None:
    record = _record()
    lane_dispatch.write_record(tmp_path, record, now=123.5)
    (read,) = lane_dispatch.read_records(tmp_path)
    assert read == record.model_copy(update={"written_at": 123.5})
    assert read.child("run-p1-l1").limits == lane_records.LaneLimits(200, 1.0e9)
    assert lane_dispatch.record_path(tmp_path, "run-p1").is_file()
    assert not list(lane_dispatch.records_dir(tmp_path).glob("*.tmp*"))


def test_a_session_without_records_reads_as_none(tmp_path: pathlib.Path) -> None:
    """An older session never wrote a records dir; a resume sees nothing to reconcile."""
    assert lane_dispatch.read_records(tmp_path) == ()


def test_records_read_in_dispatch_order(tmp_path: pathlib.Path) -> None:
    for identity in ("run-p10", "run-p2", "run-p1"):
        lane_dispatch.write_record(tmp_path, _record(identity), now=1.0)
    assert [r.identity for r in lane_dispatch.read_records(tmp_path)] == [
        "run-p1",
        "run-p2",
        "run-p10",
    ]


def test_a_newer_or_malformed_record_refuses(tmp_path: pathlib.Path) -> None:
    lane_dispatch.write_record(tmp_path, _record(), now=1.0)
    path = lane_dispatch.record_path(tmp_path, "run-p1")
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["version"] = lane_dispatch.RECORD_VERSION + 1
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(lane_dispatch.DispatchRecordError, match="newer"):
        lane_dispatch.read_records(tmp_path)
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(lane_dispatch.DispatchRecordError, match="invalid dispatch record"):
        lane_dispatch.read_records(tmp_path)


def test_advancing_returns_new_records_and_keeps_the_rest(tmp_path: pathlib.Path) -> None:
    record = _record()
    joined = record.with_child(
        record.child("run-p1-l1").model_copy(
            update={
                "join": lane_dispatch.JoinRecord(
                    status="joined", sha="b" * 40, target_before="a" * 40, target_after="b" * 40
                ),
                "usage": lane_dispatch.UsageRecord(usd=1.25),
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
