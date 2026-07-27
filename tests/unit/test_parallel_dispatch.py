# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The Workflow-free lane bookkeeping behind `/parallel` dispatch, pinned at
the unit level now that it lives outside the loop (workflows/_parallel_dispatch)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agent6.directive import DirectiveError, Segment
from agent6.workflows import _parallel_dispatch as pd
from agent6.workflows._parallel_dispatch import (
    LaneJoin,
    join_lane_result,
    lane_note,
    segment_lanes,
    segment_stamp,
    summary_text,
)
from agent6.workflows.subrun import LaneResult, LaneSpec, SubrunError


def _res(*, ok: bool, error: str = "", branch: str = "agent6/lane-1") -> LaneResult:
    spec = LaneSpec(lane=1, run_id="lane-1", workdir=Path("/nowhere"), model=None)
    return LaneResult(spec=spec, run_dir=Path("/nowhere"), branch=branch, ok=ok, error=error)


def test_segment_lanes_expands_counts_and_models() -> None:
    assert [lt.model for lt in segment_lanes(Segment(task="t", spec="3"))] == [None, None, None]
    lanes = segment_lanes(Segment(task="t", spec="m1,m2"))
    assert [lt.model for lt in lanes] == ["m1", "m2"]
    assert all(lt.task == "t" for lt in lanes)
    with pytest.raises(DirectiveError):
        segment_lanes(Segment(task="t", spec="0"))


def test_join_lane_result_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed lane, a conflicted merge, and a SubrunError each reduce to a
    non-"joined" LaneJoin instead of aborting the run."""
    assert join_lane_result(Path("/r"), _res(ok=False, error="died")) == LaneJoin(
        "lane-1", "agent6/lane-1", "failed", "", "died"
    )

    def _conflict(_root: Path, _branch: str) -> str | None:
        return None

    monkeypatch.setattr(pd, "join_branch", _conflict)
    assert join_lane_result(Path("/r"), _res(ok=True)).status == "conflict"

    def _boom(_root: Path, _branch: str) -> str | None:
        raise SubrunError("fetch failed")

    monkeypatch.setattr(pd, "join_branch", _boom)
    j = join_lane_result(Path("/r"), _res(ok=True))
    assert (j.status, j.detail) == ("failed", "fetch failed")

    def _clean(_root: Path, _branch: str) -> str | None:
        return "a" * 40

    monkeypatch.setattr(pd, "join_branch", _clean)
    j = join_lane_result(Path("/r"), _res(ok=True))
    assert (j.status, j.sha) == ("joined", "a" * 40)


def _join(status: Any, run_id: str = "lane-1", sha: str = "") -> LaneJoin:
    return LaneJoin(run_id, f"agent6/{run_id}", status, sha, "boom" if status == "failed" else "")


def test_segment_stamp_reduces_lanes() -> None:
    # Single lane keeps the old shape: passed with the join sha, or failed.
    assert segment_stamp([_join("joined", sha="abc123def4567")]) == (
        "passed",
        "lane-1 joined at abc123def456",
        "abc123def4567",
    )
    status, note, sha = segment_stamp([_join("failed")])
    assert (status, sha) == ("failed", "")
    assert "lane-1 failed: boom" in note
    # Multi-lane: any join passes, recording the LAST joined sha; the note
    # names every lane. All-conflict fails (NodeStatus has no "blocked").
    status, note, sha = segment_stamp(
        [_join("joined", "a", sha="1111111111111"), _join("joined", "b", sha="2222222222222")]
    )
    assert (status, sha) == ("passed", "2222222222222")
    assert "a joined at" in note and "b joined at" in note
    status, note, _sha = segment_stamp([_join("conflict"), _join("conflict", "b")])
    assert status == "failed"
    assert "conflicted; merge manually" in note


def test_summary_text_names_every_outcome() -> None:
    text = summary_text(
        "p1",
        [
            _join("joined", "a", sha="abc123def4567"),
            _join("conflict", "b"),
            _join("failed", "c"),
        ],
    )
    assert "group p1 complete (3 lane(s))" in text
    assert "a (agent6/a): joined at abc123def456" in text
    assert "CONFLICT" in text and "git merge agent6/b" in text
    assert "FAILED -- boom; nothing joined." in text
    assert text.endswith("Review what landed and continue.")


def test_lane_note_wordings() -> None:
    assert lane_note(_join("joined", sha="abc123def4567")) == "lane-1 joined at abc123def456"
    assert lane_note(_join("conflict")) == "lane-1 conflicted; merge manually"
    assert lane_note(_join("failed")) == "lane-1 failed: boom"
