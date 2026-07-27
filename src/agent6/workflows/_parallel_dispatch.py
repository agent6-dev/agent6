# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Lane bookkeeping for `/parallel` steer dispatch.

The Workflow owns the dispatch policy (when to cut lanes, the DAG stamps, the
events, the injected group spawner); this module owns the Workflow-free
pieces: expanding a segment into lanes, joining one returned lane's branch,
and reducing lane outcomes to the DAG stamp and the summary message the model
continues with. Unit-testable without a Workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from agent6.directive import Segment, parse_spec
from agent6.graph.models import NodeStatus
from agent6.workflows.subrun import LaneResult, LaneTask, SubrunError, join_branch


@dataclass(frozen=True, slots=True)
class LaneJoin:
    """Per-lane outcome of a `/parallel` dispatch, for the summary + events.

    ``status`` is one of "joined" (branch merged, ``sha`` set), "conflict"
    (imported but the merge conflicted; the branch exists locally for a manual
    merge), or "failed" (the lane never produced an importable branch).
    """

    run_id: str
    branch: str
    status: Literal["joined", "conflict", "failed"]
    sha: str
    detail: str


def segment_lanes(seg: Segment) -> list[LaneTask]:
    """Expand one segment into its lanes: `parse_spec` maps the spec to one
    model per lane (`None` = the worker model). Raises DirectiveError on a
    bad spec (zero lanes, empty model list)."""
    return [LaneTask(task=seg.task, model=model) for model in parse_spec(seg.spec)]


def join_lane_result(root: Path, res: LaneResult) -> LaneJoin:
    """Join one returned lane's branch into the coordinator's branch. A failed
    lane (nothing imported) or a conflicted merge yields a non-"joined" status;
    a clean merge yields "joined" with the sha. Never raises; DAG stamping is
    the segment's (see `segment_stamp`)."""
    rid = res.spec.run_id
    if not res.ok:
        return LaneJoin(rid, res.branch, "failed", "", res.error)
    try:
        sha = join_branch(root, res.branch)
    except SubrunError as exc:
        return LaneJoin(rid, res.branch, "failed", "", str(exc))
    if sha is None:
        return LaneJoin(rid, res.branch, "conflict", "", "merge conflict")
    return LaneJoin(rid, res.branch, "joined", sha, "")


def segment_stamp(lanes: list[LaneJoin]) -> tuple[NodeStatus, str, str]:
    """Reduce one segment's lane joins to its DAG stamp ``(status, note,
    sha)``. A single-lane segment reduces to the old shape (passed with the
    join sha, or failed). A multi-lane segment passes when any lane joined --
    recording the LAST joined sha -- and the note names every lane; else it
    fails. NodeStatus has no "blocked", so a conflict counts as not-joined."""
    joined = [j for j in lanes if j.status == "joined"]
    note = "; ".join(lane_note(j) for j in lanes)
    if joined:
        return "passed", note, joined[-1].sha
    return "failed", note, ""


def lane_note(j: LaneJoin) -> str:
    if j.status == "joined":
        return f"{j.run_id} joined at {j.sha[:12]}"
    if j.status == "conflict":
        return f"{j.run_id} conflicted; merge manually"
    return f"{j.run_id} failed: {j.detail}"


def summary_text(group: str, joined: list[LaneJoin]) -> str:
    """ONE user message summarizing every lane's outcome so the model
    continues informed (joined sha, conflict-to-resolve, or failure reason)."""
    lines = [f"[parallel] group {group} complete ({len(joined)} lane(s)):"]
    for j in joined:
        if j.status == "joined":
            lines.append(f"  - {j.run_id} ({j.branch}): joined at {j.sha[:12]}")
        elif j.status == "conflict":
            lines.append(
                f"  - {j.run_id} ({j.branch}): CONFLICT -- branch imported but the merge"
                f" conflicted. It exists locally; run `git merge {j.branch}` and resolve,"
                " or discard it."
            )
        else:
            lines.append(f"  - {j.run_id} ({j.branch}): FAILED -- {j.detail}; nothing joined.")
    lines.append("Review what landed and continue.")
    return "\n".join(lines)
