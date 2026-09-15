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

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from agent6.directive import Segment, parse_spec
from agent6.git_ops import CommitIdentity, GitError, chain_merge
from agent6.graph.curator import CuratorError, GraphCurator
from agent6.graph.models import (
    AddSubtaskIntent,
    NodeStatus,
    RecordCommitIntent,
    TaskNodeDraft,
    UpdateStatusIntent,
)
from agent6.workflows._dag_focus import current_task_id
from agent6.workflows.subrun import LaneResult, LaneTask


@dataclass(frozen=True, slots=True)
class LaneJoin:
    """Per-lane outcome of a `/parallel` dispatch, for the summary + events.

    `status` is one of "joined" (branch merged, `sha` set), "conflict"
    (imported but the merge conflicted; the branch exists locally for a manual
    merge), or "failed" (the lane never produced an importable branch).
    """

    session_id: str
    branch: str
    status: Literal["joined", "conflict", "failed"]
    sha: str
    detail: str


def segment_lanes(seg: Segment, pins: Sequence[str] = (), *, limit: int) -> list[LaneTask]:
    """Expand one segment into its lanes: `parse_spec` maps the spec to one
    model per lane (`None` = the worker model). Raises DirectiveError on a
    bad spec (zero lanes, empty model list, more than *limit* lanes --
    `[parallel].max_lanes`, refused before the list is built).

    Operator *pins* ride on every lane OUT-OF-BAND of the task. `/pin`
    promises an instruction "stays binding for the rest of the run", and a
    lane is work done for that run whose branch is merged back into the
    coordinator's -- so a lane that never saw the pin could violate a
    standing instruction and have it land anyway. Folded into the task text
    they would become the lane's manifest user_task (every listing and the
    judge's brief leading with the pin header); the spawner's --pin channel seeds
    the lane's own pin state instead, which renders the same block a restart
    re-shows.
    """
    lane_pins = tuple(pins)
    models = parse_spec(seg.spec, limit=limit)
    return [LaneTask(task=seg.task, model=model, pins=lane_pins) for model in models]


def join_lane_result(
    root: Path,
    res: LaneResult,
    *,
    ref: str,
    fallback_parent: str | None,
    identity: CommitIdentity | None,
    also_branch: str | None,
) -> LaneJoin:
    """Join one returned lane's branch onto the coordinator's chain at *ref*
    (`chain_merge`: HEAD and the operator's checkout stay untouched; the
    worktree gains the lane's files). A failed lane (nothing imported) or a
    conflicted merge yields a non-"joined" status; a clean merge yields
    "joined" with the sha. Never raises; DAG stamping is the segment's (see
    `segment_stamp`)."""
    rid = res.spec.session_id
    if not res.ok:
        return LaneJoin(rid, res.branch, "failed", "", res.error)
    try:
        sha = chain_merge(
            root,
            res.branch,
            f"merge {res.branch}",
            ref=ref,
            fallback_parent=fallback_parent,
            identity=identity,
            also_branch=also_branch,
        )
    except (GitError, OSError) as exc:
        return LaneJoin(rid, res.branch, "failed", "", str(exc))
    if sha is None:
        return LaneJoin(rid, res.branch, "conflict", "", "merge conflict")
    return LaneJoin(rid, res.branch, "joined", sha, "")


def segment_stamp(lanes: list[LaneJoin]) -> tuple[NodeStatus, str, str]:
    """Reduce one segment's lane joins to its DAG stamp ``(status, note,
    sha)``. A single-lane segment stamps plainly (passed with the
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
        return f"{j.session_id} joined at {j.sha[:12]}"
    if j.status == "conflict":
        return f"{j.session_id} conflicted; merge manually"
    return f"{j.session_id} failed: {j.detail}"


def summary_text(group: str, lanes: list[LaneJoin]) -> str:
    """ONE user message summarizing every lane's outcome so the model
    continues informed (joined sha, conflict-to-resolve, or failure reason)."""
    lines = [f"[parallel] group {group} complete ({len(lanes)} lane(s)):"]
    for j in lanes:
        if j.status == "joined":
            lines.append(f"  - {j.session_id} ({j.branch}): joined at {j.sha[:12]}")
        elif j.status == "conflict":
            # Git is agent6's in this run (and `.git` is read-only in the jail),
            # so the merge is the operator's to finish, not the model's.
            lines.append(
                f"  - {j.session_id} ({j.branch}): CONFLICT -- branch imported but the merge"
                f" conflicted. It exists locally for the operator (`git merge {j.branch}`);"
                " continue without it."
            )
        else:
            lines.append(f"  - {j.session_id} ({j.branch}): FAILED -- {j.detail}; nothing joined.")
    lines.append("Review what landed and continue.")
    return "\n".join(lines)


def parallel_parent_id(curator: GraphCurator | None, root_task_id: str | None) -> str | None:
    """Parent for a dispatched subtask: the curator cursor when it points at
    an open node, else the run root."""
    if curator is None:
        return root_task_id
    return current_task_id(curator.nodes(), curator.cursor()) or root_task_id


def add_parallel_node(
    curator: GraphCurator | None, task: str, parent_id: str | None, *, log: Callable[[str], None]
) -> str | None:
    """Add a steering-created DAG node for one dispatched task; None when no
    curator is wired or the add fails (the dispatch still proceeds)."""
    if curator is None:
        return None
    title = next((ln.strip() for ln in task.splitlines() if ln.strip()), "")[:200]
    try:
        node = curator.add_subtask(
            AddSubtaskIntent(
                parent_id=parent_id,
                draft=TaskNodeDraft(
                    title=title or "(parallel task)",
                    rationale="dispatched via /parallel steering",
                    created_by="steering",
                ),
            )
        )
        return node.id
    except (CuratorError, OSError, ValidationError) as exc:
        log(f"PARALLEL: DAG node add failed: {exc}")
        return None


def stamp_parallel_node(
    curator: GraphCurator | None,
    node_id: str | None,
    *,
    status: NodeStatus,
    note: str,
    sha: str = "",
    log: Callable[[str], None],
) -> None:
    """Record a dispatched node's outcome: its join sha (when given) then its
    final status. Best-effort: a curator hiccup must not break the run."""
    if curator is None or node_id is None:
        return
    try:
        if sha:
            curator.record_commit(RecordCommitIntent(id=node_id, sha=sha))
        curator.update_status(UpdateStatusIntent(id=node_id, new_status=status, note=note))
    except (CuratorError, OSError, ValidationError) as exc:
        log(f"PARALLEL: DAG node stamp failed for {node_id}: {exc}")


def stamp_segment_node(
    curator: GraphCurator | None,
    node_id: str | None,
    lanes: list[LaneJoin],
    *,
    log: Callable[[str], None],
) -> None:
    """Stamp one segment's DAG node from its lanes' joins (`segment_stamp`)."""
    status, note, sha = segment_stamp(lanes)
    stamp_parallel_node(curator, node_id, status=status, note=note, sha=sha, log=log)
