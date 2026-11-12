# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Join imported lanes and project their dispatch outcomes into the task graph."""

from __future__ import annotations

import pathlib
from collections.abc import Callable

import pydantic

from agent6 import errors, lane_records
from agent6.graph import curator as graph_curator
from agent6.graph import models as graph_models
from agent6.harness import dag_focus
from agent6.sessions import lane_dispatch
from agent6.vcs import git_ops, git_read


def join_lane_result(
    root: pathlib.Path,
    res: lane_records.LaneResult,
    *,
    ref: str,
    fallback_parent: str | None,
    identity: git_read.CommitIdentity | None,
    also_branch: str | None,
) -> lane_records.LaneJoin:
    """Join one returned lane's branch onto the coordinator's chain.

    `chain_merge` leaves HEAD and the operator's checkout alone; the worktree gains the lane's
    files. Never raises; the DAG stamp is the segment's (`segment_stamp`).

    Args:
        root: The repository root.
        res: The lane's result.
        ref: The chain ref the merge lands on.
        fallback_parent: The parent an unborn ref merges onto, or None.
        identity: The merge commit's identity, or None for the repo's.
        also_branch: A checked-out branch to move with the ref, or None.

    Returns:
        `joined` with the sha, `no_changes` for a lane whose imported tip holds no commit
        beyond its base (nothing merged), `conflict`, or `failed` for a lane that imported
        nothing.
    """
    rid = res.spec.session_id
    if not res.ok:
        return lane_records.LaneJoin(rid, res.branch, "failed", "", res.error)
    if not res.has_commits:
        return lane_records.LaneJoin(rid, res.branch, "no_changes", "", "")
    try:
        sha = git_ops.chain_merge(
            root,
            res.imported_ref or res.branch,
            f"merge {res.branch}",
            ref=ref,
            fallback_parent=fallback_parent,
            identity=identity,
            also_branch=also_branch,
        )
    except (errors.GitError, OSError) as exc:
        return lane_records.LaneJoin(rid, res.branch, "failed", "", str(exc))
    if sha is None:
        return lane_records.LaneJoin(rid, res.branch, "conflict", "", "merge conflict")
    return lane_records.LaneJoin(rid, res.branch, "joined", sha, "")


def segment_stamp(lanes: list[lane_records.LaneJoin]) -> tuple[graph_models.NodeStatus, str, str]:
    """Reduce one segment's lane joins to its DAG stamp.

    A segment passes when any lane joined or ended with no changes, recording the last
    joined sha (none for no changes); the note names every lane. A conflict counts as not
    joined.

    Args:
        lanes: The segment's joins.

    Returns:
        The status, the note and the sha ("" when failed).
    """
    joined = [j for j in lanes if j.status == "joined"]
    note = "; ".join(lane_note(j) for j in lanes)
    if joined or any(j.status == "no_changes" for j in lanes):
        return "passed", note, joined[-1].sha if joined else ""
    return "failed", note, ""


def lane_note(j: lane_records.LaneJoin) -> str:
    """Return one lane's outcome as a DAG note line."""
    if j.status == "joined":
        return f"{j.session_id} joined at {j.sha[:12]}"
    if j.status == "no_changes":
        return f"{j.session_id} made no changes"
    if j.status == "conflict":
        return f"{j.session_id} conflicted; merge manually"
    return f"{j.session_id} failed: {j.detail}"


def summary_text(group: str, lanes: list[lane_records.LaneJoin]) -> str:
    """Return the one message that gives every lane's outcome and what the lane said.

    Args:
        group: The dispatch's group id.
        lanes: Each lane's join, in dispatch order.

    Returns:
        One line per lane, with the lane's finish summary under it as its journal holds it
        (2,000 characters at most).
    """
    lines = [f"[parallel] group {group} complete ({len(lanes)} lane(s)):"]
    for j in lanes:
        if j.status == "joined":
            lines.append(f"  - {j.session_id} ({j.branch}): joined at {j.sha[:12]}")
        elif j.status == "no_changes":
            lines.append(f"  - {j.session_id} ({j.branch}): finished with no changes to join.")
        elif j.status == "conflict":
            # Git is agent6's in this run and `.git` is read-only in the jail: the operator merges.
            lines.append(
                f"  - {j.session_id} ({j.branch}): CONFLICT -- branch imported but the merge"
                f" conflicted. It exists locally for the operator (`git merge {j.branch}`);"
                " continue without it."
            )
        else:
            lines.append(f"  - {j.session_id} ({j.branch}): FAILED -- {j.detail}; nothing joined.")
        if report := j.report.strip():
            lines.append("    summary: " + report.replace("\n", "\n      "))
    lines.append("Review what landed and continue.")
    return "\n".join(lines)


def join_of(child: lane_dispatch.ChildRecord) -> lane_records.LaneJoin:
    """Return a recorded lane's join; a lane never joined reads as failed."""
    join = child.join
    if join is None:
        return lane_records.LaneJoin(child.session_id, "", "failed", "", "not joined")
    return lane_records.LaneJoin(
        child.session_id, join.branch, join.status, join.sha, join.detail, child.report
    )


def parallel_parent_id(
    curator: graph_curator.GraphCurator | None, root_task_id: str | None
) -> str | None:
    """Return the parent for a dispatched subtask: the cursor's open node, else the run root."""
    if curator is None:
        return root_task_id
    return dag_focus.current_task_id(curator.nodes(), curator.cursor()) or root_task_id


def add_parallel_node(
    curator: graph_curator.GraphCurator | None,
    task: str,
    parent_id: str | None,
    *,
    source: lane_records.DispatchSource = "operator",
    log: Callable[[str], None],
) -> str | None:
    """Add a steering-created DAG node for one dispatched task.

    Args:
        curator: The graph, or None.
        task: The task text; its first line is the title.
        parent_id: The parent node, or None.
        source: Who dispatched it, named in the node's rationale.
        log: The run's text logger.

    Returns:
        The node's id, or None when no curator is wired or the add failed; the dispatch
        proceeds either way.
    """
    if curator is None:
        return None
    title = next((ln.strip() for ln in task.splitlines() if ln.strip()), "")[:200]
    try:
        node = curator.add_subtask(
            graph_models.AddSubtaskIntent(
                parent_id=parent_id,
                draft=graph_models.TaskNodeDraft(
                    title=title or "(parallel task)",
                    rationale=f"dispatched through /parallel by the {source}",
                    created_by="steering",
                ),
            )
        )
        return node.id
    except (graph_curator.CuratorError, OSError, pydantic.ValidationError) as exc:
        log(f"PARALLEL: DAG node add failed: {exc}")
        return None


def stamp_parallel_node(
    curator: graph_curator.GraphCurator | None,
    node_id: str | None,
    *,
    status: graph_models.NodeStatus,
    note: str,
    sha: str = "",
    log: Callable[[str], None],
) -> None:
    """Record a dispatched node's join sha, when given, then its final status.

    A curator fault is logged, never raised.

    Args:
        curator: The graph, or None.
        node_id: The node, or None.
        status: The final status.
        note: The status note.
        sha: The join sha, or "".
        log: The run's text logger.
    """
    if curator is None or node_id is None:
        return
    try:
        if sha:
            curator.record_commit(graph_models.RecordCommitIntent(id=node_id, sha=sha))
        curator.update_status(
            graph_models.UpdateStatusIntent(id=node_id, new_status=status, note=note)
        )
    except (graph_curator.CuratorError, OSError, pydantic.ValidationError) as exc:
        log(f"PARALLEL: DAG node stamp failed for {node_id}: {exc}")


def stamp_segment_node(
    curator: graph_curator.GraphCurator | None,
    node_id: str | None,
    lanes: list[lane_records.LaneJoin],
    *,
    log: Callable[[str], None],
) -> None:
    """Stamp one segment's DAG node from its lanes' joins (`segment_stamp`)."""
    status, note, sha = segment_stamp(lanes)
    stamp_parallel_node(curator, node_id, status=status, note=note, sha=sha, log=log)
