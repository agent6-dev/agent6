# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Dispatch a `/parallel` steer.

`ParallelDispatcher` owns the policy: when to cut lanes, the DAG stamps, the events, the
injected group spawner. The functions below expand a segment into lanes, join one lane's
branch and reduce lane outcomes to the DAG stamp and the summary message, without a Harness.
"""

from __future__ import annotations

import dataclasses
import pathlib
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Literal

import pydantic

from agent6 import directive, git_ops
from agent6.graph import curator as graph_curator
from agent6.graph import models as graph_models
from agent6.harness import _chain, _dag_focus, subrun

if TYPE_CHECKING:
    from agent6.harness import _conversation, _loop_state


@dataclasses.dataclass(frozen=True, slots=True)
class LaneJoin:
    """One lane's outcome in a `/parallel` dispatch.

    Attributes:
        session_id: The lane's session.
        branch: The lane's branch.
        status: "joined" (merged, `sha` set), "conflict" (imported, the merge conflicted, the
            branch exists locally) or "failed" (no importable branch).
        sha: The merge commit, or "".
        detail: The failure's reason, or "".
    """

    session_id: str
    branch: str
    status: Literal["joined", "conflict", "failed"]
    sha: str
    detail: str


def segment_lanes(
    seg: directive.Segment, pins: Sequence[str] = (), *, limit: int
) -> list[subrun.LaneTask]:
    """Expand one segment into its lanes, one model per lane.

    A bad spec (zero lanes, an empty model list, more than the limit) raises `DirectiveError`
    from `parse_spec`. Operator pins ride on every lane out of band: the spawner's `--pin`
    channel seeds the lane's own pin state, so a lane never lands work that violates a
    standing instruction, and the task text stays the lane's manifest task.

    Args:
        seg: The segment.
        pins: The operator's pins.
        limit: `[parallel].max_lanes`.

    Returns:
        The lanes; a model of None is the worker model.
    """
    lane_pins = tuple(pins)
    models = directive.parse_spec(seg.spec, limit=limit)
    return [subrun.LaneTask(task=seg.task, model=model, pins=lane_pins) for model in models]


def join_lane_result(
    root: pathlib.Path,
    res: subrun.LaneResult,
    *,
    ref: str,
    fallback_parent: str | None,
    identity: git_ops.CommitIdentity | None,
    also_branch: str | None,
) -> LaneJoin:
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
        "joined" with the sha, "conflict", or "failed" for a lane that imported nothing.
    """
    rid = res.spec.session_id
    if not res.ok:
        return LaneJoin(rid, res.branch, "failed", "", res.error)
    try:
        sha = git_ops.chain_merge(
            root,
            res.branch,
            f"merge {res.branch}",
            ref=ref,
            fallback_parent=fallback_parent,
            identity=identity,
            also_branch=also_branch,
        )
    except (git_ops.GitError, OSError) as exc:
        return LaneJoin(rid, res.branch, "failed", "", str(exc))
    if sha is None:
        return LaneJoin(rid, res.branch, "conflict", "", "merge conflict")
    return LaneJoin(rid, res.branch, "joined", sha, "")


def segment_stamp(lanes: list[LaneJoin]) -> tuple[graph_models.NodeStatus, str, str]:
    """Reduce one segment's lane joins to its DAG stamp.

    A segment passes when any lane joined, recording the last joined sha; the note names every
    lane. A conflict counts as not joined.

    Args:
        lanes: The segment's joins.

    Returns:
        The status, the note and the sha ("" when failed).
    """
    joined = [j for j in lanes if j.status == "joined"]
    note = "; ".join(lane_note(j) for j in lanes)
    if joined:
        return "passed", note, joined[-1].sha
    return "failed", note, ""


def lane_note(j: LaneJoin) -> str:
    """Return one lane's outcome as a DAG note line."""
    if j.status == "joined":
        return f"{j.session_id} joined at {j.sha[:12]}"
    if j.status == "conflict":
        return f"{j.session_id} conflicted; merge manually"
    return f"{j.session_id} failed: {j.detail}"


def summary_text(group: str, lanes: list[LaneJoin]) -> str:
    """Return the one user message summarizing every lane's outcome."""
    lines = [f"[parallel] group {group} complete ({len(lanes)} lane(s)):"]
    for j in lanes:
        if j.status == "joined":
            lines.append(f"  - {j.session_id} ({j.branch}): joined at {j.sha[:12]}")
        elif j.status == "conflict":
            # Git is agent6's in this run and `.git` is read-only in the jail: the operator merges.
            lines.append(
                f"  - {j.session_id} ({j.branch}): CONFLICT -- branch imported but the merge"
                f" conflicted. It exists locally for the operator (`git merge {j.branch}`);"
                " continue without it."
            )
        else:
            lines.append(f"  - {j.session_id} ({j.branch}): FAILED -- {j.detail}; nothing joined.")
    lines.append("Review what landed and continue.")
    return "\n".join(lines)


def spawn_lanes(
    spawner: subrun.GroupLaneSpawner, lanes: list[subrun.LaneTask], group: str, *, at: str | None
) -> list[subrun.LaneResult]:
    """Run the lanes through the spawner and check the result count.

    Args:
        spawner: The ui-side group spawner.
        lanes: The lanes to run.
        group: The group's name.
        at: The commit the lanes are cut from, or None for HEAD.

    Returns:
        One result per lane, in order.

    Raises:
        SubrunError: The spawner returned a result count other than the lane count.
    """
    results = spawner(lanes, group, at=at)
    if len(results) != len(lanes):
        raise subrun.SubrunError(
            f"group spawner returned {len(results)} result(s) for {len(lanes)} lane(s)"
        )
    return results


def parallel_parent_id(
    curator: graph_curator.GraphCurator | None, root_task_id: str | None
) -> str | None:
    """Return the parent for a dispatched subtask: the cursor's open node, else the run root."""
    if curator is None:
        return root_task_id
    return _dag_focus.current_task_id(curator.nodes(), curator.cursor()) or root_task_id


def add_parallel_node(
    curator: graph_curator.GraphCurator | None,
    task: str,
    parent_id: str | None,
    *,
    log: Callable[[str], None],
) -> str | None:
    """Add a steering-created DAG node for one dispatched task.

    Args:
        curator: The graph, or None.
        task: The task text; its first line is the title.
        parent_id: The parent node, or None.
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
                    rationale="dispatched via /parallel steering",
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
    lanes: list[LaneJoin],
    *,
    log: Callable[[str], None],
) -> None:
    """Stamp one segment's DAG node from its lanes' joins (`segment_stamp`)."""
    status, note, sha = segment_stamp(lanes)
    stamp_parallel_node(curator, node_id, status=status, note=note, sha=sha, log=log)


@dataclasses.dataclass(frozen=True, slots=True)
class ParallelDispatcher:
    """The coordinator's side of a `/parallel` group.

    Cuts lanes from the chain tip, runs them through the injected spawner, joins each branch
    back and tells the model once.

    Attributes:
        chain: The run's commit chain.
        curator: The graph, or None.
        max_lanes: `[parallel].max_lanes`.
        lane_spawner: The ui-side group spawner, or None where dispatch is unavailable.
        save_snapshot: The loop's resume-snapshot writer, called before the group blocks.
        log: The run's text logger.
        emit: The run's event sink.
        emit_graph_snapshot: Publishes the graph after a write.
    """

    chain: _chain.RunChain
    curator: graph_curator.GraphCurator | None
    max_lanes: int
    lane_spawner: subrun.GroupLaneSpawner | None
    save_snapshot: Callable[..., None]
    log: Callable[[str], None]
    emit: Callable[..., None]
    emit_graph_snapshot: Callable[[], None]

    def dispatch(
        self,
        conversation: _conversation.Conversation,
        iteration: int,
        state: _loop_state.LoopState,
        segments: list[directive.Segment],
    ) -> None:
        """Dispatch a `/parallel` sibling group at the steer boundary.

        Clones the coordinator's committed HEAD into one lane per expanded lane, runs them
        through the spawner, joins each branch back in dispatch order and injects one summary.
        Runs synchronously: no provider call happens while the group is in flight. Never ends
        the run: an unavailable spawner, a bad spec, a dirty tree, a spawner fault, a failed
        lane or a join conflict each answer the steer and continue.

        Args:
            conversation: The run's conversation.
            iteration: The iteration just completed.
            state: The loop state.
            segments: The directive's segments, one DAG node each.
        """
        if self.lane_spawner is None:
            self.feedback(
                conversation,
                "parallel dispatch is not available in this front-end; continuing normally.",
            )
            return
        try:
            # One DAG node per segment; its lanes join under it.
            lanes_cap = self.max_lanes
            per_segment = [segment_lanes(seg, state.pins, limit=lanes_cap) for seg in segments]
        except directive.DirectiveError as exc:
            self.feedback(conversation, f"bad /parallel spec: {exc}; nothing dispatched.")
            return
        lanes = [lane for seg_lanes in per_segment for lane in seg_lanes]
        # Lanes cut from the chain tip only: commit a changed tree first, or refuse stale work.
        if not self.ensure_clean(iteration):
            self.feedback(
                conversation,
                "refusing to dispatch: the working tree is not clean and could not be"
                " auto-committed. Commit or discard your changes, then retry /parallel.",
            )
            return

        state.parallel_groups_dispatched += 1
        group = f"p{state.parallel_groups_dispatched}"
        # Persisted before the group blocks: a crash there would reuse this group's id on resume.
        self.save_snapshot(state, conversation.to_wire(), next_iteration=iteration + 1)
        self.log(
            f"PARALLEL: dispatching group {group} "
            f"({len(lanes)} lane(s) across {len(segments)} task(s))"
        )
        # Lane ids exist only once the spawner names them; joined and failed carry the real ids.
        self.emit(
            "loop.parallel.dispatched",
            group=group,
            lanes=len(lanes),
            tasks=[seg.task[:200] for seg in segments],
        )
        parent_id = parallel_parent_id(self.curator, state.root_task_id)
        node_ids = [
            add_parallel_node(self.curator, seg.task, parent_id, log=self.log) for seg in segments
        ]
        if any(n is not None for n in node_ids):
            self.emit_graph_snapshot()

        try:
            # Blocks until the group returns; no provider call meanwhile.
            results = spawn_lanes(self.lane_spawner, lanes, group, at=self.chain.tip() or None)
        except Exception as exc:
            # The spawner is an injected ui-side callback; any fault it leaks answers the steer.
            # Everything after this point never raises: the join and the stamp catch their own.
            self.log(f"PARALLEL: group {group} dispatch failed: {exc}")
            for nid in node_ids:
                stamp_parallel_node(
                    self.curator,
                    nid,
                    status="failed",
                    note=f"dispatch failed: {exc}",
                    log=self.log,
                )
            self.emit_graph_snapshot()
            self.emit("loop.parallel.failed", group=group, error=str(exc))
            self.feedback(
                conversation,
                f"group {group} dispatch failed: {exc}. Nothing was joined; continuing normally.",
            )
            return

        # The spawner stamps the group name `<coordinator>-<group>` on each lane's manifest.
        # Read it back off a lane (`<group id>-l<n>`): it is the id `sessions compare` takes.
        group = results[0].spec.session_id.rsplit("-l", 1)[0] if results else group

        # Joins run in order, since a merge mutates the one workspace; then one stamp per segment.
        lanes = [
            join_lane_result(
                self.chain.root,
                res,
                ref=self.chain.ref or "",
                fallback_parent=self.chain.fallback_parent,
                identity=self.chain.identity,
                also_branch=self.chain.branch,
            )
            for res in results
        ]
        cursor = 0
        for nid, seg_lanes in zip(node_ids, per_segment, strict=True):
            width = len(seg_lanes)
            stamp_segment_node(self.curator, nid, lanes[cursor : cursor + width], log=self.log)
            cursor += width
        self.emit_graph_snapshot()

        payload = [
            {
                "session_id": j.session_id,
                "branch": j.branch,
                "status": j.status,
                "sha": j.sha,
                "detail": j.detail,
            }
            for j in lanes
        ]
        self.emit("loop.parallel.joined", group=group, lanes=payload)
        failures = [p for p, j in zip(payload, lanes, strict=True) if j.status != "joined"]
        if failures:
            self.emit("loop.parallel.failed", group=group, lanes=failures)
        conversation.notice(summary_text(group, lanes))

    def ensure_clean(self, iteration: int) -> bool:
        """Chain-commit a changed tree so lanes cut from the tip see current work.

        With `commit_per_step` off a changed tree cannot be captured, and dispatch is refused.

        Args:
            iteration: The iteration just completed, for the checkpoint subject.

        Returns:
            Whether the chain tip carries the worktree's content.
        """
        if not self.chain.dirty():
            return True
        if not self.chain.per_step:
            return False
        try:
            subject = f"checkpoint before /parallel dispatch (iter {iteration})"
            sha = self.chain.commit(subject)
            if sha:
                self.log(f"  pre-dispatch checkpoint: {sha[:12]}")
                self.emit("loop.auto_commit", iteration=iteration, sha=sha, subject=subject)
                self.emit(
                    "diff.updated",
                    sha=sha,
                    patch=git_ops.commit_diff(self.chain.root, sha, max_bytes=8000),
                )
        except (git_ops.GitError, OSError) as exc:
            self.log(f"PARALLEL: pre-dispatch checkpoint failed: {exc}")
        return not self.chain.dirty()

    def feedback(self, conversation: _conversation.Conversation, msg: str) -> None:
        """Answer a `/parallel` steer with a one-line notice and continue."""
        self.log(f"PARALLEL: {msg}")
        conversation.notice(f"[parallel] {msg}")
