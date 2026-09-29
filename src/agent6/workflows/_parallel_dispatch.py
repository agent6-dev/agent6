# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`/parallel` steer dispatch: `ParallelDispatcher` owns the policy (when to
cut lanes, the DAG stamps, the events, the injected group spawner) over the
Workflow-free pieces below: expanding a segment into lanes, joining one
returned lane's branch, and reducing lane outcomes to the DAG stamp and the
summary message the model continues with. Unit-testable without a Workflow.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import ValidationError

from agent6.directive import DirectiveError, Segment, parse_spec
from agent6.git_ops import CommitIdentity, GitError, chain_merge, commit_diff
from agent6.graph.curator import CuratorError, GraphCurator
from agent6.graph.models import (
    AddSubtaskIntent,
    NodeStatus,
    RecordCommitIntent,
    TaskNodeDraft,
    UpdateStatusIntent,
)
from agent6.workflows._chain import RunChain
from agent6.workflows._dag_focus import current_task_id
from agent6.workflows.subrun import GroupLaneSpawner, LaneResult, LaneTask, SubrunError

if TYPE_CHECKING:
    from agent6.workflows._conversation import Conversation
    from agent6.workflows._loop_state import LoopState


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


@dataclass(frozen=True, slots=True)
class ParallelDispatcher:
    """The coordinator's side of a `/parallel` group: cut lanes from the chain
    tip, run them through the injected spawner, join each branch back and
    tell the model once. `save_snapshot` is the loop's resume-snapshot
    writer, called before the group blocks."""

    chain: RunChain
    curator: GraphCurator | None
    max_lanes: int
    lane_spawner: GroupLaneSpawner | None
    save_snapshot: Callable[..., None]
    log: Callable[[str], None]
    emit: Callable[..., None]
    emit_graph_snapshot: Callable[[], None]

    def dispatch(
        self,
        conversation: Conversation,
        iteration: int,
        state: LoopState,
        segments: list[Segment],
    ) -> None:
        """Dispatch a `/parallel` sibling group at the steer boundary: clone the
        coordinator's committed HEAD into one isolated lane per expanded lane
        (a segment with spec=3 -> three lanes of that task; spec=m1,m2 -> one lane
        per model), run them via the injected group spawner, join each branch back
        in dispatch order, and inject ONE summary so the model continues informed.
        Runs synchronously -- no provider calls happen while the group is in
        flight, so the run's budget is untouched by the wait.

        Never ends the run: an unavailable spawner, a bad spec, a dirty tree it
        cannot auto-commit, a spawner fault, a failed lane, or a join conflict
        each answer the steer with a message and continue."""
        if self.lane_spawner is None:
            self.feedback(
                conversation,
                "parallel dispatch is not available in this front-end; continuing normally.",
            )
            return
        try:
            # One DAG node per SEGMENT (task); its lanes join under it.
            lanes_cap = self.max_lanes
            per_segment = [segment_lanes(seg, state.pins, limit=lanes_cap) for seg in segments]
        except DirectiveError as exc:
            self.feedback(conversation, f"bad /parallel spec: {exc}; nothing dispatched.")
            return
        lanes = [lane for seg_lanes in per_segment for lane in seg_lanes]
        # Lanes cut from the chain tip only: chain-commit a changed tree first,
        # and refuse (rather than dispatch stale work) if it will not come clean.
        if not self.ensure_clean(iteration):
            self.feedback(
                conversation,
                "refusing to dispatch: the working tree is not clean and could not be"
                " auto-committed. Commit or discard your changes, then retry /parallel.",
            )
            return

        state.parallel_groups_dispatched += 1
        group = f"p{state.parallel_groups_dispatched}"
        # Persist the bump BEFORE the group blocks. This runs inside the
        # operator boundary, which is after the iteration's snapshot and before
        # the next one, so the counter would otherwise live only in memory for
        # the entire group: a crash there would resume with the stale count
        # and the next /parallel would re-use this group's id, colliding with
        # its lane clones and branches.
        self.save_snapshot(
            system=state.system,
            messages=conversation.to_wire(),
            tool_calls=state.tool_calls,
            next_iteration=iteration + 1,
            root_task_id=state.root_task_id,
            state=state,
        )
        self.log(
            f"PARALLEL: dispatching group {group} "
            f"({len(lanes)} lane(s) across {len(segments)} task(s))"
        )
        # Lane ids do not exist until the spawner names them; the dispatched
        # event carries the truth it has (per-segment tasks + group), and
        # joined/failed name the real per-lane ids from each LaneResult.
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
            # Lanes cut from the run's chain tip, which ensure_clean
            # just made current; blocks, no provider calls meanwhile.
            results = self.lane_spawner(lanes, group, at=self.chain.tip() or None)
            if len(results) != len(lanes):
                raise SubrunError(
                    f"group spawner returned {len(results)} result(s) for {len(lanes)} lane(s)"
                )
        except Exception as exc:
            # The spawner is an injected ui-side callback (clones, thread pool,
            # detached spawns); any fault it leaks -- OSError, SubrunError, a
            # result-count mismatch -- must answer the steer, never abort the
            # run. Everything after this point is never-raising by construction
            # (join_lane_result and stamp_parallel_node catch their own faults).
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

        # The spawner names the group `<coordinator>-<group>` and stamps that on
        # every lane's manifest, so it is the id `sessions compare` takes. Read
        # it back off a lane (each is `<group id>-l<n>`, the derivation
        # `run_parallel` uses too) rather than printing the local counter, which
        # names no group on disk.
        group = results[0].spec.session_id.rsplit("-l", 1)[0] if results else group

        # Join every lane sequentially in dispatch order (a merge mutates the one
        # workspace, so joins can never run concurrently), then stamp one DAG node
        # per segment from its lanes' joins.
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
        """True when the chain tip carries the worktree's content, so lanes cut
        from it see current work. Changed content is chain-committed first;
        returns whether it came clean (with commit_per_step off, a changed
        tree cannot be captured and dispatch is refused)."""
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
                    patch=commit_diff(self.chain.root, sha, max_bytes=8000),
                )
        except (GitError, OSError) as exc:
            self.log(f"PARALLEL: pre-dispatch checkpoint failed: {exc}")
        return not self.chain.dirty()

    def feedback(self, conversation: Conversation, msg: str) -> None:
        """Answer a `/parallel` steer with a one-line notice and continue."""
        self.log(f"PARALLEL: {msg}")
        conversation.notice(f"[parallel] {msg}")
