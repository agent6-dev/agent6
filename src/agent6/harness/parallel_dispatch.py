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
import time
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import pydantic

from agent6 import budget as agent6_budget
from agent6 import directive, errors, lane_records
from agent6.graph import curator as graph_curator
from agent6.graph import models as graph_models
from agent6.harness import dag_focus, run_chain, subrun
from agent6.sessions import journal, lane_dispatch
from agent6.vcs import git_ops, git_read

if TYPE_CHECKING:
    from agent6.harness import loop_conversation, loop_state

# The ceilings of a lane the model dispatches: a smaller configured cap wins, unlimited cannot.
MODEL_LANE_MAX_TURNS = 200
MODEL_LANE_DEADLINE_S = 1800.0


def segment_lanes(
    seg: directive.Segment, pins: Sequence[str] = (), *, limit: int
) -> list[lane_records.LaneTask]:
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
    return [lane_records.LaneTask(task=seg.task, model=model, pins=lane_pins) for model in models]


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
    """Return the one user message summarizing every lane's outcome."""
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
    lines.append("Review what landed and continue.")
    return "\n".join(lines)


def join_of(child: lane_dispatch.ChildRecord) -> lane_records.LaneJoin:
    """Return a recorded lane's join; a lane never joined reads as failed."""
    join = child.join
    if join is None:
        return lane_records.LaneJoin(child.session_id, "", "failed", "", "not joined")
    return lane_records.LaneJoin(child.session_id, join.branch, join.status, join.sha, join.detail)


def spawn_lanes(
    spawner: lane_records.GroupLaneSpawner,
    lanes: list[lane_records.LaneTask],
    group: str,
    *,
    at: str | None,
) -> list[lane_records.LaneResult]:
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


@dataclasses.dataclass(slots=True)
class ParallelDispatcher:
    """Dispatch sibling lane groups for one run, each recorded so a resume can finish it.

    Attributes:
        chain: The run's commit chain.
        curator: The DAG, or None.
        max_lanes: `[parallel].max_lanes`.
        lane_spawner: The group spawner; None where dispatch is unavailable.
        save_snapshot: Writes the resume snapshot.
        log: The run's text logger.
        emit: The journal.
        emit_graph_snapshot: Journals the DAG after a change.
        session_id: The run's session id, the prefix of every lane's.
        execution_id: This execution's id, the owner of its grants and records.
        budget: The run's meter; None without one.
        session_dir: Where the dispatch records live; None for a harness without a run dir.
        lane_recoverer: Finds an interrupted dispatch's lanes on resume; None when none can.
        clock: The wall clock.
    """

    chain: run_chain.RunChain
    curator: graph_curator.GraphCurator | None
    max_lanes: int
    lane_spawner: lane_records.GroupLaneSpawner | None
    save_snapshot: Callable[..., None]
    log: Callable[[str], None]
    emit: Callable[..., None]
    emit_graph_snapshot: Callable[[], None]
    session_id: str = ""
    execution_id: str = ""
    budget: agent6_budget.BudgetTracker | None = None
    session_dir: pathlib.Path | None = None
    lane_recoverer: lane_records.LaneRecoverer | None = None
    clock: Callable[[], float] = time.time

    def dispatch(  # noqa: PLR0911  # one return per refusal
        self,
        conversation: loop_conversation.Conversation,
        turn: int,
        state: loop_state.LoopState,
        segments: list[directive.Segment],
        *,
        source: lane_records.DispatchSource = "operator",
        limits: lane_records.LaneLimits | None = None,
    ) -> lane_records.DispatchResult:
        """Dispatch a sibling group at the steer boundary and return what it did.

        Reserves the parent's allowance for the lanes, writes the dispatch record before any
        launch, clones the coordinator's committed HEAD into one lane per expanded lane, runs
        them through the spawner, settles their usage and joins each branch back in dispatch
        order, advancing the record at each boundary. Runs synchronously: no provider call
        happens while the group is in flight. Never ends the run: an unavailable spawner, a
        bad spec, a dirty tree, an unwritable record, a spawner fault, a failed lane or a
        join conflict each come back as the result for the caller to deliver.

        Args:
            conversation: The run's conversation, snapshotted before the group blocks.
            turn: The turn just completed.
            state: The loop state.
            segments: The directive's segments, one DAG node each.
            source: Who asked: the operator's steer, the model's tool call or a panel.
            limits: The turn cap and deadline every lane runs under; None for the config's.

        Returns:
            The result; `dispatched` False names a refusal before any clone.
        """
        if self.lane_spawner is None:
            return self.refusal(
                "parallel dispatch is not available in this front-end; continuing normally."
            )
        try:
            # One DAG node per segment; its lanes join under it.
            lanes_cap = self.max_lanes
            per_segment = [segment_lanes(seg, state.pins, limit=lanes_cap) for seg in segments]
        except directive.DirectiveError as exc:
            return self.refusal(f"bad /parallel spec: {exc}; nothing dispatched.")
        lanes = [lane for seg_lanes in per_segment for lane in seg_lanes]
        # The cap bounds the whole sibling group, repeated segments included, before any
        # checkpoint, DAG write or clone.
        if len(lanes) > lanes_cap:
            return self.refusal(
                f"the /parallel segments request {len(lanes)} lanes together but"
                f" [parallel].max_lanes = {lanes_cap}; nothing dispatched. Request fewer, or"
                " raise [parallel].max_lanes."
            )
        # Lanes cut from the chain tip only: commit a changed tree first, or refuse stale work.
        if not self.ensure_clean(turn):
            return self.refusal(
                "refusing to dispatch: the working tree is not clean and could not be"
                " auto-committed. Commit or discard your changes, then retry /parallel."
            )
        if self.session_dir is None:
            return self.refusal(
                "parallel dispatch needs a session dir for its record; nothing dispatched."
            )

        state.parallel_groups_dispatched += 1
        group = f"p{state.parallel_groups_dispatched}"
        identity = f"{self.session_id}-{group}"
        children = [f"{identity}-l{i}" for i in range(1, len(lanes) + 1)]
        grant = None
        if self.budget is not None:
            try:
                grant = self.budget.reserve(identity, children, owner=self.execution_id)
            except agent6_budget.BudgetExceededError as exc:
                state.parallel_groups_dispatched -= 1
                return self.refusal(f"the budget is exhausted ({exc}); nothing dispatched.")
        lanes = [
            dataclasses.replace(
                lane,
                allowance=grant.share_of(child) if grant is not None else None,
                limits=limits,
            )
            for lane, child in zip(lanes, children, strict=True)
        ]
        parent_id = parallel_parent_id(self.curator, state.root_task_id)
        node_ids = [
            add_parallel_node(self.curator, seg.task, parent_id, source=source, log=self.log)
            for seg in segments
        ]
        record = lane_dispatch.DispatchRecord(
            identity=identity,
            source=source,
            owner=self.execution_id,
            turn=turn,
            base=self.chain.tip() or "",
            segments=tuple(
                lane_dispatch.SegmentRecord(
                    task=seg.task, spec=seg.spec, lanes=len(seg_lanes), node_id=nid
                )
                for seg, seg_lanes, nid in zip(segments, per_segment, node_ids, strict=True)
            ),
            children=tuple(
                lane_dispatch.ChildRecord(
                    session_id=child,
                    lane=i,
                    task=lane.task,
                    model=lane.model,
                    share=lane.allowance,
                    limits=lane.limits,
                )
                for i, (child, lane) in enumerate(zip(children, lanes, strict=True), start=1)
            ),
        )
        try:
            # The record is the launch's precondition: nothing runs that a resume cannot find.
            self.write(record)
        except OSError as exc:
            if self.budget is not None:
                self.budget.release(identity)
            state.parallel_groups_dispatched -= 1
            self.fail_nodes(record, "dispatch record could not be written")
            return self.refusal(
                f"could not persist the dispatch record ({exc}); nothing dispatched."
            )
        if any(n is not None for n in node_ids):
            self.emit_graph_snapshot()
        # Persisted before the group blocks: a crash there would reuse this group's id on resume.
        self.save_snapshot(state, conversation.to_wire(), next_turn=turn + 1)
        self.log(
            f"PARALLEL: dispatching group {identity} "
            f"({len(lanes)} lane(s) across {len(segments)} task(s))"
        )
        self.emit(
            journal.LoopParallelDispatched(
                group=group,
                lanes=len(lanes),
                tasks=[seg.task[:200] for seg in segments],
                source=source,
            )
        )
        record = record.advanced("running")
        self.write(record)
        try:
            # Blocks until the group returns; no provider call meanwhile.
            results = spawn_lanes(self.lane_spawner, lanes, group, at=record.base or None)
        except Exception as exc:
            # The spawner is an injected ui-side callback; any fault it leaks answers the steer.
            # Everything after this point never raises: the join and the stamp catch their own.
            self.log(f"PARALLEL: group {identity} dispatch failed: {exc}")
            self.fail_nodes(record, f"dispatch failed: {exc}")
            self.emit(journal.LoopParallelFailed(group=identity, error=str(exc)))
            message = (
                f"group {identity} dispatch failed: {exc}. Nothing was joined; continuing normally."
            )
            self.log(f"PARALLEL: {message}")
            record = self.settled(record, {}).advanced("complete", message=f"[parallel] {message}")
            self.write(record)
            return lane_records.DispatchResult(
                identity, True, (), f"[parallel] {message}", base=record.base
            )
        return self.land(record, results)

    def land(
        self, record: lane_dispatch.DispatchRecord, results: list[lane_records.LaneResult]
    ) -> lane_records.DispatchResult:
        """Settle, join, stamp and complete a dispatch whose lanes have returned.

        Each boundary advances the record first, so a crash in between leaves a resume one
        step to finish, never one to repeat.

        Args:
            record: The dispatch's record.
            results: The lanes' results; a lane without one never started.

        Returns:
            The result, its message the record's.
        """
        by_id = {res.spec.session_id: res for res in results}
        record = self.settled(record, by_id)
        record = self.joined(record, by_id)
        joins = [join_of(child) for child in record.children]
        cursor = 0
        for seg in record.segments:
            stamp_segment_node(
                self.curator, seg.node_id, joins[cursor : cursor + seg.lanes], log=self.log
            )
            cursor += seg.lanes
        self.emit_graph_snapshot()
        payload = [
            {
                "session_id": j.session_id,
                "branch": j.branch,
                "status": j.status,
                "sha": j.sha,
                "detail": j.detail,
            }
            for j in joins
        ]
        self.emit(journal.LoopParallelJoined(group=record.identity, lanes=payload))
        failures = [
            p
            for p, j in zip(payload, joins, strict=True)
            if j.status not in ("joined", "no_changes")
        ]
        if failures:
            self.emit(journal.LoopParallelFailed(group=record.identity, lanes=failures))
        message = summary_text(record.identity, joins)
        record = record.advanced("complete", message=message)
        self.write(record)
        return lane_records.DispatchResult(
            record.identity, True, tuple(joins), message, base=record.base
        )

    def settled(
        self,
        record: lane_dispatch.DispatchRecord,
        by_id: dict[str, lane_records.LaneResult],
    ) -> lane_dispatch.DispatchRecord:
        """Charge each lane's usage once and release its share; unknown usage stays held.

        Args:
            record: The dispatch's record.
            by_id: The lanes' results by session id.

        Returns:
            The record in the `settling` state, each lane settled or outstanding.
        """
        record = record.advanced("settling")
        children: list[lane_dispatch.ChildRecord] = []
        for child in record.children:
            if child.settled:
                children.append(child)
                continue
            res = by_id.get(child.session_id)
            usage = res.usage if res is not None else None
            if self.budget is not None and child.share is not None:
                self.budget.settle_lane(record.identity, child.session_id, usage)
            children.append(
                child.model_copy(
                    update={
                        "usage": (
                            lane_dispatch.UsageRecord(
                                usd=usage.usd, usd_partial=usage.usd_partial, tokens=usage.tokens
                            )
                            if usage is not None
                            else None
                        ),
                        "settled": usage is not None,
                        "outstanding": usage is None,
                    }
                )
            )
        record = record.model_copy(update={"children": tuple(children)})
        self.write(record)
        self.emit_budget(record.turn)
        return record

    def joined(
        self,
        record: lane_dispatch.DispatchRecord,
        by_id: dict[str, lane_records.LaneResult],
    ) -> lane_dispatch.DispatchRecord:
        """Join each lane not yet joined, in lane order, recording each join as it lands.

        Args:
            record: The dispatch's record.
            by_id: The lanes' results by session id.

        Returns:
            The record in the `joining` state with every lane's join recorded.
        """
        record = record.advanced("joining")
        self.write(record)
        for child in record.children:
            if child.join is not None:
                continue
            before = self.chain.tip() or ""
            join = self.join_child(child, by_id.get(child.session_id))
            record = record.with_child(
                child.model_copy(
                    update={
                        "join": lane_dispatch.JoinRecord(
                            branch=join.branch,
                            status=join.status,
                            sha=join.sha,
                            detail=join.detail,
                            target_before=before,
                            target_after=self.chain.tip() or "",
                        )
                    }
                )
            )
            self.write(record)
        return record

    def join_child(
        self, child: lane_dispatch.ChildRecord, res: lane_records.LaneResult | None
    ) -> lane_records.LaneJoin:
        """Join one lane, or recognise a merge that already landed on the chain.

        Args:
            child: The lane's record.
            res: Its result, None when it never started.

        Returns:
            The join.
        """
        if res is None:
            branch = git_read.run_branch_for(child.session_id)
            return lane_records.LaneJoin(child.session_id, branch, "failed", "", "never started")
        ref = self.chain.ref or ""
        if (
            res.ok
            and res.has_commits
            and res.imported_tip
            and ref
            and git_read.is_ancestor(self.chain.root, res.imported_tip, ref)
        ):
            # Merged before the record advanced: recorded now, never merged twice.
            sha = git_read.merge_commit_for(self.chain.root, ref, res.imported_tip)
            return lane_records.LaneJoin(
                res.spec.session_id, res.branch, "joined", sha or (self.chain.tip() or ""), ""
            )
        return join_lane_result(
            self.chain.root,
            res,
            ref=ref,
            fallback_parent=self.chain.fallback_parent,
            identity=self.chain.identity,
            also_branch=self.chain.branch,
        )

    def reconcile(
        self, state: loop_state.LoopState, conversation: loop_conversation.Conversation
    ) -> None:
        """Finish or diagnose every dispatch the prior execution left behind, once each.

        Runs before a resumed execution's first provider call. A complete record whose
        result never reached the conversation is delivered; an interrupted one has its
        lanes found by the recoverer (live ones stopped, finished ones imported), its grant
        restored as recorded, its usage settled and its joins landed exactly once, and its
        result delivered for the original call.

        Args:
            state: The loop state.
            conversation: The conversation the result enters.
        """
        if self.session_dir is None:
            return
        try:
            records = lane_dispatch.read_records(self.session_dir)
        except lane_dispatch.DispatchRecordError as exc:
            self.log(f"PARALLEL: {exc}")
            conversation.notice(f"[parallel] a dispatch record could not be read: {exc}")
            return
        for found in records:
            record = found
            if record.state == "complete":
                if not record.delivered:
                    self.deliver(
                        conversation,
                        lane_records.DispatchResult(
                            record.identity, True, (), record.message, base=record.base
                        ),
                    )
                continue
            self.log(
                f"PARALLEL: reconciling the interrupted dispatch {record.identity} ({record.state})"
            )
            shares = [(c.session_id, c.share) for c in record.children if c.share is not None]
            if self.budget is not None and shares:
                self.budget.restore(
                    record.identity,
                    shares,
                    owner=record.owner,
                    settled=frozenset(c.session_id for c in record.children if c.settled),
                )
            children = [c.session_id for c in record.children]
            results = (
                self.lane_recoverer(record.identity, children)
                if self.lane_recoverer is not None
                else []
            )
            if record.state in ("prepared", "running") and not results:
                if self.budget is not None:
                    self.budget.release(record.identity)
                message = (
                    f"[parallel] group {record.identity}: dispatch interrupted before launch;"
                    " nothing ran."
                )
                self.fail_nodes(record, "dispatch interrupted before launch")
                record = record.advanced("complete", message=message)
                self.write(record)
                self.deliver(
                    conversation,
                    lane_records.DispatchResult(
                        record.identity, True, (), message, base=record.base
                    ),
                )
                continue
            self.deliver(conversation, self.land(record, results))

    def deliver(
        self, conversation: loop_conversation.Conversation, result: lane_records.DispatchResult
    ) -> None:
        """Put the result's message into the conversation and record that it arrived.

        Args:
            conversation: The run's conversation.
            result: The dispatch's result, or a refusal.
        """
        conversation.notice(result.message)
        self.record_delivery(result)

    def record_delivery(self, result: lane_records.DispatchResult) -> None:
        """Mark the dispatch's record delivered, whatever surface carried its result.

        Args:
            result: The dispatch's result, or a refusal, which has no record.
        """
        if not result.dispatched or self.session_dir is None:
            return
        path = lane_dispatch.record_path(self.session_dir, result.group)
        if not path.is_file():
            return
        record = lane_dispatch.read_record(path)
        if not record.delivered:
            self.write(record.model_copy(update={"delivered": True}))

    def write(self, record: lane_dispatch.DispatchRecord) -> None:
        """Write the record atomically under the session dir.

        Args:
            record: The record.

        Raises:
            OSError: The write failed, or there is no session dir.
        """
        if self.session_dir is None:
            raise OSError("no session dir to record the dispatch in")
        lane_dispatch.write_record(self.session_dir, record, now=self.clock())

    def fail_nodes(self, record: lane_dispatch.DispatchRecord, note: str) -> None:
        """Stamp every segment's DAG node failed with the note."""
        for seg in record.segments:
            stamp_parallel_node(self.curator, seg.node_id, status="failed", note=note, log=self.log)
        self.emit_graph_snapshot()

    def emit_budget(self, turn: int) -> None:
        """Journal the meter after a settlement, so the lanes' charge is on the record."""
        if self.budget is None:
            return
        snap = self.budget.snapshot()
        cost, _ = self.budget.estimate_usd()
        self.emit(
            journal.LoopBudget(
                turn=turn,
                input_tokens=snap.input_total,
                output_tokens=snap.output_total,
                cache_read_tokens=snap.cache_read_total,
                cost_usd=round(cost, 6),
            )
        )

    def ensure_clean(self, turn: int) -> bool:
        """Chain-commit a changed tree so lanes cut from the tip see current work.

        With `commit_per_turn` off a changed tree cannot be captured, and dispatch is refused.

        Args:
            turn: The turn just completed, for the checkpoint subject.

        Returns:
            Whether the chain tip carries the worktree's content.
        """
        if not self.chain.dirty():
            return True
        if not self.chain.per_turn:
            return False
        try:
            subject = f"checkpoint before /parallel dispatch (turn {turn})"
            sha = self.chain.commit(subject)
            if sha:
                self.log(f"  pre-dispatch checkpoint: {sha[:12]}")
                self.emit(journal.LoopAutoCommit(turn=turn, sha=sha, subject=subject))
                self.emit(
                    journal.DiffUpdated(
                        sha=sha,
                        patch=git_read.commit_diff(self.chain.root, sha, max_bytes=8000),
                    )
                )
        except (errors.GitError, OSError) as exc:
            self.log(f"PARALLEL: pre-dispatch checkpoint failed: {exc}")
        return not self.chain.dirty()

    def refusal(self, msg: str) -> lane_records.DispatchResult:
        """Return the result of a dispatch refused before any lane ran, logged once."""
        self.log(f"PARALLEL: {msg}")
        return lane_records.DispatchResult("", False, (), f"[parallel] {msg}")
