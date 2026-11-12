# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Dispatch a `/parallel` steer.

`ParallelDispatcher` owns the policy: when to cut children, the DAG stamps, the events, the
injected group spawner. `parallel_join` projects child results into ordered merges, DAG
stamps and summaries; record-only panel batches share the executor without merging, and a
rank join lands the one child its injected ranker puts first.
"""

from __future__ import annotations

import dataclasses
import pathlib
import time
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from agent6 import budget as agent6_budget
from agent6 import child_records, directive, errors
from agent6.graph import curator as graph_curator
from agent6.harness import child_repo, parallel_join, run_chain
from agent6.sessions import budget_events, child_dispatch, journal
from agent6.vcs.git import read as git_read
from agent6.viewmodel.session import log as session_log

if TYPE_CHECKING:
    from agent6.harness import loop_conversation, loop_state

# The ceilings of a child the model dispatches: a smaller configured cap wins, unlimited cannot.
MODEL_CHILD_MAX_TURNS = 200
MODEL_CHILD_DEADLINE_S = 1800.0


def segment_children(
    seg: directive.Segment, pins: Sequence[str] = (), *, limit: int
) -> list[child_records.ChildTask]:
    """Expand one segment into its children, one model per child.

    A bad spec (zero children, an empty model list, more than the limit) raises `DirectiveError`
    from `parse_spec`. Operator pins ride on every child out of band: the spawner's `--pin`
    channel seeds the child's own pin state, so a child never lands work that violates a
    standing instruction, and the task text stays the child's manifest task.

    Args:
        seg: The segment.
        pins: The operator's pins.
        limit: `[harness.parallel].max_children`.

    Returns:
        The children; a model of None is the worker model.
    """
    child_pins = tuple(pins)
    models = directive.parse_spec(seg.spec, limit=limit)
    return [
        child_records.ChildTask(task=seg.task, model=model, pins=child_pins) for model in models
    ]


@dataclasses.dataclass(slots=True)
class ParallelDispatcher:
    """Dispatch sibling child groups for one run, each recorded so a resume can finish it.

    Attributes:
        chain: The run's commit chain.
        curator: The DAG, or None.
        max_children: `[harness.parallel].max_children`.
        child_spawner: The group spawner; None where dispatch is unavailable.
        save_snapshot: Writes the resume snapshot.
        log: The run's text logger.
        emit: The journal.
        emit_graph_snapshot: Journals the DAG after a change.
        session_id: The run's session id, the prefix of every child's.
        execution_id: This execution's id, the owner of its grants and records.
        budget: The run's meter; None without one.
        session_dir: Where the dispatch records live; None for a harness without a session dir.
        child_recoverer: Finds an interrupted dispatch's children on resume; None when none can.
        ranker: Orders a rank join's children; None where no dispatch joins by rank.
        stop_pending: Whether the operator asked the run to stop.
        authority: Work and harness-panel permissions from the application.
        clock: The wall clock.
    """

    chain: run_chain.RunChain
    curator: graph_curator.GraphCurator | None
    max_children: int
    child_spawner: child_records.GroupChildSpawner | None
    save_snapshot: Callable[..., None]
    log: Callable[[str], None]
    emit: Callable[..., None]
    emit_graph_snapshot: Callable[[], None]
    session_id: str = ""
    execution_id: str = ""
    budget: agent6_budget.BudgetTracker | None = None
    session_dir: pathlib.Path | None = None
    child_recoverer: child_records.ChildRecoverer | None = None
    ranker: child_records.ChildRanker | None = None
    stop_pending: Callable[[], bool] = lambda: False
    authority: child_records.DispatchAuthority = dataclasses.field(
        default_factory=child_records.DispatchAuthority
    )
    clock: Callable[[], float] = time.time

    def dispatch(  # noqa: PLR0911  # one return per refusal
        self,
        conversation: loop_conversation.Conversation,
        turn: int,
        state: loop_state.LoopState,
        segments: list[directive.Segment],
        *,
        source: child_records.DispatchSource = "operator",
        limits: child_records.ChildLimits | None = None,
        join: child_records.JoinPolicy = "merge",
    ) -> child_records.DispatchResult:
        """Dispatch a sibling group at the steer boundary and return what it did.

        Reserves the parent's allowance for the children, writes the dispatch record before any
        launch, clones the parent's committed HEAD into one child per expanded child, runs
        them through the spawner, settles their usage and joins each branch back in dispatch
        order (under `rank`, the best-ranked child alone), advancing the record at each
        boundary. Runs synchronously: no provider call happens while the group is in flight.
        Never ends the run: an unavailable spawner, a bad spec, a dirty tree, an unwritable
        record, a spawner fault, a failed child or a join conflict each come back as the
        result for the caller to deliver.

        Args:
            conversation: The run's conversation, snapshotted before the group blocks.
            turn: The turn just completed.
            state: The loop state.
            segments: The directive's segments, one DAG node each.
            source: Who asked: the operator's steer, the model's tool call or a panel.
            limits: The turn cap and deadline every child runs under; None for the config's.
            join: How the children land: each merged in order, or the best-ranked one alone.

        Returns:
            The result; `dispatched` False names a refusal before any clone.
        """
        if source == "panel" or not self.authority.work:
            return self.refusal("work dispatch is unavailable in a child; nothing dispatched.")
        if join == "rank" and self.ranker is None:
            return self.refusal("this execution has no ranker for a rank join; nothing dispatched.")
        if self.child_spawner is None:
            return self.refusal(
                "parallel dispatch is not available in this front-end; continuing normally."
            )
        try:
            # One DAG node per segment; its children join under it.
            children_cap = self.max_children
            per_segment = [
                segment_children(seg, state.pins, limit=children_cap) for seg in segments
            ]
        except directive.DirectiveError as exc:
            return self.refusal(f"bad /parallel spec: {exc}; nothing dispatched.")
        children = [child for seg_children in per_segment for child in seg_children]
        # The cap bounds the whole sibling group, repeated segments included, before any
        # checkpoint, DAG write or clone.
        if len(children) > children_cap:
            return self.refusal(
                f"the /parallel segments request {len(children)} children together but"
                f" [harness.parallel].max_children = {children_cap}; nothing dispatched. Request"
                " fewer, or raise [harness.parallel].max_children."
            )
        # Children cut from the chain tip only: commit a changed tree first, or refuse stale work.
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
        child_ids = [f"{identity}-l{i}" for i in range(1, len(children) + 1)]
        if self.budget is not None and self.budget.is_exhausted():
            state.parallel_groups_dispatched -= 1
            reason = self.budget.snapshot().exhausted_reason
            return self.refusal(f"the budget is exhausted ({reason}); nothing dispatched.")
        children = [dataclasses.replace(child, limits=limits) for child in children]
        parent_id = parallel_join.parallel_parent_id(self.curator, state.root_task_id)
        node_ids = [
            parallel_join.add_parallel_node(
                self.curator, seg.task, parent_id, source=source, log=self.log
            )
            for seg in segments
        ]
        record = child_dispatch.DispatchRecord(
            identity=identity,
            source=source,
            join_policy=join,
            owner=self.execution_id,
            turn=turn,
            base=self.chain.tip() or "",
            segments=tuple(
                child_dispatch.SegmentRecord(
                    task=seg.task, spec=seg.spec, children=len(seg_children), node_id=nid
                )
                for seg, seg_children, nid in zip(segments, per_segment, node_ids, strict=True)
            ),
            children=tuple(
                child_dispatch.ChildRecord(
                    session_id=child_id,
                    number=i,
                    task=child.task,
                    model=child.model,
                    limits=child.limits,
                )
                for i, (child_id, child) in enumerate(
                    zip(child_ids, children, strict=True), start=1
                )
            ),
        )
        try:
            # The record is the launch's precondition: nothing runs that a resume cannot find.
            self.write(record)
        except OSError as exc:
            state.parallel_groups_dispatched -= 1
            self.fail_nodes(record, "dispatch record could not be written")
            return self.refusal(
                f"could not persist the dispatch record ({exc}); nothing dispatched."
            )
        if any(n is not None for n in node_ids):
            self.emit_graph_snapshot()
        # Persisted before the group blocks: a crash there would reuse this group's id on resume.
        self.save_snapshot(state, conversation.to_wire(), next_turn=turn + 1)
        return self.execute(record, children, group)

    def dispatch_panel(  # noqa: PLR0911  # one return per refusal
        self,
        children: list[child_records.ChildTask],
        *,
        panel_id: str,
        at: str,
        turn: int = 0,
        concurrency: int | None = None,
    ) -> child_records.DispatchResult:
        """Reserve a harness-owned roster and submit bounded batches through the child executor.

        The retained target is supplied by the harness; no checkpoint or merge changes it.
        The review policy selects reviewers and concurrency; this executor bounds each batch.

        Args:
            children: The complete roster of bounded conversations with resolved reviewer settings.
            panel_id: The harness's stable panel/batch id.
            at: The retained immutable target commit.
            turn: The owning worker's completed turn.
            concurrency: The policy's concurrency cap; None uses harness.parallel.max_children.

        Returns:
            The record-only joins, or a refusal before launch.
        """
        if not self.authority.panels or self.child_spawner is None:
            return self.refusal("harness panel scheduling is unavailable in this execution.")
        if (
            self.session_dir is None
            or not at
            or not panel_id
            or not panel_id.replace("-", "").isalnum()
        ):
            return self.refusal("a panel needs a session dir, retained target and safe panel id.")
        if not children or any(child.contract is None for child in children):
            return self.refusal("a panel dispatches reviewers only.")
        group = f"r{panel_id}"
        identity = f"{self.session_id}-{group}"
        if child_dispatch.record_path(self.session_dir, identity).exists():
            return self.refusal(
                "the panel batch already has a dispatch record; reconcile it instead."
            )
        width = max(1, min(self.max_children, concurrency or self.max_children))
        child_ids = [
            f"{identity}-l{i + 1}"
            if len(children) <= width
            else f"{identity}-b{i // width + 1}-l{i % width + 1}"
            for i in range(len(children))
        ]
        if self.budget is not None and self.budget.is_exhausted():
            reason = self.budget.snapshot().exhausted_reason
            return self.refusal(f"the panel budget is exhausted ({reason}); nothing dispatched.")
        bounded: list[child_records.ChildTask] = []
        start = self.clock()
        for i, child in enumerate(children):
            assert child.contract is not None
            # Batches run one after another: each gets its own window, fixed before launch.
            deadlines = [start + MODEL_CHILD_DEADLINE_S * (i // width + 1)]
            if self.authority.deadline_at is not None:
                deadlines.append(self.authority.deadline_at)
            if child.limits is not None and child.limits.deadline_at is not None:
                deadlines.append(child.limits.deadline_at)
            bounded.append(
                dataclasses.replace(
                    child,
                    limits=child_records.ChildLimits(
                        max_turns=child.contract.max_turns, deadline_at=min(deadlines)
                    ),
                )
            )
        record = child_dispatch.DispatchRecord(
            identity=identity,
            source="panel",
            join_policy="record_only",
            owner=self.execution_id,
            turn=turn,
            base=at,
            segments=(),
            children=tuple(
                child_dispatch.ChildRecord(
                    session_id=child_id,
                    number=i,
                    task=child.task,
                    model=child.model,
                    limits=child.limits,
                )
                for i, (child_id, child) in enumerate(zip(child_ids, bounded, strict=True), start=1)
            ),
        )
        try:
            self.write(record)
        except OSError as exc:
            return self.refusal(
                f"could not persist the panel dispatch ({exc}); nothing dispatched."
            )
        return self.execute(record, bounded, group, batch_width=width)

    def execute(
        self,
        record: child_dispatch.DispatchRecord,
        children: list[child_records.ChildTask],
        group: str,
        *,
        batch_width: int | None = None,
    ) -> child_records.DispatchResult:
        """Run a published work or panel batch, then settle and land it under its recorded policy.

        Args:
            record: The durable prepared dispatch.
            children: Its bounded tasks, in child order.
            group: The group suffix used by the shared spawner.
            batch_width: The maximum children submitted together; None submits the whole group.

        Returns:
            The completed result, or a diagnosed spawner failure with unresolved holds.
        """  # noqa: DOC501  # the spawner's ChildRepoError is caught below
        assert self.child_spawner is not None
        identity = record.identity
        what = "reviewer(s)" if record.source == "panel" else "child(ren)"
        self.log(f"PARALLEL: dispatching group {identity} ({len(children)} {what})")
        self.emit(
            journal.ParallelDispatched(
                group=group,
                children=len(children),
                tasks=[child.task[:200] for child in children],
                source=record.source,
            )
        )
        record = record.advanced("running")
        self.write(record)
        try:
            width = batch_width or len(children)
            results: list[child_records.ChildResult] = []
            for offset in range(0, len(children), width):
                # Each batch starts under what the run has left then; its children share it.
                ceiling = self.budget.available() if self.budget is not None else None
                batch = [
                    dataclasses.replace(child, allowance=ceiling)
                    for child in children[offset : offset + width]
                ]
                batch_group = group if len(children) <= width else f"{group}-b{offset // width + 1}"
                results.extend(self.child_spawner(batch, batch_group, at=record.base or None))
            if len(results) != len(children):
                raise child_repo.ChildRepoError(
                    f"group spawner returned {len(results)} result(s)"
                    f" for {len(children)} child(ren)"
                )
        except Exception as exc:
            # The front-end callback may fail after launching: the children's usage is unknown.
            self.log(f"PARALLEL: group {identity} dispatch failed: {exc}")
            self.fail_nodes(record, f"dispatch failed: {exc}")
            self.emit(journal.ParallelFailed(group=identity, error=str(exc)))
            message = f"[parallel] group {identity} dispatch failed: {exc}. Nothing was joined."
            record = self.settled(record, {}).advanced("complete", message=message)
            self.write(record)
            return child_records.DispatchResult(identity, True, (), message, base=record.base)
        return self.land(record, results)

    def land(
        self, record: child_dispatch.DispatchRecord, results: list[child_records.ChildResult]
    ) -> child_records.DispatchResult:
        """Settle, join, stamp and complete a dispatch whose children have returned.

        Each boundary advances the record first, so a crash in between leaves a resume one
        step to finish, never one to repeat.

        Args:
            record: The dispatch's record.
            results: The children's results; a child without one never started.

        Returns:
            The result, its message the record's.
        """
        by_id = {res.spec.session_id: res for res in results}
        record = self.settled(record, by_id)
        if record.join_policy == "rank" and record.ranking is None and self.stop_pending():
            # No judge is asked after a stop: the record stays here and a resume ranks from it.
            self.log(f"PARALLEL: group {record.identity} stopped before its ranking")
            return child_records.DispatchResult(record.identity, True, (), "", base=record.base)
        record = self.joined(record, by_id)
        joins = [parallel_join.join_of(child) for child in record.children]
        cursor = 0
        for seg in record.segments:
            parallel_join.stamp_segment_node(
                self.curator, seg.node_id, joins[cursor : cursor + seg.children], log=self.log
            )
            cursor += seg.children
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
        self.emit(
            journal.ParallelJoined(group=record.identity, children=payload, source=record.source)
        )
        failures = [
            p
            for p, j in zip(payload, joins, strict=True)
            if j.status not in ("joined", "no_changes", "ranked")
        ]
        if failures:
            self.emit(journal.ParallelFailed(group=record.identity, children=failures))
        message = parallel_join.summary_text(record.identity, joins)
        record = record.advanced("complete", message=message)
        self.write(record)
        return child_records.DispatchResult(
            record.identity, True, tuple(joins), message, base=record.base
        )

    def settled(
        self,
        record: child_dispatch.DispatchRecord,
        by_id: dict[str, child_records.ChildResult],
    ) -> child_dispatch.DispatchRecord:
        """Charge each child's usage once; a child whose usage is unknown stays unsettled.

        Args:
            record: The dispatch's record.
            by_id: The children's results by session id.

        Returns:
            The record in the `settling` state.
        """
        record = record.advanced("settling")
        children: list[child_dispatch.ChildRecord] = []
        for child in record.children:
            if child.settled:
                children.append(child)
                continue
            res = by_id.get(child.session_id)
            usage = res.usage if res is not None else None
            if self.budget is not None:
                self.budget.settle_child(child.session_id, usage)
            children.append(
                child.model_copy(
                    update={
                        "usage": (
                            child_dispatch.UsageRecord(
                                usd=usage.usd, usd_partial=usage.usd_partial, tokens=usage.tokens
                            )
                            if usage is not None
                            else None
                        ),
                        "settled": usage is not None,
                    }
                )
            )
        record = record.model_copy(update={"children": tuple(children)})
        self.emit_budget(record.turn)
        self.write(record)
        return record

    def joined(
        self,
        record: child_dispatch.DispatchRecord,
        by_id: dict[str, child_records.ChildResult],
    ) -> child_dispatch.DispatchRecord:
        """Join each child not yet joined, in child order, recording each join as it lands.

        A rank join ranks first, once, and joins its winner alone; with no winner it joins none.

        Args:
            record: The dispatch's record.
            by_id: The children's results by session id.

        Returns:
            The record in the `joining` state with every child's join recorded.
        """
        record = record.advanced("joining")
        self.write(record)
        if record.join_policy == "rank" and record.ranking is None:
            record = self.ranked(record, by_id)
        order = record.ranking.order if record.ranking is not None else ()
        winner = record.ranking.winner if record.ranking is not None else ""
        for child in record.children:
            if child.join is not None:
                continue
            before = self.chain.tip() or ""
            res = by_id.get(child.session_id)
            if record.join_policy == "record_only":
                join = child_records.ChildJoin(
                    child.session_id,
                    "",
                    "no_changes" if res is not None and res.ok else "failed",
                    "",
                    (res.error or res.stopped_by) if res is not None else "no imported record",
                )
            elif (
                res is not None
                and child.session_id in order
                # The winner joins where the run has a chain; under `git.control = "model"`
                # it has none, and every child is placed and left on its ref.
                and (child.session_id != winner or not self.chain.ref)
            ):
                # Its branch stays imported: the operator can still merge it by hand.
                place = f"{order.index(child.session_id) + 1} of {len(order)}"
                held = res.imported_ref or res.branch
                join = child_records.ChildJoin(child.session_id, held, "ranked", "", place)
            else:
                join = self.join_child(child, res)
            record = record.with_child(
                child.model_copy(
                    update={
                        "join": child_dispatch.JoinRecord(
                            branch=join.branch,
                            status=join.status,
                            sha=join.sha,
                            detail=join.detail,
                            target_before=before,
                            target_after=self.chain.tip() or "",
                        ),
                        "report": res.report if res is not None else "",
                    }
                )
            )
            self.write(record)
        return record

    def ranked(
        self,
        record: child_dispatch.DispatchRecord,
        by_id: dict[str, child_records.ChildResult],
    ) -> child_dispatch.DispatchRecord:
        """Rank a rank join's children and write the order before any join.

        Args:
            record: The dispatch's record.
            by_id: The children's results by session id.

        Returns:
            The record with its ranking: the children that ended with a result, best first.
        """
        assert self.ranker is not None
        rankable = [
            res
            for child in record.children
            if (res := by_id.get(child.session_id)) is not None and res.ok
        ]
        ranking = (
            self.ranker(rankable, group=record.identity, base=record.base)
            if rankable
            else child_records.Ranking(order=())
        )
        ids = [res.spec.session_id for res in rankable]
        # Every rankable child has a place, whatever the ranker returned.
        order = [rid for rid in dict.fromkeys(ranking.order) if rid in ids]
        order += [rid for rid in ids if rid not in order]
        record = record.model_copy(
            update={
                "ranking": child_dispatch.RankRecord(
                    order=tuple(order),
                    ranked_by=ranking.ranked_by,
                    rationale=ranking.rationale,
                    winner=ranking.winner if ranking.winner in order else "",
                )
            }
        )
        # The judges' spend is on the meter now.
        self.emit_budget(record.turn)
        self.write(record)
        return record

    def join_child(
        self, child: child_dispatch.ChildRecord, res: child_records.ChildResult | None
    ) -> child_records.ChildJoin:
        """Join one child, or recognise a merge that already landed on the chain.

        Args:
            child: The child's record.
            res: Its result, None when it never started.

        Returns:
            The join.
        """
        if res is None:
            branch = git_read.run_branch_for(child.session_id)
            return child_records.ChildJoin(child.session_id, branch, "failed", "", "never started")
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
            return child_records.ChildJoin(
                res.spec.session_id,
                res.branch,
                "joined",
                sha or (self.chain.tip() or ""),
                res.stopped_by,
            )
        return parallel_join.join_child_result(
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
        children found by the recoverer (live ones stopped, finished ones imported), its usage
        settled and its joins landed exactly once, and its result delivered for the
        original call.

        Args:
            state: The loop state.
            conversation: The conversation the result enters.
        """
        if self.session_dir is None:
            return
        try:
            records = child_dispatch.read_records(self.session_dir)
        except child_dispatch.DispatchRecordError as exc:
            self.log(f"PARALLEL: {exc}")
            conversation.notice(f"[parallel] a dispatch record could not be read: {exc}")
            return
        for found in records:
            record = found
            scan = session_log.scan_session_log(self.session_dir / "logs.jsonl")
            accounted = scan.budget.session.settled_children
            if record.source == "panel":
                record = record.model_copy(
                    update={
                        "children": tuple(
                            child.model_copy(update={"settled": False})
                            if child.settled and child.session_id not in accounted
                            else child
                            for child in record.children
                        )
                    }
                )
            if record.state == "complete" and all(c.settled for c in record.children):
                if not record.delivered:
                    self.deliver(
                        conversation,
                        child_records.DispatchResult(
                            record.identity, True, (), record.message, base=record.base
                        ),
                    )
                continue
            self.log(
                f"PARALLEL: reconciling the interrupted dispatch {record.identity} ({record.state})"
            )
            children = [c.session_id for c in record.children]
            if self.budget is not None:
                # The journal already holds these children's usage: charged once, by that execution.
                self.budget.mark_settled(c for c in children if c in accounted)
            results = (
                self.child_recoverer(record.identity, children)
                if self.child_recoverer is not None
                else []
            )
            if (
                record.state in ("prepared", "running")
                and self.child_recoverer is not None
                and not results
            ):
                if self.budget is not None:
                    # Nothing ran, so nothing was used; the journal names each child accounted.
                    nothing = agent6_budget.UsageSettlement(usd=0, usd_partial=False, tokens=0)
                    for child in children:
                        self.budget.settle_child(child, nothing)
                message = (
                    f"[parallel] group {record.identity}: dispatch interrupted before launch;"
                    " nothing ran."
                )
                self.fail_nodes(record, "dispatch interrupted before launch")
                record = record.model_copy(
                    update={
                        "children": tuple(
                            child.model_copy(
                                update={
                                    "settled": True,
                                    "usage": child_dispatch.UsageRecord(usd=0.0),
                                }
                            )
                            for child in record.children
                        ),
                    }
                ).advanced("complete", message=message)
                self.emit_budget(record.turn)
                self.write(record)
                self.deliver(
                    conversation,
                    child_records.DispatchResult(
                        record.identity, True, (), message, base=record.base
                    ),
                )
                continue
            self.deliver(conversation, self.land(record, results))

    def deliver(
        self, conversation: loop_conversation.Conversation, result: child_records.DispatchResult
    ) -> None:
        """Put the result's message into the conversation and record that it arrived.

        Args:
            conversation: The run's conversation.
            result: The dispatch's result, or a refusal.
        """
        conversation.notice(result.message)
        self.record_delivery(result)

    def record_delivery(self, result: child_records.DispatchResult) -> None:
        """Mark the dispatch's record delivered, whatever surface carried its result.

        Args:
            result: The dispatch's result, or a refusal, which has no record.
        """
        if not result.dispatched or self.session_dir is None:
            return
        path = child_dispatch.record_path(self.session_dir, result.group)
        if not path.is_file():
            return
        record = child_dispatch.read_record(path)
        if not record.delivered:
            self.write(record.model_copy(update={"delivered": True}))

    def write(self, record: child_dispatch.DispatchRecord) -> None:
        """Write the record atomically under the session dir.

        Args:
            record: The record.

        Raises:
            OSError: The write failed, or there is no session dir.
        """
        if self.session_dir is None:
            raise OSError("no session dir to record the dispatch in")
        child_dispatch.write_record(self.session_dir, record, now=self.clock())

    def fail_nodes(self, record: child_dispatch.DispatchRecord, note: str) -> None:
        """Stamp every segment's DAG node failed with the note."""
        for seg in record.segments:
            parallel_join.stamp_parallel_node(
                self.curator, seg.node_id, status="failed", note=note, log=self.log
            )
        self.emit_graph_snapshot()

    def emit_budget(self, turn: int) -> None:
        """Journal the meter after a settlement, so the children's charge is on the record."""
        if self.budget is None:
            return
        self.emit(journal.BudgetUpdate(**budget_events.budget_update_fields(self.budget)))
        self.emit(budget_events.budget_turn(self.budget, turn))

    def ensure_clean(self, turn: int) -> bool:
        """Chain-commit a changed tree so children cut from the tip see current work.

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
                self.emit(journal.CheckpointCommitted(turn=turn, sha=sha, subject=subject))
                self.emit(
                    journal.DiffUpdated(
                        sha=sha,
                        patch=git_read.commit_diff(self.chain.root, sha, max_bytes=8000),
                    )
                )
        except (errors.GitError, OSError) as exc:
            self.log(f"PARALLEL: pre-dispatch checkpoint failed: {exc}")
        return not self.chain.dirty()

    def refusal(self, msg: str) -> child_records.DispatchResult:
        """Return the result of a dispatch refused before any child ran, logged once."""
        self.log(f"PARALLEL: {msg}")
        return child_records.DispatchResult("", False, (), f"[parallel] {msg}")
