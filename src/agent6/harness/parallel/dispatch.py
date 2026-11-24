# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Dispatch a run's children and merge their work.

`ParallelDispatcher` owns the policy: when to cut children, the DAG stamps, the events, the
injected group spawner. A work dispatch starts its children in the background and returns;
`settle` merges what finished at a turn boundary. `merge` projects child results into
ordered merges, DAG stamps and notices; `panel` runs a panel's record-only batches
through the same executor. Only the first-ranked child merges into the parent chain.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import pathlib
import time
from collections.abc import Callable, Collection, Sequence
from typing import TYPE_CHECKING

from agent6 import budget as agent6_budget
from agent6 import child_records, directive, errors, token_usage
from agent6.graph import curator as graph_curator
from agent6.harness import run_chain
from agent6.harness.parallel import merge as parallel_merge
from agent6.harness.parallel import merge_request as child_merge
from agent6.harness.parallel import rank as parallel_rank
from agent6.harness.parallel import repo as child_repo
from agent6.sessions import budget_events, child_dispatch, journal
from agent6.vcs.git import read as git_read

if TYPE_CHECKING:
    from agent6.harness import loop_conversation, loop_state


def segment_children(
    seg: directive.Segment, pins: Sequence[str] = (), *, limit: int
) -> list[child_records.ChildTask]:
    """Expand one segment into its children, one model per child.

    A bad spec (zero children, an empty model list, more than the limit) raises `DirectiveError`
    from `parse_spec`. Operator pins ride on every child out of band: the spawner's `--pin`
    channel seeds the child's own pin state, so a child never merges work that violates a
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
class OpenDispatch:
    """A work dispatch this execution started whose task is pending.

    Attributes:
        record: The dispatch's record, as last written.
        children: Its started children, in child order.
        judging: The judges ranking its attempts, while they run; None otherwise.
        completed: The record a continuation replaced; None for a first launch.
    """

    record: child_dispatch.DispatchRecord
    children: list[child_records.StartedChild]
    judging: parallel_rank.Judging | None = None
    completed: child_dispatch.DispatchRecord | None = None


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
        child_recoverable: Checks the saved workspace and record before offering continuation.
        child_recoverer: Finds an interrupted dispatch's children on resume; None when none can.
        judges: The reviewers that rank a task's attempts, `[harness.parallel].judges`
            resolved; empty for the mechanical order alone.
        reviewer_turns: A judge's reply cap, `review.max_turns`.
        reviewer_tools: The tools the run's own model is offered now, which a judge inherits.
        harness_gate: Whether the harness runs the verify gate, which decides what an
            unverified attempt may win.
        rank_recorder: Stamps and prints a ranking outside the dispatch record; None records
            it on the dispatch record alone.
        stop_pending: Whether the operator asked the run to stop.
        authority: Work and harness-panel permissions from the application.
        clock: The wall clock.
        open: This execution's pending work dispatches by identity, oldest first.
        stopping: The running children this execution asked to stop.
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
    child_recoverable: Callable[[str, child_dispatch.ChildRecord], bool] = lambda _g, _c: True
    judges: Sequence[parallel_rank.Judge] = ()
    reviewer_turns: int = 20
    reviewer_tools: Callable[[], Sequence[str]] = tuple
    harness_gate: bool = True
    rank_recorder: child_records.RankRecorder | None = None
    stop_pending: Callable[[], bool] = lambda: False
    authority: child_records.DispatchAuthority = dataclasses.field(
        default_factory=child_records.DispatchAuthority
    )
    clock: Callable[[], float] = time.time
    open: dict[str, OpenDispatch] = dataclasses.field(default_factory=dict, init=False)
    stopping: set[str] = dataclasses.field(default_factory=set, init=False)

    def dispatch(  # noqa: PLR0911  # one return per refusal
        self,
        conversation: loop_conversation.Conversation,
        turn: int,
        state: loop_state.LoopState,
        segments: list[directive.Segment],
        *,
        source: child_records.DispatchSource = "operator",
    ) -> child_records.Started:
        """Start each segment's task in the background and return what started.

        A segment is one task and one dispatch record, written before any launch.
        Several children are attempts `rank` orders. Each starts on a clone of the
        parent's committed HEAD under the parent's config and remaining budget, with no
        turn cap or deadline of its own. `settle` completes the task at a turn boundary;
        agent6-owned git merges the work, and model-owned git retains its ref.
        Never ends the run: an unavailable spawner, a bad spec, a dirty tree, an unwritable
        record or a spawner fault each come back as the refusal for the caller to deliver.

        Args:
            conversation: The run's conversation, snapshotted before the children start.
            turn: The turn just completed.
            state: The loop state.
            segments: The directive's segments, one task and one DAG node each.
            source: Who asked: the operator's steer, the model's tool call or a panel.

        Returns:
            The children started and one line per task, or the refusal with none.
        """
        if source == "panel" or not self.authority.work:
            return self.refused("work dispatch is unavailable in a child; nothing dispatched.")
        if self.child_spawner is None:
            return self.refused(
                "parallel dispatch is not available in this front-end; continuing normally."
            )
        try:
            tasks = [segment_children(seg, state.pins, limit=self.max_children) for seg in segments]
        except directive.DirectiveError as exc:
            return self.refused(f"bad /parallel spec: {exc}; nothing dispatched.")
        # The cap counts earlier dispatches' running children, before any commit, node or clone.
        asked, running = sum(len(children) for children in tasks), self.running()
        if asked + running > self.max_children:
            return self.refused(
                f"this asks for {asked} children with {running} running, over"
                f" [harness.parallel].max_children = {self.max_children}; nothing dispatched."
                " Ask for fewer, wait for one to end, stop one, or raise"
                " [harness.parallel].max_children."
            )
        # Agent6-owned git commits a changed tree first, or refuses stale work.
        entry = "the parallel tool" if source == "model" else "/parallel"
        if not self.ensure_clean(turn, f"{entry} dispatch"):
            if self.chain.ref is None:
                return self.refused(
                    f'{entry} requires a committed working tree under git.control = "model".'
                    " Nothing dispatched."
                )
            if not self.chain.per_turn:
                return self.refused(
                    f"{entry} requires a clean working tree when git.commit_per_turn = false."
                    " Nothing dispatched."
                )
            return self.refused(
                f"{entry} could not checkpoint the working tree. Nothing dispatched."
            )
        if self.session_dir is None:
            return self.refused(
                "parallel dispatch needs a session dir for its record; nothing dispatched."
            )
        if self.budget is not None and self.budget.is_exhausted():
            reason = self.budget.snapshot().exhausted_reason
            return self.refused(f"the budget is exhausted ({reason}); nothing dispatched.")
        parent_id = parallel_merge.parallel_parent_id(self.curator, state.root_task_id)
        base = self.chain.tip() or self.chain.head_sha()
        prepared: list[tuple[child_dispatch.DispatchRecord, list[child_records.ChildTask]]] = []
        lines: list[str] = []
        for seg, children in zip(segments, tasks, strict=True):
            state.parallel_groups_dispatched += 1
            identity = f"{self.session_id}-p{state.parallel_groups_dispatched}"
            # A prepared record can survive the snapshot that was to reserve its number.
            while child_dispatch.record_path(self.session_dir, identity).exists():
                state.parallel_groups_dispatched += 1
                identity = f"{self.session_id}-p{state.parallel_groups_dispatched}"
            record = child_dispatch.DispatchRecord(
                identity=identity,
                source=source,
                merge_policy="rank" if len(children) > 1 else "merge",
                owner=self.execution_id,
                turn=turn,
                base=base,
                spec=seg.spec,
                node_id=seg.node_id,
                parent_node=(
                    self.curator.get(seg.node_id).parent_id
                    if self.curator and seg.node_id
                    else parent_id
                ),
                children=tuple(
                    child_dispatch.ChildRecord(
                        session_id=f"{identity}-l{i}",
                        number=i,
                        task=child.task,
                        model=child.model,
                        limits=child.limits,
                    )
                    for i, child in enumerate(children, start=1)
                ),
            )
            try:
                # The record is the launch's precondition: nothing runs that a resume cannot find.
                self.write(record)
            except OSError as exc:
                state.parallel_groups_dispatched -= 1
                self.fail_nodes(record, "dispatch record could not be written")
                lines.append(
                    self.refused(
                        f"could not persist the dispatch record ({exc}); nothing dispatched."
                    ).message
                )
                continue
            prepared.append((record, children))
        # Persisted before the children start: a crash there would reuse a group's id on resume.
        self.save_snapshot(state, conversation.to_wire(snapshot=True), next_turn=turn + 1)
        started = [self.launch(record, children) for record, children in prepared]
        return child_records.Started(
            tuple(child for one in started for child in one.children),
            "\n".join([*lines, *(one.message for one in started)]),
            tuple(group for one in started for group in one.groups),
        )

    def launch(
        self, record: child_dispatch.DispatchRecord, children: list[child_records.ChildTask]
    ) -> child_records.Started:
        """Start a recorded task's children and return at once.

        Args:
            record: The durable prepared dispatch.
            children: Its children, in child order.

        Returns:
            The children started, or a diagnosed spawner failure with none.
        """  # noqa: DOC501  # the spawner's ChildRepoError is caught below
        assert self.child_spawner is not None
        group = record.identity.removeprefix(f"{self.session_id}-")
        if record.node_id:
            self.emit_graph_snapshot()
        record = self.announce(record, children)
        # The children start under what the run has left now, and share it.
        ceiling = self.budget.available() if self.budget is not None else None
        started: list[child_records.StartedChild] = []
        try:
            started = self.child_spawner(
                [
                    dataclasses.replace(child, allowance=ceiling, task_source=record.source)
                    for child in children
                ],
                group,
                at=record.base or None,
            )
            if len(started) != len(children):
                raise child_repo.ChildRepoError(
                    f"group spawner started {len(started)} child(ren) for {len(children)}"
                )
        except Exception as exc:
            for child in started:
                child.stop()
            # The caller delivers the failure as its answer: the record needs no second one.
            failed = self.failed_launch(record, exc, submitted={c.session_id for c in started})
            self.write(failed.model_copy(update={"delivered": True}))
            return child_records.Started((), failed.message)
        self.open[record.identity] = OpenDispatch(record, started)
        ids = tuple(child.session_id for child in started)
        return child_records.Started(
            ids, parallel_merge.started_text(ids), groups=(record.identity,)
        )

    def announce(
        self,
        record: child_dispatch.DispatchRecord,
        children: list[child_records.ChildTask],
    ) -> child_dispatch.DispatchRecord:
        """Log and journal a dispatch about to launch, and record it running.

        Args:
            record: The durable prepared dispatch.
            children: Its tasks, in child order.

        Returns:
            The record in the `running` state.
        """
        if record.source == "panel":
            what, tasks = "reviewer(s)", children
        else:
            what, tasks = "child(ren)", children[:1]
        self.log(f"PARALLEL: dispatching group {record.identity} ({len(children)} {what})")
        self.emit(
            journal.ParallelDispatched(
                group=record.identity,
                children=len(children),
                tasks=[child.task[:200] for child in tasks],
                source=record.source,
            )
        )
        record = record.advanced("running")
        self.write(record)
        return record

    def failed_launch(
        self,
        record: child_dispatch.DispatchRecord,
        exc: Exception,
        *,
        submitted: set[str],
        reviewers: Sequence[child_records.ChildResult] = (),
    ) -> child_dispatch.DispatchRecord:
        """Record a dispatch whose spawner failed: its nodes, its event, its message.

        Args:
            record: The running dispatch.
            exc: The spawner's fault.
            submitted: Children whose watches report their own ends.
            reviewers: Results collected before the panel's launch failed.

        Returns:
            The record, complete, its message the failure.
        """
        identity = record.identity
        self.log(f"PARALLEL: group {identity} dispatch failed: {exc}")
        record = self.fail_nodes(record, f"dispatch failed: {exc}")
        for child in record.children:
            if child.session_id not in submitted:
                self.emit(
                    journal.ParallelChildActivity(
                        child=child.session_id, activity="failed", detail=str(exc)
                    )
                )
        self.emit(journal.ParallelFailed(group=identity, error=str(exc)))
        message = f"[parallel] group {identity} dispatch failed: {exc}. Nothing was merged."
        by_id = {res.spec.session_id: res for res in reviewers}
        unstarted = {child.session_id for child in record.children} - submitted
        record = self.charged(record, by_id, unstarted=unstarted)
        self.emit_budget(record.turn)
        merges: list[child_records.ChildMerge] = []
        for child in record.children:
            if res := by_id.get(child.session_id):
                returned = parallel_merge.reviewer_record(child, res, self.chain.tip() or "")
                record = record.with_child(returned)
                merges.append(parallel_merge.merge_of(returned))
        return self.record_return(record, message, merges)

    def unended(self) -> list[concurrent.futures.Future[child_records.ChildResult]]:
        """Return the result of every child and judge this execution started that still runs."""
        return [
            child.result
            for task in self.open.values()
            for child in (*task.children, *(task.judging.children if task.judging else ()))
            if not child.result.done()
        ]

    def running(self) -> int:
        """Return how many children and judges this execution started have not ended."""
        return len(self.unended())

    def wait(self) -> None:
        """Block until all that runs has ended, for an entry that takes no turn meanwhile."""
        concurrent.futures.wait(self.unended())

    def settle(self, conversation: loop_conversation.Conversation, turn: int) -> bool:
        """Tell the model which children ended, then merge each dispatch whose children all have.

        Called at a turn boundary. The finished dispatches merge in the order written; one
        with a child still running merges at a later boundary and holds no other. Attempts
        are ranked first: their judges start here and run in the background, and the task
        merges at the boundary after they end.

        Args:
            conversation: The run's conversation, which takes the notices.
            turn: The turn just completed.

        Returns:
            Whether the conversation gained a notice.
        """
        told = False
        for identity, task in list(self.open.items()):
            results = [
                self.result_of(task.record, child) for child in task.children if child.result.done()
            ]
            if self.restored(task, results):
                (res,) = results
                conversation.notice(
                    f"[parallel] child {res.spec.session_id} could not continue: {res.error}."
                    " Its earlier result stands."
                )
                del self.open[identity]
                told = True
                continue
            record = self.reported(conversation, task.record, results)
            told = told or record is not task.record
            task.record = record
            if len(results) < len(task.children):
                continue
            if record.merge_policy == "rank" and record.ranking is None:
                record = task.record = parallel_rank.rank(self, task, results, turn=turn)
                if record.ranking is None:
                    # Its judges run, or a stop holds them: a later boundary or the end settles it.
                    continue
            result = self.complete(record, results, turn=turn)
            self.deliver(conversation, result)
            del self.open[identity]
            told = True
        return child_merge.settle(self, conversation, turn) or told

    def restored(self, task: OpenDispatch, results: Sequence[child_records.ChildResult]) -> bool:
        """Write back the record a continuation replaced when its child never started.

        Args:
            task: An open dispatch.
            results: The results of its children that have ended.

        Returns:
            Whether the completed record stands again, as after a refused launch.
        """
        if task.completed is None or not results or results[0].started:
            return False
        self.write(task.completed)
        if task.completed.node_id:
            self.emit_graph_snapshot()
        return True

    def reported(
        self,
        conversation: loop_conversation.Conversation,
        record: child_dispatch.DispatchRecord,
        results: Sequence[child_records.ChildResult],
    ) -> child_dispatch.DispatchRecord:
        """Put each ended child's notice in the conversation, once per child.

        Args:
            conversation: The run's conversation.
            record: The dispatch's record.
            results: The results of its children that have ended.

        Returns:
            The record with those children marked reported; the same object when none was new.
        """
        told = record
        for res in results:
            child = told.child(res.spec.session_id)
            if child.reported:
                continue
            conversation.notice(parallel_merge.end_notice(res))
            told = told.with_child(
                child.model_copy(update={"reported": True, "report": res.report})
            )
        if told is not record:
            self.write(told)
        return told

    def result_of(
        self, record: child_dispatch.DispatchRecord, child: child_records.StartedChild
    ) -> child_records.ChildResult:
        """Return an ended child's result; a watch that failed reads as a child that did.

        Args:
            record: The child's dispatch.
            child: The started child, its result resolved.

        Returns:
            The result.
        """
        try:
            return child.result.result()
        except Exception as exc:
            # The watch is a bug's victim, not the child: its process died with its thread.
            return child_records.ChildResult(
                spec=child_records.ChildSpec(
                    record.child(child.session_id).number, child.session_id, pathlib.Path(), None
                ),
                session_dir=pathlib.Path(),
                branch=git_read.run_branch_for(child.session_id),
                ok=False,
                error=f"its watch failed: {exc}",
            )

    def merges(self, group: str) -> list[child_records.ChildMerge]:
        """Return the merges a dispatch has recorded, in child order; none before it completed."""
        if self.session_dir is None:
            return []
        record = child_dispatch.read_record(child_dispatch.record_path(self.session_dir, group))
        return [
            parallel_merge.merge_of(child) for child in record.children if child.merge is not None
        ]

    def close(self, turn: int) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """End the execution's children: stop each still running, wait, charge what it used.

        A stopped child never merges. A listed task whose attempts all stopped or failed
        releases its hold. The next execution settles any outstanding merge. A task's judges
        are stopped and charged with it.

        Args:
            turn: The turn the execution ended on.

        Returns:
            The children this stopped, then the ones that had ended and whose task had not
            merged.
        """
        stopped: list[str] = []
        ended: list[str] = []
        for task in self.open.values():
            for child in task.children:
                if child.result.done():
                    continue
                child.stop()
                stopped.append(child.session_id)
            for judge in task.judging.children if task.judging is not None else ():
                judge.stop()
        for task in self.open.values():
            results = [self.result_of(task.record, child) for child in task.children]
            if self.restored(task, results):
                continue
            if task.record.node_id and all(res.stopped_by or not res.ok for res in results):
                # No work can merge. Release the listed task even if the parent never resumes.
                self.complete(task.record, results, turn=turn)
            else:
                ended.extend(c.session_id for c in task.children if c.session_id not in stopped)
                self.settled(task.record, {res.spec.session_id: res for res in results}, turn=turn)
            if (judging := task.judging) is not None:
                self.settled(
                    judging.record,
                    {j.session_id: self.result_of(judging.record, j) for j in judging.children},
                    turn=turn,
                )
        self.open.clear()
        return tuple(stopped), tuple(ended)

    def complete(
        self,
        record: child_dispatch.DispatchRecord,
        results: list[child_records.ChildResult],
        *,
        turn: int | None = None,
    ) -> child_records.DispatchResult:
        """Settle, merge, stamp and complete a dispatch whose children have returned.

        Work merges on a clean tree: the parent's uncommitted work is committed first. Each
        boundary advances the record first, so a crash in between leaves a resume one step to
        finish, never one to repeat. A rank merge comes here with its ranking written.

        Args:
            record: The dispatch's record.
            results: The children's results; a child without one never started.
            turn: The turn the merge follows; None for the turn of the dispatch.

        Returns:
            The result, its message the record's.
        """
        by_id = {res.spec.session_id: res for res in results}
        turn = record.turn if turn is None else turn
        record = self.settled(record, by_id, turn=turn)
        clean = (
            record.merge_policy == "record_only"
            or all(res.stopped_by or not res.ok for res in results)
            or self.ensure_clean(turn, "merging children")
        )
        record = self.merged(record, by_id, clean=clean)
        merges = [parallel_merge.merge_of(child) for child in record.children]
        if record.node_id is None:
            # Its id is on the record before its stamp: a resume never adds a second node.
            record = self.noded(record)
            self.write(record)
        # A detached attempt can run only after the dispatch's task is final.
        if not any(child.detached for child in record.children):
            parallel_merge.stamp_segment_node(self.curator, record.node_id, merges, log=self.log)
        message = parallel_merge.summary_text(record.identity, merges, record.ranking)
        self.record_return(record, message, merges)
        return child_records.DispatchResult(
            record.identity, True, tuple(merges), message, base=record.base
        )

    def record_return(
        self,
        record: child_dispatch.DispatchRecord,
        message: str,
        merges: Sequence[child_records.ChildMerge] = (),
    ) -> child_dispatch.DispatchRecord:
        """Journal the returned results and close the dispatch.

        Args:
            record: The dispatch to close.
            message: The notice for the parent.
            merges: The child outcomes, empty when the launch yielded no results.

        Returns:
            The completed record.
        """
        payload = [
            {
                "session_id": j.session_id,
                "branch": j.branch,
                "status": j.status,
                "sha": j.sha,
                "detail": j.detail,
                "commits": j.commits,
                "context": dataclasses.asdict(j.context),
            }
            for j in merges
        ]
        self.emit(
            journal.ParallelReturned(group=record.identity, children=payload, source=record.source)
        )
        failures = [
            p
            for p, j in zip(payload, merges, strict=True)
            # A stopped child is no failed merge: nothing of it was meant to merge.
            if j.status not in ("merged", "no_changes", "ranked", "stopped", "retained")
        ]
        if failures:
            self.emit(journal.ParallelFailed(group=record.identity, children=failures))
        record = record.advanced("complete", message=message)
        self.write(record)
        if record.node_id:
            self.emit_graph_snapshot()
        return record

    def settled(
        self,
        record: child_dispatch.DispatchRecord,
        by_id: dict[str, child_records.ChildResult],
        *,
        turn: int | None = None,
    ) -> child_dispatch.DispatchRecord:
        """Charge each child's usage once; a child whose usage is unknown stays unsettled.

        Args:
            record: The dispatch's record.
            by_id: The children's results by session id.
            turn: The turn the charge is journaled at; None for the turn of the dispatch.

        Returns:
            The record with updated charges. A completed dispatch stays complete.
        """
        if record.state != "complete":
            record = record.advanced("settling")
        record = self.charged(record, by_id)
        self.emit_budget(record.turn if turn is None else turn)
        self.write(record)
        return record

    def charged(
        self,
        record: child_dispatch.DispatchRecord,
        by_id: dict[str, child_records.ChildResult],
        *,
        unstarted: Collection[str] = (),
    ) -> child_dispatch.DispatchRecord:
        """Charge known usage without advancing the dispatch or changing its outcome.

        Args:
            record: The dispatch to account for.
            by_id: The results that came back.
            unstarted: Children known not to have started.

        Returns:
            The record with each known charge marked settled.
        """
        for child in record.children:
            if child.settled:
                continue
            res = by_id.get(child.session_id)
            if child.session_id in unstarted or (res is not None and not res.started):
                usage = agent6_budget.UsageSettlement(
                    usd=0, usd_partial=False, tokens=0, model_tokens=token_usage.ZERO
                )
            elif res is not None and res.usage is not None:
                usage = res.usage
            else:
                continue
            if self.budget is not None:
                self.budget.settle_child(child.session_id, usage)
            stored = child_dispatch.UsageRecord(
                usd=usage.usd,
                usd_observed_credits=usage.usd_observed_credits,
                credit_readings=usage.credit_readings,
                usd_partial=usage.usd_partial,
                tokens=usage.tokens,
                model_tokens=usage.model_tokens,
            )
            record = record.with_child(child.model_copy(update={"usage": stored, "settled": True}))
        return record

    def merged(
        self,
        record: child_dispatch.DispatchRecord,
        by_id: dict[str, child_records.ChildResult],
        *,
        clean: bool = True,
    ) -> child_dispatch.DispatchRecord:
        """Merge the remaining children in order and record each outcome.

        A ranked dispatch merges only its winner, if any.
        A stopped child never merges: its record names the branch that holds its commits.

        Args:
            record: The dispatch's record.
            by_id: The children's results by session id.
            clean: Whether the tree holds only committed work; a merge needs it.

        Returns:
            The record in the `merging` state with every child's merge recorded.
        """
        record = record.advanced("merging")
        self.write(record)
        order = record.ranking.order if record.ranking is not None else ()
        winner = record.ranking.winner if record.ranking is not None else ""
        for child in record.children:
            if child.merge is not None:
                continue
            before = self.chain.tip() or ""
            res = by_id.get(child.session_id)
            if record.merge_policy == "record_only":
                record = record.with_child(parallel_merge.reviewer_record(child, res, before))
                self.write(record)
                continue
            if res is not None and res.ok and res.stopped_by:
                merge = parallel_merge.stopped_merge(res)
            elif child.continuation is not None and self.chain.ref and res is not None and res.ok:
                merge = child_records.ChildMerge(
                    child.session_id,
                    res.imported_ref,
                    "retained" if res.has_commits else "no_changes",
                    "",
                    "",
                    commits=res.commits,
                )
            elif (
                res is not None
                and res.ok
                # Model-owned git records the imported ref, not its merge status.
                and (
                    not self.chain.ref or (child.session_id in order and child.session_id != winner)
                )
            ):
                # Its branch stays imported: the operator can still merge it by hand.
                place = (
                    f"{order.index(child.session_id) + 1} of {len(order)}"
                    if child.session_id in order
                    else ""
                )
                merge = child_records.ChildMerge(
                    child.session_id,
                    res.imported_ref or (res.branch if self.chain.ref else ""),
                    "ranked"
                    if self.chain.ref
                    else "retained"
                    if res.imported_ref
                    else "no_changes",
                    "",
                    place,
                    commits=res.commits,
                )
            elif not clean and res is not None and res.ok and res.has_commits:
                # Its branch stays imported: nothing merges over work no commit holds.
                merge = child_records.ChildMerge(
                    child.session_id,
                    res.imported_ref or res.branch,
                    "failed",
                    "",
                    "the working tree holds uncommitted work that could not be committed"
                    " (git.commit_per_turn is off)",
                )
            else:
                merge = self.merge_child(child, res)
            record = record.with_child(
                child.model_copy(
                    update={
                        "merge": child_dispatch.MergeRecord(
                            branch=merge.branch,
                            status=merge.status,
                            sha=merge.sha,
                            detail=merge.detail,
                            commits=merge.commits,
                            context=merge.context,
                            target_before=before,
                            target_after=self.chain.tip() or "",
                        ),
                        "report": res.report if res is not None else "",
                    }
                )
            )
            self.write(record)
        return record

    def merge_child(
        self, child: child_dispatch.ChildRecord, res: child_records.ChildResult | None
    ) -> child_records.ChildMerge:
        """Merge one child, or recognise a merge already on the chain.

        Args:
            child: The child's record.
            res: Its result, None when it never started.

        Returns:
            The merge.
        """
        if res is None:
            branch = git_read.run_branch_for(child.session_id)
            return child_records.ChildMerge(child.session_id, branch, "failed", "", "never started")
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
            return child_records.ChildMerge(
                res.spec.session_id, res.branch, "merged", sha or (self.chain.tip() or ""), ""
            )
        return parallel_merge.merge_child_result(
            self.chain.root,
            res,
            ref=ref,
            fallback_parent=self.chain.fallback_parent,
            identity=self.chain.identity,
            also_branch=self.chain.branch,
        )

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

    def noded(self, record: child_dispatch.DispatchRecord) -> child_dispatch.DispatchRecord:
        """Give a task its DAG node, now that its outcome is known.

        A text dispatch adds its node under the task it was started from.

        Args:
            record: The task's record.

        Returns:
            The record naming the node; unchanged for a panel, without a graph, or when it
            already names one.
        """
        if record.source == "panel" or record.node_id is not None or not record.children:
            return record
        if any(c.continuation for c in record.children):
            # A continuation detached from its final task creates no replacement node.
            return record
        node_id = parallel_merge.add_parallel_node(
            self.curator,
            record.children[0].task,
            record.parent_node,
            source=record.source,
            log=self.log,
        )
        return record.model_copy(update={"node_id": node_id})

    def fail_nodes(
        self, record: child_dispatch.DispatchRecord, note: str
    ) -> child_dispatch.DispatchRecord:
        """Record a text brief's failure. A listed task keeps its status on a refusal.

        Args:
            record: The task's record.
            note: Why the task failed.

        Returns:
            The record naming the node, for the caller to write.
        """
        if record.node_id is not None:
            # No child produced a result. A listed task keeps its status and can be tried again.
            return record
        record = self.noded(record)
        parallel_merge.stamp_parallel_node(
            self.curator, record.node_id, status="failed", note=note, log=self.log
        )
        self.emit_graph_snapshot()
        return record

    def emit_budget(self, turn: int) -> None:
        """Journal the meter after a settlement, so the children's charge is on the record."""
        if self.budget is None:
            return
        budget_events.emit_budget_update(self.budget, self.emit)
        self.emit(budget_events.budget_turn(self.budget, turn))

    def ensure_clean(self, turn: int, what: str) -> bool:
        """Chain-commit a changed tree, so children are cut from, and merge on, committed work.

        With `commit_per_turn` off a changed tree cannot be captured.

        Args:
            turn: The turn just completed, for the checkpoint subject.
            what: What the checkpoint precedes, for its subject.

        Returns:
            Whether the chain tip carries the worktree's content.
        """
        if not self.chain.dirty():
            return True
        if not self.chain.per_turn:
            return False
        try:
            subject = f"checkpoint before {what} (turn {turn})"
            sha = self.chain.commit(subject)
            if sha:
                self.log(f"  checkpoint before {what}: {sha[:12]}")
                self.emit(journal.CheckpointCommitted(turn=turn, sha=sha, subject=subject))
                self.emit(
                    journal.DiffUpdated(
                        sha=sha,
                        patch=git_read.commit_diff(self.chain.root, sha, max_bytes=8000),
                    )
                )
        except (errors.GitError, OSError) as exc:
            self.log(f"PARALLEL: checkpoint before {what} failed: {exc}")
        return not self.chain.dirty()

    def refused(self, msg: str) -> child_records.Started:
        """Return a work dispatch refused before any child started, logged once."""
        self.log(f"PARALLEL: {msg}")
        return child_records.Started((), f"[parallel] {msg}")
