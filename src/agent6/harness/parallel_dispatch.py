# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Dispatch a run's children and land their work.

`ParallelDispatcher` owns the policy: when to cut children, the DAG stamps, the events, the
injected group spawner. A work dispatch starts its children in the background and returns;
`settle` lands what finished at a turn boundary. `parallel_join` projects child results into
ordered merges, DAG stamps and notices; record-only panel batches share the executor, block
and merge nothing, and a rank join lands the one child its injected ranker puts first.
"""

from __future__ import annotations

import concurrent.futures
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
from agent6.tools import child_tools
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
class OpenDispatch:
    """A work dispatch this execution started whose work has not landed.

    Attributes:
        record: The dispatch's record, as last written.
        children: Its started children, in child order.
    """

    record: child_dispatch.DispatchRecord
    children: list[child_records.StartedChild]


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
        open: The work dispatches this execution started and has not landed, by identity,
            oldest first.
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
    ranker: child_records.ChildRanker | None = None
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
        limits: child_records.ChildLimits | None = None,
    ) -> child_records.Started:
        """Start each segment's task in the background and return what started.

        A segment is one task and one dispatch record, written before any launch: one child
        lands its commits, several are attempts a ranker orders. Each child starts on a
        clone of the parent's committed HEAD under what the parent has left to spend, and
        runs while the parent takes its turns; `settle` lands the work at a turn boundary.
        Never ends the run: an unavailable spawner, a bad spec, a dirty tree, an unwritable
        record or a spawner fault each come back as the refusal for the caller to deliver.

        Args:
            conversation: The run's conversation, snapshotted before the children start.
            turn: The turn just completed.
            state: The loop state.
            segments: The directive's segments, one task and one DAG node each.
            source: Who asked: the operator's steer, the model's tool call or a panel.
            limits: The turn cap and deadline every child runs under; None for the config's.

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
        if self.ranker is None and any(len(children) > 1 for children in tasks):
            return self.refused(
                "this execution has no ranker for attempts at one task; nothing dispatched."
            )
        # Children cut from the chain tip only: commit a changed tree first, or refuse stale work.
        if not self.ensure_clean(turn, "/parallel dispatch"):
            return self.refused(
                "refusing to dispatch: the working tree is not clean and could not be"
                " auto-committed. Commit or discard your changes, then retry /parallel."
            )
        if self.session_dir is None:
            return self.refused(
                "parallel dispatch needs a session dir for its record; nothing dispatched."
            )
        if self.budget is not None and self.budget.is_exhausted():
            reason = self.budget.snapshot().exhausted_reason
            return self.refused(f"the budget is exhausted ({reason}); nothing dispatched.")
        parent_id = parallel_join.parallel_parent_id(self.curator, state.root_task_id)
        base = self.chain.tip() or ""
        prepared: list[tuple[child_dispatch.DispatchRecord, list[child_records.ChildTask]]] = []
        lines: list[str] = []
        for seg, task in zip(segments, tasks, strict=True):
            state.parallel_groups_dispatched += 1
            identity = f"{self.session_id}-p{state.parallel_groups_dispatched}"
            children = [dataclasses.replace(child, limits=limits) for child in task]
            record = child_dispatch.DispatchRecord(
                identity=identity,
                source=source,
                join_policy="rank" if len(children) > 1 else "merge",
                owner=self.execution_id,
                turn=turn,
                base=base,
                spec=seg.spec,
                node_id=parallel_join.add_parallel_node(
                    self.curator, seg.task, parent_id, source=source, log=self.log
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
        if self.curator is not None:
            self.emit_graph_snapshot()
        # Persisted before the children start: a crash there would reuse a group's id on resume.
        self.save_snapshot(state, conversation.to_wire(), next_turn=turn + 1)
        started = [self.launch(record, children) for record, children in prepared]
        return child_records.Started(
            tuple(child for one in started for child in one.children),
            "\n".join([*lines, *(one.message for one in started)]),
            tuple(group for one in started for group in one.groups),
        )

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
        record = self.announce(record, children, group)
        # The children start under what the run has left now, and share it.
        ceiling = self.budget.available() if self.budget is not None else None
        started: list[child_records.StartedChild] = []
        try:
            started = self.child_spawner(
                [dataclasses.replace(child, allowance=ceiling) for child in children],
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
            failed = self.failed_launch(record, exc)
            self.write(failed.model_copy(update={"delivered": True}))
            return child_records.Started((), failed.message)
        self.open[record.identity] = OpenDispatch(record, started)
        ids = tuple(child.session_id for child in started)
        return child_records.Started(
            ids, parallel_join.started_text(ids), groups=(record.identity,)
        )

    def execute(
        self,
        record: child_dispatch.DispatchRecord,
        children: list[child_records.ChildTask],
        group: str,
        *,
        batch_width: int,
    ) -> child_records.DispatchResult:
        """Run a recorded panel in batches, waiting for each, then settle and record it.

        Args:
            record: The durable prepared dispatch.
            children: Its bounded tasks, in child order.
            group: The group suffix used by the shared spawner.
            batch_width: The maximum children submitted together.

        Returns:
            The completed result, or a diagnosed spawner failure with unresolved holds.
        """  # noqa: DOC501  # the spawner's ChildRepoError is caught below
        assert self.child_spawner is not None
        record = self.announce(record, children, group)
        started: list[child_records.StartedChild] = []
        try:
            results: list[child_records.ChildResult] = []
            for offset in range(0, len(children), batch_width):
                # Each batch starts under what the run has left then; its children share it.
                ceiling = self.budget.available() if self.budget is not None else None
                batch = [
                    dataclasses.replace(child, allowance=ceiling)
                    for child in children[offset : offset + batch_width]
                ]
                batch_group = (
                    group
                    if len(children) <= batch_width
                    else f"{group}-b{offset // batch_width + 1}"
                )
                started = self.child_spawner(batch, batch_group, at=record.base or None)
                results.extend(child.result.result() for child in started)
            if len(results) != len(children):
                raise child_repo.ChildRepoError(
                    f"group spawner returned {len(results)} result(s)"
                    f" for {len(children)} child(ren)"
                )
        except Exception as exc:
            # The front-end callback may fail after launching: the children's usage is unknown.
            for child in started:
                child.stop()
            failed = self.failed_launch(record, exc)
            return child_records.DispatchResult(
                failed.identity, True, (), failed.message, base=failed.base
            )
        return self.land(record, results)

    def announce(
        self,
        record: child_dispatch.DispatchRecord,
        children: list[child_records.ChildTask],
        group: str,
    ) -> child_dispatch.DispatchRecord:
        """Log and journal a dispatch about to launch, and record it running.

        Args:
            record: The durable prepared dispatch.
            children: Its tasks, in child order.
            group: The group suffix the event names.

        Returns:
            The record in the `running` state.
        """
        what = "reviewer(s)" if record.source == "panel" else "child(ren)"
        self.log(f"PARALLEL: dispatching group {record.identity} ({len(children)} {what})")
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
        return record

    def failed_launch(
        self, record: child_dispatch.DispatchRecord, exc: Exception
    ) -> child_dispatch.DispatchRecord:
        """Record a dispatch whose spawner failed: its nodes, its event, its message.

        Args:
            record: The running dispatch.
            exc: The spawner's fault.

        Returns:
            The record, complete, its message the failure.
        """
        identity = record.identity
        self.log(f"PARALLEL: group {identity} dispatch failed: {exc}")
        self.fail_nodes(record, f"dispatch failed: {exc}")
        self.emit(journal.ParallelFailed(group=identity, error=str(exc)))
        message = f"[parallel] group {identity} dispatch failed: {exc}. Nothing was joined."
        record = self.settled(record, {}).advanced("complete", message=message)
        self.write(record)
        return record

    def running(self) -> int:
        """Return how many children this execution started that have not ended."""
        return sum(
            1 for task in self.open.values() for child in task.children if not child.result.done()
        )

    def wait(self) -> None:
        """Block until every started child has ended, for an entry that takes no turn meanwhile."""
        concurrent.futures.wait(
            [child.result for task in self.open.values() for child in task.children]
        )

    def wait_for(self, child: str, wait_s: float, interrupted: Callable[[], bool]) -> None:
        """Wait for one running child to end, or with "" for the next of them all.

        Args:
            child: The child's session id; "" for whichever ends next.
            wait_s: The most seconds to wait.
            interrupted: Cuts the wait short: the operator's stop reaches no turn boundary
                while a tool call waits.
        """
        pending = [
            started.result
            for task in self.open.values()
            for started in task.children
            if not started.result.done() and child in ("", started.session_id)
        ]
        deadline = time.monotonic() + wait_s
        while pending and not interrupted() and (left := deadline - time.monotonic()) > 0:
            done, _ = concurrent.futures.wait(
                pending, timeout=min(left, 0.5), return_when=concurrent.futures.FIRST_COMPLETED
            )
            if done:
                return

    def stop(self, child: str) -> None:
        """Ask a running child to stop now; its end is told and its commits land as any child's.

        Args:
            child: The child's session id; one that has ended, or is no child, is left alone.
        """
        for task in self.open.values():
            for started in task.children:
                if started.session_id == child and not started.result.done():
                    started.stop()
                    self.stopping.add(child)

    def unlanded(self) -> tuple[tuple[str, str], ...]:
        """Return each child whose task has not landed, with where it stands, oldest first."""
        return tuple(
            (
                started.session_id,
                "ended, not landed"
                if started.result.done()
                else "stopping"
                if started.session_id in self.stopping
                else "running",
            )
            for task in self.open.values()
            for started in task.children
        )

    def roster(self) -> tuple[child_tools.ChildLine, ...]:
        """Return every child this run's tasks started and where it stands, oldest first."""
        if self.session_dir is None:
            return ()
        try:
            records = child_dispatch.read_records(self.session_dir)
        except child_dispatch.DispatchRecordError:
            return ()
        open_now = dict(self.unlanded())
        lines: list[child_tools.ChildLine] = []
        for record in records:
            for child in record.children if record.source != "panel" else ():
                sid = child.session_id
                state = open_now.get(sid, "ended, not landed")
                if child.join is not None:
                    state = parallel_join.join_words(parallel_join.join_of(child))
                title = next((ln.strip() for ln in child.task.splitlines() if ln.strip()), "")
                running = state in ("running", "stopping")
                lines.append(child_tools.ChildLine(sid, state, title[:80], running))
        return tuple(lines)

    def settle(self, conversation: loop_conversation.Conversation, turn: int) -> bool:
        """Tell the model which children ended, then land each finished dispatch, oldest first.

        Called at a turn boundary. A dispatch lands only after every dispatch written before
        it, so one still running holds the later ones; their children's ends are told all
        the same.

        Args:
            conversation: The run's conversation, which takes the notices.
            turn: The turn just completed.

        Returns:
            Whether the conversation gained a notice.
        """
        told = held = False
        for identity, task in list(self.open.items()):
            results = [
                self.result_of(task.record, child) for child in task.children if child.result.done()
            ]
            record = self.reported(conversation, task.record, results)
            told = told or record is not task.record
            task.record = record
            held = held or len(results) < len(task.children)
            if held:
                continue
            result = self.land(record, results, turn=turn)
            if not result.message:
                # A stop held the ranking: the execution's end settles what is left.
                assert self.session_dir is not None
                task.record = child_dispatch.read_record(
                    child_dispatch.record_path(self.session_dir, identity)
                )
                held = True
                continue
            self.deliver(conversation, result)
            del self.open[identity]
            told = True
        return told

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
            conversation.notice(parallel_join.end_notice(res))
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

    def joins(self, group: str) -> list[child_records.ChildJoin]:
        """Return the joins a dispatch has recorded, in child order; none before it landed."""
        if self.session_dir is None:
            return []
        record = child_dispatch.read_record(child_dispatch.record_path(self.session_dir, group))
        return [parallel_join.join_of(child) for child in record.children if child.join is not None]

    def close(self) -> tuple[str, ...]:
        """End the execution's children: stop each still running, wait, charge what it used.

        Their work stays on their branches and their records stay unlanded: the next
        execution lands them before its first model call.

        Returns:
            The children whose dispatch had not landed.
        """
        left = tuple(child.session_id for task in self.open.values() for child in task.children)
        for task in self.open.values():
            for child in task.children:
                if not child.result.done():
                    child.stop()
        for task in self.open.values():
            self.settled(
                task.record,
                {child.session_id: self.result_of(task.record, child) for child in task.children},
            )
        self.open.clear()
        return left

    def land(
        self,
        record: child_dispatch.DispatchRecord,
        results: list[child_records.ChildResult],
        *,
        turn: int | None = None,
    ) -> child_records.DispatchResult:
        """Settle, join, stamp and complete a dispatch whose children have returned.

        Work lands on a clean tree: the parent's uncommitted work is committed first. Each
        boundary advances the record first, so a crash in between leaves a resume one step to
        finish, never one to repeat.

        Args:
            record: The dispatch's record.
            results: The children's results; a child without one never started.
            turn: The turn the landing follows; None for the turn of the dispatch.

        Returns:
            The result, its message the record's; "" when a stop held a rank join's ranking.
        """
        by_id = {res.spec.session_id: res for res in results}
        record = self.settled(record, by_id)
        if record.join_policy == "rank" and record.ranking is None and self.stop_pending():
            # No judge is asked after a stop: the record stays here and a resume ranks from it.
            self.log(f"PARALLEL: group {record.identity} stopped before its ranking")
            return child_records.DispatchResult(record.identity, True, (), "", base=record.base)
        clean = record.join_policy == "record_only" or self.ensure_clean(
            record.turn if turn is None else turn, "landing children"
        )
        record = self.joined(record, by_id, clean=clean)
        joins = [parallel_join.join_of(child) for child in record.children]
        parallel_join.stamp_segment_node(self.curator, record.node_id, joins, log=self.log)
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
        *,
        clean: bool = True,
    ) -> child_dispatch.DispatchRecord:
        """Join each child not yet joined, in child order, recording each join as it lands.

        A rank join ranks first, once, and joins its winner alone; with no winner it joins none.

        Args:
            record: The dispatch's record.
            by_id: The children's results by session id.
            clean: Whether the tree holds only committed work; a merge needs it.

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
            elif not clean and res is not None and res.ok and res.has_commits:
                # Its branch stays imported: nothing merges over work no commit holds.
                join = child_records.ChildJoin(
                    child.session_id,
                    res.imported_ref or res.branch,
                    "failed",
                    "",
                    "the working tree holds uncommitted work that could not be committed"
                    " (git.commit_per_turn is off)",
                )
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
            self.deliver(
                conversation, self.land(self.reported(conversation, record, results), results)
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

    def fail_nodes(self, record: child_dispatch.DispatchRecord, note: str) -> None:
        """Stamp the task's DAG node failed with the note."""
        parallel_join.stamp_parallel_node(
            self.curator, record.node_id, status="failed", note=note, log=self.log
        )
        self.emit_graph_snapshot()

    def emit_budget(self, turn: int) -> None:
        """Journal the meter after a settlement, so the children's charge is on the record."""
        if self.budget is None:
            return
        self.emit(journal.BudgetUpdate(**budget_events.budget_update_fields(self.budget)))
        self.emit(budget_events.budget_turn(self.budget, turn))

    def ensure_clean(self, turn: int, what: str) -> bool:
        """Chain-commit a changed tree, so children are cut from, and land on, committed work.

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

    def refusal(self, msg: str) -> child_records.DispatchResult:
        """Return the result of a panel refused before any reviewer ran, logged once."""
        self.log(f"PARALLEL: {msg}")
        return child_records.DispatchResult("", False, (), f"[parallel] {msg}")

    def refused(self, msg: str) -> child_records.Started:
        """Return a work dispatch refused before any child started, logged once."""
        self.log(f"PARALLEL: {msg}")
        return child_records.Started((), f"[parallel] {msg}")
