# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Settle the dispatches an earlier execution left behind, before a resumed one's first call."""

from __future__ import annotations

import concurrent.futures
import dataclasses
import pathlib
from collections.abc import Sequence
from typing import TYPE_CHECKING

from agent6 import budget as agent6_budget
from agent6 import child_records, credit_usage, token_usage
from agent6.harness import parallel_dispatch, parallel_merge
from agent6.sessions import child_dispatch, journal
from agent6.vcs.git import read as git_read
from agent6.viewmodel import events, tail
from agent6.viewmodel.session import log as session_log

if TYPE_CHECKING:
    from agent6.harness import loop_conversation, loop_state


def reopened(
    record: child_dispatch.DispatchRecord, results: Sequence[child_records.ChildResult]
) -> parallel_dispatch.OpenDispatch:
    """Return an interrupted task as an open dispatch whose children have all ended.

    Args:
        record: The task's record.
        results: The results of its children that have a session dir.

    Returns:
        The dispatch, each child's result resolved; a child without one never started.
    """
    by_id = {res.spec.session_id: res for res in results}
    children: list[child_records.StartedChild] = []
    for child in record.children:
        ended: concurrent.futures.Future[child_records.ChildResult] = concurrent.futures.Future()
        ended.set_result(
            by_id.get(child.session_id)
            or child_records.ChildResult(
                spec=child_records.ChildSpec(child.number, child.session_id, pathlib.Path(), None),
                session_dir=pathlib.Path(),
                branch=git_read.run_branch_for(child.session_id),
                ok=False,
                error="never started",
            )
        )
        children.append(child_records.StartedChild(child.session_id, ended))
    return parallel_dispatch.OpenDispatch(record, children)


@dataclasses.dataclass(frozen=True, slots=True)
class _CreditHistory:
    """The journaled balance series, kept apart by account and originating execution."""

    executions: dict[str, int]
    continuations: dict[tuple[str, str, int], int]
    readings: dict[int, dict[str, set[credit_usage.Reading]]]

    @classmethod
    def read(cls, logs: pathlib.Path) -> _CreditHistory:
        """Read each balance into the series its parent meter counted.

        Args:
            logs: The resumed parent's journal.

        Returns:
            The readings by series and the executions that used them.
        """
        execution = 0
        executions: dict[str, int] = {}
        continuations: dict[tuple[str, str, int], int] = {}
        readings: dict[int, dict[str, set[credit_usage.Reading]]] = {}
        for raw in tail.tail_events(logs, follow=False):
            match raw:
                case {"type": "session.start" | "session.resume"}:
                    execution += 1
                    owner = raw.get("execution_id")
                    if isinstance(owner, str) and owner:
                        executions[owner] = execution
                case {
                    "type": "parallel.child_continued",
                    "group": str(group),
                    "child": str(child),
                    "sequence": int(sequence),
                } if execution:
                    continuations[group, child, sequence] = execution
                case {"type": "budget.update"}:
                    row = events.parse_event(raw)
                    if isinstance(row, journal.BudgetUpdate):
                        for origin, reading in row.credits.scoped():
                            series = execution if origin is None else origin
                            readings.setdefault(series, {}).setdefault(reading.account, set()).add(
                                reading
                            )
        return cls(executions, continuations, readings)

    def restore(
        self,
        meter: agent6_budget.BudgetTracker,
        record: child_dispatch.DispatchRecord,
        child: child_dispatch.ChildRecord,
        usage: agent6_budget.UsageSettlement,
    ) -> None:
        """Bank the balances from the execution that started or continued this child.

        Args:
            meter: The resumed parent's meter.
            record: The dispatch that owns the child.
            child: The child being reconciled.
            usage: Its cumulative usage.

        Raises:
            DispatchRecordError: No recorded execution owns the new credit readings.
        """
        continuation = child.continuation
        if continuation and not set(usage.credit_readings).difference(
            continuation.usage_before.credit_readings
        ):
            return
        execution = self.executions.get(continuation.owner if continuation else record.owner)
        if execution is None and continuation:
            execution = self.continuations.get(
                (record.identity, child.session_id, continuation.sequence)
            )
        if execution is None:
            raise child_dispatch.DispatchRecordError(
                f"cannot settle credit readings for {child.account_id}: "
                "the parent journal does not identify the execution that launched it"
            )
        meter.restore_child_credits(child.account_id, execution, self.readings.get(execution, {}))


def reconcile(  # noqa: PLR0912  # each durable dispatch state has its recovery path
    dispatcher: parallel_dispatch.ParallelDispatcher,
    state: loop_state.LoopState,
    conversation: loop_conversation.Conversation,
    *,
    turn: int,
) -> None:
    """Settle or diagnose every dispatch the prior execution left behind, once each.

    Runs before a resumed execution's first provider call. A complete record whose
    result never reached the conversation is delivered; an interrupted one has its
    children found by the recoverer (live ones stopped, finished ones imported) and its
    usage settled. A task is then settled as a turn boundary settles it: a stopped child is
    told with the branch that holds its work and never merges, and a task whose children
    ended by themselves merges, as at the boundary its execution did not reach.

    Args:
        dispatcher: The resumed execution's dispatcher.
        state: The loop state.
        conversation: The conversation the notices enter.
        turn: The last turn the prior execution completed.
    """
    if dispatcher.session_dir is None:
        return
    logs = dispatcher.session_dir / "logs.jsonl"
    history = None
    for found in child_dispatch.read_records(dispatcher.session_dir):
        record = found
        scan = session_log.scan_session_log(logs)
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
                dispatcher.deliver(
                    conversation,
                    child_records.DispatchResult(
                        record.identity, True, (), record.message, base=record.base
                    ),
                )
            continue
        dispatcher.log(
            f"PARALLEL: reconciling the interrupted dispatch {record.identity} ({record.state})"
        )
        children = [c.session_id for c in record.children]
        if dispatcher.budget is not None:
            # The journal already holds these children's usage: charged once, by that execution.
            dispatcher.budget.mark_settled(
                c.account_id for c in record.children if c.account_id in accounted
            )
            for child in record.children:
                if child.continuation is not None:
                    prior = child.continuation.usage_before
                    dispatcher.budget.child_execution(
                        child.session_id,
                        child.account_id,
                        agent6_budget.UsageSettlement(
                            usd=prior.usd,
                            usd_observed_credits=prior.usd_observed_credits,
                            credit_readings=prior.credit_readings,
                            usd_partial=prior.usd_partial,
                            tokens=prior.tokens,
                            model_tokens=prior.model_tokens,
                        ),
                    )
        results = (
            dispatcher.child_recoverer(record.identity, children)
            if dispatcher.child_recoverer is not None
            else []
        )
        if dispatcher.budget is not None:
            for result in results:
                child = record.child(result.spec.session_id)
                if (
                    child.account_id not in accounted
                    and result.usage
                    and result.usage.credit_readings
                ):
                    history = history or _CreditHistory.read(logs)
                    history.restore(dispatcher.budget, record, child, result.usage)
        if (
            record.state in ("prepared", "running")
            and dispatcher.child_recoverer is not None
            and not results
        ):
            if dispatcher.budget is not None:
                # Nothing ran, so nothing was used; the journal names each child accounted.
                nothing = agent6_budget.UsageSettlement(
                    usd=0, usd_partial=False, tokens=0, model_tokens=token_usage.ZERO
                )
                for child in children:
                    dispatcher.budget.settle_child(child, nothing)
            message = (
                f"[parallel] group {record.identity}: dispatch interrupted before launch;"
                " nothing ran."
            )
            record = dispatcher.fail_nodes(record, "dispatch interrupted before launch")
            record = record.model_copy(
                update={
                    "children": tuple(
                        child.model_copy(
                            update={
                                "settled": True,
                                "usage": child_dispatch.UsageRecord(
                                    usd=0.0, model_tokens=token_usage.ZERO
                                ),
                            }
                        )
                        for child in record.children
                    ),
                }
            )
            dispatcher.emit_budget(record.turn)
            dispatcher.record_return(record, message)
            dispatcher.deliver(
                conversation,
                child_records.DispatchResult(record.identity, True, (), message, base=record.base),
            )
            continue
        if record.source == "panel":
            completed = dispatcher.complete(
                dispatcher.reported(conversation, record, results), results
            )
            # A task's judges are written delivered: their result is the task's own notice.
            if not record.delivered:
                dispatcher.deliver(conversation, completed)
            continue
        dispatcher.open[record.identity] = reopened(record, results)
    dispatcher.settle(conversation, turn)
    stopped = [
        f"  - {child.session_id}: {parallel_merge.merge_words(parallel_merge.merge_of(child))}"
        for record in child_dispatch.read_records(dispatcher.session_dir)
        for child in record.children
        if record.source != "panel" and child.merge is not None and child.merge.status == "stopped"
    ]
    if stopped:
        conversation.notice("[parallel] stopped children:\n" + "\n".join(stopped))
