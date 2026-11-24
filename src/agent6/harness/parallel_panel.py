# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Run a panel's reviewers in batches through a run's dispatcher, and record them.

A panel is record-only: its reviewers run on a retained target, block their caller until the
last batch returns, and merge nothing.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from agent6 import child_records
from agent6.harness import child_repo, parallel_merge
from agent6.sessions import child_dispatch

if TYPE_CHECKING:
    from agent6.harness import parallel_dispatch

# The window one batch of a panel's reviewers runs in; a work child has no deadline of its own.
PANEL_BATCH_DEADLINE_S = 1800.0


def dispatch_panel(  # noqa: PLR0911  # one return per refusal
    dispatcher: parallel_dispatch.ParallelDispatcher,
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
        dispatcher: The run's dispatcher, whose spawner, budget and record dir the panel uses.
        children: The complete roster of bounded conversations with resolved reviewer settings.
        panel_id: The harness's stable panel/batch id.
        at: The retained immutable target commit.
        turn: The owning worker's completed turn.
        concurrency: The policy's concurrency cap; None uses harness.parallel.max_children.

    Returns:
        The reviewer outcomes, or a refusal before launch.
    """
    if not dispatcher.authority.panels or dispatcher.child_spawner is None:
        return refusal(dispatcher, "harness panel scheduling is unavailable in this execution.")
    if (
        dispatcher.session_dir is None
        or not at
        or not panel_id
        or not panel_id.replace("-", "").isalnum()
    ):
        return refusal(
            dispatcher, "a panel needs a session dir, retained target and safe panel id."
        )
    if not children or any(child.contract is None for child in children):
        return refusal(dispatcher, "a panel dispatches reviewers only.")
    group = f"r{panel_id}"
    identity = f"{dispatcher.session_id}-{group}"
    if child_dispatch.record_path(dispatcher.session_dir, identity).exists():
        return refusal(
            dispatcher, "the panel batch already has a dispatch record; reconcile it instead."
        )
    width = max(1, min(dispatcher.max_children, concurrency or dispatcher.max_children))
    child_ids = [
        f"{identity}-l{i + 1}"
        if len(children) <= width
        else f"{identity}-b{i // width + 1}-l{i % width + 1}"
        for i in range(len(children))
    ]
    if dispatcher.budget is not None and dispatcher.budget.is_exhausted():
        reason = dispatcher.budget.snapshot().exhausted_reason
        return refusal(dispatcher, f"the panel budget is exhausted ({reason}); nothing dispatched.")
    bounded: list[child_records.ChildTask] = []
    start = dispatcher.clock()
    for i, child in enumerate(children):
        assert child.contract is not None
        # Batches run one after another: each gets its own window, fixed before launch.
        deadlines = [start + PANEL_BATCH_DEADLINE_S * (i // width + 1)]
        if dispatcher.authority.deadline_at is not None:
            deadlines.append(dispatcher.authority.deadline_at)
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
        merge_policy="record_only",
        owner=dispatcher.execution_id,
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
        dispatcher.write(record)
    except OSError as exc:
        return refusal(
            dispatcher, f"could not persist the panel dispatch ({exc}); nothing dispatched."
        )
    return execute(dispatcher, record, bounded, group, batch_width=width)


def execute(
    dispatcher: parallel_dispatch.ParallelDispatcher,
    record: child_dispatch.DispatchRecord,
    children: list[child_records.ChildTask],
    group: str,
    *,
    batch_width: int,
) -> child_records.DispatchResult:
    """Run a recorded panel in batches, waiting for each, then settle and record it.

    Args:
        dispatcher: The run's dispatcher.
        record: The durable prepared dispatch.
        children: Its bounded tasks, in child order.
        group: The group suffix used by the shared spawner.
        batch_width: The maximum children submitted together.

    Returns:
        The completed result, or a diagnosed spawner failure with unresolved holds.
    """  # noqa: DOC501  # the spawner's ChildRepoError is caught below
    assert dispatcher.child_spawner is not None
    record = dispatcher.announce(record, children)
    submitted: list[child_records.StartedChild] = []
    try:
        for offset in range(0, len(children), batch_width):
            # Each batch starts under what the run has left then; its children share it.
            ceiling = dispatcher.budget.available() if dispatcher.budget is not None else None
            batch = [
                dataclasses.replace(child, allowance=ceiling)
                for child in children[offset : offset + batch_width]
            ]
            batch_group = (
                group if len(children) <= batch_width else f"{group}-b{offset // batch_width + 1}"
            )
            started = dispatcher.child_spawner(batch, batch_group, at=record.base or None)
            submitted.extend(started)
            for child in started:
                child.result.result()
        if len(submitted) != len(children):
            raise child_repo.ChildRepoError(
                f"group spawner returned {len(submitted)} result(s) for {len(children)} child(ren)"
            )
    except Exception as exc:
        # A submitted reviewer without a result stays unsettled.
        for child in submitted:
            if not child.result.done():
                child.stop()
        results = [
            c.result.result()
            for c in submitted
            if c.result.done() and not c.result.cancelled() and c.result.exception() is None
        ]
        failed = dispatcher.failed_launch(
            record, exc, submitted={c.session_id for c in submitted}, reviewers=results
        )
        outcomes = tuple(parallel_merge.merge_of(c) if c.merge else None for c in failed.children)
        return child_records.DispatchResult(
            failed.identity, True, outcomes, failed.message, base=failed.base
        )
    return dispatcher.complete(record, [c.result.result() for c in submitted])


def refusal(
    dispatcher: parallel_dispatch.ParallelDispatcher, msg: str
) -> child_records.DispatchResult:
    """Return the result of a panel refused before any reviewer ran, logged once."""
    dispatcher.log(f"PARALLEL: {msg}")
    return child_records.DispatchResult("", False, (), f"[parallel] {msg}")
