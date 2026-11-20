# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Every session's children: its children by lineage and what its dispatch records say of them.

The dispatch record names the children even when one never started. Older runs without
a dispatch record find their children by the parent named in each child's manifest.
"""

from __future__ import annotations

import dataclasses
import pathlib
import time

from agent6 import llm_json
from agent6.sessions import child_dispatch, completion, layout
from agent6.sessions import manifest as sessions_manifest
from agent6.viewmodel import format, listing
from agent6.viewmodel.session import log as session_log
from agent6.viewmodel.session import status as session_status


@dataclasses.dataclass(frozen=True, slots=True)
class ChildView:
    """One child of a session, as the surfaces show it.

    Attributes:
        session_id: The child's session id.
        number: Its number in its group.
        group: The group it was dispatched in.
        source: Who dispatched it: operator, model or panel; "" when no record says.
        task: Its task text; "" when unknown.
        model: Its selector, None for the worker's own settings.
        route: The `provider/model` its manifest records; "" when it never started.
        status: Where the child stands in its task, including a merge or a ranking.
        outcome: The recorded merge outcome, including `stopped`, `ranked` and `unmerged`.
            An empty string means the parent has recorded no outcome.
        join_sha: The merge commit, "" when none.
        imported_ref: The branch the join read, "" when none.
        error: The status detail or the reason a child failed to start or merge.
        spend_usd: The cumulative child spend.
        usd_partial: Whether unpriced calls make the spend a lower bound.
        settled: Whether the parent charged its usage; False while it is unknown.
        launched: Whether the child has a session record or a live worker.
        current_tool: The tool running now, or an empty string.
        turn: Its latest recorded turn, or None before one.
        tokens: Input, cache and output tokens in its current execution, or None if unreported.
        elapsed_s: Seconds since the child started, frozen at its last event when it ends.
        commits: The commits its imported ref holds beyond the task's base, when recorded.
        verdict_order: A reviewer's order of attempts, best first.
        verdict_summary: The summary from the reviewer's verdict.
    """

    session_id: str
    number: int
    group: str
    source: str
    task: str
    model: str | None
    route: str = ""
    status: str = "not started"
    outcome: str = ""
    join_sha: str = ""
    imported_ref: str = ""
    error: str = ""
    spend_usd: float = 0.0
    usd_partial: bool = False
    settled: bool = False
    launched: bool = False
    current_tool: str = ""
    turn: int | None = None
    tokens: int | None = None
    elapsed_s: int | None = None
    commits: int = 0
    verdict_order: tuple[str, ...] = ()
    verdict_summary: str = ""


@dataclasses.dataclass(frozen=True, slots=True)
class DispatchView:
    """One dispatch record, as the surfaces show it.

    Attributes:
        identity: The group id.
        source: Who dispatched it.
        state: The record's state, or `unreadable`.
        turn: The turn it was dispatched after.
        base: The chain tip the children were cut from.
        message: The delivered result, or why the record could not be read.
        children: Its children, in child order.
    """

    identity: str
    source: str
    state: str
    turn: int
    base: str
    message: str
    children: tuple[ChildView, ...]


def dispatches_of(session_dir: pathlib.Path) -> tuple[DispatchView, ...]:
    """Return a session's dispatch records in dispatch order; an unreadable one says so.

    Args:
        session_dir: The session's dir.

    Returns:
        The views; none for a session without records.
    """
    directory = child_dispatch.records_dir(session_dir)
    if not directory.is_dir():
        return ()
    state_dir = layout.layout_of(session_dir).state_dir
    views: list[DispatchView] = []
    for path in sorted(directory.glob("*.json"), key=lambda p: child_dispatch.order_key(p.stem)):
        try:
            record = child_dispatch.read_record(path)
        except child_dispatch.DispatchRecordError as exc:
            views.append(DispatchView(path.stem, "", "unreadable", 0, "", str(exc), ()))
            continue
        views.append(
            DispatchView(
                identity=record.identity,
                source=record.source,
                state=record.state,
                turn=record.turn,
                base=record.base,
                message=record.message,
                # A panel's children are read-only conversations, in that kind's bucket.
                children=tuple(
                    _recorded_child(
                        record,
                        child,
                        layout.bucket_dir(
                            state_dir,
                            layout.EVIDENCE_BUCKET if record.source == "panel" else "runs",
                        ),
                    )
                    for child in record.children
                ),
            )
        )
    return tuple(views)


def children_of(
    session_dir: pathlib.Path, dispatches: tuple[DispatchView, ...] | None = None
) -> tuple[ChildView, ...]:
    """Return a session's children: from its records, else from its children's lineage.

    Args:
        session_dir: The session's dir.
        dispatches: Its dispatches when already read.

    Returns:
        The children in dispatch then child order; none for a session with neither.
    """
    records = dispatches_of(session_dir) if dispatches is None else dispatches
    recorded = tuple(child for view in records for child in view.children)
    if recorded:
        return recorded
    state_dir = layout.layout_of(session_dir).state_dir
    runs = layout.bucket_dir(state_dir, "runs")
    return tuple(
        _session_child(
            ChildView(
                summary.session_id,
                summary.number or 0,
                summary.parent,
                "",
                summary.task_label.task,
                None,
            ),
            runs / summary.session_id,
        )
        for summary in listing.children_of(state_dir, session_dir.name)
    )


def _session_child(row: ChildView, child_dir: pathlib.Path) -> ChildView:
    if listing.is_session_husk(child_dir):
        return row
    scan = session_log.scan_session_log(child_dir / layout.LOGS_NAME)
    status, reason = session_status.status_for_session_dir(child_dir, scan.status_facts())
    live = status in session_status.LIVE_STATUS_WORDS
    end = time.time() if live else scan.end_ep or scan.last_ep
    usage = scan.budget.execution
    counts = (
        usage.input_tokens,
        usage.output_tokens,
        usage.cache_read_tokens,
        usage.cache_write_tokens,
    )
    try:
        manifest = sessions_manifest.read_manifest(child_dir)
    except sessions_manifest.ManifestError:
        manifest = None
    group = row.group
    if not row.source and manifest is not None and manifest.parallel is not None:
        group = manifest.parallel.group or group
    return dataclasses.replace(
        row,
        launched=True,
        group=group,
        route=format.format_model_route(manifest.models.driver) if manifest is not None else "",
        status="finished"
        if scan.finished and status in ("passed", "answered", "planned")
        else status,
        error=reason,
        current_tool=scan.current_tool if live else "",
        turn=scan.turn,
        elapsed_s=max(0, int(end - scan.start_ep))
        if end is not None and scan.start_ep is not None
        else None,
        tokens=sum(n for n in counts if n is not None)
        if any(n is not None for n in counts)
        else None,
        spend_usd=scan.budget.usd_so_far,
        usd_partial=scan.budget.session.usd_partial,
    )


def _recorded_child(
    record: child_dispatch.DispatchRecord, child: child_dispatch.ChildRecord, bucket: pathlib.Path
) -> ChildView:
    row = ChildView(
        child.session_id, child.number, record.identity, record.source, child.task, child.model
    )
    if not layout.is_safe_session_id(child.session_id):
        return dataclasses.replace(row, status="failed", error="invalid child session id")
    child_dir = bucket / child.session_id
    row = _session_child(row, child_dir)
    if record.source == "panel" and row.launched:
        try:
            verdict = completion.read_completion(child_dir)
        except completion.CompletionError as exc:
            row = dataclasses.replace(row, error=str(exc))
        else:
            obj = (
                llm_json.extract_json(verdict.text, prefer=("order", "verdict"))
                if verdict
                else None
            )
            if obj is not None:
                order = obj.get("order")
                row = dataclasses.replace(
                    row,
                    verdict_order=tuple(order)
                    if isinstance(order, list) and all(isinstance(s, str) for s in order)
                    else (),
                    verdict_summary=str(obj.get("summary") or ""),
                )
    if child.usage is not None:
        row = dataclasses.replace(
            row,
            spend_usd=child.usage.usd,
            usd_partial=child.usage.usd_partial,
            settled=child.settled,
        )
    join = child.join
    if join is not None:
        status = {
            "joined": "merged",
            "conflict": "failed",
            "failed": "failed",
            "stopped": "stopped",
        }.get(join.status)
        if not status:
            status = row.status if row.status in ("failed", "stopped") else "finished"
        return dataclasses.replace(
            row,
            status=status,
            outcome=join.status,
            imported_ref=join.branch,
            join_sha=join.sha,
            commits=join.commits,
            error=join.detail or row.error,
            current_tool="",
        )
    if record.state == "ranking" and row.status not in ("failed", "stopped"):
        return dataclasses.replace(row, status="being ranked")
    if record.state == "complete" and not row.launched:
        return dataclasses.replace(row, status="failed", error=record.message or "never started")
    return row
