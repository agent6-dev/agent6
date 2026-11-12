# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Every session's children: its lanes by lineage and what its dispatch records say of them.

The listing, the session header and the receipt read children from here. A `/parallel`
dispatch leaves a record under the parent's session dir; `run --parallel` and older
sessions leave lanes whose manifests name their parent, and nothing else.
"""

from __future__ import annotations

import dataclasses
import pathlib

from agent6.sessions import lane_dispatch, layout
from agent6.sessions import manifest as sessions_manifest
from agent6.viewmodel import listing


@dataclasses.dataclass(frozen=True, slots=True)
class ChildView:
    """One lane of a session, as the surfaces show it.

    Attributes:
        session_id: The lane's session id.
        lane: Its number in its group.
        group: The group it was dispatched in.
        source: Who dispatched it: operator, model or panel; "" when no record says.
        task: Its task text; "" when unknown.
        model: Its selector, None for the worker's own settings.
        route: The `provider/model` its manifest records; "" when it never started.
        status: Its own status word; "" when it never started.
        outcome: How it joined: `joined`, `no_changes`, `conflict` or `failed`; "" when no
            join is recorded, as for a group's lanes.
        join_sha: The merge commit, "" when none.
        imported_ref: The branch the join read, "" when none.
        error: The join's detail, "" when none.
        spend_usd: What its parent settled for it; the lane's own cost when no record says.
        usd_partial: Whether unpriced calls make the spend a lower bound.
        settled: Whether the parent charged its usage; False while it is unknown.
        launched: Whether a session dir of its own exists.
    """

    session_id: str
    lane: int
    group: str
    source: str
    task: str
    model: str | None
    route: str
    status: str
    outcome: str
    join_sha: str
    imported_ref: str
    error: str
    spend_usd: float
    usd_partial: bool
    settled: bool
    launched: bool


@dataclasses.dataclass(frozen=True, slots=True)
class DispatchView:
    """One dispatch record, as the surfaces show it.

    Attributes:
        identity: The group id.
        source: Who dispatched it.
        state: The record's state, or `unreadable`.
        turn: The turn it was dispatched after.
        base: The chain tip the lanes were cut from.
        message: The delivered result, or why the record could not be read.
        children: Its lanes, in lane order.
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
    directory = lane_dispatch.records_dir(session_dir)
    if not directory.is_dir():
        return ()
    state_dir = layout.layout_of(session_dir).state_dir
    views: list[DispatchView] = []
    for path in sorted(directory.glob("*.json"), key=lambda p: lane_dispatch.order_key(p.stem)):
        try:
            record = lane_dispatch.read_record(path)
        except lane_dispatch.DispatchRecordError as exc:
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


def children_of(session_dir: pathlib.Path) -> tuple[ChildView, ...]:
    """Return a session's children: from its records, else from its lanes' lineage.

    Args:
        session_dir: The session's dir.

    Returns:
        The children in dispatch then lane order; none for a session with neither.
    """
    recorded = tuple(
        child
        for view in dispatches_of(session_dir)
        if view.state != "unreadable"
        for child in view.children
    )
    if recorded:
        return recorded
    state_dir = layout.layout_of(session_dir).state_dir
    runs = layout.bucket_dir(state_dir, "runs")
    return tuple(
        _lineage_child(summary, runs / summary.session_id)
        for summary in listing.lanes_of(state_dir, session_dir.name)
    )


def _recorded_child(
    record: lane_dispatch.DispatchRecord, child: lane_dispatch.ChildRecord, bucket: pathlib.Path
) -> ChildView:
    lane_dir = bucket / child.session_id
    launched = lane_dir.exists()
    status = route = ""
    if launched:
        summary = listing.summarize_session_dir(lane_dir)
        status, route = summary.status, summary.model
    join = child.join
    usage = child.usage
    return ChildView(
        session_id=child.session_id,
        lane=child.lane,
        group=record.identity,
        source=record.source,
        task=child.task,
        model=child.model,
        route=route,
        status=status,
        outcome=join.status if join is not None else "",
        join_sha=join.sha if join is not None else "",
        imported_ref=join.branch if join is not None else "",
        error=join.detail if join is not None else "",
        spend_usd=usage.usd if usage is not None else 0.0,
        usd_partial=usage.usd_partial if usage is not None else False,
        settled=child.settled,
        launched=launched,
    )


def _lineage_child(summary: listing.SessionSummary, lane_dir: pathlib.Path) -> ChildView:
    group, task = summary.parent, ""
    try:
        manifest = sessions_manifest.read_manifest(lane_dir)
    except sessions_manifest.ManifestError:
        manifest = None
    if manifest is not None:
        task = manifest.user_task
        if manifest.parallel is not None and manifest.parallel.group:
            group = manifest.parallel.group
    return ChildView(
        session_id=summary.session_id,
        lane=summary.lane or 0,
        group=group,
        source="",
        task=task,
        model=None,
        route=summary.model,
        status=summary.status,
        outcome="",
        join_sha="",
        imported_ref="",
        error="",
        spend_usd=summary.cost_usd,
        usd_partial=summary.usd_partial,
        settled=False,
        launched=True,
    )
