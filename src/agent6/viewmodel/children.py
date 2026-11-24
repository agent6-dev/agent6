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

from agent6 import child_records, llm_json
from agent6.sessions import child_dispatch, completion, layout, lineage, process
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
        status: The child's own session or launch status. Read `outcome` for a parent merge.
        status_label: One line with the child's status and reason, then any parent outcome.
        outcome: The recorded dispatch outcome, including `stopped`, `ranked` and `retained`.
            An empty string means the parent has recorded no outcome.
        parent_merge_sha: The merge commit on the parent's chain, "" when none.
        merged_into_parent: The parent session whose chain received the merge, "" without a merge.
        imported_ref: The branch the merge read, "" when none.
        error: A launch, verdict or merge diagnostic, or an empty string.
        spend_usd: The cumulative child spend.
        usd_partial: Whether unpriced calls make the spend a lower bound.
        settled: Whether the parent charged its usage; False while it is unknown.
        launched: Whether the child has a session to inspect, including a live preflight worker.
        current_tool: The tool running now, or an empty string.
        turn: Its latest recorded turn, or None before one.
        tokens: Input, cache and output tokens in its current execution, or None if unreported.
        elapsed_s: Seconds since the child started, frozen at its last event when it ends.
        commits: Its measured commits beyond its start, None when unknown.
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
    status_label: str = "not started"
    outcome: str = ""
    parent_merge_sha: str = ""
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
    commits: int | None = None
    verdict_order: tuple[str, ...] = ()
    verdict_summary: str = ""
    merged_into_parent: str = ""

    def with_status(self, word: str, reason: format.StatusReason = "") -> ChildView:
        """Return the row with its session or launch status and the reason in words."""
        return dataclasses.replace(
            self, status=word, status_label=" ".join(format.status_label(word, reason).split())
        )

    @property
    def number_label(self) -> str:
        """Return the attempt or reviewer number, independent of the model selector."""
        return child_records.number_label(self.number, reviewer=self.source == "panel")

    @property
    def commit_count_text(self) -> str:
        """Return the measured count for the roster, or no label without a measurement."""
        if self.commits is None:
            return ""
        return f"{self.commits} commit{'' if self.commits == 1 else 's'} beyond start"

    @property
    def group_label(self) -> str:
        """Name the attempt or reviewer in the group that numbered it."""
        return f"{self.group} {self.number_label}"


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
                # Panel children are recorded in the reviewer bucket.
                children=tuple(
                    _recorded_child(
                        record,
                        child,
                        session_dir.name,
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
    children = (
        _session_child(ChildView(directory.name, 0, session_dir.name, "", "", None), directory)
        for directory in lineage.children_of(session_dir)
    )
    return tuple(sorted(children, key=lambda child: (child.group, child.number, child.session_id)))


def reported_start(child: child_dispatch.ChildRecord, scan: session_log.LogScan) -> bool:
    """Return whether this execution has started, ignoring any earlier continuation."""
    previous = child.continuation.sequence if child.continuation is not None else 0
    return scan.saw_start and scan.executions > previous


def _session_child(
    row: ChildView,
    child_dir: pathlib.Path,
    child: child_dispatch.ChildRecord | None = None,
    *,
    version: int = child_dispatch.RECORD_VERSION,
) -> ChildView:
    commits = (
        None
        if row.source == "panel"
        else child_dispatch.read_commit_count(
            child_dir, child.merge if child else None, version=version
        )
    )
    if listing.is_session_husk(child_dir):
        return dataclasses.replace(row, commits=commits)
    scan = session_log.scan_session_log(child_dir / layout.LOGS_NAME)
    if child is not None and not reported_start(child, scan):
        return dataclasses.replace(
            row.with_status("starting"),
            launched=process.worker_is_alive(child_dir),
            commits=commits,
        )
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
    if not row.source and manifest is not None and manifest.parallel is not None:
        row = dataclasses.replace(
            row,
            group=manifest.parallel.group or row.group,
            number=manifest.parallel.number,
            source="panel" if child_dir.parent.name == layout.EVIDENCE_BUCKET else "",
            task=scan.task_label.task or manifest.user_task,
        )
    row = dataclasses.replace(
        row.with_status(status, reason),
        launched=True,
        commits=commits,
        route=format.format_model_route(manifest.models.driver) if manifest is not None else "",
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
    if child is not None and row.source == "panel":
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
    return row


def _recorded_child(
    record: child_dispatch.DispatchRecord,
    child: child_dispatch.ChildRecord,
    parent: str,
    bucket: pathlib.Path,
) -> ChildView:
    row = ChildView(
        child.session_id, child.number, record.identity, record.source, child.task, child.model
    )
    if not layout.is_safe_session_id(child.session_id):
        return dataclasses.replace(row.with_status("failed"), error="invalid child session id")
    child_dir = bucket / child.session_id
    row = _session_child(row, child_dir, child, version=record.version)
    if child.usage is not None:
        row = dataclasses.replace(
            row,
            spend_usd=child.usage.usd,
            usd_partial=child.usage.usd_partial,
            settled=child.settled,
        )
    merge = child.merge
    if merge is None:
        if record.state == "complete" and not row.launched:
            return dataclasses.replace(
                row.with_status("failed"), error=record.message or "never started"
            )
        # A panel announces later batches before submitting them.
        if record.state == "running" and not row.launched and record.source != "panel":
            return row.with_status("unreported" if child.reported else "starting")
        return row
    if not row.launched:
        # An imported ref says nothing about the missing session's gate.
        row = row.with_status("failed" if merge.status == "failed" else "unreported")
    label = row.status_label
    if merge.status != row.status:
        outcome = (
            format.parent_merge_label(parent, merge.sha)
            if merge.status == "merged"
            else merge.status.replace("_", " ")
        )
        label += f" · {outcome}"
    error = row.error
    if merge.detail:
        if merge.status == "ranked":
            label += f" {merge.detail}"
        elif merge.status == "retained":
            label += f" (ranked {merge.detail})"
        elif merge.status in ("conflict", "failed"):
            error = merge.detail
        else:
            label += f": {merge.detail}"
    return dataclasses.replace(
        row,
        status_label=" ".join(label.split()),
        outcome=merge.status,
        imported_ref=merge.branch,
        parent_merge_sha=merge.sha,
        merged_into_parent=parent if merge.status == "merged" else "",
        error=error,
        current_tool="",
    )
