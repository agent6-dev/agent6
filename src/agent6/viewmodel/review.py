"""Fold review evidence and operator history for every session reader.

`FindingView.line_origin` reads the latest observation; `as_dict` includes it in the wire form.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any, Literal

import pydantic

from agent6 import review_records

Completion = Literal["unknown", "running", "complete", "inconclusive", "skipped"]
Freshness = Literal["unknown", "current", "stale"]


class ReviewReadError(Exception):
    """Structured review evidence contradicts the session's recorded state."""


class ActionRecord(pydantic.BaseModel):
    """Validate the journal fields retained in a finding's action history."""

    model_config = pydantic.ConfigDict(frozen=True)

    type: str
    finding_id: review_records.Identifier
    request_id: review_records.Identifier
    source: Literal["cli", "tui", "web", "acp"] | None = None
    version: int | None = pydantic.Field(default=None, ge=1, strict=True)
    resolution: review_records.FindingResolution | None = None
    reason: review_records.Reason | None = None
    target: review_records.ReviewTarget | None = None
    action: str | None = None
    execution_id: str | None = None
    error: str | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class FindingView:
    """Keep a finding's observations and operator history independently of panel verdicts."""

    finding_id: str
    observations: tuple[review_records.FindingObservation, ...]
    status: Literal["open", "resolved"] = "open"
    resolution: review_records.FindingResolution | None = None
    actions: tuple[ActionRecord, ...] = ()
    anchor_state: Literal["unknown", "current", "outdated"] = "unknown"
    fix_state: Literal["none", "delivery_unknown", "submitted", "failed"] = "none"

    @property
    def line_origin(self) -> review_records.LineOrigin:
        """Return the latest observation's evidence without discarding earlier attribution."""
        return self.observations[-1].line_origin


@dataclasses.dataclass(frozen=True, slots=True)
class SeatAssignment:
    """Identify the resolved route and settings of one panel seat."""

    seat_index: int
    model_route: str
    focus: str
    scope: Literal["diff", "repo"] = "diff"
    temperature: float | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class PanelView:
    """Keep a panel's roster, grounded verdicts and completion evidence."""

    panel_id: str
    execution_id: str
    target: review_records.ReviewTarget | None = None
    roster: tuple[SeatAssignment, ...] = ()
    seats: tuple[review_records.ReviewVerdict, ...] = ()
    result: review_records.ReviewPanel | None = None
    completion: Completion = "running"
    reason: str | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class SeverityCounts:
    """Count open findings by their grounded severity."""

    block: int = 0
    warn: int = 0
    nit: int = 0


@dataclasses.dataclass(frozen=True, slots=True)
class MergeReviewGuard:
    """Report open grounded blocks and unreadable evidence without enforcing a merge."""

    open_block_ids: tuple[str, ...] = ()
    read_failure: str | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class ReviewView:
    """Fold durable review evidence in journal append order."""

    panels: tuple[PanelView, ...] = ()
    findings: tuple[FindingView, ...] = ()
    version: int = 0
    open_counts: SeverityCounts = dataclasses.field(default_factory=SeverityCounts)
    completion: Completion = "unknown"
    freshness: Freshness = "unknown"
    legacy_notice: str = ""
    read_failure: str | None = None
    comparison: review_records.ReviewTarget | None = None
    event_ids: tuple[str, ...] = ()

    @property
    def merge_guard(self) -> MergeReviewGuard:
        """Return the merge facts without treating missing or stale review as a block."""
        return MergeReviewGuard(
            tuple(
                f.finding_id
                for f in self.findings
                if f.status == "open" and f.observations[-1].finding.severity == "block"
            ),
            self.read_failure,
        )

    def for_comparison(self, target: review_records.ReviewTarget) -> ReviewView:
        """Return freshness and anchor states for the requested trees and path coverage."""
        freshness: Freshness = "unknown"
        if self.panels and self.completion == "complete" and not self.read_failure:
            last = self.panels[-1]
            if last.target is not None and last.completion == "complete":
                covers = not last.target.paths or (
                    bool(target.paths) and set(target.paths) <= set(last.target.paths)
                )
                same = (
                    last.target.base_tree == target.base_tree
                    and last.target.head_tree == target.head_tree
                    and covers
                )
                freshness = "current" if same else "stale"
        panels = {p.panel_id: p for p in self.panels}
        findings: list[FindingView] = []
        for finding in self.findings:
            observation = finding.observations[-1]
            reviewed = panels[observation.panel_id].target
            anchor: Literal["unknown", "current", "outdated"] = "unknown"
            if reviewed is not None:
                same_tree = (
                    reviewed.base_tree == target.base_tree
                    if observation.finding.location.side == "base"
                    else reviewed.head_tree == target.head_tree
                )
                anchor = "current" if same_tree else "outdated"
            findings.append(dataclasses.replace(finding, anchor_state=anchor))
        return dataclasses.replace(
            self, comparison=target, freshness=freshness, findings=tuple(findings)
        )


class _Start(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(frozen=True)

    panel_id: review_records.Identifier
    execution_id: review_records.Identifier
    target: review_records.ReviewTarget
    seats: tuple[SeatAssignment, ...]


class _Skipped(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(frozen=True)

    panel_id: review_records.Identifier
    execution_id: review_records.Identifier
    reason: review_records.Identifier


_PANEL_EVENTS = frozenset(
    {"loop.review.start", "loop.review.seat", "loop.review.panel", "loop.review.skipped"}
)
_ACTION_EVENTS = frozenset(
    {
        "loop.review.finding.resolved",
        "loop.review.finding.reopened",
        "loop.review.fix.requested",
        "loop.review.fix.submitted",
        "loop.review.fix.failed",
    }
)


def _replace_panel(view: ReviewView, panel: PanelView) -> ReviewView:
    old = next((p for p in view.panels if p.panel_id == panel.panel_id), None)
    if old is not None and (
        old.execution_id != panel.execution_id
        or (old.target is not None and old.target != panel.target)
    ):
        raise ReviewReadError("panel identity or target changed")
    panels = tuple(panel if p.panel_id == panel.panel_id else p for p in view.panels)
    if old is None:
        panels += (panel,)
    return dataclasses.replace(view, panels=panels, completion=panel.completion)


def _observe(view: ReviewView, record: review_records.ReviewPanel) -> ReviewView:
    findings = {f.finding_id: f for f in view.findings}
    ids: set[str] = set()
    for observation in record.findings:
        if observation.panel_id != record.panel_id or observation.finding_id in ids:
            raise ReviewReadError("finding observation has a conflicting panel or duplicate ID")
        ids.add(observation.finding_id)
        prior = findings.get(observation.finding_id)
        if prior is None:
            findings[observation.finding_id] = FindingView(observation.finding_id, (observation,))
        else:
            if review_records.finding_key(
                prior.observations[-1].finding
            ) != review_records.finding_key(observation.finding):
                raise ReviewReadError("finding ID has conflicting content or location")
            findings[observation.finding_id] = dataclasses.replace(
                prior, observations=(*prior.observations, observation)
            )
    return dataclasses.replace(view, findings=tuple(findings.values()))


def _panel_event(view: ReviewView, event: Mapping[str, Any]) -> ReviewView:
    kind = event["type"]
    old = next((p for p in view.panels if p.panel_id == event.get("panel_id")), None)
    if kind == "loop.review.start":
        start = _Start.model_validate(event)
        panel = PanelView(start.panel_id, start.execution_id, start.target, start.seats)
    elif kind == "loop.review.skipped":
        skipped = _Skipped.model_validate(event)
        panel = dataclasses.replace(
            old or PanelView(skipped.panel_id, skipped.execution_id),
            completion="skipped",
            reason=skipped.reason,
        )
    elif kind == "loop.review.seat":
        if old is None:
            raise ReviewReadError("seat has no panel start")
        verdict = pydantic.TypeAdapter(review_records.ReviewVerdict).validate_python(event)
        if any(v.seat_index == verdict.seat_index for v in old.seats):
            raise ReviewReadError("panel repeats a seat index")
        panel = dataclasses.replace(old, seats=(*old.seats, verdict))
    else:
        record = review_records.ReviewPanel.model_validate(event)
        if old is not None and old.result is not None:
            raise ReviewReadError("panel has more than one completion record")
        view = _observe(view, record)
        panel = dataclasses.replace(
            old or PanelView(record.panel_id, record.execution_id, record.target),
            result=record,
            completion=record.completion,
        )
        if panel.target != record.target or panel.execution_id != record.execution_id:
            raise ReviewReadError("panel completion differs from its start")
    if panel.target is not None and not (panel.target.base_tree and panel.target.head_tree):
        raise ReviewReadError("review target is missing a tree ID")
    return _replace_panel(view, panel)


def _transition(
    view: ReviewView, finding: FindingView, action: ActionRecord, kind: str
) -> FindingView:
    if kind == "resolved":
        resolution = action.resolution
        if resolution is None:
            raise ReviewReadError("resolve is missing its resolution and reason")
        if resolution.disposition == "duplicate" and (
            resolution.duplicate_of == finding.finding_id
            or resolution.duplicate_of not in {f.finding_id for f in view.findings}
        ):
            raise ReviewReadError("duplicate resolution must name another finding in this session")
        finding = dataclasses.replace(finding, status="resolved", resolution=resolution)
    elif kind == "reopened":
        if action.reason is None:
            raise ReviewReadError("reopen is missing its reason")
        finding = dataclasses.replace(finding, status="open", resolution=None)
    elif kind == "requested":
        if action.target is None or not action.action:
            raise ReviewReadError("Fix request is missing its target or delivery action")
        finding = dataclasses.replace(finding, fix_state="delivery_unknown")
    else:
        if not any(
            a.type == "loop.review.fix.requested" and a.request_id == action.request_id
            for a in finding.actions
        ):
            raise ReviewReadError("Fix delivery has no matching request")
        if kind == "failed" and not action.error:
            raise ReviewReadError("failed Fix delivery is missing its error")
        finding = dataclasses.replace(
            finding, fix_state="failed" if kind == "failed" else "submitted"
        )
    return finding


def _action_event(view: ReviewView, event: Mapping[str, Any]) -> ReviewView:
    action = ActionRecord.model_validate(event)
    if action.resolution is not None:
        action = action.model_copy(update={"reason": action.resolution.reason})
    finding = next((f for f in view.findings if f.finding_id == action.finding_id), None)
    if finding is None:
        raise ReviewReadError("action names an unknown finding")
    for previous in finding.actions:
        if previous.request_id == action.request_id and previous.type == action.type:
            if previous != action:
                raise ReviewReadError("request ID has conflicting action records")
            return view
    kind = action.type.rsplit(".", 1)[-1]
    if kind in {"resolved", "reopened", "requested"} and action.source is None:
        raise ReviewReadError("operator action is missing its source")
    if kind in {"resolved", "reopened"} and action.version != view.version + 1:
        raise ReviewReadError("disposition has a conflicting review-state version")
    finding = _transition(view, finding, action, kind)
    finding = dataclasses.replace(finding, actions=(*finding.actions, action))
    return dataclasses.replace(
        view,
        findings=tuple(finding if f.finding_id == finding.finding_id else f for f in view.findings),
    )


def apply_event(view: ReviewView, event: Mapping[str, Any]) -> ReviewView:
    """Return the shared live and journal-scan review fold for one appended event."""
    kind = event.get("type")
    if kind not in _PANEL_EVENTS and kind not in _ACTION_EVENTS:
        return view
    event_id = event.get("event_id")
    if isinstance(event_id, str) and event_id in view.event_ids:
        return view
    try:
        if kind in _PANEL_EVENTS and "panel_id" not in event and "target" not in event:
            result = dataclasses.replace(
                view,
                completion="unknown",
                freshness="unknown",
                legacy_notice="Legacy review counts contain no finding evidence.",
            )
        elif kind in _PANEL_EVENTS:
            result = _panel_event(view, event)
        else:
            result = _action_event(view, event)
    except (pydantic.ValidationError, ReviewReadError) as exc:
        result = dataclasses.replace(
            view,
            read_failure=view.read_failure or str(exc),
            completion="unknown",
            freshness="unknown",
        )
    if result is view:
        return view
    counts = {
        severity: sum(
            f.status == "open" and f.observations[-1].finding.severity == severity
            for f in result.findings
        )
        for severity in ("block", "warn", "nit")
    }
    result = dataclasses.replace(
        result,
        version=view.version + 1,
        open_counts=SeverityCounts(**counts),
        event_ids=(*view.event_ids, event_id) if isinstance(event_id, str) else view.event_ids,
    )
    return result.for_comparison(result.comparison) if result.comparison is not None else result


def notice_texts(event: Mapping[str, Any]) -> tuple[str, ...]:
    """Render review evidence and dispositions as shared transcript notices.

    Args:
        event: The journal event to render.

    Returns:
        Plain text notices, or none for events without an operator-facing result.
    """
    kind = event.get("type")
    try:
        if kind == "loop.review.panel":
            if "target" not in event:
                return ("Review has legacy count-only evidence; findings are unavailable.",)
            panel = review_records.ReviewPanel.model_validate(event)
            lines = [
                f"Review {panel.panel_id}: {panel.completion}; "
                f"block_on={panel.block_on}, rejected={panel.blocked}, disarmed={panel.disarmed}"
            ]
            lines.extend(
                f"Finding {o.finding_id} · {o.finding.severity} · {o.finding.location}\n"
                f"{o.finding.title}\n{o.finding.detail}"
                for o in panel.findings
            )
            return tuple(lines)
        if kind == "loop.review.skipped":
            skipped = _Skipped.model_validate(event)
            return (f"Review {skipped.panel_id} skipped: {skipped.reason}",)
        if kind not in _ACTION_EVENTS:
            return ()
        action = ActionRecord.model_validate(event)
        label = f"Finding {action.finding_id}"
        if kind == "loop.review.finding.resolved" and action.resolution is not None:
            resolution = action.resolution
            duplicate = (
                f" (duplicate of {resolution.duplicate_of})" if resolution.duplicate_of else ""
            )
            body = f"{label} resolved as {resolution.disposition}{duplicate}: {resolution.reason}"
        elif kind == "loop.review.finding.reopened":
            body = f"{label} reopened: {action.reason}"
        else:
            detail = {
                "loop.review.fix.requested": (
                    "Fix requested; delivery unknown until submission is recorded"
                ),
                "loop.review.fix.submitted": "steer submitted; finding stays open",
                "loop.review.fix.failed": f"Fix delivery failed: {action.error}",
            }.get(kind, "incomplete disposition record")
            body = f"{label}: {detail}"
        return (body,)
    except pydantic.ValidationError as exc:
        return (f"Cannot read review notice: {exc}",)


def as_dict(view: ReviewView) -> dict[str, Any]:
    """Return JSON-ready evidence, including nested boundary models and merge facts."""
    result = pydantic.TypeAdapter(ReviewView).dump_python(view, mode="json")
    result["merge_guard"] = dataclasses.asdict(view.merge_guard)
    for row, finding in zip(result["findings"], view.findings, strict=True):
        row["line_origin"] = finding.line_origin.model_dump(mode="json")
    return result
