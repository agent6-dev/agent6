"""Decide cold starts from the last ordinary request in the session journal."""

from __future__ import annotations

import dataclasses
import datetime
import json
import pathlib
from typing import Any, Literal

from agent6 import kinds
from agent6.sessions import journal


@dataclasses.dataclass(frozen=True, slots=True)
class Trigger:
    """The route and clock facts that made this start cold."""

    reason: Literal["route_changed", "idle"]
    previous_route: str
    route: str
    gap_s: float | None
    idle_compact_s: float

    def record(
        self, mode: kinds.ColdStart, action: journal.ColdStartAction, saved: int
    ) -> journal.CompactColdStart:
        """Describe the action taken, separate from any ordinary size checks.

        Args:
            mode: The configured mode.
            action: What happened to the prepared history.
            saved: The reduction in its character count.

        Returns:
            The journal row with its operator-facing line.
        """
        cause = (
            f"route changed from {self.previous_route} to {self.route}"
            if self.reason == "route_changed"
            else f"{self.gap_s:.1f}s since the last worker request (limit {self.idle_compact_s:g}s)"
        )
        detail = {
            "resend": "resend history",
            "drop": "drop eligible old results",
            "summarize": "summarize with the helper",
            "truncate": "truncate without a summary",
            "summary_failed": "helper summary failed, kept history",
            "endpoint": f"endpoint compacts, local {mode} does not apply",
        }[action]
        return journal.CompactColdStart(
            reason=self.reason,
            previous_route=self.previous_route,
            route=self.route,
            gap_s=self.gap_s,
            idle_compact_s=self.idle_compact_s,
            mode=mode,
            action=action,
            chars_not_sent=max(0, saved),
            message=f"Cold start: {cause}; {detail}; {max(0, saved)} characters not sent.",
        )


def _last_request(path: pathlib.Path, actor: str) -> dict[str, Any]:
    """Find the last HTTP attempt or confirmed process call, excluding summary calls.

    Args:
        path: The session journal.
        actor: The driving role.

    Returns:
        The last ordinary request row, or an empty dict.
    """
    last: dict[str, Any] = {}
    opening: dict[str, Any] = {}
    summary_id: str | None = None
    types = (
        b'"provider.request"',
        b'"model.call"',
        b'"model.attempt"',
        b'"compact.summarise.call"',
    )
    try:
        stream = path.open("rb")
    except FileNotFoundError:
        return last
    with stream:
        for line in stream:
            if not any(tag in line for tag in types):
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if not isinstance(row, dict):
                continue
            kind = row.get("type")
            if kind == "compact.summarise.call":
                summary_id = row.get("model_call_id")
                continue
            if row.get("actor") != actor or not row.get("execution_id"):
                continue  # A commit subject runs outside a worker turn.
            call_id = row.get("model_call_id")
            if not call_id or call_id == summary_id:
                continue
            if kind == "model.call":
                opening = row
            elif kind == "provider.request" and row.get("target") == "inference":
                last = row
                opening = {}
            elif (
                kind == "model.attempt"
                and row.get("ok") is True
                and row.get("request_id") is None
                and call_id == opening.get("model_call_id")
            ):
                # A process has no HTTP row. Its reply confirms the call was issued at its opening.
                last = opening
                opening = {}
    return last


def detect(
    path: pathlib.Path, *, actor: str, route: kinds.ModelRoute, limit: float, now: float
) -> Trigger | None:
    """Check the last request's route and age, never inferring a missing fact.

    Args:
        path: The resumed session's journal.
        actor: The role driving this conversation.
        route: The next request's configured route.
        limit: The current provider's idle limit in seconds, or -1.
        now: Wall time in epoch seconds.

    Returns:
        The rule that fired, or None to leave the start alone.
    """
    row = _last_request(path, actor)
    provider, model = row.get("provider"), row.get("model")
    previous = f"{provider}/{model}" if provider and model else ""
    gap = None
    if isinstance(ts := row.get("ts"), str):
        try:
            started = datetime.datetime.fromisoformat(ts)
            if started.tzinfo is not None:
                gap = now - started.timestamp()
        except (ValueError, OverflowError, OSError):
            pass  # An unknown age does not prevent a route comparison.
    if previous and route.provider and route.model and previous != route.spec:
        return Trigger("route_changed", previous, route.spec, gap, limit)
    if limit > 0 and gap is not None and gap >= limit:
        return Trigger("idle", previous, route.spec, gap, limit)
    return None
