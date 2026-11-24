"""Decide cold starts from the last ordinary request in the session journal."""

from __future__ import annotations

import dataclasses
import datetime
import json
import pathlib
from typing import Any, Literal

from agent6 import kinds


@dataclasses.dataclass(frozen=True, slots=True)
class Trigger:
    """The route and clock facts that made this start cold."""

    reason: Literal["route_changed", "idle"]
    previous_route: str
    route: str
    gap_s: float | None
    idle_compact_s: float


def _last_request(path: pathlib.Path, actor: str) -> dict[str, Any]:
    """Find the driving role's last inference attempt, excluding summary calls.

    Args:
        path: The session journal.
        actor: The driving role.

    Returns:
        The last ordinary request row, or an empty dict.
    """
    last: dict[str, Any] = {}
    summary_id: str | None = None
    try:
        stream = path.open("rb")
    except FileNotFoundError:
        return last
    with stream:
        for line in stream:
            if b'"provider.request"' not in line and b'"compact.summarise.call"' not in line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if not isinstance(row, dict):
                continue
            if row.get("type") == "compact.summarise.call":
                summary_id = row.get("model_call_id")
            elif (
                row.get("type") == "provider.request"
                and row.get("actor") == actor
                and row.get("target") == "inference"
                and row.get("execution_id")  # A commit subject runs outside a worker turn.
                and row.get("model_call_id") != summary_id
            ):
                last = row
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
