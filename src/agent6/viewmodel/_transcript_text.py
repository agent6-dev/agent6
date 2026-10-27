# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Render transcript tool previews, salient arguments and marker text."""

from __future__ import annotations

import difflib
import shlex
from collections.abc import Callable
from typing import Any

from agent6 import budget
from agent6.viewmodel import events as viewmodel_events
from agent6.viewmodel import format, review

PRIMARY_ARGS = ("path", "file", "pattern", "query", "command", "cmd", "url", "title", "summary")


def clip(text: str, n: int = 60) -> str:
    """Return the text as one line of at most `n` characters, an ellipsis marking a cut."""
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 3] + "…"


def call_preview(name: str, args: Any) -> str:
    """Return a bounded preview of an `apply_edit` call's first hunk, "" for other tools.

    Args:
        name: The tool's name.
        args: The raw args.

    Returns:
        The changed lines of the first edit, clipped, with counts of what was left out.
    """
    if name != "apply_edit" or not isinstance(args, dict):
        return ""
    edits = args.get("edits")
    if not isinstance(edits, (list, tuple)) or not edits:
        return ""
    first = edits[0]
    if not isinstance(first, dict):
        return ""
    old = str(first.get("old_string", ""))
    new = str(first.get("new_string", ""))
    # The changed lines only: an anchor line the edit re-emits unchanged is not news.
    changed = [
        ln
        for ln in difflib.ndiff(old.splitlines(), new.splitlines())
        if ln[:1] in "+-" and ln[2:].strip()
    ]
    lines = [clip(ln, 120) for ln in changed[:4]]
    if len(changed) > 4:
        lines.append(f"…(+{len(changed) - 4} more changed lines)")
    more = len(edits) - 1
    if more > 0:
        lines.append(f"…(+{more} more edit{'' if more == 1 else 's'})")
    return "\n".join(lines)


def salient_arg(args: Any) -> str:
    """Return the one argument worth showing beside a tool name.

    Args:
        args: The raw args; a non-dict is tolerated.

    Returns:
        An argv as a shell line, a question's text, the first primary arg found, else
        the first pair; "" when there are no args.
    """
    if not isinstance(args, dict) or not args:
        return ""
    argv = args.get("argv")
    if isinstance(argv, (list, tuple)) and argv:
        return clip(shlex.join(str(a) for a in argv))
    questions = args.get("questions")
    if isinstance(questions, (list, tuple)) and questions:
        first = questions[0]
        q = first.get("question", "") if isinstance(first, dict) else str(first)
        more = f" (+{len(questions) - 1})" if len(questions) > 1 else ""
        return clip(str(q)) + more
    for key in PRIMARY_ARGS:
        value = args.get(key)
        if isinstance(value, (str, int)):
            text = format.tool_path(str(value)) if key in format.PATH_KEYS else str(value)
            return clip(text)
    key, value = next(iter(args.items()))
    return clip(f"{key}={value}")


def parallel_group_label(event: dict[str, Any]) -> str:
    """Return the group's label: `group <name>`, or "parallel" for an unnamed one."""
    group = str(event.get("group", "")).strip()
    return f"group {group}" if group else "parallel"


def parallel_dispatched_body(event: dict[str, Any]) -> str:
    """Word a dispatched group: how many lanes for how many tasks, and which.

    Args:
        event: The `loop.parallel.dispatched` event.

    Returns:
        The head line, then one bullet per task; an event carrying no lane count
        names the tasks alone.
    """
    tasks_raw = event.get("tasks")
    tasks = [str(t).strip() for t in tasks_raw] if isinstance(tasks_raw, list) else []
    n = len(tasks)
    lanes = event.get("lanes")
    if isinstance(lanes, int) and not isinstance(lanes, bool):
        what = format.lane_count(lanes) + (f" for {n} tasks" if n > 1 else "")
    else:
        what = f"{n} parallel task{'' if n == 1 else 's'}"
    head = (
        f"dispatched {what} ({parallel_group_label(event)});"
        " lanes run detached, listed under this session"
    )
    return "\n".join([head, *(f"• {clip(t, 80)}" for t in tasks if t)])


def parallel_compared_body(event: dict[str, Any]) -> str:
    """Word a fan-out's ranking: best first, each with its gate verdict and cost.

    Args:
        event: The `loop.parallel.compared` event.

    Returns:
        The head line naming who ranked, one line per candidate, then the judge's rationale.
    """
    raw = event.get("ranking")
    rows = [r for r in raw if isinstance(r, dict)] if isinstance(raw, list) else []
    by = str(event.get("ranked_by", "")).strip() or "?"
    head = f"compared {parallel_group_label(event)}: {len(rows)} candidate(s), ranked by {by}"
    lines = [head]
    for rank, r in enumerate(rows, start=1):
        cost = r.get("cost_usd")
        cost_s = budget.format_usd(float(cost)) if isinstance(cost, (int, float)) else "?"
        # The listings' cell, when the event carries it (plan points, or both).
        cell = r.get("cost")
        if isinstance(cell, str) and cell:
            cost_s = cell
        lines.append(f"{rank}. {r.get('session_id', '?')}  {r.get('verify', '?')}  {cost_s}")
    if rationale := str(event.get("rationale", "")).strip():
        lines.append(f"judge: {rationale}")
    return "\n".join(lines)


def parallel_joined_body(event: dict[str, Any]) -> str:
    """Word a joined group: one line per lane with its status, id, branch and sha.

    Args:
        event: The `loop.parallel.joined` event.

    Returns:
        The head line, then one line per lane.
    """
    lanes_raw = event.get("lanes")
    lanes = [ln for ln in lanes_raw if isinstance(ln, dict)] if isinstance(lanes_raw, list) else []
    head = f"joined {parallel_group_label(event)}: {len(lanes)} lane(s)"
    rows: list[str] = []
    for ln in lanes:
        status = str(ln.get("status", "?"))
        parts = [str(ln.get("session_id", "?"))]
        branch = str(ln.get("branch", "")).strip()
        if branch:
            parts.append(branch)
        sha = str(ln.get("sha", "")).strip()
        if sha:
            parts.append(sha[:12])
        detail = str(ln.get("detail", "")).strip()
        if detail:
            parts.append(detail)
        rows.append(f"{status}  {'  '.join(parts)}")
    return "\n".join([head, *rows])


def parallel_failed_body(event: dict[str, Any]) -> str | None:
    """Word a dispatch failure or a fan-out's failed lanes.

    Args:
        event: The `loop.parallel.failed` event.

    Returns:
        The dispatch error when the event carries one; a fan-out's failed lanes
        otherwise; None for a post-join failure, whose joined event already showed
        each lane's status.
    """
    error = str(event.get("error", "")).strip()
    if error:
        return f"{parallel_group_label(event)} dispatch failed: {error}"
    if event.get("fanout") is not True:
        return None
    raw = event.get("lanes")
    lanes = [lane for lane in raw if isinstance(lane, dict)] if isinstance(raw, list) else []
    lines = [f"failed lanes in {parallel_group_label(event)}:"]
    for lane in lanes:
        session_id = str(lane.get("session_id", "?")).strip() or "?"
        detail = str(lane.get("detail", "")).strip()
        lines.append(f"{session_id}: {detail}" if detail else session_id)
    return "\n".join(lines)


def mcp_unavailable_body(event: dict[str, Any]) -> str:
    """Word an MCP server that did not start: the error and the tools it costs.

    Args:
        event: The `mcp.server_unavailable` event.

    Returns:
        The error, which names the server, and the consequence.
    """
    error = str(event.get("error", "")).strip()
    if not error:
        server = str(event.get("server", "")).strip() or "(unnamed)"
        return f"MCP server {server!r} is unavailable; its tools are missing"
    return f"{error}; its tools are missing"


def compact_requested_body(event: dict[str, Any]) -> str:
    """Return the line for a compaction request, with its focus."""
    focus = str(event.get("focus", "")).strip()
    return f"compaction requested: {focus}" if focus else "compaction requested"


def task_queued_body(event: dict[str, Any]) -> str:
    """Return the line for a queued task; it reaches the model when the frontier gets there."""
    return f"task queued: {str(event.get('title', '')).strip()}".rstrip(": ")


def task_retired_body(event: dict[str, Any]) -> str:
    """Return the line for a retired task."""
    return f"task retired: {str(event.get('title', '')).strip()}".rstrip(": ")


def standing_set_body(event: dict[str, Any]) -> str:
    """Return the line for the standing goal the operator set."""
    return f"standing goal: {str(event.get('title', '')).strip()}".rstrip(": ")


def compact_done_body(event: dict[str, Any]) -> str:
    """Return the line for a tier-2 compaction: the summary's size and the messages kept."""
    chars = viewmodel_events.as_int(event.get("summary_chars"))
    kept = viewmodel_events.as_int(event.get("kept_messages"))
    return f"context compacted: {chars:,}-char summary, {kept} recent messages kept verbatim"


def request_refused_body(event: dict[str, Any]) -> str:
    """Return the line for a refused operator request, which takes the composer's word back."""
    kind, text = str(event.get("kind", "")), str(event.get("text", "")).strip()
    error = str(event.get("error", "")).strip()
    return f"{kind} request refused ({text[:60]}): {error}"


def compact_failed_body(event: dict[str, Any]) -> str:
    """Return the line for a failed compaction, with its error."""
    error = str(event.get("error", "")).strip()
    return f"compaction failed: {error}" if error else "compaction failed"


def compact_refused_body(event: dict[str, Any]) -> str:
    """Return the line for a refused compaction, with its reason."""
    reason = str(event.get("reason", "")).strip()
    return f"compaction refused: {reason}" if reason else "compaction refused"


def jail_degraded_body(event: dict[str, Any]) -> str:
    """Return the line for a degraded sandbox, with its reason."""
    detail = " ".join(str(event.get("detail", "")).split())
    return f"sandbox degraded: {detail}" if detail else "sandbox degraded"


# Events that render as a marker between turns; a builder returning None renders nothing.
MARKER_BODIES: dict[str, Callable[[dict[str, Any]], str | None]] = {
    "loop.review.start": review.rules_notice,
    "mcp.server_unavailable": mcp_unavailable_body,
    "jail.degraded": jail_degraded_body,
    "loop.parallel.dispatched": parallel_dispatched_body,
    "loop.parallel.joined": parallel_joined_body,
    "loop.parallel.failed": parallel_failed_body,
    "loop.parallel.compared": parallel_compared_body,
    "loop.task.queued": task_queued_body,
    "loop.task.retired": task_retired_body,
    "loop.standing.set": standing_set_body,
    "loop.request.refused": request_refused_body,
    "loop.compact.requested": compact_requested_body,
    "loop.compact.summarise.done": compact_done_body,
    "loop.compact.summarise.failed": compact_failed_body,
    "loop.compact.refused": compact_refused_body,
}
