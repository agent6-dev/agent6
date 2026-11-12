# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Render transcript tool previews, salient arguments and marker text."""

from __future__ import annotations

import difflib
import shlex
from collections.abc import Callable
from typing import Any

from agent6 import budget, evidence, terminal
from agent6.viewmodel import events as viewmodel_events
from agent6.viewmodel import format, review

PRIMARY_ARGS = ("path", "file", "pattern", "query", "command", "cmd", "url", "title", "summary")


def verify_body(command: str, detail: str = "") -> str:
    """Return a verification marker naming what it waits on or its result."""
    return f"verify: {detail or f'running `{command}`'}"


def verify_result(event: dict[str, Any]) -> str:
    """Return the verification result's badge, reason and duration."""
    label = evidence.result_label(
        event.get("exit_code"), evidence.outcome(event), str(event.get("reason") or "")
    )
    badge = "✓ pass" if event.get("exit_code") == 0 else f"✗ {label}"
    duration = float(event.get("duration_s", 0) or 0)
    return terminal.scrub_terminal_controls(f"{badge} · {duration:.1f}s")


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
    """Word a dispatched group: how many children for how many tasks, and which.

    Args:
        event: The `parallel.dispatched` event.

    Returns:
        The head line, then one bullet per task; an event carrying no child count
        names the tasks alone.
    """
    tasks_raw = event.get("tasks")
    tasks = [str(t).strip() for t in tasks_raw] if isinstance(tasks_raw, list) else []
    n = len(tasks)
    if event.get("source") == "panel":
        # A panel's reviewers are read-only sessions of their own, on no child listing.
        head = (
            f"dispatched {n} reviewer{'' if n == 1 else 's'} ({parallel_group_label(event)});"
            " each runs as a read-only session"
        )
        return "\n".join([head, *(f"• {clip(t, 80)}" for t in tasks if t)])
    children = event.get("children")
    if isinstance(children, int) and not isinstance(children, bool):
        what = format.child_count(children) + (f" for {n} tasks" if n > 1 else "")
    else:
        what = f"{n} parallel task{'' if n == 1 else 's'}"
    head = (
        f"dispatched {what} ({parallel_group_label(event)});"
        " children run detached, listed under this session"
    )
    return "\n".join([head, *(f"• {clip(t, 80)}" for t in tasks if t)])


def parallel_compared_body(event: dict[str, Any]) -> str:
    """Word a group's ranking: best first, each with its gate verdict and cost.

    Args:
        event: The `parallel.compared` event.

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
    """Word a joined group: one line per child with its status, id, branch and sha.

    Args:
        event: The `parallel.joined` event.

    Returns:
        The head line, then one line per child.
    """
    children_raw = event.get("children")
    children = (
        [ln for ln in children_raw if isinstance(ln, dict)]
        if isinstance(children_raw, list)
        else []
    )
    panel = event.get("source") == "panel"
    head = (
        f"reviewers returned ({parallel_group_label(event)}): {len(children)}"
        if panel
        else f"joined {parallel_group_label(event)}: {len(children)} child(ren)"
    )
    rows: list[str] = []
    for ln in children:
        status = str(ln.get("status", "?"))
        if panel and status == "no_changes":
            status = "returned"  # a reviewer has no work to join; its verdict follows in the panel
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
    """Word a dispatch failure.

    Args:
        event: The `parallel.failed` event.

    Returns:
        The dispatch error when the event carries one; None for a failure after the children
        ran, whose joined event or end summary already showed each child's status.
    """
    error = str(event.get("error", "")).strip()
    return f"{parallel_group_label(event)} dispatch failed: {error}" if error else None


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
    "skills.warning": lambda event: f"skills: WARNING: {event.get('warning', '')}",
    "review.start": review.rules_notice,
    "mcp.server_unavailable": mcp_unavailable_body,
    "jail.degraded": jail_degraded_body,
    "parallel.dispatched": parallel_dispatched_body,
    "parallel.joined": parallel_joined_body,
    "parallel.failed": parallel_failed_body,
    "parallel.compared": parallel_compared_body,
    "task.queued": task_queued_body,
    "task.retired": task_retired_body,
    "standing.set": standing_set_body,
    "task.request_refused": request_refused_body,
    "compact.requested": compact_requested_body,
    "compact.summarise.done": compact_done_body,
    "compact.summarise.failed": compact_failed_body,
    "compact.refused": compact_refused_body,
}
