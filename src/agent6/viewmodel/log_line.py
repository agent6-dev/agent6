# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Render a run's events as the one-line rows the log views paint."""

from __future__ import annotations

import shlex
from typing import Any

from agent6.viewmodel import events, transcript


def _edit_kind(edit: dict[str, object]) -> str:
    """Return the kind an `apply_edit` pair resolves to, as the tool resolves it."""
    kind = str(edit.get("kind") or "")
    if kind:
        return kind
    return "replace" if edit.get("old_string") else "create"


def _render_arg_value(key: str, value: Any) -> str:
    """Render one arg value for a reader.

    Args:
        key: The arg's name.
        value: The arg's raw value.

    Returns:
        An argv as a shell line, questions as the first one's text, edits as their
        kinds, a string as is, anything else as its repr.
    """
    if key == "argv" and isinstance(value, (list, tuple)) and value:
        return shlex.join(str(a) for a in value)
    if key == "questions" and isinstance(value, (list, tuple)) and value:
        first = value[0]
        q = first.get("question", "") if isinstance(first, dict) else str(first)
        return str(q) + (f" (+{len(value) - 1})" if len(value) > 1 else "")
    if key == "edits" and isinstance(value, (list, tuple)) and value:
        return ", ".join(_edit_kind(e) if isinstance(e, dict) else str(e) for e in value)
    return value if isinstance(value, str) else repr(value)


def _as_dict(value: Any) -> dict[str, Any]:
    """Return an untrusted event field as a dict, empty for any other type."""
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    """Return an untrusted event field as a list, empty for any other type."""
    return list(value) if isinstance(value, (list, tuple)) else []


def render_args(args: dict[str, Any], *, max_value: int = 80) -> str:
    """Render an args dict as `k=v, ...`.

    Args:
        args: The tool call's args.
        max_value: The characters each value keeps; the inline row uses the default,
            the detail view a generous cap.

    Returns:
        The pairs joined by ", ".
    """
    pairs: list[str] = []
    for k, v in args.items():
        s = _render_arg_value(k, v)
        if len(s) > max_value:
            s = s[:max_value] + "…"
        pairs.append(f"{k}={s}")
    return ", ".join(pairs)


def _cut(text: str, limit: int) -> str:
    """Return the text cut to `limit` characters, an ellipsis marking the cut."""
    return text if len(text) <= limit else text[: limit - 1] + "\u2026"


def format_log_line(event: dict[str, Any]) -> str:  # noqa: C901, PLR0912, PLR0915  # one branch per event type
    """Render one event as a log row: timestamp, type, then the salient field.

    The row embeds model-authored text, so it is scrubbed of terminal controls and
    kept to one line.

    Args:
        event: The raw event dict.

    Returns:
        The row.
    """
    ts = str(event.get("ts", ""))
    etype = str(event.get("type", "?"))
    salient = ""
    match etype:
        case "graph.update":
            nodes = event.get("nodes", {})
            salient = f"{len(nodes)} tasks" if isinstance(nodes, dict) else ""
        case "diff.updated":
            salient = f"{len(str(event.get('patch', '')).splitlines())} lines"
        case "loop.task.queued" | "loop.task.retired" | "loop.standing.set":
            salient = str(event.get("title", ""))
        case "loop.request.refused":
            salient = f"{event.get('kind', '')} {event.get('text', '')}: {event.get('error', '')}"
        case "loop.auto_commit":
            salient = f"{str(event.get('sha', ''))[:12]} {event.get('subject', '')}".strip()
        case "tool.call":
            salient = f"{event.get('name', '')}({render_args(_as_dict(event.get('args')))})"
        case "tool.result":
            summ = events.readable_summary(event.get("summary", ""))
            salient = f"{event.get('name', '')} ok={event.get('ok')} {summ}"
            # A hint of the latest stderr (else stdout), so an outcome reads without the transcript.
            tail = str(event.get("stderr_tail") or event.get("stdout_tail") or "")
            snippet = _cut(" ".join(tail.split()), 100)
            if snippet:
                salient = f"{salient.rstrip()} | {snippet}"
        case "role.call":
            salient = f"{event.get('role', '')}/{event.get('model', '')}"
        case "role.result":
            role = event.get("role", "")
            if event.get("error"):
                salient = f"{role} error: {_cut(str(event.get('error')), 160)}"
            else:
                tin = event.get("tokens_in")
                tout = event.get("tokens_out")
                salient = f"{role} in={tin} out={tout}"
        case "loop.provider.retry":
            salient = f"attempt {event.get('attempt')}: {_cut(str(event.get('error', '')), 160)}"
        case "loop.pin.added":
            salient = f"pinned ({event.get('chars')} chars): {_cut(str(event.get('text', '')), 80)}"
        case "loop.pin.refused":
            salient = f"pin refused: over the {event.get('limit')}-char cap"
        case "loop.pin.restored":
            pins = [str(p) for p in _as_list(event.get("pins"))]
            salient = (
                f"{len(pins)} pinned: " + " | ".join(_cut(p, 80) for p in pins)
                if pins
                else "no pinned instructions"
            )
        case "loop.compact.dropped":
            calls = _as_list(event.get("calls"))
            named = ", ".join(str(c) for c in calls)
            salient = f"elided {event.get('n')} old tool results"
            if named:
                salient += f": {_cut(named, 160)}"
        case "loop.compact.deduped":
            calls = _as_list(event.get("calls"))
            named = ", ".join(str(c) for c in calls)
            salient = f"deduplicated {event.get('n')} identical tool results"
            if named:
                salient += f": {_cut(named, 160)}"
        case "loop.compact.thinking_dropped":
            salient = (
                f"dropped thinking from {event.get('turns')} old turns ({event.get('chars')} chars)"
            )
        case "loop.compact.gists":
            parts = []
            if event.get("gisted"):
                paths = ", ".join(str(p) for p in _as_list(event.get("paths")))
                parts.append(f"{event.get('gisted')} distilled ({_cut(paths, 120)})")
            if event.get("demoted"):
                dem = ", ".join(str(p) for p in _as_list(event.get("demoted_paths")))
                parts.append(f"{event.get('demoted')} demoted ({_cut(dem, 120)})")
            salient = "; ".join(parts)
        case "loop.compact.summarise.done":
            salient = f"restarted on a {event.get('summary_chars')}-char progress summary"
        case "loop.compact.summarise.failed" | "loop.compact.gist.failed":
            salient = _cut(str(event.get("error", "")), 160)
        case "loop.compact.requested":
            focus = str(event.get("focus", ""))
            salient = f"focus: {_cut(focus, 120)}" if focus else "no focus"
        case "loop.compact.restored":
            salient = f"{event.get('elided')} elided, {event.get('gists')} gists in context"
        case "loop.compact.refused":
            salient = _cut(str(event.get("reason", "")), 160)
        case "jail.degraded":
            salient = _cut(" ".join(str(event.get("detail", "")).split()), 160)
        case "mcp.server_unavailable":
            salient = f"{event.get('server')} unavailable: {_cut(str(event.get('error', '')), 120)}"
        case "loop.skills.warning":
            salient = _cut(str(event.get("warning", "")), 160)
        case "loop.resume.start":
            salient = f"iteration={event.get('iteration')} messages={event.get('messages')}"
        case "budget.update":
            usd = event.get("usd_total")
            usd_s = f"${usd:.4f}" if isinstance(usd, (int, float)) else f"${usd}"
            salient = f"in={event.get('input_total')} out={event.get('output_total')} {usd_s}"
        case "session.start":
            salient = _cut(str(event.get("user_task", "")), 80)
        case "verify.end":
            dur = event.get("duration_s")
            dur_s = f"{dur:.1f}s" if isinstance(dur, (int, float)) else f"{dur}s"
            salient = f"exit={event.get('exit_code')} dur={dur_s}"
        case "approval.prompt":
            salient = _cut(str(event.get("prompt", "")), 80)
        case "approval.answer":
            salient = f"id={event.get('id')} approved={event.get('approved')}"
        case "question.prompt":
            qs = _as_list(event.get("questions"))
            first = str(qs[0].get("question", "")) if qs and isinstance(qs[0], dict) else ""
            salient = (f"[{len(qs)}] " if len(qs) > 1 else "") + _cut(first, 80)
        case "question.answer":
            ans = _as_list(event.get("answers"))
            salient = f"id={event.get('id')} answers={len(ans)}"
        case "session.end":
            salient = f"{event.get('reason', '')} all_passed={event.get('all_passed')}"
        case _:
            salient = ""
    line = f"{ts[11:23] if len(ts) > 23 else ts}  {etype:<18}"
    # The scrubber keeps newlines for transcripts; a provider error's SSE dump would paint rows.
    if not salient:
        return line
    scrubbed = transcript.scrub_terminal_controls(f"{line} {salient}")
    return " ".join(scrubbed.split("\n"))
