# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Fold a session's event stream into the ordered conversation items every front-end paints.

`TranscriptFold` walks the events in emission order and yields what is worth showing
as plain data: a reasoning block, an assistant message, a tool call (in flight, then
settled with its result), a commit, the final verdict. An item lands when it
completes, so a settled tool call lands after anything that landed during the call;
until then the call shows in flight. `fold_transcript` folds a whole stream; the
live tailers feed the same fold one event at a time.
"""

from __future__ import annotations

import dataclasses
import difflib
import re
import shlex
from collections.abc import Callable, Iterable
from typing import Any, Literal

from agent6 import budget, kinds
from agent6.viewmodel import events as viewmodel_events
from agent6.viewmodel import format, listing

# Default-deny: CSI alone would let an OSC 52 clipboard write and DCS/SOS/PM/APC payloads
# through, and a C1 byte opens the same doors 8-bit. Sequences drop whole; \n and \t stay.
_CONTROL_RE = re.compile(
    r"\x1b\[[0-9;?]*[ -/]*[@-~]"  # CSI
    r"|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)?"  # OSC, BEL- or ST-terminated (or cut off)
    r"|\x1b[PX^_][^\x1b]*(?:\x1b\\)?"  # DCS / SOS / PM / APC, ST-terminated
    r"|\x1b."  # any other escape, with its final byte
    r"|[\x00-\x08\x0b-\x1f\x7f\x80-\x9f]",  # stray C0 (keep \t \n) + DEL + C1
    re.DOTALL,
)


def scrub_terminal_controls(text: str) -> str:
    """Return the text without terminal control sequences and stray control characters.

    Idempotent, so an accumulating stream tail re-scrubs for free.
    """
    return _CONTROL_RE.sub("", text)


# SGR styling passes, conceal (SGR 8) excepted; a colour cannot move the cursor or rewrite a
# line. The spinners' erase idiom goes under `ui/cli/_terminal_guard.raw_stream` instead.
_TERMINAL_RE = re.compile(
    r"(?P<own>\x1b\[(?!(?:[0-9;]*;)?0*8(?:;|m))[0-9;]*m)|" + _CONTROL_RE.pattern, re.DOTALL
)


def scrub_terminal_output(text: str) -> str:
    """Return the text the CLI may write to its terminal: SGR styling kept, controls dropped."""
    return _TERMINAL_RE.sub(lambda m: m.group("own") or "", text)


# Text characters, not graphics, so every terminal font renders them.
CALL = "→"
RESULT = "└"
COMMIT = "✎"
THINK = "·"
DONE = "●"
OPERATOR = "❯"  # noqa: RUF001  # a prompt glyph, not a mistyped >

# Literals, so the viewmodel needs no tools import.
_FINISH_TOOLS = frozenset({"finish_session", "finish_planning"})

ItemKind = Literal["thinking", "text", "tool", "commit", "marker", "done", "operator"]


# Events that render between turns: type -> (kind, the field holding the text).
_BETWEEN_TURNS: dict[str, tuple[ItemKind, str]] = {
    "loop.steer.injected": ("operator", "text"),
    "btw.answered": ("marker", "block"),
}

# Where the operator's own words live in the journal; `tools.sessions._SPEAKER` quotes them.
_OPERATOR_TEXT = {"session.start": "user_task", "loop.steer.injected": "text"}


def worker_models(events: Iterable[dict[str, Any]]) -> tuple[str, ...]:
    """Return the models that wrote code in a session, in first-seen order.

    Commit trailers read this; a message-writing model never joins the list.

    Args:
        events: The raw events, in order.

    Returns:
        Every `role.call` worker model once, the primary worker first.
    """
    seen: dict[str, None] = {}
    for event in events:
        if event.get("type") == "role.call" and event.get("role") == "worker":
            model = str(event.get("model", "")).strip()
            if model:
                seen.setdefault(model, None)
    return tuple(seen)


def operator_inputs(events: Iterable[dict[str, Any]]) -> list[str]:
    """Return the operator's typed messages, oldest first, consecutive repeats collapsed.

    Args:
        events: The raw events, in order; the journal spans every execution and surface.

    Returns:
        The opening task, then every steer.
    """
    out: list[str] = []
    for event in events:
        field = _OPERATOR_TEXT.get(str(event.get("type", "")))
        if field is None:
            continue
        text = str(event.get(field, "")).strip()
        if text and (not out or out[-1] != text):
            out.append(text)
    return out


@dataclasses.dataclass(frozen=True, slots=True)
class TranscriptItem:
    """One rendered conversation step; only the fields its `kind` needs are set.

    Attributes:
        kind: The item's kind.
        body: The thinking, text or marker prose; the final summary for `done`.
        name: The tool's name; the status label for `done`.
        arg: The tool's salient argument (a path, a pattern, a command).
        ok: The tool or run outcome; None while in flight or when not applicable.
        detail: The tool result's summary, the verify badge or the commit and done
            metadata; for a call in flight, why it waits ("awaiting approval").
        tail: A tool's captured output tail.
        call_id: The stamped call id, so a surface pairs a call's start with its
            outcome by identity where two identical calls would collide on name and arg.
    """

    kind: ItemKind
    body: str = ""
    name: str = ""
    arg: str = ""
    ok: bool | None = None
    detail: str = ""
    tail: str = ""
    call_id: str = ""


_PRIMARY_ARGS = ("path", "file", "pattern", "query", "command", "cmd", "url", "title", "summary")


def _clip(text: str, n: int = 60) -> str:
    """Return the text as one line of at most `n` characters, an ellipsis marking a cut."""
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 3] + "…"


def _call_preview(name: str, args: Any) -> str:
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
    lines = [_clip(ln, 120) for ln in changed[:4]]
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
        return _clip(shlex.join(str(a) for a in argv))
    questions = args.get("questions")
    if isinstance(questions, (list, tuple)) and questions:
        first = questions[0]
        q = first.get("question", "") if isinstance(first, dict) else str(first)
        more = f" (+{len(questions) - 1})" if len(questions) > 1 else ""
        return _clip(str(q)) + more
    for key in _PRIMARY_ARGS:
        value = args.get(key)
        if isinstance(value, (str, int)):
            return _clip(str(value))
    key, value = next(iter(args.items()))
    return _clip(f"{key}={value}")


def _parallel_group_label(event: dict[str, Any]) -> str:
    """Return the group's label: `group <name>`, or "parallel" for an unnamed one."""
    group = str(event.get("group", "")).strip()
    return f"group {group}" if group else "parallel"


def _parallel_dispatched_body(event: dict[str, Any]) -> str:
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
        f"dispatched {what} ({_parallel_group_label(event)});"
        " lanes run detached, listed under this session"
    )
    return "\n".join([head, *(f"• {_clip(t, 80)}" for t in tasks if t)])


def _parallel_compared_body(event: dict[str, Any]) -> str:
    """Word a fan-out's ranking: best first, each with its gate verdict and cost.

    Args:
        event: The `loop.parallel.compared` event.

    Returns:
        The head line naming who ranked, then one line per candidate.
    """
    raw = event.get("ranking")
    rows = [r for r in raw if isinstance(r, dict)] if isinstance(raw, list) else []
    by = str(event.get("ranked_by", "")).strip() or "?"
    head = f"compared {_parallel_group_label(event)}: {len(rows)} candidate(s), ranked by {by}"
    lines = [head]
    for rank, r in enumerate(rows, start=1):
        cost = r.get("cost_usd")
        cost_s = budget.format_usd(float(cost)) if isinstance(cost, (int, float)) else "?"
        lines.append(f"{rank}. {r.get('session_id', '?')}  {r.get('verify', '?')}  {cost_s}")
    return "\n".join(lines)


def _parallel_joined_body(event: dict[str, Any]) -> str:
    """Word a joined group: one line per lane with its status, id, branch and sha.

    Args:
        event: The `loop.parallel.joined` event.

    Returns:
        The head line, then one line per lane.
    """
    lanes_raw = event.get("lanes")
    lanes = [ln for ln in lanes_raw if isinstance(ln, dict)] if isinstance(lanes_raw, list) else []
    head = f"joined {_parallel_group_label(event)}: {len(lanes)} lane(s)"
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


def _parallel_failed_body(event: dict[str, Any]) -> str | None:
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
        return f"{_parallel_group_label(event)} dispatch failed: {error}"
    if event.get("fanout") is not True:
        return None
    raw = event.get("lanes")
    lanes = [lane for lane in raw if isinstance(lane, dict)] if isinstance(raw, list) else []
    lines = [f"failed lanes in {_parallel_group_label(event)}:"]
    for lane in lanes:
        session_id = str(lane.get("session_id", "?")).strip() or "?"
        detail = str(lane.get("detail", "")).strip()
        lines.append(f"{session_id}: {detail}" if detail else session_id)
    return "\n".join(lines)


def _mcp_unavailable_body(event: dict[str, Any]) -> str:
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


def _compact_requested_body(event: dict[str, Any]) -> str:
    """Return the line for a compaction request, with its focus."""
    focus = str(event.get("focus", "")).strip()
    return f"compaction requested: {focus}" if focus else "compaction requested"


def _task_queued_body(event: dict[str, Any]) -> str:
    """Return the line for a queued task; it reaches the model when the frontier gets there."""
    return f"task queued: {str(event.get('title', '')).strip()}".rstrip(": ")


def _task_retired_body(event: dict[str, Any]) -> str:
    """Return the line for a retired task."""
    return f"task retired: {str(event.get('title', '')).strip()}".rstrip(": ")


def _standing_set_body(event: dict[str, Any]) -> str:
    """Return the line for the standing goal the operator set."""
    return f"standing goal: {str(event.get('title', '')).strip()}".rstrip(": ")


def _compact_done_body(event: dict[str, Any]) -> str:
    """Return the line for a tier-2 compaction: the summary's size and the turns kept."""
    chars = viewmodel_events.as_int(event.get("summary_chars"))
    kept = viewmodel_events.as_int(event.get("kept_turns"))
    return f"context compacted: {chars:,}-char summary, {kept} recent turns kept verbatim"


def _request_refused_body(event: dict[str, Any]) -> str:
    """Return the line for a refused operator request, which takes the composer's word back."""
    kind, text = str(event.get("kind", "")), str(event.get("text", "")).strip()
    error = str(event.get("error", "")).strip()
    return f"{kind} request refused ({text[:60]}): {error}"


def _compact_failed_body(event: dict[str, Any]) -> str:
    """Return the line for a failed compaction, with its error."""
    error = str(event.get("error", "")).strip()
    return f"compaction failed: {error}" if error else "compaction failed"


def _compact_refused_body(event: dict[str, Any]) -> str:
    """Return the line for a refused compaction, with its reason."""
    reason = str(event.get("reason", "")).strip()
    return f"compaction refused: {reason}" if reason else "compaction refused"


def _jail_degraded_body(event: dict[str, Any]) -> str:
    """Return the line for a degraded sandbox, with its reason."""
    detail = " ".join(str(event.get("detail", "")).split())
    return f"sandbox degraded: {detail}" if detail else "sandbox degraded"


# Events that render as a marker between turns; a builder returning None renders nothing.
_MARKER_BODIES: dict[str, Callable[[dict[str, Any]], str | None]] = {
    "mcp.server_unavailable": _mcp_unavailable_body,
    "jail.degraded": _jail_degraded_body,
    "loop.parallel.dispatched": _parallel_dispatched_body,
    "loop.parallel.joined": _parallel_joined_body,
    "loop.parallel.failed": _parallel_failed_body,
    "loop.parallel.compared": _parallel_compared_body,
    "loop.task.queued": _task_queued_body,
    "loop.task.retired": _task_retired_body,
    "loop.standing.set": _standing_set_body,
    "loop.request.refused": _request_refused_body,
    "loop.compact.requested": _compact_requested_body,
    "loop.compact.summarise.done": _compact_done_body,
    "loop.compact.summarise.failed": _compact_failed_body,
    "loop.compact.refused": _compact_refused_body,
}


def _pending_key(event: dict[str, Any], name: str) -> int | str:
    """Return the pairing key of a tool event: the stamped call id, else the name."""
    cid = event.get("call_id")
    return cid if isinstance(cid, int) else name


class TranscriptFold:
    """Fold events one at a time into conversation items.

    A tool call is several items under one `call_id`, each superseding the last: in
    flight at `tool.call`, marked awaiting while the prompt naming it is open,
    settled at `tool.result`. A consumer keeping a list drops the superseded one, as
    `fold_transcript` does. An execution boundary settles every call still open; a
    reader that knows the worker died calls `settle_open_calls` itself.
    """

    def __init__(self) -> None:
        self._thinking: list[str] = []
        self._text: list[str] = []
        # Keyed by call id, since a concurrent panel interleaves same-name calls; an id-less
        # event pairs by name.
        self._pending: dict[int | str, tuple[TranscriptItem, str]] = {}
        # The call each open prompt gates, by prompt id: the answer names only the prompt.
        self._gated: dict[str, int] = {}
        self._verify: tuple[bool, str] | None = None
        self._finish = ""
        self._tools = 0
        self._commits = 0
        self._mode = ""
        self._usd = 0.0
        self._usd_partial = False
        self._first_ep: float | None = None
        self._last_ep: float | None = None
        self._commit_subject = ""
        # A pin renders once, where it enters the conversation, never again at a resume.
        self._pins_shown: set[str] = set()

    def _fold_receipt(self, event: dict[str, Any], etype: str) -> bool:
        """Track the done item's receipt: wall-clock span, cost, last commit subject.

        Args:
            event: The raw event.
            etype: Its type.

        Returns:
            True when the event carried only receipt state.
        """
        if (ep := viewmodel_events.event_epoch(event.get("ts"))) is not None:
            self._last_ep = ep
            if self._first_ep is None:
                self._first_ep = ep
        if etype in viewmodel_events.SESSION_START_EVENTS:
            self._mode = str(event.get("mode", "")) or self._mode
            # The receipt is the execution's own.
            self._first_ep = ep
            self._tools = 0
            self._commits = 0
            self._commit_subject = ""
            self._usd = 0.0
            self._usd_partial = False
            self._verify = None
            self._finish = ""
        if etype == "budget.update":
            self._usd = float(event.get("usd_total", 0) or 0)
            self._usd_partial = bool(event.get("usd_partial")) or self._usd_partial
            return True
        if etype == "loop.auto_commit":
            self._commit_subject = str(event.get("subject", "")).strip()
            return True
        return False

    def _receipt_detail(self) -> str:
        """Return the done item's detail: cost, wall time, counts and commit subject.

        Returns:
            The pieces the journal carried, joined by " · ".
        """
        tools = f"{self._tools} tool{'' if self._tools == 1 else 's'}"
        commits = f"{self._commits} commit{'' if self._commits == 1 else 's'}"
        parts = []
        if self._usd:
            parts.append(budget.format_usd(self._usd, partial=self._usd_partial))
        if self._first_ep is not None and self._last_ep is not None:
            parts.append(f"{max(0, round(self._last_ep - self._first_ep))}s")
        # An ask or a plan never commits, so "0 commits" there is noise.
        counts = (
            tools if self._mode in ("ask", "plan") and not self._commits else f"{tools} · {commits}"
        )
        parts.append(counts)
        if self._commit_subject:
            parts.append(_clip(self._commit_subject, 60))
        return " · ".join(parts)

    def _pin_items(self, event: dict[str, Any], etype: str) -> list[TranscriptItem]:
        """Return the pins this fold has not shown yet as one operator item.

        Args:
            event: The pin event.
            etype: Its type, `loop.pin.added` or `loop.pin.restored`.

        Returns:
            The flushed message, then the pins item; nothing when every pin was shown.
        """
        raw = [event.get("text", "")] if etype == "loop.pin.added" else event.get("pins") or ()
        texts = [str(t).strip() for t in raw] if isinstance(raw, (list, tuple)) else []
        fresh = [t for t in texts if t and t not in self._pins_shown]
        if not fresh:
            return []
        self._pins_shown.update(fresh)
        out = self._flush_message()
        out.append(TranscriptItem("operator", body="pinned: " + " | ".join(fresh)))
        return out

    def feed(self, event: dict[str, Any]) -> list[TranscriptItem]:  # noqa: PLR0911, PLR0912, PLR0915  # one branch per event type
        """Fold one event.

        Args:
            event: The raw event.

        Returns:
            The items the event produced, usually none or one.
        """
        etype = event.get("type", "")
        if self._fold_receipt(event, etype):
            return []
        if etype == "role.call":
            self._thinking.clear()
            self._text.clear()
            return []
        if etype in ("role.thinking_delta", "role.text_delta"):
            buffer = self._thinking if etype == "role.thinking_delta" else self._text
            buffer.append(str(event.get("text", "")))
            return []
        if etype == "role.result":
            # Only the driving role speaks: a side call's answer would read as the agent's.
            settled = "" if self._is_side_call(event) else str(event.get("text", ""))
            return self._flush_message(settled=settled)
        if etype == "tool.call":
            out = self._flush_message()
            out.extend(self._start_tool(event))
            return out
        if etype in ("approval.prompt", "question.prompt"):
            why = "awaiting approval" if etype == "approval.prompt" else "awaiting answer"
            return self._mark_gated_call(event, why)
        if etype in ("approval.answer", "question.answer"):
            return self._release_gated_call(event)
        if etype == "verify.end":
            code = event.get("exit_code")
            dur = float(event.get("duration_s", 0) or 0)
            badge = "✓ pass" if code == 0 else f"✗ exit {code}"
            self._verify = (code == 0, f"{badge} · {dur:.1f}s")
            return []
        if etype == "tool.result":
            return self._complete_tool(event)
        if etype == "diff.updated":
            self._commits += 1
            n = len(str(event.get("patch", "")).splitlines())
            sha = str(event.get("sha", ""))[:12]
            return [TranscriptItem("commit", detail=f"{sha} · {n} lines" if sha else f"{n} lines")]
        build = _MARKER_BODIES.get(etype)
        if build is not None:
            body = build(event)
            if body is None:
                return []
            out = self._flush_message()
            out.append(TranscriptItem("marker", body=body))
            return out
        if etype in ("loop.pin.added", "loop.pin.restored"):
            return self._pin_items(event, etype)
        aside = _BETWEEN_TURNS.get(etype)
        if aside is not None:
            kind, field = aside
            out = self._flush_message()
            body = str(event.get(field, "")).strip()
            if body:
                out.append(TranscriptItem(kind, body=body))
            return out
        if etype in viewmodel_events.SESSION_START_EVENTS:
            return self.settle_open_calls("the run ended")
        if etype == "session.end":
            out = self.settle_open_calls("the run ended")
            out.extend(self._flush_message())
            counts = self._receipt_detail()
            reason = str(event.get("reason", ""))
            all_passed = event.get("all_passed")
            word, detail = listing.status_word(
                finished=True,
                all_passed=all_passed if isinstance(all_passed, bool) else None,
                end_reason=reason,
                scoped=bool(event.get("scoped", False)),
                gate_red=self._verify is not None and not self._verify[0],
            )
            # On a failure or stop the summary is an earlier finish call's and reads as success.
            body = self._finish if reason in ("", "finish_session", "finish_planning") else ""
            out.append(
                TranscriptItem(
                    "done",
                    body=body,
                    # The gate's tri-state: a stop or a gateless finish is neither pass nor fail.
                    ok=all_passed if isinstance(all_passed, bool) else None,
                    detail=counts,
                    name=format.status_label(word, detail),
                )
            )
            return out
        return []

    def _is_side_call(self, event: dict[str, Any]) -> bool:
        """Return whether the result is a side call's."""
        return kinds.is_side_role(str(event.get("role", "")))

    def _flush_message(self, *, settled: str = "") -> list[TranscriptItem]:
        """Emit the buffered thinking and text as items.

        Args:
            settled: The result's text, used only when no deltas arrived.

        Returns:
            A thinking item and a text item, each only when non-empty.
        """
        out: list[TranscriptItem] = []
        thinking = "".join(self._thinking).strip()
        self._thinking.clear()
        if thinking:
            out.append(TranscriptItem("thinking", body=thinking))
        text = "".join(self._text).strip() or settled.strip()
        self._text.clear()
        if text:
            out.append(TranscriptItem("text", body=text))
        return out

    def _start_tool(self, event: dict[str, Any]) -> list[TranscriptItem]:
        """Start a dispatched call's in-flight item, kept until its result.

        Args:
            event: The `tool.call` event.

        Returns:
            The in-flight item, after a superseded same-key call's settled item when an
            id-less journal pairs by name; nothing for a finish tool, whose summary is
            the done line's.
        """
        name = str(event.get("name", ""))
        raw_args = event.get("args")
        args = raw_args if isinstance(raw_args, dict) else {}
        if name in _FINISH_TOOLS:
            self._finish = str(args.get("summary", "")).strip()
            return []
        self._tools += 1
        key = _pending_key(event, name)
        out: list[TranscriptItem] = []
        if key in self._pending:
            first, _preview = self._pending.pop(key)
            out.append(dataclasses.replace(first, ok=False, detail="no result (superseded)"))
        pending = TranscriptItem("tool", name=name, arg=salient_arg(args), call_id=str(key))
        self._pending[key] = (pending, _call_preview(name, args))
        self._verify = None
        out.append(pending)
        return out

    def _mark_gated_call(self, event: dict[str, Any], why: str) -> list[TranscriptItem]:
        """Re-emit the call a prompt gates, marked with why it waits.

        Args:
            event: The prompt event.
            why: The waiting detail.

        Returns:
            The marked item; nothing for a prompt naming no call in flight.
        """
        key = event.get("call_id")
        if not isinstance(key, int) or key not in self._pending:
            return []
        self._gated[str(event.get("id", ""))] = key
        return [self._redetail(key, why)]

    def _release_gated_call(self, event: dict[str, Any]) -> list[TranscriptItem]:
        """Return an answered prompt's call re-emitted as running, nothing when none is gated."""
        key = self._gated.pop(str(event.get("id", "")), None)
        if key is None or key not in self._pending:
            return []
        return [self._redetail(key, "")]

    def _redetail(self, key: int, detail: str) -> TranscriptItem:
        """Return the pending call's item with a new detail, kept as the pending one."""
        pending, preview = self._pending[key]
        marked = dataclasses.replace(pending, detail=detail)
        self._pending[key] = (marked, preview)
        return marked

    def _complete_tool(self, event: dict[str, Any]) -> list[TranscriptItem]:
        """Settle a call with its result.

        Args:
            event: The `tool.result` event.

        Returns:
            The settled item; nothing for a finish tool's result or an unmatched one.
        """
        name = str(event.get("name", ""))
        key = _pending_key(event, name)
        if key not in self._pending:
            return []
        pending, call_preview = self._pending.pop(key)
        if name == "run_verify_command" and self._verify is not None:
            ok, detail = self._verify
            self._verify = None
        else:
            ok = viewmodel_events.tool_result_ok(event.get("ok"))
            detail = str(event.get("summary", "")).strip()
        # A failed tool's tail is why; a passed one's is its substance.
        if not ok:
            tail = str(event.get("stderr_tail") or event.get("stdout_tail") or "").strip()
        elif name in ("run_command", "run_metric_command"):
            tail = str(event.get("stdout_tail") or "").strip()
        elif name == "read_file":
            head = str(event.get("head_tail") or "").strip("\n")
            total = event.get("lines_total")
            tail = f"{head}\n…({total} lines)" if head and total else head
        else:
            tail = call_preview
        return [
            dataclasses.replace(
                pending,
                ok=ok,
                detail=scrub_terminal_controls(detail),
                tail=scrub_terminal_controls(tail),
            )
        ]

    def settle_open_calls(self, why: str) -> list[TranscriptItem]:
        """Settle every call still in flight as one that never returned.

        Args:
            why: The reason, "the run ended" at an execution boundary or "the run died"
                from a reader whose probe found the worker gone.

        Returns:
            The settled items.
        """
        out = [
            dataclasses.replace(pending, ok=False, detail=f"no result ({why})")
            for pending, _preview in self._pending.values()
        ]
        self._pending.clear()
        self._gated.clear()
        return out


def _land(out: list[TranscriptItem], item: TranscriptItem) -> None:
    """Append an item, a tool item first dropping the in-flight one it supersedes."""
    if item.kind == "tool":
        for i in range(len(out) - 1, -1, -1):
            earlier = out[i]
            if earlier.kind == "tool" and earlier.ok is None and earlier.call_id == item.call_id:
                del out[i]
                break
    out.append(item)


def fold_transcript(
    events: list[dict[str, Any]], *, worker_dead: bool = False
) -> list[TranscriptItem]:
    """Fold a whole event stream into its conversation items, one per tool call.

    Args:
        events: The raw events, in order.
        worker_dead: The caller probed the worker and it is gone, so the calls still
            open at the end are settled as never returning.

    Returns:
        The items, in landing order.
    """
    fold = TranscriptFold()
    out: list[TranscriptItem] = []
    for event in events:
        for item in fold.feed(event):
            _land(out, item)
    if worker_dead:
        for item in fold.settle_open_calls("the run died"):
            _land(out, item)
    return out


def _outcome_word(item: TranscriptItem) -> str:
    """Return a tool item's bracket word: its verdict, or why it waits, else "running"."""
    if item.ok is None:
        return item.detail or "running"
    return "ok" if item.ok else "FAILED"


def restate(events: list[dict[str, Any]], *, worker_dead: bool = False) -> str:
    """Restate the conversation since the operator's last prompt or steer.

    Rendered from the journal, never a model call, so every surface answers
    `/restate` locally and free.

    Args:
        events: The raw events, in order.
        worker_dead: The caller probed the worker and it is gone.

    Returns:
        The operator's words, then assistant prose kept whole with tool calls and
        markers one line each; a notice when there is no operator input yet.
    """
    last: int | None = None
    for i, event in enumerate(events):
        if str(event.get("type", "")) in _OPERATOR_TEXT:
            last = i
    if last is None:
        return "nothing to restate: this session has no operator input yet"
    anchor = events[last]
    said = str(anchor.get(_OPERATOR_TEXT[str(anchor["type"])], "")).strip()
    lines = [f"you said: {_clip(said, 200)}"]
    for item in fold_transcript(events[last:], worker_dead=worker_dead):
        if item.kind in ("thinking", "operator"):
            continue
        if item.kind == "text":
            body = item.body.strip()
            if body:
                lines.extend(("", body))
        elif item.kind == "tool":
            arg = f" {item.arg}" if item.arg else ""
            detail = f": {_clip(item.detail, 80)}" if item.detail else ""
            lines.append(f"  [{_outcome_word(item)}] {item.name}{arg}{detail}")
        else:
            body = (item.body or item.detail).strip()
            if body:
                lines.append(f"  {body}")
    if len(lines) == 1:
        lines.append("(nothing has happened since)")
    return "\n".join(lines)
