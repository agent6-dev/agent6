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
from collections.abc import Iterable, Mapping
from typing import Any, Literal

from agent6 import budget, evidence, kinds, terminal
from agent6.viewmodel import _session_status, _transcript_text, format, review
from agent6.viewmodel import events as viewmodel_events

# Text characters, not graphics, so every terminal font renders them.
CALL = "→"
RESULT = "└"
COMMIT = "✎"
THINK = "·"
DONE = "●"
OPERATOR = "❯"  # noqa: RUF001  # a prompt glyph, not a mistyped >

ItemKind = Literal["thinking", "text", "tool", "commit", "marker", "done", "operator", "task"]


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
        arg: The tool's salient argument (a path, a pattern, a command); the end reason
            for `done`, which says whose sentence the body is.
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


def _pending_key(event: dict[str, Any], name: str) -> int | str:
    """Return the pairing key of a tool event: the stamped call id, else the name."""
    cid = event.get("call_id")
    return cid if isinstance(cid, int) else name


def queued_question_line(event: Mapping[str, Any]) -> str:
    """Return the line naming a question the model queued: its id, each question, its options."""
    asked = "; ".join(
        str(q.get("question", ""))
        + (f" (options: {', '.join(str(o) for o in q['options'])})" if q.get("options") else "")
        for q in (event.get("questions") or [])
        if isinstance(q, dict)
    )
    return f"{event.get('id', '')} queued: {asked}"


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
        # The last gate run a tool call has not claimed: passed, its badge, ended on its timeout.
        self._verify: tuple[bool, str, bool] | None = None
        # The last gate verdict (ok, timed out), kept past the tool result that consumed it.
        self._gate: tuple[bool, bool] | None = None
        self._stranded = False
        self._ended = False
        # Items a receipt event produced, handed out with the next feed.
        self._deferred: list[TranscriptItem] = []
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
            self._gate = None
            self._stranded = False
            self._ended = False
            self._finish = ""
        if notices := review.notice_texts(event):
            self._deferred.extend(self._flush_message())
            self._deferred.extend(TranscriptItem("marker", body=t) for t in notices)
            return True
        if etype == "budget.update":
            self._usd = float(event.get("usd_total", 0) or 0)
            self._usd_partial = bool(event.get("usd_partial")) or self._usd_partial
            return True
        if etype == "loop.auto_commit":
            self._commit_subject = str(event.get("subject", "")).strip()
            return True
        if etype == "session.stranded":
            self._stranded = True
            if self._ended:
                # The end line is out: the stranded fact, journaled after it, gets its own.
                self._deferred.append(
                    TranscriptItem(
                        "marker",
                        body="no commit landed: the edits sit uncommitted (not committed)",
                    )
                )
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
            parts.append(_transcript_text.clip(self._commit_subject, 60))
        return " · ".join(parts)

    def _pin_items(self, event: dict[str, Any], etype: str) -> list[TranscriptItem]:
        """Return the pins this fold has not shown yet as one marker.

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
        out.append(TranscriptItem("marker", body="pinned: " + " | ".join(fresh)))
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
            deferred, self._deferred = self._deferred, []
            return deferred
        if etype == "role.call":
            self._thinking.clear()
            self._text.clear()
            return []
        # Side streams belong to the live pane, not the driving role's settled speech.
        if etype in ("role.thinking_delta", "role.text_delta") and not self._is_side_call(event):
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
        if (prompted := self._prompt_items(etype, event)) is not None:
            return prompted
        if etype == "verify.end":
            code = event.get("exit_code")
            dur = float(event.get("duration_s", 0) or 0)
            label = evidence.result_label(
                code, evidence.outcome(event), str(event.get("reason") or "")
            )
            badge = "✓ pass" if code == 0 else f"✗ {label}"
            timed_out = viewmodel_events.verify_limit_hit(code, event.get("timeout_s")) > 0
            self._verify = (code == 0, f"{badge} · {dur:.1f}s", timed_out)
            self._gate = (code == 0, timed_out)
            return []
        if etype == "tool.result":
            return self._complete_tool(event)
        if etype == "diff.updated":
            self._commits += 1
            n = len(str(event.get("patch", "")).splitlines())
            sha = str(event.get("sha", ""))[:12]
            return [TranscriptItem("commit", detail=f"{sha} · {n} lines" if sha else f"{n} lines")]
        build = _transcript_text.MARKER_BODIES.get(etype)
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
            out = self.settle_open_calls("the run ended")
            if etype == "session.start":
                task = _session_status.task_snippet(str(event.get("user_task", "")), max_chars=200)
                if task:
                    out.append(TranscriptItem("task", body=task))
            return out
        if etype == "session.end":
            out = self.settle_open_calls("the run ended")
            out.extend(self._flush_message())
            counts = self._receipt_detail()
            reason = str(event.get("reason", ""))
            all_passed = event.get("all_passed")
            word, detail = _session_status.status_word(
                finished=True,
                all_passed=all_passed if isinstance(all_passed, bool) else None,
                end_reason=reason,
                scoped=bool(event.get("scoped", False)),
                gate_red=self._gate is not None and not self._gate[0],
                gate_timed_out=self._gate is not None and self._gate[1],
                stranded=self._stranded,
            )
            body = _session_status.end_text(reason, str(event.get("summary", "")), self._finish)
            self._ended = True
            out.append(
                TranscriptItem(
                    "done",
                    body=body,
                    # The gate's tri-state: a stop or a gateless finish is neither pass nor fail.
                    ok=all_passed if isinstance(all_passed, bool) else None,
                    detail=counts,
                    name=format.status_label(word, detail),
                    arg=reason,
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
        if name in viewmodel_events.FINISH_TOOLS:
            self._finish = str(args.get("summary", "")).strip()
            return []
        self._tools += 1
        key = _pending_key(event, name)
        out: list[TranscriptItem] = []
        if key in self._pending:
            first, _preview = self._pending.pop(key)
            out.append(dataclasses.replace(first, ok=False, detail="no result (superseded)"))
        pending = TranscriptItem(
            "tool", name=name, arg=_transcript_text.salient_arg(args), call_id=str(key)
        )
        self._pending[key] = (pending, _transcript_text.call_preview(name, args))
        self._verify = None
        out.append(pending)
        return out

    def _prompt_items(self, etype: str, event: dict[str, Any]) -> list[TranscriptItem] | None:
        """Return the items for a prompt or answer event; None for any other event.

        A waiting prompt marks the call it gates and its answer releases it; a queued question
        gates nothing, so it and its late answer read as markers naming them.

        Args:
            etype: The event type.
            event: The event.

        Returns:
            The items, or None when the event is not a prompt or an answer.
        """
        if etype == "question.prompt" and event.get("wait") is False:
            return [TranscriptItem("marker", body=queued_question_line(event))]
        if etype == "question.answer" and event.get("source") == "late":
            given = ", ".join(str(a) for a in (event.get("answers") or []))
            return [TranscriptItem("marker", body=f"{event.get('id', '')} answered: {given}")]
        if etype in ("approval.prompt", "question.prompt"):
            why = "awaiting approval" if etype == "approval.prompt" else "awaiting answer"
            return self._mark_gated_call(event, why)
        if etype in ("approval.answer", "question.answer"):
            return self._release_gated_call(event)
        return None

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
            ok, detail, _timed_out = self._verify
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
                detail=terminal.scrub_terminal_controls(detail),
                tail=terminal.scrub_terminal_controls(tail),
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
    lines = [f"you said: {_transcript_text.clip(said, 200)}"]
    for item in fold_transcript(events[last:], worker_dead=worker_dead):
        if item.kind in ("thinking", "operator"):
            continue
        if item.kind == "text":
            body = item.body.strip()
            if body:
                lines.extend(("", body))
        elif item.kind == "tool":
            arg = f" {item.arg}" if item.arg else ""
            detail = f": {_transcript_text.clip(item.detail, 80)}" if item.detail else ""
            lines.append(f"  [{_outcome_word(item)}] {item.name}{arg}{detail}")
        else:
            body = (item.body or item.detail).strip()
            if body:
                lines.append(f"  {body}")
    if len(lines) == 1:
        lines.append("(nothing has happened since)")
    return "\n".join(lines)
