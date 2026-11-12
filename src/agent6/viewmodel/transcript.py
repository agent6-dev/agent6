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

from collections.abc import Iterable
from typing import Any

from agent6 import budget, evidence, kinds
from agent6.sessions import journal
from agent6.viewmodel import events as viewmodel_events
from agent6.viewmodel import format, review, session_status, transcript_items, transcript_text

# Text characters, not graphics, so every terminal font renders them.
CALL = "→"
RESULT = "└"
COMMIT = "✎"
THINK = "·"
DONE = "●"
OPERATOR = "❯"  # noqa: RUF001  # a prompt glyph, not a mistyped >

# Events that render between turns: type -> (kind, the field holding the text).
_BETWEEN_TURNS: dict[str, tuple[transcript_items.ItemKind, str]] = {
    "loop.steer.feedback": ("marker", "text"),
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
        text = str(event.get(field, ""))
        if text.strip() and (not out or out[-1] != text):
            out.append(text)
    return out


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
        self._calls = transcript_items.PendingItems()
        # The last gate verdict (ok, timed out), kept past the tool result that consumed it.
        self._gate: tuple[bool, bool] | None = None
        self._stranded = False
        self._ended = False
        # Items a receipt event produced, handed out with the next feed.
        self._deferred: list[transcript_items.TranscriptItem] = []
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
        if etype in journal.SESSION_START_EVENTS:
            self._mode = str(event.get("mode", "")) or self._mode
            # The receipt is the execution's own.
            self._first_ep = ep
            self._tools = 0
            self._commits = 0
            self._commit_subject = ""
            self._usd = 0.0
            self._usd_partial = False
            self._gate = None
            self._stranded = False
            self._ended = False
            self._finish = ""
        if notices := review.notice_texts(event):
            self._deferred.extend(self._flush_message())
            self._deferred.extend(
                transcript_items.TranscriptItem("marker", body=t) for t in notices
            )
            return True
        if etype == "budget.update":
            self._usd = float(event.get("usd_total", 0) or 0)
            self._usd_partial = bool(event.get("usd_partial")) or self._usd_partial
            return True
        if etype == "loop.auto_commit":
            self._commit_subject = str(event.get("subject", "")).strip()
            return True
        if etype == "verify.end":
            self._gate = (
                event.get("exit_code") == 0,
                viewmodel_events.verify_limit_hit(evidence.outcome(event), event.get("timeout_s"))
                > 0,
            )
        if etype == "session.stranded":
            self._stranded = True
            if self._ended:
                # The end line is out: the stranded fact, journaled after it, gets its own.
                self._deferred.append(
                    transcript_items.TranscriptItem(
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
            parts.append(transcript_text.clip(self._commit_subject, 60))
        return " · ".join(parts)

    def _pin_items(
        self, event: dict[str, Any], etype: str
    ) -> list[transcript_items.TranscriptItem]:
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
        out.append(transcript_items.TranscriptItem("marker", body="pinned: " + " | ".join(fresh)))
        return out

    def feed(self, event: dict[str, Any]) -> list[transcript_items.TranscriptItem]:  # noqa: PLR0911, PLR0912, PLR0915  # one branch per event type
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
            name = str(event.get("name", ""))
            if name in viewmodel_events.FINISH_TOOLS:
                args = event.get("args")
                self._finish = (
                    str(args.get("summary", "")).strip() if isinstance(args, dict) else ""
                )
            else:
                self._tools += 1
            out.extend(self._calls.start_tool(event))
            return out
        if (prompted := self._calls.prompt_items(etype, event)) is not None:
            return prompted
        if etype in ("verify.start", "verify.end"):
            return [*self._flush_message(), self._calls.verification(event)]
        if etype == "tool.result":
            return self._calls.complete_tool(event)
        if etype == "diff.updated":
            self._commits += 1
            n = len(str(event.get("patch", "")).splitlines())
            sha = str(event.get("sha", ""))[:12]
            return [
                transcript_items.TranscriptItem(
                    "commit", detail=f"{sha} · {n} lines" if sha else f"{n} lines"
                )
            ]
        build = transcript_text.MARKER_BODIES.get(etype)
        if build is not None:
            body = build(event)
            if body is None:
                return []
            out = self._flush_message()
            out.append(transcript_items.TranscriptItem("marker", body=body))
            return out
        if etype in ("loop.pin.added", "loop.pin.restored"):
            return self._pin_items(event, etype)
        aside = _BETWEEN_TURNS.get(etype)
        if aside is not None:
            kind, field = aside
            out = self._flush_message()
            body = str(event.get(field, ""))
            if body:
                out.append(transcript_items.TranscriptItem(kind, body=body))
            return out
        if etype in journal.SESSION_START_EVENTS:
            out = self.settle_open_calls("the run ended")
            if etype == "session.start":
                task = str(event.get("user_task", ""))
                if task:
                    out.append(transcript_items.TranscriptItem("task", body=task))
            return out
        if etype == "session.stop":
            out = self.settle_open_calls("the run ended")
            out.extend(self._flush_message())
            counts = self._receipt_detail()
            reason = str(event.get("reason", ""))
            verification = viewmodel_events.verification_word(event.get("verification"))
            word, detail = session_status.status_word(
                finished=True,
                verification=verification,
                stop_reason=reason,
                scoped=bool(event.get("scoped", False)),
                gate_red=self._gate is not None and not self._gate[0],
                gate_timed_out=self._gate is not None and self._gate[1],
                stranded=self._stranded,
            )
            body = session_status.end_text(reason, str(event.get("summary", "")), self._finish)
            self._ended = True
            out.append(
                transcript_items.TranscriptItem(
                    "done",
                    body=body,
                    # The gate's tri-state: a stop or a gateless finish is neither pass nor fail.
                    ok=(verification == "passed") if verification in ("passed", "failed") else None,
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

    def _flush_message(self, *, settled: str = "") -> list[transcript_items.TranscriptItem]:
        """Emit the buffered thinking and text as items.

        Args:
            settled: The result's text, used only when no deltas arrived.

        Returns:
            A thinking item and a text item, each only when non-empty.
        """
        out: list[transcript_items.TranscriptItem] = []
        thinking = "".join(self._thinking).strip()
        self._thinking.clear()
        if thinking:
            out.append(transcript_items.TranscriptItem("thinking", body=thinking))
        text = "".join(self._text).strip() or settled.strip()
        self._text.clear()
        if text:
            out.append(transcript_items.TranscriptItem("text", body=text))
        return out

    def settle_open_calls(self, why: str) -> list[transcript_items.TranscriptItem]:
        """Return the settled tools and verification attempts that did not return."""
        return self._calls.settle_open_calls(why)


def _land(
    out: list[transcript_items.TranscriptItem], item: transcript_items.TranscriptItem
) -> None:
    """Replace a verification in place, or append an item after dropping its pending call."""
    if item.kind in ("tool", "verify"):
        for i in range(len(out) - 1, -1, -1):
            earlier = out[i]
            if (
                earlier.kind in ("tool", "verify")
                and earlier.ok is None
                and earlier.call_id == item.call_id
            ):
                if item.kind == "verify":
                    out[i] = item
                    return
                del out[i]
                break
    out.append(item)


def fold_transcript(
    events: list[dict[str, Any]], *, worker_dead: bool = False
) -> list[transcript_items.TranscriptItem]:
    """Fold a whole event stream into its conversation items, one per tool call.

    Args:
        events: The raw events, in order.
        worker_dead: The caller probed the worker and it is gone, so the calls still
            open at the end are settled as never returning.

    Returns:
        The items, in landing order.
    """
    fold = TranscriptFold()
    out: list[transcript_items.TranscriptItem] = []
    for event in events:
        for item in fold.feed(event):
            _land(out, item)
    if worker_dead:
        for item in fold.settle_open_calls("the run died"):
            _land(out, item)
    return out


def _outcome_word(item: transcript_items.TranscriptItem) -> str:
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
    lines = [f"you said: {transcript_text.clip(said, 200)}"]
    for item in fold_transcript(events[last:], worker_dead=worker_dead):
        if item.kind in ("thinking", "operator"):
            continue
        if item.kind == "text":
            body = item.body.strip()
            if body:
                lines.extend(("", body))
        elif item.kind == "tool":
            arg = f" {item.arg}" if item.arg else ""
            detail = f": {transcript_text.clip(item.detail, 80)}" if item.detail else ""
            lines.append(f"  [{_outcome_word(item)}] {item.name}{arg}{detail}")
        else:
            body = (item.body or item.detail).strip()
            if body:
                lines.append(f"  {body}")
    if len(lines) == 1:
        lines.append("(nothing has happened since)")
    return "\n".join(lines)
