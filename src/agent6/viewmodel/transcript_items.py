"""Pair the transcript's tool calls, verification attempts and approval prompts."""

from __future__ import annotations

import dataclasses
import shlex
from collections.abc import Mapping
from typing import Any, Literal

from agent6 import terminal
from agent6.viewmodel import events as viewmodel_events
from agent6.viewmodel import log_line, transcript_text

ItemKind = Literal[
    "thinking", "text", "tool", "verify", "commit", "marker", "done", "operator", "task"
]


@dataclasses.dataclass(frozen=True, slots=True)
class TranscriptItem:
    """One rendered conversation step; only the fields its `kind` needs are set.

    Attributes:
        kind: The item's kind.
        body: The thinking, text or marker prose; the final summary for `done`.
        name: The tool's name; the status label for `done`.
        arg: The tool's salient argument (a path, a pattern, a command); the end reason
            for `done`, which says whose sentence the body is.
        args_full: Every tool argument without abbreviation, for the expanded item.
        ok: The tool or run outcome; None while in flight or when not applicable.
        detail: The tool result's summary, the verify badge or the commit and done
            metadata; for a call in flight, why it waits ("awaiting approval").
        tail: A tool's captured output tail.
        call_id: The tool call or verification attempt identity, retained through its result.
    """

    kind: ItemKind
    body: str = ""
    name: str = ""
    arg: str = ""
    ok: bool | None = None
    detail: str = ""
    tail: str = ""
    call_id: str = ""
    args_full: str = ""


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


class PendingItems:
    """Pair tool calls, verification attempts and the prompts that gate them."""

    def __init__(self) -> None:
        self._pending: dict[int | str, tuple[TranscriptItem, str]] = {}
        self._gated: dict[str, int | str] = {}
        self._verifications: dict[str, int | str] = {}
        self._next_verify = 0

    def start_tool(self, event: dict[str, Any]) -> list[TranscriptItem]:
        """Start a dispatched call's in-flight item, kept until its result.

        Args:
            event: The `tool.call` event.

        Returns:
            The in-flight item, after a superseded same-key call's settled item when an
            id-less journal pairs by name; nothing for a finish tool, whose summary is
            the done line's.
        """
        name = str(event.get("name", ""))
        args = viewmodel_events.call_args(event)
        if name in viewmodel_events.FINISH_TOOLS:
            return []
        key = _pending_key(event, name)
        out: list[TranscriptItem] = []
        if key in self._pending:
            first, _preview = self._pending.pop(key)
            out.append(dataclasses.replace(first, ok=False, detail="no result (superseded)"))
        pending = TranscriptItem(
            "tool",
            name=name,
            arg=transcript_text.salient_arg(args),
            call_id=str(key),
            args_full=terminal.scrub_terminal_controls(log_line.render_args(args, max_value=None)),
        )
        self._pending[key] = (pending, transcript_text.call_preview(name, args))
        out.append(pending)
        return out

    def prompt_items(self, etype: str, event: dict[str, Any]) -> list[TranscriptItem] | None:
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
        if key is None:
            # The harness gate's prompt names no call: it gates the verification waiting on it.
            key = next(
                (
                    k
                    for k, (item, _) in reversed(self._pending.items())
                    if item.kind == "verify" and item.detail == "waiting for approval"
                ),
                None,
            )
        if not isinstance(key, (int, str)) or key not in self._pending:
            return []
        self._gated[str(event.get("id", ""))] = key
        return [self._redetail(key, why)]

    def _release_gated_call(self, event: dict[str, Any]) -> list[TranscriptItem]:
        """Return an answered prompt's call re-emitted as running, nothing when none is gated."""
        key = self._gated.pop(str(event.get("id", "")), None)
        if key is None or key not in self._pending:
            return []
        detail = "denied" if event.get("approved") is False else ""
        return [self._redetail(key, detail)]

    def _redetail(self, key: int | str, detail: str) -> TranscriptItem:
        """Return the pending call's item with a new detail, kept as the pending one."""
        pending, preview = self._pending[key]
        body = pending.body
        if pending.kind == "verify":
            detail = "waiting for approval" if detail == "awaiting approval" else detail
            body = transcript_text.verify_body(pending.arg, detail)
        marked = dataclasses.replace(pending, detail=detail, body=body)
        self._pending[key] = (marked, preview)
        return marked

    def verification(self, event: dict[str, Any]) -> TranscriptItem:
        """Return a verification item, replacing its model tool item when present."""
        starting = event.get("type") == "verify.start"
        verify_id = str(event.get("verify_id", ""))
        key: int | str | None = None
        if not starting:
            if verify_id:
                key = self._verifications.pop(verify_id, None)
            elif self._verifications:
                _, key = self._verifications.popitem()
        if key is None:
            cid = event.get("call_id")
            if isinstance(cid, int):
                key = cid
            elif "call_id" not in event:
                key = next(
                    (
                        k
                        for k, (it, _) in self._pending.items()
                        if it.kind == "tool" and it.name == "run_verify_command"
                    ),
                    None,
                )
        if key is None:
            self._next_verify += 1
            key = f"verify:{verify_id or self._next_verify}"
        previous = self._pending.pop(key, (TranscriptItem("verify", call_id=str(key)), ""))[0]
        cmd = event.get("command")
        command = (
            shlex.join(str(a) for a in cmd) if isinstance(cmd, (list, tuple)) else previous.arg
        )
        command = terminal.scrub_terminal_controls(command)
        detail = (
            ("waiting for approval" if event.get("approval_required") else "")
            if starting
            else transcript_text.verify_result(event)
        )
        item = dataclasses.replace(
            previous,
            kind="verify",
            name="run_verify_command",
            arg=command,
            body=transcript_text.verify_body(command, detail),
            detail=detail,
            ok=None if starting else event.get("exit_code") == 0,
            tail=terminal.scrub_terminal_controls(
                str(event.get("stderr_tail") or event.get("stdout_tail") or "").strip()
            )
            if not starting and event.get("exit_code") != 0
            else "",
        )
        if starting:
            self._verifications[verify_id or str(key)] = key
            self._pending[key] = (item, "")
        return item

    def complete_tool(self, event: dict[str, Any]) -> list[TranscriptItem]:
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
            dataclasses.replace(
                pending,
                ok=False,
                detail=f"no result ({why})",
                body=transcript_text.verify_body(pending.arg, f"no result ({why})")
                if pending.kind == "verify"
                else pending.body,
            )
            for pending, _preview in self._pending.values()
        ]
        self._pending.clear()
        self._gated.clear()
        self._verifications.clear()
        return out
