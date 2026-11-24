# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Project a run's journal into the session's updates as the run writes it.

A tail reads the journal a poll behind the run and sends each row's updates. The
lifecycle's own lines wait in `ProseOrder` until the tail has read past the point
they were said at.
"""

from __future__ import annotations

import dataclasses
import pathlib
import sys
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING

from agent6 import kinds
from agent6.sessions import journal
from agent6.ui.acp import updates
from agent6.viewmodel import tail as viewmodel_tail
from agent6.viewmodel import transcript, transcript_items

if TYPE_CHECKING:
    from agent6.ui.acp import runner
    from agent6.ui.acp import server as acp_server
    from agent6.ui.acp import session as acp_session


@dataclasses.dataclass
class ProseOrder:
    """Queue the lifecycle's lines for the journal tail to emit in order.

    The tail projects the journal a poll behind the run thread, so a line sent as
    said would land before the execution's last tool calls. Each line is stamped with
    the journal's size when said, and emitted once the tail has read past it.

    Attributes:
        server: The connection.
        acp_session_id: The conversation the lines address.
        logs_path: The journal.
    """

    server: acp_server.ACPServer
    acp_session_id: str
    logs_path: pathlib.Path
    _lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)
    _pending: list[tuple[int, str]] = dataclasses.field(default_factory=list)

    def say(self, text: str) -> None:
        """Queue a line, stamped with the journal's size now."""
        with self._lock:
            self._pending.append((viewmodel_tail.journal_size(self.logs_path), text))

    def flush(self, consumed: int | None) -> None:
        """Emit every line stamped at or before the consumed journal size; all for None."""
        with self._lock:
            due = [t for stamp, t in self._pending if consumed is None or stamp <= consumed]
            self._pending = [
                (s, t) for s, t in self._pending if consumed is not None and s > consumed
            ]
        for text in due:
            # A note or warning already carries the marker `notice` adds.
            self.server.update(self.acp_session_id, updates.notice(text.removeprefix("[agent6] ")))


@dataclasses.dataclass(frozen=True, slots=True)
class Sending:
    """What the tail knows about one fold item as it sends it.

    Attributes:
        paths: The paths the tool result named.
        spoken: The last message streamed; an end summary repeating it is not sent again,
            and the status line keeps its own paragraph.
        prompt: The prompt this execution runs.
        objects_dir: The run's object directory; None sends an edit without its diffs.
        message_id: The id of the message the item becomes.
        call_message_id: The id of the model's message in flight.
        cancelled: The item's row ends a run the operator stopped.
    """

    paths: tuple[str, ...]
    spoken: str
    prompt: str
    objects_dir: pathlib.Path | None
    message_id: str = ""
    call_message_id: str = ""
    cancelled: bool = False


def project(  # noqa: PLR0912
    server: acp_server.ACPServer,
    session: acp_session.Session,
    logs_path: pathlib.Path,
    stop: Callable[[], bool],
    journal_before: int | None,
    announced: runner.Announced,
    order: ProseOrder | None = None,
    info: updates.RunInfo | None = None,
    prompt: str = "",
    objects_dir: pathlib.Path | None = None,
) -> None:
    """Project the run's journal into `session/update` as it is written.

    The lifecycle's own lines take their place between events; the ending also
    goes to stderr, since the lifecycle prints none of its own.

    Args:
        server: The connection.
        session: The session.
        logs_path: The journal.
        stop: Polled by the tail; True ends it.
        journal_before: Where to start on a resumed run, whose earlier executions the
            editor already rendered; None replays the whole journal.
        announced: The execution's register of announced calls.
        order: The lifecycle's queued lines.
        info: The title, commands and usage projection; None sends none.
        prompt: The prompt this execution runs; the journal's echo of it is not sent back.
        objects_dir: The run's object directory; None sends an edit without its diffs.
    """
    fold = transcript.TranscriptFold()
    plan = updates.TaskPlan()
    ids = updates.MessageIds()
    streamed: set[str] = set()
    # The last message streamed: an end summary repeating it is not sent again.
    spoken: list[str] = []
    consumed = [0]

    def _at(position: int) -> None:
        consumed[0] = position

    try:
        for event in viewmodel_tail.tail_events(
            logs_path,
            stop_when_finished=True,
            should_stop=stop,
            start_at=journal_before,
            on_position=_at,
        ):
            event_type = str(event.get("type", ""))
            ids.feed(event)
            if event_type in journal.SESSION_START_EVENTS:
                announced.execution_id = str(event.get("execution_id") or "")
                if session.turn is not None:
                    # The engine holds the prompt: recorded as its task, or queued as its steer.
                    session.turn.begin(prompt)
            for update in info.feed(event) if info is not None else ():
                server.update(session.acp_id, update)
            for update in plan.feed(event):
                server.update(session.acp_id, update)
            paths = updates.result_paths(event) if event_type == "tool.result" else ()
            if event_type == "model.call":
                streamed.clear()
            is_delta = event_type in ("model.thinking_delta", "model.text_delta")
            side_delta = is_delta and kinds.is_side_actor(str(event.get("actor", "")))
            items = [] if side_delta else fold.feed(event)
            if is_delta and not side_delta:
                kind = "thinking" if event_type == "model.thinking_delta" else "text"
                if kind == "text":
                    if "text" not in streamed:
                        spoken.clear()  # a new message begins
                    spoken.append(str(event.get("text", "")))
                streamed.add(kind)
                for update in updates.updates_for(
                    transcript_items.TranscriptItem(kind, body=str(event.get("text", ""))),
                    streamed=True,
                    message_id=ids.call,
                ):
                    server.update(session.acp_id, update)
            for index, item in enumerate(items):
                if item.kind in streamed:
                    continue
                sending = Sending(
                    paths,
                    "".join(spoken),
                    prompt,
                    objects_dir,
                    message_id=ids.of(item, event, index),
                    call_message_id=ids.call,
                    cancelled=updates.operator_stopped(event),
                )
                send_item(server, session, item, announced, sending)
            if event_type == "model.result":
                streamed.clear()
            if order is not None:
                order.flush(consumed[0])
    finally:
        announced.close()
        for item in fold.settle_open_calls("the run died"):
            within = updates.execution_call_id(item)
            for update in updates.updates_for(
                item,
                wire_id=updates.wire_call_id(session.session_id, announced.execution_id, within),
                announced=within in announced,
                cwd=session.cwd,
                cancelled=session.cancelled,
            ):
                server.update(session.acp_id, update)
        if order is not None:
            order.flush(None)


def send_item(
    server: acp_server.ACPServer,
    session: acp_session.Session,
    item: transcript_items.TranscriptItem,
    announced: runner.Announced,
    sending: Sending,
) -> None:
    """Send one fold item's updates; the ending also goes to stderr.

    Args:
        server: The connection.
        session: The session.
        item: The fold item.
        announced: The execution's register of announced calls.
        sending: What the tail knows about the item.
    """
    if item.kind == "task" or (
        item.kind == "operator" and item.body.strip() == sending.prompt.strip()
    ):
        # The execution's start row reported its own prompt; another front-end's steer is news.
        return
    if item.kind == "operator" and sending.message_id in session.inserted:
        return  # reported when this connection took it as a prompt
    within = updates.execution_call_id(item) if item.kind in ("tool", "verify") else ""
    wire_id = (
        updates.wire_call_id(session.session_id, announced.execution_id, within) if within else ""
    )
    wire_item = item
    spoken = sending.spoken
    if item.kind == "done" and spoken.strip() and item.body.strip() == spoken.strip():
        wire_item = dataclasses.replace(item, body="")
        trailing = len(spoken) - len(spoken.rstrip("\n"))
        if pad := "\n" * max(0, 2 - trailing):
            for update in updates.updates_for(
                transcript_items.TranscriptItem("text", body=pad),
                streamed=True,
                message_id=sending.call_message_id,
            ):
                server.update(session.acp_id, update)
    for update in updates.updates_for(
        wire_item,
        wire_id=wire_id,
        announced=within in announced,
        cwd=session.cwd,
        paths=sending.paths,
        objects_dir=sending.objects_dir,
        message_id=sending.message_id,
        cancelled=sending.cancelled,
    ):
        server.update(session.acp_id, update)
    if within:
        announced.add(within)
    elif item.kind == "done":
        ending = (item.body.strip(), updates.status_line(item))
        print("\n\n".join(part for part in ending if part), file=sys.stderr)
