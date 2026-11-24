# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Project the shared transcript fold into the session's updates.

Projecting the fold every other surface renders keeps a fourth surface from
disagreeing about what happened. The updates are `model`'s; `wire` renders them.
"""

from __future__ import annotations

import dataclasses
import pathlib
import uuid
from collections.abc import Callable, Iterable
from typing import Any

from agent6 import directive, evidence, kinds
from agent6.app.parallel import launch as child_launch
from agent6.sessions import journal
from agent6.ui.acp import model
from agent6.viewmodel import events as viewmodel_events
from agent6.viewmodel import receipt, transcript, transcript_items
from agent6.viewmodel.session import records as state_types
from agent6.viewmodel.session import state as session_state
from agent6.viewmodel.session import status as session_status
from agent6.viewmodel.session import view as state_view

# Whose message a fold item is; the operator's own words are a user message.
_ROLE: dict[str, model.Role] = {
    "thinking": "thought",
    "text": "agent",
    "operator": "user",
    "task": "user",
}

# ACP's kind per built-in tool, the editor's icon; an MCP tool reads `other`.
_TOOL_KINDS = {
    "read_file": "read",
    "list_dir": "read",
    "agent6_docs": "read",
    "read_session": "read",
    "parallel": "execute",  # children run and merge; the closest ACP kind
    "merge_child": "execute",
    "steer_child": "other",
    "list_findings": "read",
    "read_background": "read",
    "use_skill": "read",
    "outline": "search",
    "find_definition": "search",
    "find_references": "search",
    "list_tasks": "read",
    "apply_edit": "edit",
    "apply_patch": "edit",
    "run_verify_command": "execute",
    "run_command": "execute",
    "run_metric_command": "execute",
    "stop_background": "execute",
    "fetch": "fetch",
    "ask_user": "other",
    "add_task": "think",
    "update_task": "think",
    "finish_session": "other",  # the finish tools never fold to a call; listed for coverage
    "finish_planning": "other",
}


def available_commands(mode: str) -> tuple[model.Command, ...]:
    """Return the registered commands an editor can offer for a session in this mode."""
    can_dispatch = child_launch.can_dispatch_work(mode)
    offered: list[model.Command] = []
    for command in directive.commands_for("acp"):
        if command.kind == "starts children" and not can_dispatch:
            continue
        word, _, hint = command.usage.partition(" ")
        offered.append(model.Command(word.removeprefix("/"), command.help, hint))
    return tuple(offered)


def row_id(event: dict[str, Any]) -> str:
    """Return a journal row's id: its `event_id`, or its timestamp in a journal without one."""
    return str(event.get("event_id") or event.get("ts") or "")


@dataclasses.dataclass
class MessageIds:
    """Name each message by the journal row it came from, the same live as replayed.

    The model's text and thinking take the id of their `model.call` row, so the deltas a
    live run streams and the whole message a replay sends are one message.

    Attributes:
        call: The id of the worker's model call in flight.
    """

    call: str = dataclasses.field(default_factory=lambda: uuid.uuid4().hex, init=False)

    def feed(self, event: dict[str, Any]) -> None:
        """Take the next call's id at the worker's `model.call` row."""
        if event.get("type") == "model.call" and not kinds.is_side_actor(
            str(event.get("actor", ""))
        ):
            self.call = row_id(event) or uuid.uuid4().hex

    def of(self, item: transcript_items.TranscriptItem, event: dict[str, Any], index: int) -> str:
        """Return the id of the message a fold item becomes.

        Args:
            item: The fold item.
            event: The row the fold yielded it at.
            index: The item's place among that row's items.

        Returns:
            The call's id for the model's words, else the row's, with the place appended
            after the row's first item; "" for a row with no id.
        """
        if item.kind in ("text", "thinking"):
            return self.call
        base = row_id(event)
        return f"{base}:{index}" if base and index else base


@dataclasses.dataclass
class RunInfo:
    """The session's title, slash commands and context use, projected from the journal.

    `used` over `size` is the rule behind every surface's `ctx N%` readout
    (`state_view.context_fill`): the last completed call's whole prompt over its model's window.

    Attributes:
        window: The context window of a provider's model, None when unknown.
        banked_usd: The spend of the run's earlier executions; each execution's budget
            starts at 0.
    """

    window: Callable[[str, str], int | None] = state_view.cached_context_window
    banked_usd: float = 0.0
    _verification: evidence.VerificationEvidence | None = dataclasses.field(
        default=None, init=False
    )
    _size: int | None = dataclasses.field(default=None, init=False)
    _used: int = dataclasses.field(default=0, init=False)
    _cost: float | None = dataclasses.field(default=None, init=False)
    # A call's result and its budget update arrive as a pair, in either order in old journals.
    _owed: set[str] = dataclasses.field(default_factory=set, init=False)

    def feed(self, event: dict[str, Any]) -> list[model.Update]:
        """Return the updates the event owes beyond the fold's items.

        Args:
            event: One rows event.

        Returns:
            The title and the commands at `session.start`; the usage once a call's
            `model.result` and `budget.update` have both landed.
        """
        match str(event.get("type", "")):
            case "session.start":
                return self._start(str(event.get("user_task", "")), str(event.get("mode", "")))
            case "verify.end":
                try:
                    self._verification = evidence.from_event(event)
                except ValueError:
                    self._verification = None
            case "session.stop":
                tree_id = f"{row}:tree" if (row := row_id(event)) else ""
                if child := journal.ChildVerification.from_stop(event):
                    return [notice(f"Final tree passed ({child.label})", tree_id)]
                record = self._verification
                if record is not None:
                    final = record.verify_id == event.get("final_verify_id") and record.certifies(
                        event.get("tree")
                    )
                    word = "Final tree passed" if final else "Final tree not certified"
                    return [notice(word + "\n" + receipt.verification_summary(record), tree_id)]
            case "model.call":
                self._size = self.window(
                    str(event.get("provider", "")), str(event.get("model", ""))
                )
                self._owed = {"model.result", "budget.update"}
            case "model.result":
                used = sum(
                    int(event.get(k) or 0)
                    for k in ("input_tokens", "cache_read_tokens", "cache_write_tokens")
                )
                self._used = used if used > 0 else self._used
                return self._settle("model.result")
            case "budget.update":
                # The listing's cell: the dollars, $0 included, unless unpriced or a plan's points.
                row = viewmodel_events.parse_event(event)
                if isinstance(row, journal.BudgetUpdate):
                    total = self.banked_usd + row.usd_total
                    plan = (row.plan_observed_points or 0.0) > 0 or (
                        row.account is not None and row.account.used_percent > 0
                    )
                    priced = not row.usd_partial and (total > 0 or not plan)
                    self._cost = total if priced else None
                return self._settle("budget.update")
            case _:
                pass
        return []

    def _start(self, user_task: str, mode: str) -> list[model.Update]:
        """Return the title, when the task has a headline, and the commands."""
        out: list[model.Update] = []
        if title := printable(session_status.task_snippet(user_task)):
            out.append(model.SessionInfoUpdate(title))
        out.append(model.AvailableCommandsUpdate(available_commands(mode)))
        return out

    def _settle(self, arrived: str) -> list[model.Update]:
        """Return the call's usage once its pair is complete, the window and a prompt known."""
        self._owed.discard(arrived)
        if self._owed or not self._size or self._used <= 0:
            return []
        return [model.UsageUpdate(self._used, self._size, self._cost)]


# ACP's plan status per task status; it has no failed status, and a failed task is still open.
_PLAN_STATUS = {
    "pending": "pending",
    "in_progress": "in_progress",
    "passed": "completed",
    "failed": "pending",
}


def plan_entries(tasks: tuple[state_types.TaskNodeView, ...]) -> tuple[model.PlanEntry, ...]:
    """Return the task list as ACP plan entries, in the read model's depth-first order.

    Args:
        tasks: The folded session state's tasks.

    Returns:
        One entry per task that is neither skipped nor obsolete. The task the worker is on
        reads `in_progress`; a failed one keeps " (failed)" after its title.
    """
    return tuple(
        model.PlanEntry(
            printable(task.title) + (" (failed)" if task.status == "failed" else ""),
            "in_progress" if task.is_cursor else _PLAN_STATUS.get(task.status, "pending"),
        )
        for task in tasks
        if task.status not in ("skipped", "obsolete")
    )


@dataclasses.dataclass
class TaskPlan:
    """Project the run's task list into ACP's plan, from the state every surface folds."""

    _state: state_types.SessionState = dataclasses.field(
        default_factory=session_state.initial_state, init=False
    )
    # The entries the editor holds from this projection; None before the first graph.
    _sent: tuple[model.PlanEntry, ...] | None = dataclasses.field(default=None, init=False)

    def feed(self, event: dict[str, Any]) -> list[model.Update]:
        """Return the plan when the event changed it.

        Args:
            event: One rows event.

        Returns:
            One plan update with the complete list, empty included, at a `graph.update` and
            at the `session.stop` that clears the focus; none while the list is unchanged, or
            at a stop before the execution journaled a graph.
        """
        kind = event.get("type")
        if kind != "graph.update" and (kind != "session.stop" or self._sent is None):
            return []
        self._state = session_state.apply_event(self._state, event)
        entries = plan_entries(self._state.tasks)
        if entries == self._sent:
            return []
        self._sent = entries
        return [model.PlanUpdate(entries)]


def replay(
    rows: Iterable[dict[str, Any]],
    *,
    session_id: str,
    cwd: pathlib.Path,
    objects_dir: pathlib.Path,
    notify: Callable[[model.Update], None],
) -> None:
    """Replay a recorded run through the shared fold and ACP projection.

    Args:
        rows: The run's events in emission order.
        session_id: The recorded run's id.
        cwd: The working directory for tool locations.
        objects_dir: The run's object directory, holding the text of each edit's diff.
        notify: Sends each update before the replay returns.
    """
    fold = transcript.TranscriptFold()
    plan = TaskPlan()
    ids = MessageIds()
    execution, executions = "", 0
    announced: set[str] = set()

    def send(
        item: transcript_items.TranscriptItem,
        paths: tuple[str, ...] = (),
        message_id: str = "",
        cancelled: bool = False,
    ) -> None:
        wire_id = tool_call_id(item, session_id, execution)
        for update in updates_for(
            item,
            wire_id=wire_id,
            announced=wire_id in announced,
            cwd=cwd,
            paths=paths,
            objects_dir=objects_dir,
            message_id=message_id,
            cancelled=cancelled,
        ):
            notify(update)
        if item.kind in ("tool", "verify"):
            announced.add(wire_id)

    for event in rows:
        event_type = event.get("type")
        ids.feed(event)
        for update in plan.feed(event):
            notify(update)
        paths = result_paths(event) if event_type == "tool.result" else ()
        for index, item in enumerate(fold.feed(event)):
            wire_item = item
            if item.kind == "task":
                # The fold's task is a headline; a loaded conversation keeps the full prompt.
                wire_item = dataclasses.replace(item, body=str(event.get("user_task", "")))
            send(wire_item, paths, ids.of(item, event, index), operator_stopped(event))
        if event_type in journal.SESSION_START_EVENTS:
            # The fold settles the previous execution's open calls before this boundary.
            executions += 1
            # A journal from before execution ids numbers its executions instead.
            execution = str(event.get("execution_id") or executions)
            announced.clear()
    for item in fold.settle_open_calls("the run died"):
        send(item)


def operator_stopped(event: dict[str, Any]) -> bool:
    """Return whether the row is the `session.stop` of a run the operator stopped."""
    return (
        event.get("type") == "session.stop" and event.get("reason") in session_status.OPERATOR_STOPS
    )


def result_paths(event: dict[str, Any]) -> tuple[str, ...]:
    """Return the paths a tool.result journaled, for the editor's follow-along."""
    raw = event.get("paths")
    if not isinstance(raw, list):
        return ()
    return tuple(path for path in raw if isinstance(path, str) and path.strip())


def updates_for(
    item: transcript_items.TranscriptItem,
    *,
    wire_id: str = "",
    announced: bool = False,
    cwd: pathlib.Path | None = None,
    paths: tuple[str, ...] = (),
    streamed: bool = False,
    objects_dir: pathlib.Path | None = None,
    message_id: str = "",
    prompt: bool = False,
    cancelled: bool = False,
) -> list[model.Update]:
    """Return the updates one fold item becomes.

    The first update of a tool call creates it and names it; the ones after patch it, so
    an editor shows work in progress.

    Args:
        item: The fold item.
        wire_id: The call's `toolCallId`; "" takes the execution's own stamp.
        announced: The editor already has the call.
        cwd: The working directory the locations and diffs resolve against.
        paths: The paths the tool result named.
        streamed: The item is one delta of a message in flight, sent whole.
        objects_dir: The run's object directory; None sends an edit without its diffs.
        message_id: The id of the message the item is, or is a piece of; "" mints one.
        prompt: The item is the prompt this connection sent.
        cancelled: The operator stopped the run, so a call it never returned from reads
            `cancelled`.

    Returns:
        The updates, in sending order; none for a blank message.
        An unmapped kind becomes an agent message naming its kind.
    """
    if item.kind == "done":
        summary = item.body.strip()
        result: list[model.Update] = []
        if summary and item.arg in session_status.MODEL_ENDS:
            result.append(
                model.Message("agent", message_id or _new_id(), printable(summary + "\n\n"))
            )
        elif summary:
            # A stop's sentence (a spend ceiling, an operator stop) is agent6's, not the model's.
            result.append(notice(summary, message_id))
        result.append(notice(status_line(item), f"{message_id}:status" if message_id else ""))
        return result
    if item.kind == "commit":
        text = " ".join(part for part in ("committed", item.arg, item.detail) if part)
        return [notice(text, message_id)]
    if item.kind in ("tool", "verify"):
        text, changes = _tool_content(item, cwd, objects_dir)
        # The changes name every file, a deleted one and one outside the workspace too.
        located = tuple(change.path for change in item.changes) or paths
        return [
            model.ToolCallUpdate(
                tool_call_id=wire_id or execution_call_id(item),
                status=_tool_status(item, cancelled),
                title="" if announced else printable(f"{item.name} {item.arg}".strip()),
                name="" if announced else item.name,
                kind="" if announced else _TOOL_KINDS.get(item.name, "other"),
                text=text,
                changes=changes,
                locations=_tool_locations(located, cwd) if cwd else (),
            )
        ]
    if item.kind == "marker":
        # Harness prose: a compaction, a btw answer, a notice.
        body = item.body.strip()
        return [notice(body, message_id)] if body else []
    role = _ROLE.get(item.kind)
    # A streamed delta keeps every byte; a whole message is stripped, and a blank one is none.
    body = item.body if streamed else item.body.strip()
    if role is None:
        detail = "\n".join(
            part for part in (body, item.name, item.arg, item.detail, item.tail) if part
        )
        return [notice(f"{item.kind}: {detail}".rstrip(), message_id)]
    message_id = message_id or _new_id()
    message = (
        model.MessageChunk(role, message_id, printable(body))
        if streamed
        else model.Message(role, message_id, printable(body), prompt=prompt)
    )
    return [message] if body else []


def status_line(item: transcript_items.TranscriptItem) -> str:
    """Return the run status without the model's summary.

    Args:
        item: The fold's `done` item.

    Returns:
        "Session" and the status label ("ended" without one), then " - " and the item's
        detail (cost, wall time, counts, commit subject) when it has one.
    """
    parts = [f"Session {item.name or 'ended'}"]
    if item.detail:
        parts.append(f"- {item.detail}")
    return " ".join(parts)


def notice(text: str, message_id: str = "") -> model.Message:
    """Return one line of agent6's own prose as an agent message, marked as its own.

    The message ends its line: a v1 editor appends messages as they come, and the model's
    next words would otherwise start on the notice's line.

    Args:
        text: The line, without the `[agent6]` marker this adds.
        message_id: The message's id; "" mints one.

    Returns:
        The message.
    """
    return model.Message("agent", message_id or _new_id(), printable(f"[agent6] {text}\n"))


def _new_id() -> str:
    """Return an id for a message no journal row names."""
    return uuid.uuid4().hex


def printable(text: str) -> str:
    """Return the text with control characters dropped.

    Every model-authored string on the wire goes through here: message text,
    tool titles, and the permission request an operator reads before granting.
    A diff's file text is the one exception.

    Args:
        text: A string bound for the wire.

    Returns:
        The text, keeping only printable characters, newlines and tabs.
    """
    return "".join(c for c in text if c.isprintable() or c in "\n\t")


def _tool_status(item: transcript_items.TranscriptItem, cancelled: bool) -> model.ToolCallStatus:
    """Return the call's status; a call waiting on an answer is `paused`."""
    if item.ok is None:
        return "paused" if item.detail else "in_progress"
    if item.ok:
        return "completed"
    return "cancelled" if cancelled else "failed"


def _tool_content(
    item: transcript_items.TranscriptItem,
    cwd: pathlib.Path | None = None,
    objects_dir: pathlib.Path | None = None,
) -> tuple[str, tuple[model.DiffChange, ...]]:
    """Return what the tool produced: its text, and the files it changed.

    `tail` is the failure's output, the reason an editor shows beside "failed".

    Args:
        item: A tool item of the fold.
        cwd: The working directory a change's path resolves against.
        objects_dir: The run's object directory.

    Returns:
        The item's detail and tail as one text, and one change per file the call wrote. A
        call whose stored text is gone (pruned history) has no change, and the text says so.
    """
    parts = [item.detail, item.tail]
    changes: tuple[model.DiffChange, ...] | None = ()
    if cwd is not None and objects_dir is not None:
        changes = _changes(item.changes, cwd, objects_dir)
    if changes is None:
        parts.append("diff unavailable: its stored text is gone")
    return printable("\n".join(part for part in parts if part)), changes or ()


def _changes(
    stored: tuple[transcript_items.StoredChange, ...], cwd: pathlib.Path, objects_dir: pathlib.Path
) -> tuple[model.DiffChange, ...] | None:
    """Return one change per file with both texts, or None when a stored text is gone."""
    changes: list[model.DiffChange] = []
    for change in stored:
        texts = transcript_items.change_texts(change, objects_dir)
        if texts is None:
            return None
        # The text is the file's, kept whole: a scrubbed side would differ from the file.
        changes.append(model.DiffChange(str((cwd / change.path).resolve()), texts[0], texts[1]))
    return tuple(changes)


def _tool_locations(paths: tuple[str, ...], cwd: pathlib.Path) -> tuple[str, ...]:
    """Return ACP's absolute follow-along locations, each path once."""
    resolved = (str((cwd / path).resolve()) for path in paths)
    return tuple(dict.fromkeys(resolved))


def wire_call_id(session_id: str, execution: str, within_execution: str) -> str:
    """Return a tool call's wire id, `<run>:<execution>:<call>`, unique for the ACP session.

    Args:
        session_id: The run id; "" leaves the stamp bare.
        execution: The execution's id from its start row.
        within_execution: The dispatcher's stamp, a counter that restarts every execution.

    Returns:
        The `toolCallId` the call's updates and permission request carry.
    """
    return f"{session_id}:{execution}:{within_execution}" if session_id else within_execution


def execution_call_id(item: transcript_items.TranscriptItem) -> str:
    """Return the item's stamped call id, or its name and arg for an event with no stamp."""
    return item.call_id or (f"{item.name}:{item.arg}" if item.arg else item.name)


def tool_call_id(item: transcript_items.TranscriptItem, session_id: str, execution: str) -> str:
    """Return the wire id for a fold item."""
    return wire_call_id(session_id, execution, execution_call_id(item))
