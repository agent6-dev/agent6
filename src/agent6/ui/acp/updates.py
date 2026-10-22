# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Project the shared transcript fold into ACP `session/update` notifications.

Projecting the fold every other surface renders keeps a fourth surface from
disagreeing about what happened. Nothing here touches the wire.
"""

from __future__ import annotations

import pathlib
from typing import Any

from agent6.viewmodel import transcript

# The update a fold item becomes; the operator's own words echo back as a user message.
_CHUNK_KIND = {
    "thinking": "agent_thought_chunk",
    "text": "agent_message_chunk",
    "operator": "user_message_chunk",
}

# ACP's kind per built-in tool, the editor's icon; an MCP tool reads `other`.
_TOOL_KINDS = {
    "read_file": "read",
    "list_dir": "read",
    "agent6_docs": "read",
    "read_session": "read",
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


def updates_for(
    item: transcript.TranscriptItem,
    *,
    acp_session_id: str,
    wire_id: str = "",
    announced: bool = False,
    cwd: pathlib.Path | None = None,
    paths: tuple[str, ...] = (),
    streamed: bool = False,
) -> list[dict[str, Any]]:
    """Return the `session/update` notifications one fold item becomes.

    A tool call is announced once and updated after that, so an editor shows work
    in progress.

    Args:
        item: The fold item.
        acp_session_id: The conversation the notifications address.
        wire_id: The call's `toolCallId`; "" takes the execution's own stamp.
        announced: The editor already has the call.
        cwd: The working directory the locations resolve against.
        paths: The paths the tool result named.
        streamed: The item is one delta of a message in flight, sent whole.

    Returns:
        The notifications, in the order to send them; none for a blank message or a `task`
        item.
    """
    if item.kind == "done":
        summary = item.body.strip()
        result: list[dict[str, Any]] = []
        if summary:
            result.append(
                _update(
                    acp_session_id,
                    {"sessionUpdate": "agent_message_chunk", "content": _text(summary + "\n\n")},
                )
            )
        result.append(message_update(acp_session_id, status_line(item)))
        return result
    if item.kind == "commit":
        text = " ".join(part for part in ("committed", item.arg, item.detail) if part)
        return [message_update(acp_session_id, text)]
    if item.kind == "tool":
        wire_id = wire_id or _execution_call_id(item)
        if item.ok is None and not announced:
            update = {"sessionUpdate": "tool_call", **_tool_call(item, wire_id)}
        else:
            update = {
                "sessionUpdate": "tool_call_update",
                "toolCallId": wire_id,
                "status": _tool_status(item),
                **({"content": _tool_content(item)} if _tool_content(item) else {}),
                **({"locations": _tool_locations(paths, cwd)} if paths and cwd else {}),
            }
        return [_update(acp_session_id, update)]
    if item.kind == "marker":
        # Harness prose: a compaction, a btw answer, a notice.
        body = item.body.strip()
        return [message_update(acp_session_id, body)] if body else []
    chunk = _CHUNK_KIND.get(item.kind)
    # A streamed delta keeps every byte; a whole message is stripped, and a blank one is none.
    body = item.body if streamed else item.body.strip()
    if chunk is None or not body:
        return []
    return [_update(acp_session_id, {"sessionUpdate": chunk, "content": _text(body)})]


def status_line(item: transcript.TranscriptItem) -> str:
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


def message_update(acp_session_id: str, text: str) -> dict[str, Any]:
    """Return one line of agent6's own prose as a `session/update`, marked as its own.

    The chunk ends its line: an editor appends chunks as they come, and the model's next
    words would otherwise start on the notice's line.

    Args:
        acp_session_id: The conversation the notification addresses.
        text: The line, without the `[agent6]` marker this adds.

    Returns:
        An `agent_message_chunk` notification.
    """
    return _update(
        acp_session_id,
        {"sessionUpdate": "agent_message_chunk", "content": _text(f"[agent6] {text}\n")},
    )


def _update(acp_session_id: str, update: dict[str, Any]) -> dict[str, Any]:
    """Return one `session/update` notification."""
    return {
        "jsonrpc": "2.0",
        "method": "session/update",
        "params": {"sessionId": acp_session_id, "update": update},
    }


def printable(text: str) -> str:
    """Return the text with control characters dropped.

    Every model-authored string on the wire goes through here: content blocks,
    tool titles, and the permission request an operator reads before granting.

    Args:
        text: A string bound for the wire.

    Returns:
        The text, keeping only printable characters, newlines and tabs.
    """
    return "".join(c for c in text if c.isprintable() or c in "\n\t")


def _text(text: str) -> dict[str, Any]:
    """Return a text content block, scrubbed; the editor is a third-party renderer."""
    return {"type": "text", "text": printable(text)}


def _tool_status(item: transcript.TranscriptItem) -> str:
    """Return ACP's status for the item; a call waiting on an answer is `pending`."""
    if item.ok is None:
        return "pending" if item.detail else "in_progress"
    return "completed" if item.ok else "failed"


def _tool_content(item: transcript.TranscriptItem) -> list[dict[str, Any]]:
    """Return what the tool produced, in ACP's tagged shape.

    A bare content-block array makes a strict client reject the notification, and
    the call then stays pending; `tail` is the failure's output, the reason an
    editor shows beside "failed".

    Args:
        item: A tool item of the fold.

    Returns:
        One `content` entry holding the item's detail and tail; empty when neither is set.
    """
    body = "\n".join(part for part in (item.detail, item.tail) if part)
    return [{"type": "content", "content": _text(body)}] if body else []


def _tool_locations(paths: tuple[str, ...], cwd: pathlib.Path) -> list[dict[str, str]]:
    """Return ACP's absolute follow-along locations, each path once."""
    resolved = ((cwd / path).resolve() for path in paths)
    return [{"path": str(path)} for path in dict.fromkeys(resolved)]


def wire_call_id(session_id: str, execution: int, within_execution: str) -> str:
    """Return a tool call's wire id, `<run>:<execution>:<call>`, unique for the ACP session.

    Args:
        session_id: The run id; "" leaves the stamp bare.
        execution: The execution's number within the run.
        within_execution: The dispatcher's stamp, a counter that restarts every execution.

    Returns:
        The `toolCallId` the call's announcement, updates and permission request carry.
    """
    return f"{session_id}:{execution}:{within_execution}" if session_id else within_execution


def _execution_call_id(item: transcript.TranscriptItem) -> str:
    """Return the item's stamped call id, or its name and arg for an event with no stamp."""
    return item.call_id or (f"{item.name}:{item.arg}" if item.arg else item.name)


def tool_call_id(item: transcript.TranscriptItem, session_id: str, execution: int) -> str:
    """Return the wire id for a fold item."""
    return wire_call_id(session_id, execution, _execution_call_id(item))


def _tool_call(item: transcript.TranscriptItem, wire_id: str) -> dict[str, Any]:
    """Return the `tool_call` announcement's fields; the model wrote the arg, so it is scrubbed."""
    title = printable(f"{item.name} {item.arg}".strip())
    return {
        "toolCallId": wire_id,
        "title": title,
        "kind": _TOOL_KINDS.get(item.name, "other"),
        "status": _tool_status(item),
    }
