# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Project the shared transcript fold into ACP `session/update` notifications.

Projecting the fold every other surface renders keeps a fourth surface from
disagreeing about what happened. Nothing here touches the wire.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent6.viewmodel.transcript import TranscriptItem

# The update a fold item becomes; the operator's own words echo back as a user message.
_CHUNK_KIND = {
    "thinking": "agent_thought_chunk",
    "text": "agent_message_chunk",
    "operator": "user_message_chunk",
    "marker": "agent_message_chunk",  # harness prose: a compaction, a btw answer, a notice
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
    item: TranscriptItem,
    *,
    acp_session_id: str,
    wire_id: str = "",
    announced: bool = False,
    cwd: Path | None = None,
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
    """
    if item.kind == "done":
        return [
            _update(
                acp_session_id,
                {"sessionUpdate": "agent_message_chunk", "content": _text(ending(item))},
            )
        ]
    if item.kind == "commit":
        text = " ".join(part for part in ("committed", item.arg, item.detail) if part)
        return [
            _update(
                acp_session_id, {"sessionUpdate": "agent_message_chunk", "content": _text(text)}
            )
        ]
    if item.kind == "tool":
        wire_id = wire_id or _execution_call_id(item)
        if item.ok is None and not announced:
            return [
                _update(acp_session_id, {"sessionUpdate": "tool_call", **_tool_call(item, wire_id)})
            ]
        return [
            _update(
                acp_session_id,
                {
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": wire_id,
                    "status": _tool_status(item),
                    **({"content": _tool_content(item)} if _tool_content(item) else {}),
                    **({"locations": _tool_locations(paths, cwd)} if paths and cwd else {}),
                },
            )
        ]
    chunk = _CHUNK_KIND.get(item.kind)
    # A streamed delta keeps every byte; a whole message is stripped, and a blank one is none.
    body = item.body if streamed else item.body.strip()
    if chunk is None or not body:
        return []
    return [_update(acp_session_id, {"sessionUpdate": chunk, "content": _text(body)})]


def ending(item: TranscriptItem) -> str:
    """Return how a run ended, in the status words every surface uses.

    The fold sets `body` only for a clean `finish_session`; the end reason rides in
    `name` and `detail`, so a provider error or a red gate is never silence.
    """
    parts = [f"Session {item.name or 'ended'}"]
    if item.detail:
        parts.append(f"- {item.detail}")
    ending = " ".join(parts)
    return f"{item.body}\n\n{ending}" if item.body.strip() else ending


def message_update(acp_session_id: str, text: str) -> dict[str, Any]:
    """Return one line of agent6's own prose as a `session/update`, marked as its own."""
    return _update(
        acp_session_id,
        {"sessionUpdate": "agent_message_chunk", "content": _text(f"[agent6] {text}")},
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
    """
    return "".join(c for c in text if c.isprintable() or c in "\n\t")


def _text(text: str) -> dict[str, Any]:
    """Return a text content block, scrubbed; the editor is a third-party renderer."""
    return {"type": "text", "text": printable(text)}


def _tool_status(item: TranscriptItem) -> str:
    """Return ACP's status for the item; a call waiting on an answer is `pending`."""
    if item.ok is None:
        return "pending" if item.detail else "in_progress"
    return "completed" if item.ok else "failed"


def _tool_content(item: TranscriptItem) -> list[dict[str, Any]]:
    """Return what the tool produced, in ACP's tagged shape.

    A bare content-block array makes a strict client reject the notification, and
    the call then stays pending; `tail` is the failure's output, the reason an
    editor shows beside "failed".
    """
    body = "\n".join(part for part in (item.detail, item.tail) if part)
    return [{"type": "content", "content": _text(body)}] if body else []


def _tool_locations(paths: tuple[str, ...], cwd: Path) -> list[dict[str, str]]:
    """Return ACP's absolute follow-along locations, each path once."""
    resolved = ((cwd / path).resolve() for path in paths)
    return [{"path": str(path)} for path in dict.fromkeys(resolved)]


def wire_call_id(session_id: str, turn: int, within_execution: str) -> str:
    """Return a tool call's wire id, `<run>:<turn>:<call>`, unique for the ACP session.

    Args:
        session_id: The run id; "" leaves the stamp bare.
        turn: The turn.
        within_execution: The dispatcher's stamp, a counter that restarts every turn.
    """
    return f"{session_id}:{turn}:{within_execution}" if session_id else within_execution


def _execution_call_id(item: TranscriptItem) -> str:
    """Return the item's stamped call id, or its name and arg for an event with no stamp."""
    return item.call_id or (f"{item.name}:{item.arg}" if item.arg else item.name)


def tool_call_id(item: TranscriptItem, session_id: str, turn: int) -> str:
    """Return the wire id for a fold item."""
    return wire_call_id(session_id, turn, _execution_call_id(item))


def _tool_call(item: TranscriptItem, wire_id: str) -> dict[str, Any]:
    """Return the `tool_call` announcement's fields; the model wrote the arg, so it is scrubbed."""
    title = printable(f"{item.name} {item.arg}".strip())
    return {
        "toolCallId": wire_id,
        "title": title,
        "kind": _TOOL_KINDS.get(item.name, "other"),
        "status": _tool_status(item),
    }
