# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Render the session's updates as the frames each protocol version reads.

`model` names everything in v2's vocabulary, so v1 is the translation: its own update
names, its statuses, and the fields v2 renamed.
"""

from __future__ import annotations

from typing import Any

from agent6.ui.acp import model, rpc

_CHUNK = {
    "user": "user_message_chunk",
    "agent": "agent_message_chunk",
    "thought": "agent_thought_chunk",
}
# v1 has neither status: a waiting call reads `pending`, a cancelled one `failed`.
_V1_STATUS = {"paused": "pending", "cancelled": "failed"}
# v1 tags the input of these two commands as text; its schema makes the tag optional.
_V1_TEXT_INPUTS = frozenset({"pin", "review"})
# v1 has no code for a missing session: it is a bad parameter.
_V1_CODES = {rpc.RESOURCE_NOT_FOUND: rpc.INVALID_PARAMS}


def notification(acp_session_id: str, update: dict[str, Any]) -> dict[str, Any]:
    """Return one `session/update` notification carrying a rendered update."""
    return {
        "jsonrpc": "2.0",
        "method": "session/update",
        "params": {"sessionId": acp_session_id, "update": update},
    }


def _text(text: str) -> dict[str, Any]:
    """Return a text content block."""
    return {"type": "text", "text": text}


def v1(update: model.Update) -> dict[str, Any] | None:
    """Return the update as v1's `update` object.

    Args:
        update: The update.

    Returns:
        The object, or None for an update v1 has no form for: the echo of the editor's own
        prompt.
    """
    body: dict[str, Any] | None
    match update:
        case model.Message(prompt=True):
            body = None
        case model.Message() | model.MessageChunk():
            body = {"sessionUpdate": _CHUNK[update.role], "content": _text(update.text)}
        case model.ToolCallUpdate():
            body = _v1_tool_call(update)
        case model.PlanUpdate():
            entries = [
                {"content": entry.content, "priority": "medium", "status": entry.status}
                for entry in update.entries
            ]
            body = {"sessionUpdate": "plan", "entries": entries}
        case model.AvailableCommandsUpdate():
            commands = [_v1_command(command) for command in update.commands]
            body = {"sessionUpdate": "available_commands_update", "availableCommands": commands}
        case model.ConfigOptionUpdate():
            options = v1_config_options(update.config_options)
            body = {"sessionUpdate": "config_option_update", "configOptions": options}
        case model.SessionInfoUpdate():
            body = {"sessionUpdate": "session_info_update", "title": update.title}
        case model.UsageUpdate():
            body = {"sessionUpdate": "usage_update", "used": update.used, "size": update.size}
            if update.cost_usd is not None:
                body["cost"] = {"amount": update.cost_usd, "currency": "USD"}
    return body


def _v1_tool_call(update: model.ToolCallUpdate) -> dict[str, Any]:
    """Return v1's `tool_call` for a call still in flight that the update creates, else its update.

    A v1 diff is one entry per file: `oldText` null for an added file, `newText` empty for
    a deleted one.
    """
    status = _V1_STATUS.get(update.status, update.status)
    if update.title and update.status in ("pending", "in_progress", "paused"):
        return {
            "sessionUpdate": "tool_call",
            "toolCallId": update.tool_call_id,
            "title": update.title,
            "name": update.name,
            "kind": update.kind,
            "status": status,
        }
    content: list[dict[str, Any]] = (
        [{"type": "content", "content": _text(update.text)}] if update.text else []
    )
    content += [
        {
            "type": "diff",
            "path": change.path,
            "oldText": change.old_text,
            "newText": change.new_text or "",
        }
        for change in update.changes
    ]
    return {
        "sessionUpdate": "tool_call_update",
        "toolCallId": update.tool_call_id,
        "status": status,
        **({"content": content} if content else {}),
        **({"locations": _locations(update)} if update.locations else {}),
    }


def _locations(update: model.ToolCallUpdate) -> list[dict[str, Any]]:
    """Return the call's follow-along locations, each without a line."""
    return [{"path": path} for path in update.locations]


def _v1_command(command: model.Command) -> dict[str, Any]:
    """Return one command as v1 lists it."""
    entry: dict[str, Any] = {"name": command.name, "description": command.description}
    if command.hint:
        entry["input"] = {"hint": command.hint}
        if command.name in _V1_TEXT_INPUTS:
            entry["input"]["type"] = "text"
    return entry


def v1_config_options(options: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the config options with v1's key for an option's id."""
    return [
        {("id" if key == "configId" else key): value for key, value in option.items()}
        for option in options
    ]


def v1_result(result: dict[str, Any]) -> dict[str, Any]:
    """Return a session method's result as v1 reads it."""
    if "configOptions" not in result:
        return result
    return {**result, "configOptions": v1_config_options(result["configOptions"])}


def v1_error(error: rpc.RpcError) -> dict[str, Any]:
    """Return the error as v1's error object: its code, and no data."""
    return {"code": _V1_CODES.get(error.code, error.code), "message": error.message}
