# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Render the session's updates as the frames each protocol version reads.

`model` names everything in v2's vocabulary, so v2 is the plain rendering and v1 is the
translation: its own update names, its statuses, and the fields v2 renamed.
"""

from __future__ import annotations

import difflib
from typing import Any

from agent6.ui.acp import model, rpc

_CHUNK = {
    "user": "user_message_chunk",
    "agent": "agent_message_chunk",
    "thought": "agent_thought_chunk",
}
_MESSAGE = {"user": "user_message", "agent": "agent_message", "thought": "agent_thought"}
# The run has one task list, so its plan has one id.
PLAN_ID = "tasks"
# v1 has neither status: a waiting call reads `pending`, a cancelled one `failed`.
_V1_STATUS = {"paused": "pending", "cancelled": "failed"}
# v1 tags the input of these two commands as text; its schema makes the tag optional.
_V1_TEXT_INPUTS = frozenset({"pin", "review"})
# v1 has no code for a missing session: it is a bad parameter.
_V1_CODES = {rpc.RESOURCE_NOT_FOUND: rpc.INVALID_PARAMS}
# The bytes Git escapes by name in a quoted path.
_GIT_ESCAPES = {0x07: "a", 0x08: "b", 0x09: "t", 0x0A: "n", 0x0B: "v", 0x0C: "f", 0x0D: "r"}


def notification(acp_session_id: str, update: dict[str, Any]) -> dict[str, Any]:
    """Return one `session/update` notification carrying a rendered update."""
    return {
        "jsonrpc": "2.0",
        "method": "session/update",
        "params": {"sessionId": acp_session_id, "update": update},
    }


def render(version: int, update: model.Update) -> dict[str, Any] | None:
    """Return the update as the version's `update` object, or None when it has no form."""
    return v1(update) if version == 1 else v2(update)


def _text(text: str) -> dict[str, Any]:
    """Return a text content block."""
    return {"type": "text", "text": text}


def v2(update: model.Update) -> dict[str, Any]:
    """Return the update as v2's `update` object.

    Args:
        update: The update.

    Returns:
        The object. A whole message replaces what its id held, so a replay needs no
        emptying update first.
    """
    body: dict[str, Any]
    match update:
        case model.Message():
            body = {"sessionUpdate": _MESSAGE[update.role], "messageId": update.message_id}
            body["content"] = [_text(update.text)]
        case model.MessageChunk():
            body = {"sessionUpdate": _CHUNK[update.role], "messageId": update.message_id}
            body["content"] = _text(update.text)
        case model.ToolCallUpdate():
            body = _v2_tool_call(update)
        case model.PlanUpdate():
            plan = {"type": "items", "planId": PLAN_ID, "entries": _plan_entries(update)}
            body = {"sessionUpdate": "plan_update", "plan": plan}
        case model.AvailableCommandsUpdate():
            commands = v2_commands(update.commands)
            body = {"sessionUpdate": "available_commands_update", "availableCommands": commands}
        case model.ConfigOptionUpdate():
            options = update.config_options
            body = {"sessionUpdate": "config_option_update", "configOptions": options}
        case model.SessionInfoUpdate():
            body = {"sessionUpdate": "session_info_update", "title": update.title}
        case model.UsageUpdate():
            # `cost` is a patch: null clears a total an earlier update set.
            cost = None if update.cost_usd is None else _cost(update.cost_usd)
            body = {"sessionUpdate": "usage_update", "used": update.used, "size": update.size}
            body["cost"] = cost
        case model.StateUpdate():
            body = {"sessionUpdate": "state_update", "state": update.state}
            if update.stop_reason:
                body["stopReason"] = update.stop_reason
            if update.error is not None:
                body["error"] = error(2, update.error)
        case model.ReplayComplete():
            body = {"sessionUpdate": "replay_complete"}
    return body


def _plan_entries(update: model.PlanUpdate) -> list[dict[str, Any]]:
    """Return the plan's entries, each `medium` priority: the task list ranks none."""
    return [
        {"content": entry.content, "priority": "medium", "status": entry.status}
        for entry in update.entries
    ]


def _cost(usd: float) -> dict[str, Any]:
    """Return a run's spend as ACP's cost object."""
    return {"amount": usd, "currency": "USD"}


def _v2_tool_call(update: model.ToolCallUpdate) -> dict[str, Any]:
    """Return v2's upsert: the fields the update sets, the title on the one that creates the call.

    The files an edit changed are one `diff`: each file's operation, and a patch of them all.

    Args:
        update: The tool call's update.

    Returns:
        The `tool_call_update` body.
    """
    body: dict[str, Any] = {
        "sessionUpdate": "tool_call_update",
        "toolCallId": update.tool_call_id,
        "status": update.status,
    }
    if update.title:
        body |= {"title": update.title, "name": update.name, "kind": update.kind}
    content: list[dict[str, Any]] = (
        [{"type": "content", "content": _text(update.text)}] if update.text else []
    )
    if update.changes:
        changes = [
            {"operation": change.operation, "path": change.path, "fileType": "text"}
            for change in update.changes
        ]
        patch = {"format": "git_patch", "text": git_patch(update.changes)}
        content.append({"type": "diff", "changes": changes, "patch": patch})
    if content:
        body["content"] = content
    if update.locations:
        body["locations"] = _locations(update)
    return body


def git_patch(changes: tuple[model.DiffChange, ...]) -> str:
    """Return the changes as one patch in Git's format, every path absolute and unprefixed.

    Args:
        changes: The files, each with the text of both sides.

    Returns:
        One `diff --git` section per file, `/dev/null` for the side an added or deleted file
        lacks. Such a file reads as a regular one (mode 100644), which Git needs to tell an
        add from an edit: the stored change holds the text alone.
    """
    out: list[str] = []
    for change in changes:
        path = _git_path(change.path)
        out.append(f"diff --git {path} {path}\n")
        if change.operation != "modify":
            word = "new" if change.operation == "add" else "deleted"
            out.append(f"{word} file mode 100644\n")
        lines = difflib.unified_diff(
            (change.old_text or "").splitlines(keepends=True),
            (change.new_text or "").splitlines(keepends=True),
            "/dev/null" if change.old_text is None else path,
            "/dev/null" if change.new_text is None else path,
        )
        out.extend(
            line if line.endswith("\n") else f"{line}\n\\ No newline at end of file\n"
            for line in lines
        )
    return "".join(out)


def _git_path(path: str) -> str:
    """Return the path as Git writes it in a patch: quoted when it holds an unusual byte."""
    raw = path.encode("utf-8", "surrogateescape")
    if all(0x20 <= byte < 0x7F and byte not in b'"\\' for byte in raw):
        return path
    return '"' + "".join(_git_byte(byte) for byte in raw) + '"'


def _git_byte(byte: int) -> str:
    """Return one byte of a quoted Git path: itself, a named escape, or three octal digits."""
    if byte in b'"\\':
        return f"\\{chr(byte)}"
    if byte in _GIT_ESCAPES:
        return f"\\{_GIT_ESCAPES[byte]}"
    return chr(byte) if 0x20 <= byte < 0x7F else f"\\{byte:03o}"


def v2_commands(commands: tuple[model.Command, ...]) -> list[dict[str, Any]]:
    """Return the commands as v2 lists them, each and its input tagged with its type."""
    return [
        {
            "type": "command",
            "name": command.name,
            "description": command.description,
            **({"input": {"type": "text", "hint": command.hint}} if command.hint else {}),
        }
        for command in commands
    ]


def v1(update: model.Update) -> dict[str, Any] | None:
    """Return the update as v1's `update` object.

    Args:
        update: The update.

    Returns:
        The object, or None for an update v1 has no form for: the echo of the editor's own
        prompt, a state, the end of a replay.
    """
    body: dict[str, Any] | None
    match update:
        case model.Message(prompt=True) | model.StateUpdate() | model.ReplayComplete():
            body = None
        case model.Message() | model.MessageChunk():
            body = {"sessionUpdate": _CHUNK[update.role], "content": _text(update.text)}
        case model.ToolCallUpdate():
            body = _v1_tool_call(update)
        case model.PlanUpdate():
            body = {"sessionUpdate": "plan", "entries": _plan_entries(update)}
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
                body["cost"] = _cost(update.cost_usd)
    return body


def _v1_tool_call(update: model.ToolCallUpdate) -> dict[str, Any]:
    """Return v1's `tool_call` for a call still in flight that the update creates, else its update.

    A v1 diff is one entry per file: `oldText` null for an added file, `newText` empty for
    a deleted one.

    Args:
        update: The tool call's update.

    Returns:
        The `tool_call` or `tool_call_update` body.
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


def result(version: int, result: dict[str, Any]) -> dict[str, Any]:
    """Return a session method's result as the version reads it.

    Args:
        version: The connection's protocol version.
        result: The result, its config options and commands as the model holds them.

    Returns:
        v2's result with the commands rendered. v1's without them, since a v1 editor
        learns the commands from an update, and with v1's key for an option's id.
    """
    out = dict(result)
    commands = out.pop("availableCommands", None)
    if version == 1:
        if "configOptions" in out:
            out["configOptions"] = v1_config_options(out["configOptions"])
    elif commands is not None:
        out["availableCommands"] = v2_commands(commands)
    return out


def error(version: int, error: rpc.RpcError) -> dict[str, Any]:
    """Return the error object: v1 has its own code for a missing session, and no data for it."""
    code = _V1_CODES.get(error.code, error.code) if version == 1 else error.code
    body: dict[str, Any] = {"code": code, "message": error.message}
    if error.data is not None and code == error.code:
        body["data"] = error.data
    return body


def initialize_result(version: int, agent_version: str) -> dict[str, Any]:
    """Return the `initialize` result in the selected version's shape.

    Args:
        version: The protocol version selected for the connection.
        agent_version: agent6's version.

    Returns:
        The version, agent6's name and what it serves: image blocks in a prompt, and
        under v1 the three session methods v2 makes the baseline. No MCP transport is
        advertised, so an editor sends no server of its own.
    """
    info = {"name": "agent6", "version": agent_version}
    if version == 1:
        return {
            "protocolVersion": 1,
            "agentCapabilities": {
                "loadSession": True,
                # A resource block's uri is client-controlled; passing it through is path injection.
                # An image block's inline data is taken; its uri is never opened.
                "promptCapabilities": {"embeddedContext": False, "image": True},
                # An editor lists recorded runs, reopens one after a restart and closes a thread.
                "sessionCapabilities": {"list": {}, "resume": {}, "close": {}},
            },
            "agentInfo": info,
            "authMethods": [],
        }
    return {
        "protocolVersion": version,
        "info": info,
        "capabilities": {"session": {"prompt": {"image": {}}}},
        "authMethods": [],
    }


def permission_request(
    version: int, acp_session_id: str, request: model.PermissionRequest, call_id: str
) -> dict[str, Any]:
    """Return the params of `session/request_permission`.

    Args:
        version: The connection's protocol version.
        acp_session_id: The conversation the request addresses.
        request: The prompt, the call it gates and its answers.
        call_id: The call v1 names: the gated one, or one made up for a prompt gating none.

    Returns:
        v2's request: the prompt as its `title`, and the gated call as its subject by id
        alone. v1's request: a `toolCall` whose title is the prompt.
    """
    options = [
        # An index, not the text: the text can be model-written.
        {"optionId": str(index), "name": option.name, "kind": option.kind}
        for index, option in enumerate(request.options)
    ]
    if version == 1:
        tool_call: dict[str, Any] = {"toolCallId": call_id, "title": request.title}
        if not request.tool_call_id:
            tool_call["kind"] = "other"
        tool_call["status"] = "pending"
        return {"sessionId": acp_session_id, "toolCall": tool_call, "options": options}
    params: dict[str, Any] = {"sessionId": acp_session_id, "title": request.title}
    if request.tool_call_id:
        subject = {"type": "tool_call", "toolCall": {"toolCallId": request.tool_call_id}}
        params["subject"] = subject
    return {**params, "options": options}


def selected_option(answer: dict[str, Any], offered: int) -> int | None:
    """Return the place of the option the editor chose, or None for no usable answer.

    The answer must be an offered index, or an unknown string could become an "allow" by
    prefix.

    Args:
        answer: The client's result for `session/request_permission`; empty when none came.
        offered: How many options the request offered; each went out with its index as its id.

    Returns:
        The index the client selected; None for no `selected` outcome or an id not offered.
    """
    outcome = answer.get("outcome")
    if not isinstance(outcome, dict) or outcome.get("outcome") != "selected":
        return None
    chosen = outcome.get("optionId")
    if not isinstance(chosen, str):
        return None
    return {str(index): index for index in range(offered)}.get(chosen)
