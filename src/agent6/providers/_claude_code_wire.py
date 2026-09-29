# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The claude_code provider's wire vocabulary, independent of a live process.

The child's argv and environment, the plan reading off a `rate_limit_event`, the
history rendered for a replay, the stdin line shapes and the helpers over
Anthropic-shaped messages; `claude_code` owns the process.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from agent6.budget import PlanUsage, PlanWindow
from agent6.child_env import curated_env

# Above this size Claude Code persists a tool result to disk and hands the model a 2 KB preview.
CLAUDE_CODE_PERSIST_BYTES = 50_000


MCP_SERVER = "agent6"
# The model sees `mcp__<server>__<tool>`; `tools/call` carries the bare name.
TOOL_PREFIX = f"mcp__{MCP_SERVER}__"
_MCP_CONFIG = json.dumps(
    {"mcpServers": {MCP_SERVER: {"type": "sdk", "name": MCP_SERVER}}}, separators=(",", ":")
)

# Every capability beyond the model is off; the child dials only the API and its login refresh.
CLAUDE_CODE_ENV: dict[str, str] = {
    "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",
    "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
    "DISABLE_AUTO_COMPACT": "1",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "DISABLE_AUTOUPDATER": "1",
    "DISABLE_TELEMETRY": "1",
    "DISABLE_ERROR_REPORTING": "1",
}
# The one operator variable that reaches the child: it selects a login, never a credential.
_PASSTHROUGH = ("CLAUDE_CONFIG_DIR",)
_HARNESS_REPLAY = (
    "[harness] This session continues an earlier one on the same repository. The"
    " transcript so far follows, oldest first; every tool call in it was executed, and"
    " its result follows as this session holds it (older ones elided to a placeholder"
    " or a gist). Continue from the end of it; do not redo finished steps."
)


def claude_argv(
    binary: str, model: str, effort: str | None, system_prompt_file: Path
) -> tuple[str, ...]:
    """Build the child's argv from operator config and literals only.

    The system prompt is passed as a file path and prompts ride stdin, so no model
    or repo text reaches the argv.

    Args:
        binary: The Claude Code binary.
        model: The model id.
        effort: The reasoning effort; None omits the flag.
        system_prompt_file: The file holding the system prompt.

    Returns:
        The argv.
    """
    argv = [
        binary,
        "-p",
        "--verbose",
        "--model",
        model,
        "--input-format",
        "stream-json",
        "--output-format",
        "stream-json",
        "--include-partial-messages",
        "--tools",
        "",
        "--allowedTools",
        f"mcp__{MCP_SERVER}",
        "--mcp-config",
        _MCP_CONFIG,
        "--strict-mcp-config",
        "--setting-sources",
        "",
        "--disable-slash-commands",
        "--no-session-persistence",
        "--system-prompt-file",
        str(system_prompt_file),
    ]
    if effort:
        argv += ["--effort", effort]
    return tuple(argv)


def child_env() -> dict[str, str]:
    """Return the child's environment.

    The curated base (HOME included, so the binary finds its login), `CLAUDE_CONFIG_DIR`
    when set, and the fixed toggles; no other `ANTHROPIC_*` or `CLAUDE*` variable
    reaches it, since a key would override the subscription login.
    """
    return curated_env(passthrough=_PASSTHROUGH, extra=CLAUDE_CODE_ENV, desktop=False)


def bare_tool_name(name: str) -> str:
    """Return the agent6-side tool name behind Claude Code's `mcp__agent6__` prefix."""
    return name[len(TOOL_PREFIX) :] if name.startswith(TOOL_PREFIX) else name


def _window_minutes(name: str) -> int:
    """Return a plan window's length in minutes; 0 for a window agent6 does not know."""
    if name == "five_hour":
        return 300
    return 10_080 if name.startswith("seven_day") else 0


def plan_usage_from_rate_limit(info: Mapping[str, Any]) -> PlanUsage | None:
    """Read the plan usage off one `rate_limit_event.rate_limit_info`.

    Args:
        info: The event's `rate_limit_info` object.

    Returns:
        Every `unifiedWindows` entry as a window (utilization is a fraction on the
        wire), the backend's exhausted verdict and whether extra usage is enabled;
        None when the event names no window.
    """
    raw = info.get("unifiedWindows")
    if not isinstance(raw, Mapping):
        return None
    windows: list[PlanWindow] = []
    for name, window in raw.items():
        if not isinstance(window, Mapping):
            continue
        used, resets_at = window.get("utilization"), window.get("resetsAt")
        if not isinstance(used, (int, float)) or not isinstance(resets_at, (int, float)):
            continue
        windows.append(
            PlanWindow(str(name), float(used) * 100.0, _window_minutes(str(name)), float(resets_at))
        )
    if not windows:
        return None
    return PlanUsage(
        windows=tuple(windows),
        has_credits=info.get("overageStatus") == "allowed" or bool(info.get("isUsingOverage")),
        limit_reached=info.get("status") == "rejected",
    )


def message_blocks(message: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return a message's content as blocks, wrapping a string as one text block."""
    content = message.get("content")
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return [b for b in (content or ()) if isinstance(b, dict)]


def message_texts(message: Mapping[str, Any]) -> list[str]:
    """Return the text of a message's text blocks, in order."""
    return [str(b.get("text", "")) for b in message_blocks(message) if b.get("type") == "text"]


def tool_use_ids(message: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the ids of a message's tool_use blocks, in order."""
    return tuple(
        str(b.get("id", "")) for b in message_blocks(message) if b.get("type") == "tool_use"
    )


def result_text(content: Any) -> str:
    """Return a tool result's content as text.

    A string as is, a block list's text joined, absent content as "", anything
    else as its JSON.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(b.get("text", "")) for b in content if isinstance(b, dict) and "text" in b
        )
    return json.dumps(content, ensure_ascii=False)


def tool_results(message: Mapping[str, Any]) -> dict[str, str]:
    """Return `tool_use_id -> content` for a message's tool_result blocks, in wire order."""
    return {
        str(b.get("tool_use_id", "")): result_text(b.get("content"))
        for b in message_blocks(message)
        if b.get("type") == "tool_result"
    }


Skeleton = tuple[str, tuple[tuple[str, str], ...]]


def message_skeleton(message: Mapping[str, Any]) -> Skeleton:
    """Return what identifies a message the process has consumed.

    The role and, per block, the tool_use id, the tool_result id or the text. A
    tool_result's content and thinking are left out: elision and thinking strips
    rewrite those in place without changing what the process was sent.
    """
    keys = {"text": "text", "tool_use": "id", "tool_result": "tool_use_id"}
    return (
        str(message.get("role", "")),
        tuple(
            (str(kind), str(block.get(keys[kind], "")))
            for block in message_blocks(message)
            if (kind := block.get("type")) in keys
        ),
    )


def history_skeleton(messages: Sequence[Mapping[str, Any]]) -> tuple[Skeleton, ...]:
    """Return the skeleton of every message, in order."""
    return tuple(message_skeleton(m) for m in messages)


def render_history(messages: Sequence[Mapping[str, Any]]) -> str:
    """Render the history as one user message for a replay.

    The first user text verbatim, then, for a longer history, a harness paragraph
    and every later turn as labelled text: tool calls with their inputs, results
    as the history holds them, thinking dropped.

    Args:
        messages: The conversation in Anthropic shape.

    Returns:
        The rendered text; "" for an empty history.
    """
    if not messages:
        return ""
    first = "\n\n".join(message_texts(messages[0]))
    if len(messages) == 1:
        return first
    parts = [first, _HARNESS_REPLAY]
    names: dict[str, str] = {}
    for message in messages[1:]:
        role = str(message.get("role", "user"))
        for block in message_blocks(message):
            kind = block.get("type")
            if kind == "text":
                parts.append(f"### {role}\n{block.get('text', '')}")
            elif kind == "tool_use":
                name = str(block.get("name", ""))
                names[str(block.get("id", ""))] = name
                parts.append(
                    f"[tool_use {name}] {json.dumps(block.get('input'), ensure_ascii=False)}"
                )
            elif kind == "tool_result":
                name = names.get(str(block.get("tool_use_id", "")), "tool")
                parts.append(f"### tool_result {name}\n{result_text(block.get('content'))}")
    return "\n\n".join(parts)


def user_line(text: str) -> dict[str, Any]:
    """Return the stdin line that sends one user text."""
    return {
        "type": "user",
        "message": {"role": "user", "content": [{"type": "text", "text": text}]},
    }


def mcp_answer(request_id: str, rpc_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    """Return the stdin line that answers one MCP request from the child."""
    return {
        "type": "control_response",
        "response": {
            "subtype": "success",
            "request_id": request_id,
            "response": {"mcp_response": {"jsonrpc": "2.0", "id": rpc_id, "result": result}},
        },
    }
