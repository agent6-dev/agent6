# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Translate Anthropic-shaped messages and tools into the Chat Completions request shape.

The request-building half of the OpenAI provider; `providers/openai.py` states
the translation rationale.
"""

from __future__ import annotations

import json
from typing import Any

from agent6.providers.types import ToolDefinition


def tool_result_text(tr_content: Any) -> str:
    """Return a tool_result's content as the one string the OpenAI wires carry.

    The text blocks joined, else the content as JSON, else as text.
    """
    if isinstance(tr_content, list):
        parts = [
            str(b.get("text", ""))
            for b in tr_content
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        return "".join(parts) if parts else json.dumps(tr_content)
    return str(tr_content)


def anthropic_to_openai_messages(  # noqa: PLR0912
    system: str, anthropic_msgs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Translate Anthropic-shaped messages into the Chat Completions `messages` array.

    Text blocks join into the message's string content; an assistant's `tool_use`
    blocks move into `tool_calls`; each `tool_result` becomes its own `role="tool"`
    message, since the wire puts tool replies in their own role.

    Args:
        system: The system prompt, the first message.
        anthropic_msgs: The conversation in Anthropic content-block shape.

    Returns:
        The messages array, system first.
    """
    out: list[dict[str, Any]] = [{"role": "system", "content": system}]
    # A blank-name tool_use is dropped, and its paired tool_result with it, or strict backends 400.
    dropped_tool_use_ids: set[str] = set()
    for msg in anthropic_msgs:
        role = str(msg.get("role", "user"))
        content = msg.get("content", "")
        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            out.append({"role": role, "content": str(content)})
            continue
        text_chunks: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                text_chunks.append(str(block.get("text", "")))
            elif btype == "tool_use" and role == "assistant":
                if not str(block.get("name") or "").strip():
                    dropped_tool_use_ids.add(str(block.get("id", "")))
                    continue
                tool_calls.append(
                    {
                        "id": str(block.get("id", "")),
                        "type": "function",
                        "function": {
                            "name": str(block.get("name", "")),
                            # The wire carries arguments as a JSON string, not an object.
                            "arguments": json.dumps(block.get("input") or {}),
                        },
                    }
                )
            elif btype == "tool_result":
                if str(block.get("tool_use_id", "")) in dropped_tool_use_ids:
                    continue
                # A string result is what every backend accepts (Ollama, Kimi included).
                tr_text = tool_result_text(block.get("content", ""))
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(block.get("tool_use_id", "")),
                        "content": tr_text,
                    }
                )
        if role == "assistant":
            assistant_msg: dict[str, Any] = {"role": "assistant"}
            if text_chunks:
                assistant_msg["content"] = "\n\n".join(text_chunks)
            elif tool_calls:
                assistant_msg["content"] = None
            else:
                # A thinking-only turn has neither; null content without tool_calls 400s.
                assistant_msg["content"] = ""
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            out.append(assistant_msg)
        else:
            # A role=tool message must directly follow the tool_calls it answers; text comes after.
            for tr in tool_results:
                out.append(tr)
            if text_chunks:
                out.append({"role": role, "content": "\n\n".join(text_chunks)})
    return out


def tools_to_openai(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
    """Return the tools as Chat Completions function-tool entries."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema,
            },
        }
        for t in tools
    ]
