# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Translate Anthropic-shaped messages and tools into the Chat Completions request shape.

The request-building half of the OpenAI provider; `providers/openai.py` states
the translation rationale.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

from agent6.providers import types


def tool_result_text(tr_content: Any) -> str:
    """Return a tool_result's content as the one string the OpenAI wires carry.

    Args:
        tr_content: A tool_result block's `content`.

    Returns:
        A block list's text blocks joined, else the list as JSON; anything else as text.
    """
    if isinstance(tr_content, list):
        parts = [
            str(b.get("text", ""))
            for b in tr_content
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        return "".join(parts) if parts else json.dumps(tr_content)
    return str(tr_content)


def tool_result_media(
    block: dict[str, Any], path: str, *, responses: bool = False
) -> list[dict[str, Any]]:
    """Encode a canonical tool attachment as labelled user-message parts.

    Args:
        block: The canonical tool result.
        path: The source path from its paired tool call.
        responses: Use the Responses dialect instead of Chat Completions.

    Returns:
        Text labels and inline image or file parts, empty for a text-only result.
    """
    content = block.get("content")
    if not isinstance(content, list):
        return []
    parts: list[dict[str, Any]] = []
    for item in content:
        if not isinstance(item, dict) or item.get("type") not in ("image", "document"):
            continue
        source = item.get("source")
        if not isinstance(source, dict) or source.get("type") != "base64":
            continue
        label = f"{item['type']} from tool call {block.get('tool_use_id', '')}: {path}"
        parts.append({"type": "input_text" if responses else "text", "text": label})
        url = f"data:{source['media_type']};base64,{source['data']}"
        if item["type"] == "image":
            parts.append(
                {"type": "input_image", "image_url": url}
                if responses
                else {"type": "image_url", "image_url": {"url": url}}
            )
        else:
            file = {"filename": pathlib.PurePosixPath(path).name, "file_data": url}
            parts.append(
                {"type": "input_file", **file} if responses else {"type": "file", "file": file}
            )
    return parts


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
    paths: dict[str, str] = {}
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
        attachments: list[dict[str, Any]] = []
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
                if isinstance(call_input := block.get("input"), dict):
                    paths[str(block.get("id", ""))] = str(call_input.get("path", ""))
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
                attachments.extend(
                    tool_result_media(block, paths.get(str(block.get("tool_use_id", "")), ""))
                )
                # A string result is what every backend accepts (Ollama, Kimi included).
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(block.get("tool_use_id", "")),
                        "content": tool_result_text(block.get("content", "")),
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
            out.extend(tool_results)
            if attachments:
                out.append({"role": "user", "content": attachments})
            if text_chunks:
                out.append({"role": role, "content": "\n\n".join(text_chunks)})
    return out


def tools_to_openai(tools: list[types.ToolDefinition]) -> list[dict[str, Any]]:
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
