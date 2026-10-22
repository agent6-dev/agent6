# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Fold a session's per-call provider transcripts into conversation messages and render them.

agent6 writes one JSON file per round-trip under `<run>/transcripts/`, the redacted
`{request, response}`; each request carries the whole conversation up to that call.
The fold walks them in seq order across the Chat Completions, Anthropic and Responses
wire shapes, emitting only the messages each call introduced, so the cumulative
snapshots are not double-printed and a compaction restart shows as a marker.
`agent6 sessions transcript` is the CLI front end.
The rendered transcript is a reading surface with labelled paths, not the wire arguments.
"""

from __future__ import annotations

import collections
import dataclasses
import json
import pathlib
import re
from typing import Any

from agent6.providers import result_text
from agent6.viewmodel import format

# Equals ELISION_PREFIX in harness/_compaction.py (a test pins it), so the read model
# needs no engine import.
ELISION_MARKER_PREFIX = "<elided by context compaction"
_GIST_MARKER_PREFIX = ELISION_MARKER_PREFIX + " (distilled)"
_ELIDED_IDENTITY_RE = re.compile(r": the result of (.+?) was replaced")


@dataclasses.dataclass
class Message:
    """One normalized conversation message, provider-agnostic.

    Mutable: a shape helper builds the message without knowing its call, and
    `fold_conversation` stamps `seq` on it after.

    Attributes:
        role: "system", "user", "assistant", "tool" or "marker".
        text: The message's text.
        thinking: The assistant's reasoning.
        tool_calls: The assistant's calls as (name, args JSON).
        tool_name: The tool a "tool" message is the result of.
        seq: The round trip the message belongs to: a result's is its call's.
    """

    role: str
    text: str = ""
    thinking: str = ""
    tool_calls: list[tuple[str, str]] = dataclasses.field(default_factory=list)
    tool_name: str = ""
    seq: int = 0


# The driving seats; a side call's one-message request would read as a compaction restart.
CONVERSATION_SEATS = frozenset({"worker", "planner"})


def transcript_seq(t: dict[str, Any]) -> int:
    """Return a transcript's seq, 0 when the record carries no integer one."""
    seq = t.get("seq", 0)
    return seq if isinstance(seq, int) else 0


def load_transcripts(transcripts_dir: pathlib.Path) -> list[dict[str, Any]]:
    """Load every transcript under a session's transcripts dir, in seq order, all seats.

    Args:
        transcripts_dir: The session's transcripts dir.

    Returns:
        The transcript objects; an unreadable file is skipped, a missing dir yields none.
    """
    if not transcripts_dir.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for p in sorted(transcripts_dir.glob("*.json")):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(obj, dict):
            out.append(obj)
    out.sort(key=transcript_seq)
    return out


def conversation_transcripts(transcripts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the driving seats' round-trips; a transcript with no seat is the driver's."""
    return [t for t in transcripts if str(t.get("seat", "") or "worker") in CONVERSATION_SEATS]


def _as_dict(value: Any) -> dict[str, Any]:
    """Return the value as a dict, a JSON string decoded, empty for anything else."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return {}
    return value if isinstance(value, dict) else {}


def _shape(req: dict[str, Any], resp: dict[str, Any]) -> str:
    """Return the provider wire shape of one transcript: openai, anthropic or responses."""
    if isinstance(resp.get("choices"), list):
        return "openai"
    if isinstance(resp.get("content"), list) and resp.get("role"):
        return "anthropic"
    if isinstance(resp.get("output"), list) or isinstance(req.get("input"), list):
        return "responses"
    # Anthropic carries a top-level `system`; OpenAI a system message.
    return "anthropic" if "system" in req else "openai"


def _request_items(req: dict[str, Any], shape: str) -> list[Any]:
    """Return the request's conversation list: Responses `input`, else `messages`."""
    return (req.get("input") if shape == "responses" else req.get("messages")) or []


def _item_text(item: dict[str, Any]) -> str:
    """Return a Responses message item's text parts, joined."""
    content = item.get("content")
    if isinstance(content, str):
        return content
    return "".join(
        str(part.get("text", ""))
        for part in content or []
        if isinstance(part, dict) and part.get("type") in ("input_text", "output_text", "text")
    )


def _responses_messages(items: list[Any], names: dict[str, str]) -> list[Message]:
    """Fold Responses items into messages.

    One model response spans several items (reasoning, a message, function calls),
    so consecutive assistant-side items fold into one assistant message.

    Args:
        items: The Responses items.
        names: The call id to tool name map, filled as calls are seen.

    Returns:
        The messages.
    """
    messages: list[Message] = []
    current: Message | None = None

    def flush() -> None:
        """Close the assistant message being built, if any."""
        nonlocal current
        if current is not None:
            messages.append(current)
            current = None

    for item in items:
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        if kind == "message" and item.get("role") != "assistant":
            flush()
            messages.append(Message(role=str(item.get("role") or "user"), text=_item_text(item)))
            continue
        if kind == "function_call_output":
            flush()
            call_id = str(item.get("call_id", ""))
            output = str(item.get("output", ""))
            messages.append(Message(role="tool", text=output, tool_name=names.get(call_id, "")))
            continue
        if current is None:
            current = Message(role="assistant")
        if kind == "reasoning":
            summary = "\n".join(
                str(part.get("text", ""))
                for part in item.get("summary") or []
                if isinstance(part, dict) and part.get("text")
            )
            if summary:
                current.thinking = f"{current.thinking}\n{summary}".strip()
        elif kind == "message":
            text = _item_text(item)
            if text:
                current.text = f"{current.text}\n\n{text}".strip()
        elif kind == "function_call":
            name = str(item.get("name", ""))
            call_id = str(item.get("call_id") or item.get("id") or "")
            if call_id:
                names[call_id] = name
            current.tool_calls.append((name, _pretty_args(item.get("arguments", ""))))
    flush()
    return messages


def _same_item(a: Any, b: Any) -> bool:
    """Return whether a replayed Responses input item is the recorded output item.

    By id when both carry one, else by what it says, since a replayed message drops
    the id, status and annotations the response carried.

    Args:
        a: The input item, from the next request.
        b: The output item, from the previous response.

    Returns:
        True when `a` replays `b`.
    """
    if not isinstance(a, dict) or not isinstance(b, dict) or a.get("type") != b.get("type"):
        return False
    key = "call_id" if a.get("type") == "function_call" else "id"
    if a.get(key) and b.get(key):
        return a.get(key) == b.get(key)
    if a.get("type") == "message":
        return a.get("role") == b.get("role") and _item_text(a) == _item_text(b)
    return a == b


def _pretty_args(raw: Any) -> str:
    """Return tool-call arguments as compact one-line JSON, or as text when not JSON."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return raw.strip()
    if isinstance(raw, dict):
        raw = {
            key: format.tool_path(value)
            if key in format.PATH_KEYS and isinstance(value, str)
            else value
            for key, value in raw.items()
        }
    try:
        return json.dumps(raw, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(raw)


def _openai_messages(m: dict[str, Any], names: dict[str, str]) -> list[Message]:
    """Normalize one Chat Completions message.

    Args:
        m: The message.
        names: The call id to tool name map, filled as calls are seen.

    Returns:
        The messages.
    """
    role = m.get("role", "")
    if role == "tool":
        name = names.get(str(m.get("tool_call_id", "")), "")
        return [Message(role="tool", text=str(m.get("content", "")), tool_name=name)]
    if role == "assistant":
        calls: list[tuple[str, str]] = []
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function", {}) if isinstance(tc, dict) else {}
            name = str(fn.get("name", ""))
            calls.append((name, _pretty_args(fn.get("arguments", ""))))
            if isinstance(tc, dict) and tc.get("id"):
                names[str(tc["id"])] = name
        return [
            Message(
                role="assistant",
                text=str(m.get("content") or ""),
                thinking=str(m.get("reasoning_content") or ""),
                tool_calls=calls,
            )
        ]
    return [Message(role=role or "user", text=str(m.get("content") or ""))]


def _anthropic_messages(m: dict[str, Any], names: dict[str, str]) -> list[Message]:
    """Normalize one Anthropic message.

    Args:
        m: The message.
        names: The tool-use id to tool name map, filled as calls are seen.

    Returns:
        The messages; a user message's tool results come first, then its text.
    """
    role = m.get("role", "user")
    content = m.get("content")
    if isinstance(content, str):
        return [Message(role=role, text=content)]
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    calls: list[tuple[str, str]] = []
    tool_results: list[Message] = []
    for b in content or []:
        if not isinstance(b, dict):
            continue
        match b.get("type"):
            case "text":
                text_parts.append(str(b.get("text", "")))
            case "thinking":
                thinking_parts.append(str(b.get("thinking", "")))
            case "tool_use":
                nm = str(b.get("name", ""))
                calls.append((nm, _pretty_args(b.get("input", {}))))
                if b.get("id"):
                    names[str(b["id"])] = nm
            case "tool_result":
                nm = names.get(str(b.get("tool_use_id", "")), "")
                tool_results.append(
                    Message(role="tool", text=result_text(b.get("content")), tool_name=nm)
                )
            case _:
                pass
    if role == "assistant":
        return [
            Message(
                role="assistant",
                text="\n\n".join(text_parts).strip(),
                thinking="\n".join(thinking_parts).strip(),
                tool_calls=calls,
            )
        ]
    # The loop may append a notice after a batch of tool results in the same message.
    text = "\n\n".join(text_parts).strip()
    if tool_results:
        return [*tool_results, *([Message(role=role, text=text)] if text else [])]
    return [Message(role=role, text=text)]


def _request_messages(m: dict[str, Any], shape: str, names: dict[str, str]) -> list[Message]:
    """Return normalized messages from one request message by wire shape."""
    return _openai_messages(m, names) if shape == "openai" else _anthropic_messages(m, names)


def _response_messages(resp: dict[str, Any], shape: str, names: dict[str, str]) -> list[Message]:
    """Return a response body's assistant messages by wire shape."""
    if shape == "responses":
        return _responses_messages(resp.get("output") or [], names)
    if shape == "openai":
        choices = resp.get("choices") or []
        if not choices:
            return []
        # The streaming path synthesises the message without a role.
        message = {**_as_dict(choices[0].get("message")), "role": "assistant"}
        return _openai_messages(message, names)
    if resp.get("content") is not None:
        return _anthropic_messages({"role": "assistant", "content": resp.get("content")}, names)
    return []


def _elided_strings(msg: dict[str, Any]) -> list[str]:
    """Return every elision-placeholder string one wire message carries, either shape."""
    out: list[str] = []
    content = msg.get("content") if "content" in msg else msg.get("output")
    if isinstance(content, str):
        if content.startswith(ELISION_MARKER_PREFIX):
            out.append(content)
    elif isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                inner = item.get("content")
                if isinstance(inner, str) and inner.startswith(ELISION_MARKER_PREFIX):
                    out.append(inner)
    return out


def _elision_identity(placeholder: str) -> str:
    """Return the elided call's identity, recovered from the placeholder's own words."""
    m = _ELIDED_IDENTITY_RE.search(placeholder)
    return m.group(1) if m else "a tool result"


def _elision_label(placeholder: str) -> str:
    """Return the placeholder's identity, noting a kept gist."""
    label = _elision_identity(placeholder)
    if placeholder.startswith(_GIST_MARKER_PREFIX):
        label += " (distilled gist kept)"
    return label


def _elision_marker(prev: list[Any], msgs: list[Any], upto: int) -> str:
    """Word the tool results elided between two request snapshots.

    The conversation view keeps showing the original results; this line says what
    the model still sees. Identity counts are compared, not placeholder bytes: a gist
    demoting to the bare marker is not re-reported, a second result of the same
    identity elided later is.

    Args:
        prev: The prior request's messages.
        msgs: This request's messages.
        upto: How many leading messages both requests share.

    Returns:
        The marker text, or "" when nothing new was elided.
    """
    labels: list[str] = []
    for i in range(min(upto, len(prev), len(msgs))):
        cur_m, prev_m = msgs[i], prev[i]
        if not isinstance(cur_m, dict) or not isinstance(prev_m, dict):
            continue
        before = collections.Counter(_elision_identity(s) for s in _elided_strings(prev_m))
        for s in _elided_strings(cur_m):
            ident = _elision_identity(s)
            if before[ident] > 0:
                before[ident] -= 1
            else:
                labels.append(_elision_label(s))
    if not labels:
        return ""
    shown = ", ".join(labels[:6]) + (f", +{len(labels) - 6} more" if len(labels) > 6 else "")
    noun = "result" if len(labels) == 1 else "results"
    return (
        f"context compaction: elided {len(labels)} older tool {noun}"
        f" from the model's context: {shown}"
    )


def fold_conversation(transcripts: list[dict[str, Any]]) -> list[Message]:
    """Fold per-call transcripts into one ordered conversation.

    Each request is reconciled against the prior one rather than predicted: a
    recorded response reappears as the next request's message only when the history
    grew, since an error transcript or an empty-response retry re-sends the identical
    list. Only the driving seats fold.

    Args:
        transcripts: The transcripts, in seq order.

    Returns:
        The messages, a marker for a compaction restart or an unreadable seq included.
    """
    transcripts = conversation_transcripts(transcripts)
    messages: list[Message] = []
    names: dict[str, str] = {}
    prev_len = 0
    prev_msgs: list[Any] = []
    pending_response = False
    prev_output: list[Any] = []
    call_seq = 0
    for t in transcripts:
        seq = t.get("seq", 0)
        if not isinstance(seq, int):
            messages.append(Message(role="marker", text=f"unreadable seq {seq!r}: call skipped"))
            continue
        req = _as_dict(_as_dict(t.get("request")).get("body"))
        resp = _as_dict(_as_dict(t.get("response")).get("body"))
        shape = _shape(req, resp)
        msgs = _request_items(req, shape)
        # Anthropic and Responses keep the system prompt out of the message list.
        sys = req.get("system") if shape == "anthropic" else req.get("instructions")
        if shape in ("anthropic", "responses") and prev_len == 0 and sys:
            messages.append(
                Message(
                    role="system", seq=seq, text=sys if isinstance(sys, str) else json.dumps(sys)
                )
            )
        if len(msgs) < prev_len:
            messages.append(Message(role="marker", text="context summarised / restarted", seq=seq))
            prev_len = 0
            pending_response = False
        elif marker := _elision_marker(prev_msgs, msgs, prev_len):
            messages.append(Message(role="marker", text=marker, seq=seq))
        # A Responses call's output is several items, echoed back verbatim.
        start = prev_len
        if shape == "responses":
            while (
                start - prev_len < len(prev_output)
                and start < len(msgs)
                and _same_item(msgs[start], prev_output[start - prev_len])
            ):
                start += 1
        elif pending_response and len(msgs) > prev_len:
            start = prev_len + 1
        fresh = msgs[start:]
        if shape == "responses":
            new_messages = _responses_messages(fresh, names)
        else:
            new_messages = [
                tt
                for m in fresh
                if isinstance(m, dict)
                for tt in _request_messages(m, shape, names)
            ]
        for tt in new_messages:
            # A result answers the previous round's call and reads under that round's seq; a
            # result the first request carries (a fork's inherited history) under that request's.
            tt.seq = (call_seq or seq) if tt.role == "tool" else seq
            messages.append(tt)
        response_messages = _response_messages(resp, shape, names)
        for rt in response_messages:
            rt.seq = seq
            messages.append(rt)
        call_seq = seq if response_messages else call_seq
        prev_len = len(msgs)
        prev_msgs = list(msgs)
        pending_response = bool(response_messages)
        prev_output = list(resp.get("output") or []) if shape == "responses" else []
    return messages


def _clip(s: str, n: int) -> str:
    """Return the text cut to `n` characters, the cut noted with its size."""
    return s if len(s) <= n else s[:n] + f"… (+{len(s) - n} chars)"


def render_markdown(
    messages: list[Message],
    *,
    session_id: str,
    show_thinking: bool = True,
    tools: str = "both",
    result_cap: int = 4000,
) -> str:
    """Render folded messages as a Markdown conversation.

    Args:
        messages: The messages to render.
        session_id: The heading's session id.
        show_thinking: Include the assistant's reasoning.
        tools: "both" for calls and results, "calls" for calls alone, "none".
        result_cap: The characters a tool result keeps.

    Returns:
        The Markdown text.
    """
    out: list[str] = [f"# Transcript: {session_id}", ""]
    for tn in messages:
        if tn.role == "marker":
            out.append(f"\n--- {tn.text} ---\n")
            continue
        if tn.role == "tool":
            if tools != "both":
                continue
            label = f" {tn.tool_name}" if tn.tool_name else ""
            out.append(f"  <-{label}: {_clip(tn.text, result_cap)}")
            out.append("")
            continue
        header = f"## {tn.role}"
        if tn.role == "assistant":
            header += f"  (seq {tn.seq})"
        body: list[str] = []
        if tn.thinking and show_thinking:
            body.append(f"<thinking>\n{tn.thinking}\n</thinking>")
        if tn.text:
            body.append(tn.text)
        if tn.tool_calls and tools != "none":
            body.extend(f"-> {name}({args})" for name, args in tn.tool_calls)
        if not body and (tn.thinking or tn.tool_calls):
            continue  # the flags hide all the message holds; an empty message keeps its heading
        out.extend([header, *body, ""])
    return "\n".join(out).rstrip() + "\n"
