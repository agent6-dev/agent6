# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Claude Code stream events, assistant blocks and email scrubbing."""

from __future__ import annotations

import contextlib
import dataclasses
from collections.abc import Callable, Mapping
from typing import Any

from agent6.providers import _claude_code_session, _claude_code_wire, _stream

EMAIL_PLACEHOLDER = "<operator-email>"


def email_prefix_len(email: str, text: str) -> int:
    """Return the length of the longest proper prefix of the email that ends the text."""
    return next((k for k in range(len(email) - 1, 0, -1) if text.endswith(email[:k])), 0)


@dataclasses.dataclass(slots=True)
class Round:
    """One API round as it streams in.

    Attributes:
        blocks: The assistant blocks, in Anthropic shape.
        usage: The usage, from message_start and message_delta.
        stop_reason: The stop reason from message_delta.
        message_id: The message id from message_start.
        ended: Whether message_stop landed.
        cb: The open block's delta callback.
        held: The streamed tail the email scrub holds back.
    """

    blocks: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    usage: dict[str, Any] = dataclasses.field(default_factory=dict)
    stop_reason: str = ""
    message_id: str = ""
    ended: bool = False
    cb: Callable[[str], None] | None = None
    held: str = ""


def stream_event(
    s: _claude_code_session.Session,
    r: Round,
    event: Mapping[str, Any],
    watch: _claude_code_session.Watch,
    text_cb: Callable[[str], None] | None,
    thinking_cb: Callable[[str], None] | None,
) -> None:
    """Fold one Messages stream event into the round."""
    kind = event.get("type")
    if kind == "message_start":
        message = event.get("message") or {}
        r.message_id = str(message.get("id", ""))
        r.usage.update(message.get("usage") or {})
        watch.limit = _stream.STREAM_IDLE_TIMEOUT_S
    elif kind == "content_block_start":
        thinking = (event.get("content_block") or {}).get("type") == "thinking"
        watch.limit = (
            _stream.STREAM_THINKING_IDLE_TIMEOUT_S if thinking else _stream.STREAM_IDLE_TIMEOUT_S
        )
    elif kind == "content_block_stop":
        watch.limit = _stream.STREAM_IDLE_TIMEOUT_S
        emit_delta(s, r, "", r.cb, final=True)
    elif kind == "content_block_delta":
        delta = event.get("delta") or {}
        if delta.get("type") == "text_delta":
            emit_delta(s, r, str(delta.get("text", "")), text_cb)
        elif delta.get("type") == "thinking_delta":
            emit_delta(s, r, str(delta.get("thinking", "")), thinking_cb)
    elif kind == "message_delta":
        r.stop_reason = str((event.get("delta") or {}).get("stop_reason") or "")
        r.usage.update(event.get("usage") or {})
    elif kind == "message_stop":
        r.ended = True


def emit_delta(
    s: _claude_code_session.Session,
    r: Round,
    piece: str,
    cb: Callable[[str], None] | None,
    *,
    final: bool = False,
) -> None:
    """Stream a delta with the email scrubbed across delta boundaries.

    A tail that could start an occurrence waits for the next delta; the block's
    stop flushes it.

    Args:
        s: The session.
        r: The round.
        piece: The delta's text.
        cb: The delta callback; None streams nothing.
        final: Whether this flushes the held tail.
    """
    if cb is None:
        return
    r.cb = cb
    out = scrub(s, r.held + piece)
    keep = 0 if final else email_prefix_len(s.account_email, out)
    r.held = out[len(out) - keep :] if keep else ""
    if keep:
        out = out[:-keep]
    if out:
        with contextlib.suppress(Exception):
            cb(out)


def block(s: _claude_code_session.Session, raw: Mapping[str, Any]) -> dict[str, Any]:
    """Return one assistant block in Anthropic shape.

    Args:
        s: The session holding the email to scrub.
        raw: A content block of the CLI's `assistant` line.

    Returns:
        The block with tool names bare, the login's email scrubbed from text and
        thinking, and the CLI's `caller` dropped.
    """
    kind = raw.get("type")
    if kind == "text":
        return {"type": "text", "text": scrub(s, str(raw.get("text", "")))}
    if kind == "thinking":
        out: dict[str, Any] = {
            "type": "thinking",
            "thinking": scrub(s, str(raw.get("thinking", ""))),
        }
        if raw.get("signature"):
            out["signature"] = raw["signature"]
        return out
    if kind == "tool_use":
        return {
            "type": "tool_use",
            "id": str(raw.get("id", "")),
            "name": _claude_code_wire.bare_tool_name(str(raw.get("name", ""))),
            "input": raw.get("input"),
        }
    return dict(raw)


def scrub(s: _claude_code_session.Session, text: str) -> str:
    """Return the text with the login's email replaced by the placeholder."""
    return text.replace(s.account_email, EMAIL_PLACEHOLDER) if s.account_email else text
