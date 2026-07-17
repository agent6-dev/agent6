# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Helpers for the in-loop reviewing critic.

The critic is an optional sub-agent the loop can run to second-guess the
worker (on verify failure, periodically, or before a finish_run). This module
holds the pure pieces: the `CritiqueResult` record, rendering a compact
transcript tail for the critic call, and parsing its VERDICT line. The critic's
system prompt lives in agent6.prompts.revision; the loop owns running the call.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class CritiqueResult:
    text: str
    satisfied: bool


def format_messages_tail_for_critic(
    messages: list[dict[str, Any]], *, max_messages: int = 6, max_chars: int = 6000
) -> str:
    """Render the last few messages as a plain-text transcript for the
    critic. Tool calls / results are shown as compact summaries; long
    payloads are truncated so the critic call stays cheap.
    """
    tail = messages[-max_messages:]
    parts: list[str] = []
    for msg in tail:
        role = str(msg.get("role", "?"))
        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(f"[{role}] {content[:1500]}")
            continue
        if not isinstance(content, list):
            parts.append(f"[{role}] {str(content)[:1500]}")
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                parts.append(f"[{role}:text] {str(block.get('text', ''))[:1500]}")
            elif btype == "tool_use":
                inp = json.dumps(block.get("input") or {}, ensure_ascii=False)
                parts.append(f"[{role}:tool_use {block.get('name', '')}] {inp[:800]}")
            elif btype == "tool_result":
                body = block.get("content", "")
                if not isinstance(body, str):
                    body = json.dumps(body, ensure_ascii=False)
                parts.append(f"[{role}:tool_result] {body[:800]}")
            elif btype == "thinking":
                # Skip reasoning blocks - the critic doesn't need them.
                continue
    joined = "\n".join(parts)
    if len(joined) > max_chars:
        joined = joined[-max_chars:]
    return joined


def parse_critic_verdict(text: str) -> bool:
    """Return True iff the critic's last non-empty line is ``VERDICT:
    SATISFIED``. Anything else is treated as NEEDS_WORK."""
    last = ""
    for raw in reversed(text.splitlines()):
        line = raw.strip()
        if line:
            last = line
            break
    return last.upper() == "VERDICT: SATISFIED"
