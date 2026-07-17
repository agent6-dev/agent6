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
from collections.abc import Sequence
from dataclasses import dataclass

from agent6.workflows._conversation import AssistantTurn, Notice, Turn


@dataclass(frozen=True, slots=True)
class CritiqueResult:
    text: str
    satisfied: bool


def format_tail_for_critic(
    turns: Sequence[Turn], *, max_messages: int = 6, max_chars: int = 6000
) -> str:
    """Render the last few turns as a plain-text transcript for the critic.
    Tool calls / results are shown as compact summaries; long payloads are
    truncated so the critic call stays cheap. Assistant thinking blocks are
    skipped - the critic doesn't need them."""
    parts: list[str] = []
    for turn in turns[-max_messages:]:
        if isinstance(turn, AssistantTurn):
            for block in turn.raw_content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text":
                    parts.append(f"[assistant:text] {str(block.get('text', ''))[:1500]}")
                elif btype == "tool_use":
                    inp = json.dumps(block.get("input") or {}, ensure_ascii=False)
                    parts.append(f"[assistant:tool_use {block.get('name', '')}] {inp[:800]}")
            continue
        for item in turn.items:
            if isinstance(item, Notice):
                parts.append(f"[user:text] {item.text[:1500]}")
            else:
                parts.append(f"[user:tool_result] {item.content[:800]}")
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
