# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Rolling prompt-cache breakpoints for the agent loop.

Anthropic prompt caching bills a request's prefix up to a ``cache_control``
breakpoint at 0.1x once cached (1.25x to write). The provider marks the
system prompt and the tool list, but the conversation itself dominates input
tokens in a long run and grows every turn: without a breakpoint near the
tail, each turn re-bills the whole history at full price (quadratic in run
length).

``roll_cache_breakpoints`` keeps exactly two breakpoints inside ``messages``:

- the block marked before the PREVIOUS call (its prefix is now cached, so
  this breakpoint is the guaranteed cache HIT for the current call), and
- the final block of the current last message (the cache WRITE that next
  call's hit lands on).

Everything older is unmarked. Together with the provider's two static marks
(system + last tool) that is 4 breakpoints, Anthropic's per-request maximum.

The markers live in the loop-owned ``messages`` list (and therefore in resume
snapshots), so continuity survives crash-resume. OpenAI-format providers
rebuild content block-by-block and never forward the field. Tier-1 elision
mutates old blocks and so invalidates the cached prefix; that costs one 1.25x
re-write on the next call and the rolling pair keeps caching from there.
"""

from __future__ import annotations

from typing import Any

# Block types that may legally carry ``cache_control`` in a user message.
_MARKABLE_TYPES = frozenset({"text", "tool_result"})


def _marked_blocks(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Blocks carrying ``cache_control``, in message order."""
    marked: list[dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and "cache_control" in block:
                marked.append(block)
    return marked


def _tail_markable_block(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The newest block that may carry a breakpoint (scanning tail-first)."""
    for msg in reversed(messages):
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in reversed(content):
            if isinstance(block, dict) and block.get("type") in _MARKABLE_TYPES:
                return block
    return None


def roll_cache_breakpoints(messages: list[dict[str, Any]]) -> None:
    """Advance the rolling cache breakpoints to the current tail.

    Mutates ``messages`` in place: unmarks every ``cache_control`` block
    except the most recent one (the previous call's write position), then
    marks the final markable block of the last message. Idempotent; safe on
    histories rewritten by compaction or restored from a resume snapshot.
    String-content messages (no block list) are skipped.
    """
    marked = _marked_blocks(messages)
    target = _tail_markable_block(messages)

    keep_ids = set()
    if target is not None:
        keep_ids.add(id(target))
        # marked[] is in message order, so its newest non-target entry is the
        # previous call's breakpoint: the position whose prefix the cache
        # already holds. Keeping it (rather than marked[-1], which is the
        # target itself when nothing was appended, e.g. a crash-resume
        # re-issuing the same call) makes the roll idempotent.
        prev = next((b for b in reversed(marked) if b is not target), None)
        if prev is not None:
            keep_ids.add(id(prev))
    for block in marked:
        if id(block) not in keep_ids:
            block.pop("cache_control", None)
    if target is not None and "cache_control" not in target:
        target["cache_control"] = {"type": "ephemeral"}
