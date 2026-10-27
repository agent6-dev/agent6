# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Measure context, retain recent exchanges and parse summary checkoffs."""

from __future__ import annotations

import dataclasses
import json
import re
from collections.abc import Sequence

from agent6 import media
from agent6.harness import _compaction_result, _conversation
from agent6.providers import Provider, types

# Chars, not tokens: tokens are roughly chars/4 for English-shaped content.
DROP_BLOCKS_AT_CHARS = 256_000  # ~64k tokens of tool_result content


SUMMARISE_AT_CHARS = 768_000  # ~192k tokens: full context restart


CHECKOFF_FENCE_RE = re.compile(r"```checkoff\s*\n(.*?)\n```", re.DOTALL)


def parse_checkoff(text: str) -> tuple[list[str], list[str]]:
    """Extract the tier-2 check-off block from the summariser's reply.

    The summariser appends a fenced checkoff block holding `completed_ids` and
    `new_tasks`, so the DAG stays accurate without the worker calling
    update_task.

    Args:
        text: The summariser's reply.

    Returns:
        (completed ids, new task titles); ([], []) for a missing or malformed
        block, so a bad summary never breaks the run.
    """
    m = CHECKOFF_FENCE_RE.search(text)
    if m is None:
        return [], []
    try:
        data = json.loads(m.group(1))
    except (ValueError, TypeError):
        return [], []
    if not isinstance(data, dict):
        return [], []
    return nonempty_strs(data.get("completed_ids")), nonempty_strs(data.get("new_tasks"))


def nonempty_strs(value: object) -> list[str]:
    """Return the stripped, non-empty strings in a JSON value.

    Args:
        value: The parsed field; `null`, a number or a bool are natural
            summariser output.

    Returns:
        The strings, [] when the value is not a list.
    """
    if not isinstance(value, list):
        return []
    return [s.strip() for s in value if isinstance(s, str) and s.strip()]


def strip_checkoff(text: str) -> str:
    """Remove the checkoff block from a summary before it re-enters context.

    Args:
        text: The summariser's reply.

    Returns:
        The narrative alone, stripped.
    """
    return CHECKOFF_FENCE_RE.sub("", text).strip()


def context_chars(conversation: _conversation.Conversation) -> int:
    """Return the character size of the conversation context.

    Notice text, tool_result content and every value of every assistant raw
    block count, since `to_wire` sends those blocks verbatim; counting known
    keys alone would score a thinking block as zero. The tier-2 trigger
    measures this against about 80% of the model's window.

    Args:
        conversation: The loop's history.

    Returns:
        The sum over the messages.
    """
    return sum(message_chars(message) for message in conversation.messages)


def message_chars(message: _conversation.Message) -> int:
    """Return one message's contribution to `context_chars`.

    Args:
        message: The message.

    Returns:
        Its character count, "type" keys excluded.
    """
    if isinstance(message, _conversation.AssistantMessage):
        total = 0
        for item in message.raw_content:
            if not isinstance(item, dict):
                total += len(str(item))
                continue
            # "type" is the discriminator, not payload.
            total += sum(
                len(v if isinstance(v, str) else str(v))
                for k, v in item.items()
                if k != "type" and v is not None
            )
        return total
    return sum(
        (len(item.content) + (media.MEDIA_CONTEXT_CHARS if item.attachment else 0))
        if isinstance(item, _conversation.ToolResultItem)
        else len(item.text)
        for item in message.items
    )


# The verbatim tail a tier-2 restart keeps, about 20k tokens; `[context]
# keep_recent_chars` overrides.
KEEP_RECENT_CHARS = 80_000


@dataclasses.dataclass(frozen=True, slots=True)
class CompactionSettings:
    """Context compaction as the run configures it.

    Attributes:
        drop_at_chars: Tier 1 turns the oldest tool results into placeholders
            past this.
        summarise_at_chars: Tier 2 summarises and restarts past this.
        tool_result_cap_bytes: The bound on one result before it enters the
            conversation.
        keep_recent_chars: The verbatim tail a restart keeps.
        keep_thinking_turns: At tier-1 moments, keep thinking for the newest N
            assistant messages (N >= 1); -1 keeps all.
        elision_gists: A large read decays through a model-written gist first.
        summary_max_tokens: The summariser's output cap.
        summariser: The reviewer role's provider; the worker's when None.
    """

    drop_at_chars: int = DROP_BLOCKS_AT_CHARS
    summarise_at_chars: int = SUMMARISE_AT_CHARS
    tool_result_cap_bytes: int = _compaction_result.TOOL_RESULT_CAP_BYTES
    keep_recent_chars: int = KEEP_RECENT_CHARS
    keep_thinking_turns: int = -1
    elision_gists: bool = True
    summary_max_tokens: int = 2048
    summariser: Provider | None = None


def strip_old_thinking(
    conversation: _conversation.Conversation, *, keep_messages: int
) -> tuple[int, int]:
    """Drop thinking blocks from assistant messages older than the newest few.

    Anthropic requires the signed thinking block of a tool_use still being
    answered, so callers pass at least 1.

    Args:
        conversation: The loop's history.
        keep_messages: How many of the newest assistant messages keep their thinking.

    Returns:
        (messages stripped, chars removed).
    """
    assistant_idxs = [
        i
        for i, message in enumerate(conversation.messages)
        if isinstance(message, _conversation.AssistantMessage)
    ]
    n = chars = 0
    for idx in assistant_idxs[:-keep_messages]:
        removed = conversation.strip_thinking(idx)
        if removed:
            n += 1
            chars += removed
    return n, chars


def request_prefix_chars(system: str, tools: Sequence[types.ToolDefinition]) -> int:
    """Return the chars every request carries besides the conversation.

    The window bounds the whole request, so a threshold on the conversation
    alone leaves a band the size of this prefix where the loop sees room and
    the provider refuses. AGENTS.md rides in the system prompt whole.

    Args:
        system: The system prompt.
        tools: The tool definitions.

    Returns:
        The system prompt's length plus each tool's name, description and
        compact schema.
    """
    return len(system) + sum(
        len(t.name) + len(t.description) + len(json.dumps(t.input_schema, separators=(",", ":")))
        for t in tools
    )


def recent_tail_start(messages: Sequence[_conversation.Message], cap_chars: int) -> int:
    """Return the index where a tier-2 restart's verbatim tail begins.

    A safe start is any message except a user message carrying tool_results, which
    answers the assistant message before it. Message 0, the task, is never part of
    the tail; the restart keeps it separately.

    Args:
        messages: The conversation's messages.
        cap_chars: The tail's size cap.

    Returns:
        The start of the largest tail of whole messages within the cap on a safe
        boundary; `len(messages)` when nothing is kept.
    """
    if cap_chars <= 0:
        return len(messages)
    total = 0
    start = len(messages)
    i = len(messages) - 1
    while i >= 1:
        size = message_chars(messages[i])
        if total + size > cap_chars:
            break
        total += size
        start = i
        i -= 1
    while start < len(messages) and starts_with_results(messages[start]):
        start += 1
    if start == len(messages):
        # The newest exchange exceeds the cap alone: keep it anyway, since paraphrasing
        # undelivered results away is the one loss the tail exists to prevent.
        assistant_idxs = [
            i
            for i in range(1, len(messages))
            if isinstance(messages[i], _conversation.AssistantMessage)
        ]
        if assistant_idxs:
            start = assistant_idxs[-1]
    return start


def starts_with_results(message: _conversation.Message) -> bool:
    return isinstance(message, _conversation.UserMessage) and any(
        isinstance(item, _conversation.ToolResultItem) for item in message.items
    )
