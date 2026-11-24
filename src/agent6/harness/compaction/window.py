# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Measure context, retain recent exchanges and parse summary checkoffs."""

from __future__ import annotations

import dataclasses
import json
import re
from collections.abc import Sequence
from typing import Literal

from agent6 import kinds, media
from agent6.harness import loop_conversation
from agent6.harness.compaction import elision as compaction_elision
from agent6.providers import Provider, provider_types

# Chars, not tokens: tokens are roughly chars/4 for English-shaped content.
DROP_BLOCKS_AT_CHARS = 256_000  # ~64k tokens of tool_result content


SUMMARIZE_AT_CHARS = 768_000  # ~192k tokens: full context restart


CHECKOFF_FENCE_RE = re.compile(r"```checkoff\s*\n(.*?)\n```", re.DOTALL)


def parse_checkoff(text: str) -> tuple[list[str], list[str]]:
    """Extract the tier-2 check-off block from the summarizer's reply.

    The summarizer appends a fenced checkoff block holding `completed_ids` and
    `new_tasks`, so the DAG stays accurate without the worker calling
    update_task.

    Args:
        text: The summarizer's reply.

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
            summarizer output.

    Returns:
        The strings, [] when the value is not a list.
    """
    if not isinstance(value, list):
        return []
    return [s.strip() for s in value if isinstance(s, str) and s.strip()]


def strip_checkoff(text: str) -> str:
    """Remove the checkoff block from a summary before it re-enters context.

    Args:
        text: The summarizer's reply.

    Returns:
        The narrative alone, stripped.
    """
    return CHECKOFF_FENCE_RE.sub("", text).strip()


def context_chars(conversation: loop_conversation.Conversation) -> int:
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


def message_chars(message: loop_conversation.Message) -> int:
    """Return one message's contribution to `context_chars`.

    Args:
        message: The message.

    Returns:
        Its character count, "type" keys excluded.
    """
    if isinstance(message, loop_conversation.AssistantMessage):
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
        if isinstance(item, loop_conversation.ToolResultItem)
        else len(item.text) + media.MEDIA_CONTEXT_CHARS * len(item.images)
        for item in message.items
    )


# The verbatim tail a tier-2 restart keeps, about 20k tokens; `[compaction]
# keep_recent_chars` overrides.
KEEP_RECENT_CHARS = 80_000


@dataclasses.dataclass(frozen=True, slots=True)
class IdleCompaction:
    """Hold the driving provider's local wait policy.

    Attributes:
        provider: The provider entry that owns the limits.
        seconds: The idle limit from the start of the last worker request.
        min_chars: The character floor for the whole context.
    """

    provider: str
    seconds: float
    min_chars: int


@dataclasses.dataclass(frozen=True, slots=True)
class CompactionSettings:
    """Context compaction as the run configures it.

    Attributes:
        cold_start: What to do with older local history on a cold resume.
        drop_at_chars: Tier 1 turns the oldest tool results into placeholders
            past this, aiming for half the threshold where protections allow.
        summarize_at_chars: Tier 2 summarizes and restarts past this.
        tool_result_cap_bytes: The bound on one result before it enters the
            conversation.
        keep_operator_messages: How many recent operator messages to reserve, -1 for all.
        keep_recent_turns: How many recent turns to replay, -1 for all that fit.
        keep_recent_chars: The shared character cap on local replay.
        keep_thinking_turns: How many recent turns keep thinking during a rewrite, -1 for all.
        elision_gists: A large read decays through a model-written gist first.
        summary_max_tokens: The summarizer's output cap.
        summarizer: The writer selected for tier 2.
        helper: The helper role's provider; the worker's when None.
        helper_context_tokens: The helper's known window; None refuses a helper summary.
        endpoint: The history operations of the driving route.
        idle: The driving provider's idle settings, or None without an HTTP provider.
    """

    drop_at_chars: int = DROP_BLOCKS_AT_CHARS
    summarize_at_chars: int = SUMMARIZE_AT_CHARS
    tool_result_cap_bytes: int = compaction_elision.TOOL_RESULT_CAP_BYTES
    keep_operator_messages: int = -1
    keep_recent_turns: int = -1
    keep_recent_chars: int = KEEP_RECENT_CHARS
    keep_thinking_turns: int = -1
    elision_gists: bool = True
    summary_max_tokens: int = 2048
    summarizer: Literal["worker", "helper"] = "worker"
    helper: Provider | None = None
    helper_context_tokens: int | None = None
    endpoint: provider_types.EndpointCompaction = dataclasses.field(
        default_factory=provider_types.EndpointCompaction
    )
    idle: IdleCompaction | None = None
    cold_start: kinds.ColdStart = "drop"


def strip_old_thinking(
    conversation: loop_conversation.Conversation, *, keep_messages: int
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
        if isinstance(message, loop_conversation.AssistantMessage)
    ]
    n = chars = 0
    for idx in assistant_idxs[:-keep_messages]:
        removed = conversation.strip_thinking(idx)
        if removed:
            n += 1
            chars += removed
    return n, chars


def request_prefix_chars(system: str, tools: Sequence[provider_types.ToolDefinition]) -> int:
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
