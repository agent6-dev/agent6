# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Select the operator quotes and whole turns for a local restart."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Sequence

from agent6.harness import loop_conversation as history
from agent6.harness.compaction import window


def operator_words(item: history.Notice | history.ToolResultItem) -> tuple[str, ...]:
    """Read operator input from its typed notice or answered question result.

    Args:
        item: A user message block.

    Returns:
        One message per delivery, including the quotes carried by a prior restart.
    """
    if isinstance(item, history.Notice):
        if item.origin == "operator":
            return item.operator_words or (item.text,)
        return item.operator_words if item.origin == "restart" else ()
    if item.for_call.name != "ask_user":
        return ()
    try:
        value = json.loads(item.content)
    except (ValueError, RecursionError):
        return ()
    if isinstance(value, dict) and value.get("status") == "answered":
        answers = value.get("answers")
        if isinstance(answers, list) and answers and all(isinstance(a, str) for a in answers):
            return ("\n".join(answers),)
    return ()


@dataclasses.dataclass(frozen=True, slots=True)
class Replay:
    """Keep one chronological suffix after the summary and quote older operator messages."""

    replaced: tuple[history.Message, ...]
    quotes: tuple[str, ...]
    tail: tuple[history.Message, ...]


def _turns(
    messages: Sequence[history.Message],
    keep_thinking_turns: int,
) -> tuple[list[list[history.Message]], list[history.Message]]:
    """Group each reply with the inputs it consumed and the results it produced.

    Args:
        messages: History since the last restart.
        keep_thinking_turns: How many recent replies keep their thinking, or -1 for all.

    Returns:
        The turns and the inputs the worker has yet to read.
    """
    turns: list[list[history.Message]] = []
    inputs: list[history.Message] = []
    strip = (
        sum(isinstance(m, history.AssistantMessage) for m in messages) - keep_thinking_turns
        if keep_thinking_turns > 0
        else 0
    )
    for msg in messages:
        if isinstance(msg, history.AssistantMessage):
            reply = msg.without_thinking()[0] if len(turns) < strip else msg
            if reply.raw_content:
                inputs.append(reply)
            turns.append(inputs)
            inputs = []
        elif msg.items and isinstance(msg.items[0], history.ToolResultItem):
            results = tuple(it for it in msg.items if isinstance(it, history.ToolResultItem))
            # Preserve the capped result on the wire. Carry the full operator answer beside it.
            answers = tuple(
                history.Notice(it.unshown_answer, "operator")
                for it in results
                if it.unshown_answer is not None
            )
            if answers:
                turns[-1].append(
                    history.UserMessage(
                        tuple(dataclasses.replace(it, unshown_answer=None) for it in results)
                    )
                )
                turns[-1].append(history.UserMessage(answers))
            else:
                turns[-1].append(
                    msg if len(results) == len(msg.items) else history.UserMessage(results)
                )
            if len(results) < len(msg.items):
                inputs.append(history.UserMessage(msg.items[len(results) :]))
        else:
            inputs.append(msg)
    return turns, inputs


def _partition(
    messages: Sequence[history.Message],
) -> tuple[list[history.Message], list[history.Message], list[history.Message]]:
    """Separate the notices with a fixed place on replay.

    Args:
        messages: The local history.

    Returns:
        The review opening, prior summaries and selectable history.
    """
    contract: list[history.Message] = []
    prior: list[history.Message] = []
    history_messages: list[history.Message] = []
    for m in messages:
        notice = m.items[0] if isinstance(m, history.UserMessage) and len(m.items) == 1 else None
        origin = notice.origin if isinstance(notice, history.Notice) else None
        if origin == "contract":
            contract.append(m)
        elif origin == "restart":
            prior.append(m)
        elif origin not in ("plan", "reply_count"):
            history_messages.append(m)
    return contract, prior, history_messages


def select(messages: Sequence[history.Message], settings: window.CompactionSettings) -> Replay:
    """Reserve operator messages, then extend the kept suffix while whole turns fit.

    Args:
        messages: The live local history.
        settings: The resolved caps.

    Returns:
        The helper transcript, verbatim quotes and paired messages to replay.
    """
    contract, prior, history_messages = _partition(messages)
    turns, live = _turns(history_messages, settings.keep_thinking_turns)
    start = len(turns)
    remaining = settings.keep_recent_chars
    optional = remaining > 0 and settings.keep_recent_turns != 0
    if turns and not optional:
        # Even at zero, unanswered calls and results without a subsequent reader are live work.
        last = turns[-1]
        for i, m in enumerate(last):
            if isinstance(m, history.AssistantMessage) and m.tool_uses:
                live = [*last[i:], *live]
                del last[i:]
                break

    groups = [prior, *turns, live]
    words = [
        (i - 1, word)
        for i, group in enumerate(groups)
        for m in group
        if isinstance(m, history.UserMessage)
        for it in m.items
        for word in operator_words(it)
    ]
    reserved: list[tuple[int, str]] = []
    costs: dict[int, int] = {}
    for n, (turn, word) in enumerate(reversed(words)):
        if n == settings.keep_operator_messages or remaining <= 0:
            break
        if turn >= start:
            continue  # Already paid for, but counts as one of the newest N inputs.
        if len(word) > remaining:
            break
        reserved.append((turn, word))
        remaining -= len(word)
        costs[turn] = costs.get(turn, 0) + len(word)

    while optional and start > 0 and len(turns) - start != settings.keep_recent_turns:
        cost = sum(map(window.message_chars, turns[start - 1])) - costs.get(start - 1, 0)
        # Only the newest turn can exceed the room left after reserving operator words.
        if cost > remaining and start != len(turns):
            break
        remaining -= cost
        start -= 1
    return Replay(
        (*prior, *(m for turn in turns[:start] for m in turn)),
        tuple(word for turn, word in reversed(reserved) if turn < start),
        (*contract, *(m for turn in turns[start:] for m in turn), *live),
    )
