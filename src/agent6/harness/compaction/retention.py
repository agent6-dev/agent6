# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Select the task, the operator quotes and the whole turns for a local restart."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterator, Sequence

from agent6.harness import loop_conversation as history
from agent6.harness.compaction import window


def inputs(message: history.UserMessage) -> Iterator[history.ToolResultItem | history.Notice]:
    """Read results in call order, using the full answer notice where one was replayed.

    Args:
        message: A user message in wire order, results before notices.

    Yields:
        Each result or its answer notice, then the other notices.
    """
    answers = {
        it.result_index: it
        for it in message.items
        if isinstance(it, history.Notice) and it.result_index is not None
    }
    for index, item in enumerate(message.items):
        if isinstance(item, history.ToolResultItem):
            yield answers.get(index, item)
        elif item.result_index is None:
            yield item


def operator_words(item: history.Notice | history.ToolResultItem) -> tuple[str, ...]:
    """Read the verbatim input of a notice or an answered question result.

    Args:
        item: A user message block.

    Returns:
        One message per delivery: the opening's task, the quotes carried by a prior restart.
    """
    if isinstance(item, history.Notice):
        return item.operator_words or ((item.text,) if item.origin == "operator" else ())
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
    """Keep one chronological suffix after the summary and quote older operator messages.

    Attributes:
        replaced: The history the restart removes.
        quotes: Reserved operator messages older than the kept turns.
        tail: The kept turns and pending inputs.
        task: The run's task when it is reserved and no kept turn states it.
    """

    replaced: tuple[history.Message, ...]
    quotes: tuple[str, ...]
    tail: tuple[history.Message, ...]
    task: str = ""


def _result_messages(
    msg: history.UserMessage,
) -> tuple[history.UserMessage, list[history.UserMessage]]:
    """Keep full answers with their results and leave other notices for the next turn.

    Args:
        msg: The message answering the calls.

    Returns:
        The paired results and any inputs for the next reply.
    """
    results: list[history.ToolResultItem] = []
    answers: list[history.Notice] = []
    notices: list[history.Notice] = []
    for item in msg.items:
        if isinstance(item, history.ToolResultItem):
            if item.unshown_answer is not None:
                answers.append(
                    history.Notice(item.unshown_answer, "operator", result_index=len(results))
                )
                results.append(dataclasses.replace(item, unshown_answer=None))
            else:
                results.append(item)
        elif item.result_index is not None:
            answers.append(item)
        else:
            notices.append(item)
    following = [history.UserMessage(tuple(notices))] if notices else []
    return history.UserMessage((*results, *answers)), following


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
            paired, following = _result_messages(msg)
            turns[-1].append(paired)
            inputs.extend(following)
        else:
            inputs.append(msg)
    return turns, inputs


def _partition(
    messages: Sequence[history.Message],
    plan: history.UserMessage | None,
) -> tuple[list[history.Message], list[history.Message], list[history.Message]]:
    """Separate the notices with a fixed place on replay.

    Args:
        messages: The local history.
        plan: The plan notice to keep; other copies are superseded.

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
        elif origin != "reply_count":
            if isinstance(m, history.UserMessage) and m is not plan and m.plan_text is not None:
                continue
            history_messages.append(m)
    return contract, prior, history_messages


def select(
    messages: Sequence[history.Message],
    settings: window.CompactionSettings,
    task: str = "",
    pins: Sequence[str] = (),
) -> Replay:
    """Reserve the task and operator messages, then extend the kept suffix while whole turns fit.

    Args:
        messages: The live local history.
        settings: The resolved caps.
        task: The task the run was given, the oldest reservable message.
        pins: The pinned instructions, which the restart states in their own block.

    Returns:
        The helper transcript, verbatim quotes and paired messages to replay.
    """
    plan = history.latest_plan(messages)
    opening, prior, history_messages = _partition(messages, plan)
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
        for it in inputs(m)
        for word in operator_words(it)
    ]
    # A review contract's opening is its task and keeps its own place.
    task = "" if opening else task
    if task:
        # The run holds its task, so a rewrite that removed the opening cannot lose it.
        stated = next((turn for turn, word in words if word == task), -1)
        words = [(stated, task), *(w for w in words if w[1] != task)]
    # The pinned block states a pin, so its message is neither reserved nor counted.
    words = [w for w in words if w[1] not in pins]
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
    replaced = (*prior, *(m for turn in turns[:start] for m in turn))
    # A fork may have no plan.md. Keep its latest plan even outside the selected turns.
    if plan is not None and plan in replaced:
        opening.append(plan)
        replaced = tuple(m for m in replaced if m is not plan)
    quotes = tuple(word for turn, word in reversed(reserved) if turn < start)
    tail = (*opening, *(m for turn in turns[start:] for m in turn), *live)
    if task and quotes[:1] == (task,):
        # Reserved and older than the kept turns, the task leaves the quotes for its own label.
        return Replay(replaced, quotes[1:], tail, task)
    return Replay(replaced, quotes, tail)
