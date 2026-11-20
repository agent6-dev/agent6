"""Build the two summary requests and validate their shared output contract."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Sequence
from typing import Literal

from agent6.harness import loop_conversation
from agent6.harness.compaction import window
from agent6.prompts import compaction
from agent6.providers import provider_types


@dataclasses.dataclass(frozen=True, slots=True)
class Summary:
    """An accepted summary and its writer's provenance."""

    raw: str
    writer: Literal["worker", "helper"]
    fallback_reason: str
    usage: provider_types.ProviderUsage


def helper_input(
    messages: Sequence[loop_conversation.Message],
    instruction: str,
    *,
    context_tokens: int | None,
    max_tokens: int,
) -> str:
    """Keep replaced text and tool inputs, dropping results and the oldest excess.

    One UTF-8 byte per token is conservative; the reserve covers provider output floors.

    Args:
        messages: The replaced history, excluding the original task.
        instruction: The task, check-off, pins, focus and prior summary.
        context_tokens: The helper's known context window.
        max_tokens: The configured output cap.

    Returns:
        The helper's bounded user message.

    Raises:
        ProviderError: The window is unknown or the fixed instruction cannot fit.
    """
    if context_tokens is None:
        raise provider_types.ProviderError("helper context window is unknown")
    parts: list[str] = []
    for message in messages:
        if isinstance(message, loop_conversation.AssistantMessage):
            for block in message.raw_content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    parts.append(f"[assistant:text] {block.get('text', '')}")
                elif block.get("type") == "tool_use":
                    parts.append(
                        f"[assistant:tool_use {block.get('name', '')}] "
                        + json.dumps(block.get("input") or {}, ensure_ascii=False)
                    )
        else:
            parts.extend(
                f"[user:text] {item.text}"
                for item in message.items
                if isinstance(item, loop_conversation.Notice)
            )
    prefix = instruction + "\n\nTRANSCRIPT (oldest first, tool output removed):\n"
    missing = "[The beginning is missing because it does not fit the helper's window.]\n"
    # Reserve message framing as well as the largest built-in reasoning output floor.
    room = (
        context_tokens
        - max(max_tokens, 32768)
        - 256
        - len(compaction.CONTEXT_SUMMARY_SYSTEM_PROMPT.encode())
        - len(prefix.encode())
    )
    if room <= len(missing.encode()):
        raise provider_types.ProviderError("helper instructions exceed its context window")
    transcript = "\n".join(parts).encode()
    if len(transcript) > room:
        transcript = transcript[-(room - len(missing.encode())) :]
        return prefix + missing + transcript.decode("utf-8", errors="ignore")
    return prefix + transcript.decode()


def rejection(response: provider_types.ProviderResponse) -> str:
    """Name an unusable summary, or return an empty string for accepted prose.

    Args:
        response: Either writer's response.

    Returns:
        The reason the conversation must stay unchanged.
    """
    if response.tool_uses:
        return "summary contains a tool call"
    if provider_types.output_cap_truncated(response):
        return "summary reached the output cap"
    raw = response.raw
    refusal = any(
        isinstance(block, dict) and block.get("type") == "refusal"
        for block in raw.get("content", [])
    ) or any(
        isinstance(choice, dict) and (choice.get("message") or {}).get("refusal")
        for choice in raw.get("choices", [])
    )
    if refusal or response.stop_reason.lower() in ("refusal", "content_filter"):
        return "provider refusal"
    if response.stop_reason.lower() not in ("end_turn", "stop", "stop_sequence"):
        return f"incomplete summary ({response.stop_reason})"
    if not window.strip_checkoff(response.text.strip()):
        return "empty summary"
    return ""


def instruction(
    *,
    writer: Literal["worker", "helper"],
    task: str,
    open_tasks: list[tuple[str, str]],
    focus: str,
    pins: Sequence[str],
    messages: Sequence[loop_conversation.Message],
) -> str:
    """Build the summary's goal, task check-off, focus, pins and prior summary.

    The worker also gets the summary prompt, which the helper has as its system text.

    Args:
        writer: Who receives the instruction.
        task: The original goal.
        open_tasks: Open task ids and titles.
        focus: The operator's summary focus.
        pins: Instructions repeated verbatim after restart.
        messages: Replaced history, read for the prior summary a helper cannot see.

    Returns:
        The instruction appended to the worker or sent to the helper.
    """
    if open_tasks:
        task_lines = "\n".join(f"- {tid}: {title}" for tid, title in open_tasks)
        checkoff_req = (
            "\n\nThe worker is tracking these OPEN tasks:\n"
            f"{task_lines}\n\n"
            "After the summary, append a fenced block exactly like:\n"
            "```checkoff\n"
            '{"completed_ids": ["<ids the transcript clearly shows finished>"], '
            '"new_tasks": ["<short title of work discovered but not yet tracked>"]}\n'
            "```\n"
            "Mark a task completed ONLY if the transcript clearly shows it done;"
            " leave the rest open. Use [] when none apply."
        )
    else:
        checkoff_req = ""
    focus_req = (
        f"\n\nOperator focus for this summary, weigh these aspects heavily:\n{focus}"
        if focus
        else ""
    )
    pins_req = ""
    if pins:
        pin_lines = "\n".join(f"{i}. {p}" for i, p in enumerate(pins, start=1))
        pins_req = compaction.PINS_NO_RESTATE_CLAUSE + pin_lines
    # A helper tail clip must not erase the previous restart summary; the worker has it in view.
    prior_req = ""
    for message in messages if writer == "helper" else ():
        for item in getattr(message, "items", ()):
            if isinstance(item, loop_conversation.Notice) and (
                prior := compaction.progress_summary_from_notice(item.text)
            ):
                prior_req = (
                    "\n\nThis conversation was ALREADY compacted; the summary from"
                    " that restart follows, and the transcript below covers only what"
                    " happened SINCE. Carry anything still relevant into the new"
                    f" summary:\n{prior}"
                )
    body = f"TASK (the goal, verbatim):\n{task}{checkoff_req}{focus_req}{pins_req}{prior_req}"
    if writer == "helper":
        return body
    return (
        "Write the summary now. Do not make a tool call; no tool call will be run.\n"
        + compaction.CONTEXT_SUMMARY_SYSTEM_PROMPT
        + "\nThe conversation remains in view, including its previous summary.\n\n"
        + body
    )
