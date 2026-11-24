# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Run-control handlers: ask the operator or signal the end of the session.

The harness exits the loop after dispatching a finish signal.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent6.tools import results, schema

if TYPE_CHECKING:
    from agent6.tools import dispatch


def finish_session(ctx: dispatch.ToolContext, raw: dict[str, Any]) -> results.FinishSessionResult:
    """Echo the validated summary and the structured `result` payload.

    Args:
        ctx: Unused for finish signals.
        raw: The tool call's arguments.

    Returns:
        The summary, the `result` payload state-machine agent states read, and the stale gate.
    """
    args = schema.FinishSessionInput.model_validate(raw)
    return results.FinishSessionResult(
        summary_text=args.summary, result=args.result, stale_gate=args.stale_gate
    )


def finish_planning(ctx: dispatch.ToolContext, raw: dict[str, Any]) -> results.FinishPlanningResult:
    """Echo the validated summary of a planning pass.

    The harness writes `plan_markdown` to disk and exits after dispatching the call.

    Args:
        ctx: Unused for finish signals.
        raw: The tool call's arguments.

    Returns:
        The summary and the plan's size in bytes.
    """
    args = schema.FinishPlanningInput.model_validate(raw)
    return results.FinishPlanningResult(
        summary_text=args.summary,
        plan_bytes=len(args.plan_markdown.encode("utf-8")),
    )


def ask_user(ctx: dispatch.ToolContext, raw: dict[str, Any]) -> results.AnswersResult:
    """Return the operator's answers to the model's questions."""
    args = schema.AskUserInput.model_validate(raw)
    answer = ctx.prompts.ask(args.questions, call_id=ctx.call_id, wait=args.wait)
    return results.AnswersResult(
        id=answer.id,
        answers=None if answer.status == "pending" else answer.answers,
        status=answer.status,
        note=answer.note,
        asked=tuple(q.question for q in args.questions),
    )
