# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Run-control signal handlers: finish_session and finish_planning.

Neither acts; the harness checks for the tool name in the response's tool uses and exits the
loop after dispatching it.
"""

from __future__ import annotations

from typing import Any

from agent6.tools import results, schema


def finish_session(raw: dict[str, Any]) -> results.FinishSessionResult:
    """Echo the validated summary and the structured `result` payload.

    Args:
        raw: The tool call's arguments.

    Returns:
        The summary, the `result` payload state-machine agent states read, and the stale gate.
    """
    args = schema.FinishSessionInput.model_validate(raw)
    return results.FinishSessionResult(
        summary_text=args.summary, result=args.result, stale_gate=args.stale_gate
    )


def finish_planning(raw: dict[str, Any]) -> results.FinishPlanningResult:
    """Echo the validated summary of a planning pass.

    The harness writes `plan_markdown` to disk and exits after dispatching the call.

    Args:
        raw: The tool call's arguments.

    Returns:
        The summary and the plan's size in bytes.
    """
    args = schema.FinishPlanningInput.model_validate(raw)
    return results.FinishPlanningResult(
        summary_text=args.summary,
        plan_bytes=len(args.plan_markdown.encode("utf-8")),
    )
