# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Run-control signal handlers: ask_user, finish_run, finish_planning.

finish_run/finish_planning don't act; the workflow checks for the tool name in
the response's tool_uses and exits the loop after dispatching it. ask_user
poses the validated questions to the injected questioner (TUI modal / stdin /
headless skip), which owns the question.prompt/answer events itself."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agent6.tools.schema import AskUserInput, FinishPlanningInput, FinishRunInput, UserQuestion


def ask_user(
    questioner: Callable[[tuple[UserQuestion, ...]], tuple[str, ...]],
    raw: dict[str, Any],
) -> dict[str, Any]:
    """Pose one or more questions to the operator and return the answers.
    Answers align to `questions` by index."""
    args = AskUserInput.model_validate(raw)
    answers = questioner(args.questions)
    return {"answers": list(answers)}


def finish_run(raw: dict[str, Any]) -> dict[str, Any]:
    """Signal the workflow to terminate. Handler echoes the validated summary
    (and any structured ``result`` payload, used by state-machine agent
    states)."""
    args = FinishRunInput.model_validate(raw)
    return {"acknowledged": True, "summary": args.summary, "result": args.result}


def finish_planning(raw: dict[str, Any]) -> dict[str, Any]:
    """Signal the planning pass is done. Plan-mode counterpart of finish_run;
    the workflow writes ``plan_markdown`` to disk and exits after dispatching
    it. Handler echoes the validated summary."""
    args = FinishPlanningInput.model_validate(raw)
    return {
        "acknowledged": True,
        "summary": args.summary,
        "plan_bytes": len(args.plan_markdown.encode("utf-8")),
    }
