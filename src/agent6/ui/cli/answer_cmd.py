# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Answer a run's `ask_user` question from a script (`agent6 answer`).

`steer` refuses a run blocked on a question, and every other way in needs a person at a
screen. This writes the same answer file those front-ends write, so a listener that forwards
questions elsewhere can send the reply back. A queued question takes an answer the run reads
at its next turn; an ended run's question takes one its resume delivers.
"""

from __future__ import annotations

import pathlib
import sys

from agent6.sessions import id, ipc
from agent6.sessions import layout as sessions_layout
from agent6.ui.cli import _common
from agent6.viewmodel import QuestionPrompt, pending_questions, session_is_live


def _print_pending(
    layout: sessions_layout.SessionLayout, prompts: tuple[QuestionPrompt, ...]
) -> None:
    """Print each pending question set and the command that answers it."""
    session_id = layout.session_id
    several = len(prompts) > 1
    for prompt in prompts:
        saved = ipc.question_answers_written(layout.session_dir, prompt.id)
        kind = "answer saved" if saved else prompt.kind()
        print(f"{session_id} {prompt.id} ({kind}): {len(prompt.questions)} question(s)")
        for i, q in enumerate(prompt.questions, start=1):
            print(f"  {i}. {q.question}")
            if q.options:
                print(f"     options: {', '.join(q.options)}")
        pick = f" --question {prompt.id}" if several else ""
        texts = " ".join(["TEXT"] * len(prompt.questions))
        print(f"  answer with: agent6 answer {session_id}{pick} {texts}")


def _select(prompts: tuple[QuestionPrompt, ...], question_id: str) -> QuestionPrompt | str:
    """Return the set to answer, or the refusal when the choice is missing or unknown."""
    if question_id:
        for prompt in prompts:
            if prompt.id == question_id:
                return prompt
        pending = ", ".join(p.id for p in prompts)
        return f"no pending question {question_id!r}; pending: {pending}"
    if len(prompts) > 1:
        return "several questions are pending; pick one with --question ID (a bare call lists them)"
    return next(iter(prompts))


def _write(
    layout: sessions_layout.SessionLayout, prompt: QuestionPrompt, answers: tuple[str, ...]
) -> int:
    """Write the answers to one set and say when the run reads them.

    Args:
        layout: The run.
        prompt: The set answered.
        answers: One answer per question, in order.

    Returns:
        The exit code: 0 when written or the set was printed, 2 on a refusal.
    """
    session_id = layout.session_id
    if len(answers) != len(prompt.questions):
        # Answers align to the questions by index; a short list would answer the wrong one.
        _print_pending(layout, (prompt,))
        if not answers:
            return 0
        print(
            f"\nERROR: {prompt.id} has {len(prompt.questions)} question(s);"
            f" {len(answers)} answer(s) given.",
            file=sys.stderr,
        )
        return 2
    if not ipc.write_question_answers(layout.session_dir, prompt.id, answers):
        _common.refuse(f"{session_id}: {ipc.ANSWERED_ELSEWHERE}.")
        return 2
    given = f"answered {session_id} {prompt.id}: {', '.join(answers)}"
    if not session_is_live(layout.session_dir):
        print(f"{given}; saved: `agent6 resume {session_id}` delivers it")
    elif prompt.blocking:
        print(given)
    else:
        print(f"{given}; the run reads it at its next turn")
    return 0


def _cmd_answer(target: str, answers: tuple[str, ...], question_id: str | None = None) -> int:
    """Answer one of the run's pending questions, or print them when no answer is given.

    Args:
        target: The session id or prefix.
        answers: One answer per question, in order.
        question_id: The set to answer; needed when several are pending.

    Returns:
        The exit code: 0 when answered or printed, 2 on a refusal.
    """
    try:
        layout = _common.resolve_session_layout(pathlib.Path.cwd(), target)
    except id.SessionIdError as exc:
        _common.error(f"{exc}")
        return 2
    session_id = layout.session_id
    prompts = pending_questions(layout.session_dir)
    if not prompts:
        _common.refuse(
            f"{session_id} has no pending question (an approval is answered by attaching:"
            f" agent6 attach {session_id})."
        )
        return 2
    if not answers and question_id is None:
        _print_pending(layout, prompts)
        return 0
    prompt = _select(prompts, question_id or "")
    if isinstance(prompt, str):
        _common.refuse(f"{session_id}: {prompt}.")
        return 2
    return _write(layout, prompt, answers)
