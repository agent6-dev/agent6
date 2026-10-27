# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Read the questions a resumed execution carries from the journal."""

from __future__ import annotations

import dataclasses
import pathlib
import re

from agent6.tools import operator_prompts, schema
from agent6.viewmodel import QuestionPrompt, tail
from agent6.viewmodel import state as viewmodel_state


@dataclasses.dataclass(frozen=True, slots=True)
class CarriedQuestions:
    """What the journal's questions hand the next execution.

    Attributes:
        next_number: The number the next question id takes.
        pending: The questions nobody has answered, oldest first.
        late_answers: The answers taken at a boundary; the snapshot says which the model read.
    """

    next_number: int
    pending: tuple[operator_prompts.QuestionRequest, ...]
    late_answers: tuple[operator_prompts.LateAnswer, ...]


def carried_questions(logs_path: pathlib.Path) -> CarriedQuestions:
    """Read the journal's questions for the execution that resumes.

    A question agent6 asked outside an execution (the start gate) is left out: nothing
    delivers its answer.

    Args:
        logs_path: The session journal.

    Returns:
        The carried questions.
    """
    state = viewmodel_state.fold_session(tail.tail_events(logs_path, follow=False))
    asked = [q for q in state.pending_questions if not q.from_harness]
    numbers = [
        int(match.group(1)) for q in asked if (match := re.fullmatch(r"question-(\d+)", q.id))
    ]

    def questions(prompt: QuestionPrompt) -> tuple[schema.UserQuestion, ...]:
        return tuple(
            schema.UserQuestion(question=q.question, options=q.options) for q in prompt.questions
        )

    return CarriedQuestions(
        next_number=max(numbers, default=0) + 1,
        pending=tuple(
            operator_prompts.QuestionRequest(q.id, questions(q), None, wait=q.wait)
            for q in asked
            if not q.answered
        ),
        late_answers=tuple(
            operator_prompts.LateAnswer(q.id, questions(q), q.answers)
            for q in asked
            if q.answered and q.source == "late"
        ),
    )
