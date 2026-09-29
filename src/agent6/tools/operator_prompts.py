# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The one gate every prompt to the operator goes through.

`OperatorPrompts` mints the prompt ids, clears a prompt's answer slot, journals the prompt
before asking and the answer after, and names the tool call a prompt gates. A front-end
supplies only the two callables that answer (`Approver`, `Questioner`) and say which source
answered; it never journals.
"""

from __future__ import annotations

import dataclasses
import itertools
import pathlib
import sys
from typing import Any, Literal, Protocol

from agent6.sessions import ipc
from agent6.tools import schema

# Who answered, as the answer events journal it: the CLI's terminal ("stdin"), a live TUI, web
# or attach ("frontend"), a park until one attached ("await-frontend"), the detach choice
# ("away-deny", approvals only; "away-wait", questions only), no terminal and no front-end
# ("headless-default", empty answers), a standing grant ("session"), a machine state with nobody
# to ask ("headless"), or an editor over ACP ("acp").
Source = Literal[
    "stdin",
    "frontend",
    "await-frontend",
    "away-deny",
    "away-wait",
    "headless-default",
    "session",
    "headless",
    "acp",
]

UNANSWERED_NOTE = (
    "no operator was attached to answer these questions; decide on your own judgement and go on"
)


@dataclasses.dataclass(frozen=True, slots=True)
class ApprovalRequest:
    """One approval put to the operator, as journaled in `approval.prompt`.

    Attributes:
        id: The answer slot's key; a file-bridge front-end answers into `approvals/<id>.answer`.
        prompt: The line shown.
        scope: What a standing answer grants, "command" or "mcp.<server>"; None offers none, so
            the operator is asked every time (`fetch`). One scope's grant is never consent for
            another.
        call_id: The dispatched tool call the prompt gates; None for a verify the harness runs.
    """

    id: str
    prompt: str
    scope: str | None
    call_id: int | None


@dataclasses.dataclass(frozen=True, slots=True)
class QuestionRequest:
    """One `ask_user` (or a pre-run question), as journaled in `question.prompt`.

    Attributes:
        id: The answer slot's key.
        questions: The questions put.
        call_id: The dispatched tool call the prompt gates; None for a pre-run question.
    """

    id: str
    questions: tuple[schema.UserQuestion, ...]
    call_id: int | None


@dataclasses.dataclass(frozen=True, slots=True)
class ApprovalAnswer:
    """One approval's answer and who gave it."""

    approved: bool
    source: Source


@dataclasses.dataclass(frozen=True, slots=True)
class QuestionAnswer:
    """One question set's answers and who gave them.

    Attributes:
        answers: Aligned to the request's questions by index; a shorter tuple leaves the rest "".
        source: Who answered.
        unseen: Nobody could see the questions (no terminal and no front-end, or a park that
            ended empty); a person who left them blank is not unseen.
    """

    answers: tuple[str, ...]
    source: Source
    unseen: bool = False


class Approver(Protocol):
    """Answer one approval and say who answered."""

    def __call__(self, request: ApprovalRequest, /) -> ApprovalAnswer:
        """Answer the request."""
        ...


class Questioner(Protocol):
    """Answer one question set and say who answered."""

    def __call__(self, request: QuestionRequest, /) -> QuestionAnswer:
        """Answer the request."""
        ...


class Journal(Protocol):
    """Where the gate's events go: a run's `EventSink.emit`."""

    def __call__(self, event_type: str, /, **fields: Any) -> None:
        """Write one event."""
        ...


def unjournaled(event_type: str, /, **fields: Any) -> None:
    """Write nothing: the journal of a gate built with no sink."""


def _default_approver(request: ApprovalRequest, /) -> ApprovalAnswer:  # pragma: no cover
    try:
        ans = input(f"{request.prompt} [y/N] ").strip().lower()
    except EOFError:
        return ApprovalAnswer(False, "stdin")
    return ApprovalAnswer(ans in {"y", "yes"}, "stdin")


def _default_questioner(request: QuestionRequest, /) -> QuestionAnswer:  # pragma: no cover
    """Ask on stdin, one numbered prompt per question, when no front-end is wired.

    Returns:
        The answers; a stdin that is not a terminal answers "" for each so a run never hangs.
    """
    if not sys.stdin.isatty():
        return QuestionAnswer(tuple("" for _ in request.questions), "headless-default", unseen=True)
    answers: list[str] = []
    for q in request.questions:
        lines = [q.question, *(f"  {i}) {opt}" for i, opt in enumerate(q.options, start=1))]
        try:
            ans = input("\n".join(lines) + "\n> ").strip()
        except EOFError:
            ans = ""
        if ans.isdigit() and 1 <= int(ans) <= len(q.options):
            ans = q.options[int(ans) - 1]
        answers.append(ans)
    return QuestionAnswer(tuple(answers), "stdin")


class OperatorPrompts:
    """Every prompt to the operator for one execution, journaled once.

    Args:
        approver: Answers approvals; stdin when None.
        questioner: Answers questions; stdin when None.
        journal: Takes every event the gate writes (a run's `EventSink.emit`).
        session_dir: Where the file bridge lives (`sessions.ipc`: the answer slots and the
            operator's standing choices); None keeps no bridge.
    """

    def __init__(
        self,
        *,
        approver: Approver | None = None,
        questioner: Questioner | None = None,
        journal: Journal = unjournaled,
        session_dir: pathlib.Path | None = None,
    ) -> None:
        self._approver: Approver = approver or _default_approver
        self._questioner: Questioner = questioner or _default_questioner
        self._journal = journal
        self._session_dir = session_dir
        self._approvals = itertools.count(1)
        self._questions = itertools.count(1)

    def approve(self, prompt: str, *, scope: str | None = None, call_id: int | None = None) -> bool:
        """Ask, unless a standing grant for the scope already answers.

        The answer slot is cleared before the prompt is journaled: ids are predictable
        counters, so an answer written ahead of its prompt must never be the one consumed.
        The prompt's `standing` tells a front-end whether to offer an "allow all".

        Args:
            prompt: The line shown.
            scope: What a standing answer grants; None offers none.
            call_id: The tool call the prompt gates.

        Returns:
            Whether the call was approved.
        """
        request = ApprovalRequest(
            id=f"approval-{next(self._approvals)}", prompt=prompt, scope=scope, call_id=call_id
        )
        if (
            scope
            and self._session_dir is not None
            and ipc.session_allow_set(self._session_dir, scope)
        ):
            self._journal("approval.answer", id=request.id, approved=True, source="session")
            return True
        if self._session_dir is not None:
            ipc.clear_answer(self._session_dir, request.id)
        self._journal(
            "approval.prompt",
            id=request.id,
            prompt=prompt,
            standing=bool(scope),
            call_id=call_id,
        )
        answer = self._approver(request)
        self._journal(
            "approval.answer", id=request.id, approved=answer.approved, source=answer.source
        )
        return answer.approved

    def ask(
        self, questions: tuple[schema.UserQuestion, ...], *, call_id: int | None = None
    ) -> QuestionAnswer:
        """Put questions to the operator.

        Args:
            questions: The questions.
            call_id: The tool call the prompt gates.

        Returns:
            The answers aligned to the questions by index ("" for one left unanswered), who
            answered, and whether anyone saw the questions.

        Raises:
            ValueError: The questioner returned more answers than questions.
        """
        request = QuestionRequest(
            id=f"question-{next(self._questions)}", questions=questions, call_id=call_id
        )
        if self._session_dir is not None:
            ipc.clear_question_answers(self._session_dir, request.id)
        self._journal(
            "question.prompt",
            id=request.id,
            questions=[{"question": q.question, "options": list(q.options)} for q in questions],
            call_id=call_id,
        )
        answer = self._questioner(request)
        if len(answer.answers) > len(questions):
            raise ValueError(
                f"{answer.source} answered {len(questions)} questions"
                f" with {len(answer.answers)} answers"
            )
        answers = answer.answers + ("",) * (len(questions) - len(answer.answers))
        self._journal(
            "question.answer",
            id=request.id,
            answers=list(answers),
            source=answer.source,
            unseen=answer.unseen,
        )
        return QuestionAnswer(answers, answer.source, answer.unseen)


def unanswered_note(answer: QuestionAnswer) -> str:
    """Return the note the model reads when nobody saw its questions, else ""."""
    return UNANSWERED_NOTE if answer.unseen else ""
