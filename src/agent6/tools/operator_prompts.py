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

from agent6 import portable
from agent6.sessions import ipc, process
from agent6.tools import schema

# Who answered, as the answer events journal it: the CLI's terminal ("stdin"), a live TUI, web
# or attach ("frontend"), a park until one attached ("await-frontend"), the detach choice
# ("away-deny"), no terminal and no front-end ("headless-default", empty answers), a standing
# grant ("session"), a machine state with nobody to ask ("headless"), an editor over ACP ("acp"),
# a queued question nobody has answered yet ("queued") or its answer taken at a later execution
# boundary ("late").
Source = Literal[
    "stdin",
    "frontend",
    "await-frontend",
    "away-deny",
    "headless-default",
    "session",
    "headless",
    "acp",
    "queued",
    "late",
]
# `answered`: the answers are in; `pending`: queued, they arrive later; `unanswered`: nobody could
# answer and nothing is queued.
QuestionStatus = Literal["answered", "pending", "unanswered"]


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
        wait: Whether the call waits for the answers; a queued prompt is answered later.
    """

    id: str
    questions: tuple[schema.UserQuestion, ...]
    call_id: int | None
    wait: bool = True


@dataclasses.dataclass(frozen=True, slots=True)
class ApprovalAnswer:
    """One approval's answer and who gave it."""

    approved: bool
    source: Source


@dataclasses.dataclass(frozen=True, slots=True)
class QuestionAnswer:
    """One question set's outcome: the answers, who gave them, and the state the call met.

    Attributes:
        answers: Aligned to the request's questions by index; a shorter tuple leaves the rest "";
            empty while pending.
        source: Who answered.
        status: `answered`, `pending` or `unanswered`; a person who left blanks answered.
        note: The state of the world the model reads when the status is not `answered`: the
            front-end claim, the away mode, whether the questions are queued.
        id: The prompt's id, set by the gate.
    """

    answers: tuple[str, ...]
    source: Source
    status: QuestionStatus = "answered"
    note: str = ""
    id: str = ""


@dataclasses.dataclass(frozen=True, slots=True)
class LateAnswer:
    """The answers to a queued question, taken at an execution boundary."""

    id: str
    questions: tuple[schema.UserQuestion, ...]
    answers: tuple[str, ...]


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

    Args:
        request: The questions to ask.

    Returns:
        The answers; a stdin that is not a terminal answers "" for each so a run never hangs.
    """
    if not sys.stdin.isatty():
        return QuestionAnswer(
            tuple("" for _ in request.questions),
            "headless-default",
            status="unanswered",
            note="stdin is not a terminal: no answer, not queued",
        )
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


def presence_note(session_dir: pathlib.Path) -> str:
    """Return the operator's presence as the model reads it.

    The front-end claim, else the terminal the run was started from, else the away mode.

    Args:
        session_dir: The session directory.

    Returns:
        "a front-end is attached", "a terminal is attached", or "no front-end or terminal
        attached; away mode <mode or unset>".
    """
    if process.frontend_is_live(session_dir):
        return "a front-end is attached"
    if portable.has_controlling_tty():
        return "a terminal is attached"
    return f"no front-end or terminal attached; away mode {ipc.away_mode(session_dir) or 'unset'}"


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
        first_question: int = 1,
        pending: tuple[QuestionRequest, ...] = (),
        undelivered: tuple[LateAnswer, ...] = (),
    ) -> None:
        self._approver: Approver = approver or _default_approver
        self._questioner: Questioner = questioner or _default_questioner
        self._journal = journal
        self._session_dir = session_dir
        self._approvals = itertools.count(1)
        # Question ids run on across executions, so an answer names one prompt in the journal.
        self._questions = itertools.count(first_question)
        self._pending: dict[str, QuestionRequest] = {request.id: request for request in pending}
        # Answers an earlier execution journaled and never read: the first collection has them.
        self._undelivered = list(undelivered)

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
        self,
        questions: tuple[schema.UserQuestion, ...],
        *,
        call_id: int | None = None,
        wait: bool = True,
    ) -> QuestionAnswer:
        """Put questions to the operator.

        A waiting call returns with the questioner's outcome. A queued call journals the prompt
        and returns pending at once; `collect_answers` takes the answers when they land.

        Args:
            questions: The questions.
            call_id: The tool call the prompt gates.
            wait: Whether the call waits for the answers.

        Returns:
            The answers aligned to the questions by index ("" for one left unanswered), who
            answered, the status and the state the call met.

        Raises:
            ValueError: The questioner returned more answers than questions.
        """
        request = QuestionRequest(
            id=f"question-{next(self._questions)}", questions=questions, call_id=call_id, wait=wait
        )
        if self._session_dir is not None:
            ipc.clear_question_answers(self._session_dir, request.id)
        self._journal(
            "question.prompt",
            id=request.id,
            questions=[{"question": q.question, "options": list(q.options)} for q in questions],
            call_id=call_id,
            wait=wait,
        )
        if not wait:
            self._pending[request.id] = request
            presence = (
                presence_note(self._session_dir)
                if self._session_dir is not None
                else "no front-end or terminal attached; away mode unset"
            )
            return QuestionAnswer(
                (),
                "queued",
                status="pending",
                note=f"{presence}; queued: the answers arrive as a message at a later turn,"
                " or on resume",
                id=request.id,
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
            unseen=answer.status == "unanswered",
        )
        return dataclasses.replace(answer, answers=answers, id=request.id)

    def pending(self) -> tuple[QuestionRequest, ...]:
        """Return the queued questions nobody has answered yet, oldest first."""
        return tuple(self._pending.values())

    def presence(self) -> str:
        """Return the operator's presence (`presence_note`), "" without a session dir."""
        return presence_note(self._session_dir) if self._session_dir is not None else ""

    def collect_answers(self) -> tuple[LateAnswer, ...]:
        """Take the answers that landed for queued questions, journaling each as `late`.

        Returns:
            The answered prompts, oldest first; each answer file is consumed once.
        """
        taken, self._undelivered = self._undelivered, []
        if self._session_dir is None:
            return tuple(taken)
        for request in list(self._pending.values()):
            answers = ipc.peek_question_answers(self._session_dir, request.id)
            if answers is None:
                continue
            aligned = (answers + ("",) * len(request.questions))[: len(request.questions)]
            # The journal row first: a crash after it leaves a file the next sweep drops.
            self._journal(
                "question.answer",
                id=request.id,
                answers=list(aligned),
                source="late",
                unseen=False,
            )
            ipc.clear_question_answers(self._session_dir, request.id)
            del self._pending[request.id]
            taken.append(LateAnswer(request.id, request.questions, aligned))
        return tuple(taken)
