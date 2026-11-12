# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Put a child session's prompt to its parent's operator.

A parent is its child's front-end: it reads the child's open approval or question from the
child's journal, asks through its own prompts and writes the outcome where an attached
front-end would. A child runs with the away mode `wait`, so a chain relays itself: a parent
that is a child parks on the prompt it relays, and its own parent relays that. The operator
of the root answers every prompt.
"""

from __future__ import annotations

import pathlib
import threading

from agent6.sessions import ipc, layout
from agent6.tools import operator_prompts, schema
from agent6.viewmodel import state as session_fold
from agent6.viewmodel import state_types, tail


class PromptRelay:
    """Answer a child's open prompt through the parent's own prompts, one at a time.

    Args:
        prompts: The parent execution's prompts, which journal on the parent and honour its
            standing answers.
        parent_dir: The parent's session dir, where its standing answers live.
    """

    def __init__(self, prompts: operator_prompts.OperatorPrompts, parent_dir: pathlib.Path) -> None:
        self._prompts = prompts
        self._parent_dir = parent_dir
        # Lanes wait on their own threads; the parent's prompts take one caller.
        self._lock = threading.Lock()

    def __call__(self, child_dir: pathlib.Path, label: str) -> None:
        """Relay the prompt the child is paused on, if nobody answered it yet.

        A queued question pauses nothing and stays on the child's own row.

        Args:
            child_dir: The child's session dir.
            label: How the parent's operator reads the child, such as `lane 2 <session id>`.
        """
        with self._lock:
            state = session_fold.fold_session(
                tail.tail_events(child_dir / layout.LOGS_NAME, follow=False)
            )
            approval = session_fold.open_approval_of(state)
            if approval is not None and not ipc.answer_written(child_dir, approval.id):
                self._approval(child_dir, label, approval)
                return
            question = next((q for q in state.questions if q.blocking and not q.answered), None)
            if question is not None and not ipc.question_answers_written(child_dir, question.id):
                self._question(child_dir, label, question)

    def _approval(
        self, child_dir: pathlib.Path, label: str, prompt: state_types.ApprovalPrompt
    ) -> None:
        """Ask the parent's operator a child's approval and write the verdict down.

        A standing answer the parent holds for the prompt's scope goes down with the verdict,
        so the child stops asking.

        Args:
            child_dir: The child's session dir.
            label: The child, as the operator reads it.
            prompt: The child's open approval.
        """
        scope = prompt.scope
        if scope and ipc.session_deny_set(self._parent_dir, scope):
            ipc.write_answer(child_dir, prompt.id, "session-deny")
            return
        approved = self._prompts.approve(f"[{label}] {prompt.prompt}", scope=scope)
        if scope and approved and ipc.session_allow_set(self._parent_dir, scope):
            answer = "session"
        elif scope and not approved and ipc.session_deny_set(self._parent_dir, scope):
            answer = "session-deny"
        else:
            answer = "yes" if approved else "no"
        ipc.write_answer(child_dir, prompt.id, answer)

    def _question(
        self, child_dir: pathlib.Path, label: str, prompt: state_types.QuestionPrompt
    ) -> None:
        """Ask the parent's operator a child's questions and write the outcome down.

        Args:
            child_dir: The child's session dir.
            label: The child, as the operator reads it.
            prompt: The question prompt the child is paused on.
        """
        asked = tuple(
            schema.UserQuestion(question=_labelled(label, q.question), options=q.options)
            for q in prompt.questions
        )
        answer = self._prompts.ask(asked)
        if answer.status == "answered":
            ipc.write_question_answers(child_dir, prompt.id, answer.answers)
        else:
            ipc.write_question_unanswered(child_dir, prompt.id, answer.note)


def _labelled(label: str, question: str) -> str:
    """Return a child's question under its label, within the length a question may have."""
    text = f"[{label}] {question}"
    cap = schema.QUESTION_MAX_CHARS
    return text if len(text) <= cap else text[: cap - 1] + "\u2026"
