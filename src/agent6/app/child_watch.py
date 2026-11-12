# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""What a parent does for a child session while it waits on it.

A parent is its child's front-end: it reads the child's open approval or question from the
child's journal, asks through its own prompts and writes the outcome where an attached
front-end would. A child runs with the away mode `wait`, so a chain relays itself: a parent
that is a child parks on the prompt it relays, and its own parent relays that. The operator
of the root answers every prompt.

A parent also counts its child's spend as it grows: the child's journal carries its usage
so far, its own running children included, and the parent's tracker bounds the sum.
"""

from __future__ import annotations

import pathlib
import threading
import time
from collections.abc import Callable

from agent6 import budget as agent6_budget
from agent6.sessions import budget_events, ipc, journal, layout
from agent6.tools import operator_prompts, schema
from agent6.viewmodel import state as session_fold
from agent6.viewmodel import state_types, tail
from agent6.viewmodel.session import log as session_log

# The least time between two budget rows the watch writes for running lanes.
ROW_INTERVAL_S = 10.0


class ChildWatch:
    """Answer a running child's prompts and count its spend, on each poll of its wait.

    Args:
        prompts: The parent execution's prompts, which journal on the parent and honour its
            standing answers.
        parent_dir: The parent's session dir, where its standing answers live.
        budget: The parent execution's tracker; None counts nothing.
        emit: Journals on the parent; with it the meter is journaled as the running children's
            spend moves, so the parent's own parent reads the whole subtree.
    """

    def __init__(
        self,
        prompts: operator_prompts.OperatorPrompts,
        parent_dir: pathlib.Path,
        budget: agent6_budget.BudgetTracker | None = None,
        emit: Callable[[journal.Event], None] | None = None,
    ) -> None:
        self._prompts = prompts
        self._parent_dir = parent_dir
        self._budget = budget
        self._emit = emit
        # Lanes wait on their own threads; the parent's prompts take one caller.
        self._lock = threading.Lock()
        self._row_lock = threading.Lock()
        self._journaled: tuple[float, int, float] = (0.0, 0, 0.0)
        self._journaled_at = 0.0

    def __call__(self, child_dir: pathlib.Path, label: str, status: str) -> bool:
        """Count the child's spend so far, and relay its prompt when it waits on one.

        Args:
            child_dir: The child's session dir, named for its session id.
            label: How the parent's operator reads the child, such as `lane 2 <session id>`.
            status: The child's status word at this poll.

        Returns:
            Whether the parent's budget is spent, which ends the child.
        """
        if self._budget is not None:
            self._budget.note_running(child_dir.name, running_usage(child_dir))
            over = self._budget.is_exhausted()
            self._journal_meter(self._budget, now=over)
            if over:
                return True
        if status == "waiting":
            self._relay(child_dir, label)
        return False

    def _journal_meter(self, budget: agent6_budget.BudgetTracker, *, now: bool) -> None:
        """Journal the meter when the running children moved it, `ROW_INTERVAL_S` apart.

        Args:
            budget: The parent's tracker.
            now: Whether to skip the interval, as when a cap was reached.
        """
        if self._emit is None:
            return
        snap = budget.snapshot()
        moved = (snap.usd_running_lanes, snap.tokens_running_lanes, snap.plan_observed_points)
        at = time.monotonic()
        with self._row_lock:
            if moved == self._journaled or (not now and at - self._journaled_at < ROW_INTERVAL_S):
                return
            self._journaled, self._journaled_at = moved, at
        self._emit(journal.BudgetUpdate(**budget_events.budget_update_fields(budget)))

    def _relay(self, child_dir: pathlib.Path, label: str) -> None:
        """Relay the prompt the child is paused on, if nobody answered it yet.

        A queued question pauses nothing and stays on the child's own row.

        Args:
            child_dir: The child's session dir.
            label: The child, as the operator reads it.
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


def running_usage(child_dir: pathlib.Path) -> agent6_budget.UsageSettlement:
    """Return what a running child has used so far, from its journal.

    Args:
        child_dir: The child's session dir.

    Returns:
        Its spend across its executions with its own running children's, its fallback
        tokens the same way, and its account's latest plan reading.
    """
    usage = session_log.scan_session_log(child_dir / layout.LOGS_NAME).budget
    account = usage.account
    readings: tuple[tuple[str, agent6_budget.PlanUsage], ...] = ()
    if account is not None and account.provider and account.windows:
        readings = ((account.provider, agent6_budget.PlanUsage(windows=account.windows)),)
    return agent6_budget.UsageSettlement(
        usd=usage.usd_so_far,
        usd_partial=usage.session.usd_partial,
        tokens=usage.session.unmetered_tokens + usage.execution.tokens_running_lanes,
        plan_readings=readings,
    )


def _labelled(label: str, question: str) -> str:
    """Return a child's question under its label, within the length a question may have."""
    text = f"[{label}] {question}"
    cap = schema.QUESTION_MAX_CHARS
    return text if len(text) <= cap else text[: cap - 1] + "\u2026"
