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

import dataclasses
import json
import os
import pathlib
import threading
import time
from collections.abc import Callable

from agent6 import budget as agent6_budget
from agent6.sessions import budget_events, ipc, journal, layout, process
from agent6.tools import operator_prompts, schema
from agent6.viewmodel.session import log as session_log
from agent6.viewmodel.session import records as state_types
from agent6.viewmodel.session import state as session_fold

# The least time between two budget rows the watch writes for running children.
ROW_INTERVAL_S = 10.0


@dataclasses.dataclass(frozen=True, slots=True)
class ChildStartup:
    """The spawned pid and the end of the child's complete journal rows before launch."""

    pid: int
    start_at: int


class _PromptJournal:
    """Fold complete rows for one worker, starting at the launch boundary."""

    def __init__(
        self, directory: pathlib.Path, worker: process.ProcessIdentity, start_at: int
    ) -> None:
        self.binding = directory, worker, start_at
        self.position = start_at
        self.seen: os.stat_result | None = None
        self.state = session_fold.initial_state()

    def pending(self) -> state_types.ApprovalPrompt | state_types.QuestionPrompt | None:
        """Return the same prompt object only while its journal slot stays open."""
        directory, _, start_at = self.binding
        try:
            with (directory / layout.LOGS_NAME).open("rb") as fh:
                # An unreadable or replaced file cannot confirm a cached slot. Stamp the open fd.
                stamp = os.fstat(fh.fileno())
                if self.seen is None or (
                    not os.path.samestat(stamp, self.seen)
                    or stamp.st_size < self.seen.st_size
                    or (
                        stamp.st_size == self.seen.st_size
                        and stamp.st_ctime_ns != self.seen.st_ctime_ns
                    )
                ):
                    self.position, self.state = start_at, session_fold.initial_state()
                self.seen = stamp
                fh.seek(self.position)
                while self.position < stamp.st_size:
                    line = fh.readline(stamp.st_size - self.position)
                    if not line.endswith(b"\n"):
                        break
                    self.position += len(line)
                    try:
                        event = json.loads(line.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        continue
                    if isinstance(event, dict):
                        self.state = session_fold.apply_event(self.state, event)
        except OSError:
            self.seen = None
            return None
        if self.state.finished:
            return None
        approval = session_fold.open_approval_of(self.state)
        if approval is not None and not ipc.answer_written(directory, approval.id):
            return approval
        question = next((q for q in self.state.questions if q.blocking and not q.answered), None)
        if question is not None and not ipc.question_answers_written(directory, question.id):
            return question
        return None


class _WatchThread(threading.local):
    """Drop the child's fold when its supervision thread exits."""

    journal: _PromptJournal | None = None


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
        self._thread = _WatchThread()
        self._row_lock = threading.Lock()
        self._journaled: tuple[float, int, float] = (0.0, 0, 0.0)
        self._journaled_at = 0.0

    def __call__(
        self,
        child_dir: pathlib.Path,
        label: str,
        status: str | ChildStartup,
        interrupted: Callable[[], bool] = operator_prompts.never_interrupted,
    ) -> bool:
        """Count the child's spend so far, and relay its prompt when it waits on one.

        Args:
            child_dir: The child's session dir, named for its session id.
            label: How the parent's operator reads the child, such as `attempt 2 <session id>`.
            status: The launch being watched, or the status after its start report.
            interrupted: Ends a relay on a stop or deadline.

        Returns:
            Whether the parent's budget is spent, which ends the child.
        """
        startup = status if isinstance(status, ChildStartup) else None
        worker = process.read_live_worker_identity(child_dir)
        owned = startup is None or (worker is not None and worker[0] == startup.pid)
        if self._budget is not None:
            if owned:
                self._budget.note_running(child_dir.name, running_usage(child_dir))
            over = self._budget.is_exhausted()
            self._journal_meter(self._budget, now=over)
            if over:
                return True
        if not owned or worker is None or (startup is None and status != "waiting"):
            return False
        # Retry next poll rather than block supervision behind a sibling's prompt.
        if not self._prompts.one_at_a_time.acquire(blocking=False):
            return False
        try:
            start_at = startup.start_at if startup is not None else 0
            reader = self._thread.journal
            if reader is None or reader.binding != (child_dir, worker, start_at):
                reader = self._thread.journal = _PromptJournal(child_dir, worker, start_at)
            prompt = reader.pending()
            if prompt is None:
                return False

            ended = False

            def cancelled() -> bool:
                nonlocal ended
                ended = (
                    ended
                    or interrupted()
                    or ipc.stop_request_pending(child_dir)
                    or (self._budget is not None and self._budget.is_exhausted())
                    or reader.pending() is not prompt
                    or process.read_live_worker_identity(child_dir) != worker
                )
                return ended

            if not cancelled():
                if isinstance(prompt, state_types.ApprovalPrompt):
                    self._approval(child_dir, label, prompt, cancelled)
                else:
                    self._question(child_dir, label, prompt, cancelled)
        finally:
            self._prompts.one_at_a_time.release()
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
        moved = (snap.usd_running_children, snap.tokens_running_children, snap.plan_observed_points)
        at = time.monotonic()
        with self._row_lock:
            if moved == self._journaled or (not now and at - self._journaled_at < ROW_INTERVAL_S):
                return
            self._journaled, self._journaled_at = moved, at
        self._emit(journal.BudgetUpdate(**budget_events.budget_update_fields(budget)))

    def _approval(
        self,
        child_dir: pathlib.Path,
        label: str,
        prompt: state_types.ApprovalPrompt,
        cancelled: Callable[[], bool],
    ) -> None:
        """Ask the parent's operator a child's approval and write the verdict down.

        A standing answer the parent holds for the prompt's scope goes down with the verdict,
        so the child stops asking.

        Args:
            child_dir: The child's session dir.
            label: The child, as the operator reads it.
            cancelled: Ends the relay without writing to the child.
            prompt: The child's open approval.
        """
        scope = prompt.scope
        if scope and ipc.session_deny_set(self._parent_dir, scope):
            answer = "session-deny"
        else:
            approved = self._prompts.approve(
                f"[{label}] {prompt.prompt}", scope=scope, interrupted=cancelled
            )
            if scope and approved and ipc.session_allow_set(self._parent_dir, scope):
                answer = "session"
            elif scope and not approved and ipc.session_deny_set(self._parent_dir, scope):
                answer = "session-deny"
            else:
                answer = "yes" if approved else "no"
        if not cancelled():
            ipc.write_answer(child_dir, prompt.id, answer)

    def _question(
        self,
        child_dir: pathlib.Path,
        label: str,
        prompt: state_types.QuestionPrompt,
        cancelled: Callable[[], bool],
    ) -> None:
        """Ask the parent's operator a child's questions and write the outcome down.

        Args:
            child_dir: The child's session dir.
            label: The child, as the operator reads it.
            cancelled: Ends the relay without writing to the child.
            prompt: The question prompt the child is paused on.
        """
        asked = tuple(
            schema.UserQuestion(question=_labelled(label, q.question), options=q.options)
            for q in prompt.questions
        )
        answer = self._prompts.ask(asked, interrupted=cancelled)
        if cancelled():
            return
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
        tokens=usage.session.unmetered_tokens + usage.execution.tokens_running_children,
        plan_readings=readings,
    )


def _labelled(label: str, question: str) -> str:
    """Return a child's question under its label, within the length a question may have."""
    text = f"[{label}] {question}"
    cap = schema.QUESTION_MAX_CHARS
    return text if len(text) <= cap else text[: cap - 1] + "\u2026"
