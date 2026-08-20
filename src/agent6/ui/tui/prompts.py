# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Prompt dispatch for a TUI view over a session's fold: one modal per
unanswered approval or question, its answer written to that session's file
bridge. Shared by the run dashboard and the machine watch view."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from textual.app import App

from agent6.sessions.ipc import write_answer, write_question_answers
from agent6.ui.tui.modals import ApprovalModal, QuestionModal
from agent6.viewmodel.state import SessionState


class PromptDispatcher:
    """Pops each pending prompt once, keyed by (session dir, prompt id): a
    session boundary restarts the id counters (`reset`), and a machine's next
    agent state has its own dir. An answer submitted once *answerable* turns
    false (the worker died mid-modal) is dropped with the *lost* warning
    instead of written to a file nobody polls."""

    def __init__(self, app: App[Any], *, answerable: Callable[[], bool], lost: str) -> None:
        self._app = app
        self._answerable = answerable
        self._lost = lost
        self._seen: set[str] = set()

    def reset(self) -> None:
        self._seen.clear()

    def dispatch(self, session_dir: Path, state: SessionState) -> None:
        for ap in state.pending_approvals:
            if not ap.answered and self._first(session_dir, ap.id):
                self._app.push_screen(
                    ApprovalModal(ap.id, ap.prompt, standing=ap.standing),
                    self._on_approval(session_dir, ap.id),
                )
        for qp in state.pending_questions:
            if not qp.answered and self._first(session_dir, qp.id):
                self._app.push_screen(
                    QuestionModal(qp.id, qp.questions, from_harness=qp.from_harness),
                    self._on_question(session_dir, qp.id),
                )

    def _first(self, session_dir: Path, prompt_id: str) -> bool:
        key = f"{session_dir}|{prompt_id}"
        if key in self._seen:
            return False
        self._seen.add(key)
        return True

    def _on_approval(self, session_dir: Path, prompt_id: str) -> Callable[[str | None], None]:
        def cb(answer: str | None) -> None:
            if not self._answerable():
                self._app.notify(self._lost, severity="warning", timeout=6.0)
                return
            write_answer(session_dir, prompt_id, answer or "no")

        return cb

    def _on_question(
        self, session_dir: Path, prompt_id: str
    ) -> Callable[[tuple[str, ...] | None], None]:
        def cb(answers: tuple[str, ...] | None) -> None:
            if not self._answerable():
                self._app.notify(self._lost, severity="warning", timeout=6.0)
                return
            write_question_answers(session_dir, prompt_id, answers or ())

        return cb
