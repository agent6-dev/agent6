# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Prompt dispatch for a TUI view over a session's fold.

One modal per unanswered question the run waits on (approvals dock as a row
instead), its answer written to the session's file bridge, and the claim every
surface takes a prompt through. A queued question is announced once and opened
on request, so no modal stacks over a run that went on. A question another
front-end answered closes its modal, which writes nothing. Shared by the run
views and the machine watch view.
"""

from __future__ import annotations

import pathlib
from collections.abc import Callable
from typing import Any

from textual import app as textual_app

from agent6.sessions import ipc
from agent6.ui.tui import modals
from agent6.viewmodel import SessionState


class PromptDispatcher:
    """Pop each pending prompt once, keyed by session dir and prompt id.

    A session boundary restarts the id counters (`reset`); a machine's next agent
    state has its own dir. An answer submitted once `answerable` turns false (the
    worker died mid-modal) is dropped with the `lost` warning instead of written
    to a file nobody polls.
    """

    def __init__(
        self, app: textual_app.App[Any], *, answerable: Callable[[], bool], lost: str
    ) -> None:
        """Bind the dispatcher to the app, its liveness check and the lost-answer text."""
        self._app = app
        self._answerable = answerable
        self._lost = lost
        self._seen: set[str] = set()
        # The modals still waiting on the operator, by claim key.
        self._open: dict[str, modals.QuestionModal] = {}

    def reset(self) -> None:
        """Forget every claimed prompt, at a session boundary."""
        self._seen.clear()

    def dispatch(self, session_dir: pathlib.Path, state: SessionState) -> None:
        """Push a modal for each waiting, unclaimed question; announce a queued one once.

        A question answered elsewhere closes its modal. A question a dead run left
        waiting is a queued one: its answer waits for the resume.

        Args:
            session_dir: The session the answers are written to.
            state: That session's fold.
        """
        for qp in state.pending_questions:
            key = f"{session_dir}|{qp.id}"
            if qp.answered:
                # The journal holds an answer this modal did not write: it takes none now.
                closed = self._open.pop(key, None)
                if closed is not None:
                    closed.close_answered_elsewhere(on_top=self._app.screen is closed)
            elif not qp.blocking or not self._answerable():
                if self.claim(session_dir, qp.id):
                    first = qp.questions[0].question if qp.questions else ""
                    self._app.notify(
                        f"{qp.id} {qp.kind()}: {first}\nRun menu: Answer a question", timeout=8.0
                    )
            elif self.claim(session_dir, qp.id):
                modal = modals.QuestionModal(qp.id, qp.questions, from_harness=qp.from_harness)
                self._open[key] = modal
                self._app.push_screen(modal, self._on_question(session_dir, qp.id))

    def open_queued(self, session_dir: pathlib.Path, state: SessionState) -> bool:
        """Push the modal for the oldest queued question not already open.

        Args:
            session_dir: The session the answer is written to.
            state: That session's fold.

        Returns:
            Whether a modal opened; False when no queued question is pending.
        """
        live = self._answerable()
        for qp in state.pending_questions:
            key = f"{session_dir}|{qp.id}"
            if qp.answered or (qp.blocking and live) or key in self._open:
                continue
            if ipc.question_answers_written(session_dir, qp.id):
                continue  # answered here already; the run reads the file at its next turn
            modal = modals.QuestionModal(
                qp.id, qp.questions, from_harness=qp.from_harness, queued=True
            )
            self._open[key] = modal
            self._app.push_screen(modal, self._on_question(session_dir, qp.id, queued=True))
            return True
        return False

    def claim(self, session_dir: pathlib.Path, prompt_id: str) -> bool:
        """Claim a prompt for one surface.

        Every surface asks here, so a prompt answered on one screen never reopens
        on another before its answer event folds.

        Args:
            session_dir: The prompt's session.
            prompt_id: The prompt's id within it.

        Returns:
            True the first time, False once claimed.
        """
        key = f"{session_dir}|{prompt_id}"
        if key in self._seen:
            return False
        self._seen.add(key)
        return True

    def seen(self, session_dir: pathlib.Path, prompt_id: str) -> bool:
        """Return whether the prompt was claimed."""
        return f"{session_dir}|{prompt_id}" in self._seen

    def _on_question(
        self, session_dir: pathlib.Path, prompt_id: str, *, queued: bool = False
    ) -> Callable[[tuple[str, ...] | None], None]:
        def cb(answers: tuple[str, ...] | None) -> None:
            if self._open.pop(f"{session_dir}|{prompt_id}", None) is None:
                self._app.notify(ipc.ANSWERED_ELSEWHERE, severity="warning", timeout=6.0)
                return
            if queued and answers is None:
                return  # Esc: the question stays pending
            if not queued and not self._answerable():
                self._app.notify(self._lost, severity="warning", timeout=6.0)
                return
            if not ipc.write_question_answers(session_dir, prompt_id, answers or ()):
                self._app.notify(ipc.ANSWERED_ELSEWHERE, severity="warning", timeout=6.0)
                return
            if queued:
                self._app.notify(f"{prompt_id} answered; the run reads it at its next turn")

        return cb
