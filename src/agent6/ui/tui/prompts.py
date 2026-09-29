# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Prompt dispatch for a TUI view over a session's fold.

One modal per unanswered question (approvals dock as a row instead), its answer
written to the session's file bridge, and the claim every surface takes a prompt
through. Shared by the run views and the machine watch view.
"""

from __future__ import annotations

import pathlib
from collections.abc import Callable
from typing import Any

from textual import app as textual_app

from agent6.sessions import ipc
from agent6.ui.tui import modals
from agent6.viewmodel import state as viewmodel_state


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

    def reset(self) -> None:
        """Forget every claimed prompt, at a session boundary."""
        self._seen.clear()

    def dispatch(self, session_dir: pathlib.Path, state: viewmodel_state.SessionState) -> None:
        """Push a modal for each unanswered, unclaimed question in the state."""
        for qp in state.pending_questions:
            if not qp.answered and self.claim(session_dir, qp.id):
                self._app.push_screen(
                    modals.QuestionModal(qp.id, qp.questions, from_harness=qp.from_harness),
                    self._on_question(session_dir, qp.id),
                )

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
        self, session_dir: pathlib.Path, prompt_id: str
    ) -> Callable[[tuple[str, ...] | None], None]:
        def cb(answers: tuple[str, ...] | None) -> None:
            if not self._answerable():
                self._app.notify(self._lost, severity="warning", timeout=6.0)
                return
            if not ipc.write_question_answers(session_dir, prompt_id, answers or ()):
                self._app.notify(ipc.ANSWERED_ELSEWHERE, severity="warning", timeout=6.0)

        return cb
