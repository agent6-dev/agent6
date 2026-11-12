# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Put a child session's approval to its parent's operator.

A parent is its child's front-end: it reads the child's open approval from the child's journal,
asks through its own prompts and writes the answer where an attached front-end would. A child
runs with the away mode `wait`, so a chain relays itself: a parent that is a child parks on the
prompt it relays, and its own parent relays that.
"""

from __future__ import annotations

import pathlib
import threading

from agent6.sessions import ipc, layout
from agent6.tools import operator_prompts
from agent6.viewmodel import state as session_fold
from agent6.viewmodel import tail


class ApprovalRelay:
    """Answer a child's open approval through the parent's own prompts, one at a time.

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
        """Relay the child's open approval, if it has one nobody answered yet.

        A standing answer the parent holds for the prompt's scope is handed down with the
        verdict, so the child stops asking.

        Args:
            child_dir: The child's session dir.
            label: How the parent's operator reads the child, such as `lane 2 <session id>`.
        """
        with self._lock:
            state = session_fold.fold_session(
                tail.tail_events(child_dir / layout.LOGS_NAME, follow=False)
            )
            prompt = session_fold.open_approval_of(state)
            if prompt is None or ipc.answer_written(child_dir, prompt.id):
                return
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
