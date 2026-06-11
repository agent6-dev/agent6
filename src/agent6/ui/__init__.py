# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""UI subtree for agent6, read-only viewers over the JSONL event stream.

Everything in this package is **optional, side-effect-free, and consumes
`.agent6/runs/<id>/logs.jsonl` from disk**. Nothing in here is part of the
core agent loop; reviewers can skip this directory and still understand
how agent6 actually plans and edits code.

Layout:
    state.py     pure event-fold: list[event] -> RunState. The data
                 contract a VS Code extension (or any other viewer) can
                 mirror in its own language.
    tail.py      stdlib JSONL file tailer.
    approval.py  file-based bridges (approve / ask_user / steer) workflow<->TUI.
    modals.py    textual modal screens (approve / steer / question).
    app.py       the run dashboard (Agent6TUI + run_tui).
    home.py      the `agent6 tui` hub: list runs + launch run/plan/ask.

Everything is launched out-of-process and only reads `logs.jsonl` + writes the
small answer files the workflow polls, so the core loop is untouched and any
other front-end (VS Code, web, desktop) can mirror the same file contract.
"""

from __future__ import annotations

from agent6.ui.approval import (
    APPROVAL_DIR_NAME,
    write_answer,
)
from agent6.ui.state import RunState, TaskNodeView, apply_event, initial_state
from agent6.ui.tail import tail_events

__all__ = [
    "APPROVAL_DIR_NAME",
    "RunState",
    "TaskNodeView",
    "apply_event",
    "initial_state",
    "tail_events",
    "write_answer",
]
