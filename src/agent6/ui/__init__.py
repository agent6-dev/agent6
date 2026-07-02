# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""UI subtree for agent6, read-only viewers over the JSONL event stream.

Everything in this package is **optional, side-effect-free, and consumes
`<run-dir>/logs.jsonl` from disk**. Nothing in here is part of the
core agent loop; reviewers can skip this directory and still understand
how agent6 actually plans and edits code.

The render-ready state and the JSONL tailer live in `agent6.viewmodel` (shared
with the CLI and any future web client); this package is the textual painting of
that state.

Layout:
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

__all__ = [
    "APPROVAL_DIR_NAME",
    "write_answer",
]
