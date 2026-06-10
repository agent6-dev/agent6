# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""UI subtree for agent6 — read-only viewers over the JSONL event stream.

Everything in this package is **optional, side-effect-free, and consumes
`.agent6/runs/<id>/logs.jsonl` from disk**. Nothing in here is part of the
core agent loop; reviewers can skip this directory and still understand
how agent6 actually plans and edits code.

Layout:
    state.py     pure event-fold: list[event] -> RunState. The data
                 contract a VS Code extension (or any other viewer) can
                 mirror in its own language.
    tail.py      stdlib JSONL file tailer.
    approval.py  file-based approval bridge (workflow <-> TUI).
    tui.py       Textual app. `textual` ships in the base install;
                 importing this module raises a clear error if it's missing.

The TUI is launched out-of-process by `agent6 run` (when stdout is a TTY
and textual is installed) so the workflow process stays exactly as it is
today — the TUI only reads files. `agent6 watch <run-id>` is a
standalone read-only viewer.
"""

from __future__ import annotations

from agent6.ui.approval import (
    APPROVAL_DIR_NAME,
    write_answer,
)
from agent6.ui.state import RunState, StepView, apply_event, initial_state
from agent6.ui.tail import tail_events

__all__ = [
    "APPROVAL_DIR_NAME",
    "RunState",
    "StepView",
    "apply_event",
    "initial_state",
    "tail_events",
    "write_answer",
]
