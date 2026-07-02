# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The shared view-model: the JSONL event stream folded into render-ready state.

This is the data contract every front-end consumes. The CLI, the textual TUI,
and (later) the web UI all read the same `<run-dir>/logs.jsonl`, fold it through
the same pure functions here, and only differ in how they paint the result.
Keeping the fold in one place is what stops the front-ends from drifting.

Layout:
    state.py          pure event-fold: list[event] -> RunState (a run / agent state).
    machine_state.py  pure fold: a machine journal -> MachineState (the watch view).
    tail.py           stdlib JSONL file tailer (the event source).

No I/O in the folds, no textual, no async: just frozen dataclasses and pure
functions, so a viewer in any language (a VS Code extension, a web client) can
mirror `RunState` / `MachineState` field-for-field.
"""

from __future__ import annotations

from agent6.viewmodel.machine_state import (
    MachineEndView,
    MachineState,
    MachineStateView,
    TransitionView,
    fold_machine,
    newest_state_log,
)
from agent6.viewmodel.state import (
    MAX_LOG_TAIL,
    ApprovalPrompt,
    BudgetView,
    QuestionPrompt,
    RoleCall,
    RunState,
    TaskNodeView,
    ToolCallView,
    VerifyView,
    apply_event,
    format_log_line,
    initial_state,
)
from agent6.viewmodel.tail import tail_events

__all__ = [
    "MAX_LOG_TAIL",
    "ApprovalPrompt",
    "BudgetView",
    "MachineEndView",
    "MachineState",
    "MachineStateView",
    "QuestionPrompt",
    "RoleCall",
    "RunState",
    "TaskNodeView",
    "ToolCallView",
    "TransitionView",
    "VerifyView",
    "apply_event",
    "fold_machine",
    "format_log_line",
    "initial_state",
    "newest_state_log",
    "tail_events",
]
