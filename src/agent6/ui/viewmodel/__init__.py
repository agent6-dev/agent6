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

from agent6.ui.viewmodel.listing import (
    RunSummary,
    first_task_line,
    run_mtime,
    status_word,
    summarize_run_dir,
    task_snippet,
)
from agent6.ui.viewmodel.machine_state import (
    MachineEndView,
    MachineState,
    MachineStateView,
    MachineWatchCursor,
    NotificationView,
    TransitionView,
    fold_machine,
    machine_state_as_dict,
    newest_state_log,
    notification_key,
    read_complete_lines,
)
from agent6.ui.viewmodel.state import (
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
    fold_run,
    format_log_line,
    initial_state,
    run_state_as_dict,
    run_status_label,
)
from agent6.ui.viewmodel.tail import LogTail, tail_events
from agent6.ui.viewmodel.transcript import (
    TranscriptFold,
    TranscriptItem,
    fold_transcript,
    salient_arg,
)

__all__ = [
    "MAX_LOG_TAIL",
    "ApprovalPrompt",
    "BudgetView",
    "LogTail",
    "MachineEndView",
    "MachineState",
    "MachineStateView",
    "MachineWatchCursor",
    "NotificationView",
    "QuestionPrompt",
    "RoleCall",
    "RunState",
    "RunSummary",
    "TaskNodeView",
    "ToolCallView",
    "TranscriptFold",
    "TranscriptItem",
    "TransitionView",
    "VerifyView",
    "apply_event",
    "first_task_line",
    "fold_machine",
    "fold_run",
    "fold_transcript",
    "format_log_line",
    "initial_state",
    "machine_state_as_dict",
    "newest_state_log",
    "notification_key",
    "read_complete_lines",
    "run_mtime",
    "run_state_as_dict",
    "run_status_label",
    "salient_arg",
    "status_word",
    "summarize_run_dir",
    "tail_events",
    "task_snippet",
]
