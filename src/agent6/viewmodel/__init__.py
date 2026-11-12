# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Fold the JSONL event stream into the render-ready state every front-end paints.

The CLI, the TUI, the web UI and ACP read the same `<run-dir>/logs.jsonl`, fold it
through the same pure functions here, and differ only in how they paint the result.
The folds do no I/O and hold no async: frozen dataclasses and pure functions, so a
viewer in any language mirrors `SessionState` and `MachineState` field for field.
"""

from __future__ import annotations

from agent6.viewmodel.comparison import is_winner, session_compare  # re-export
from agent6.viewmodel.events import event_epoch  # re-export
from agent6.viewmodel.listing import (  # re-export
    SessionSummary,
    is_session_husk,
    newest_session_dir,
    session_dirs,
    session_is_live,
    session_mtime,
    summarize_session_dir,
    summary_row,
)
from agent6.viewmodel.log_line import format_log_line  # re-export
from agent6.viewmodel.machine_state import (  # re-export
    InstanceProbes,
    MachineState,
    MachineSummary,
    NotificationView,
    TransitionView,
    armed_wait,
    fold_machine,
    machine_files,
    machine_instance_dirs,
    machine_state_as_dict,
    machine_status_word,
    machine_verb_refusal,
    machine_verb_refusals,
    machine_word_for_dir,
    probe_instance,
    summarize_machine_dir,
    verb_answer,
)
from agent6.viewmodel.policy import session_policy  # re-export
from agent6.viewmodel.session_log import (  # re-export
    LogScan,
    scan_session_log,
)
from agent6.viewmodel.session_status import (  # re-export
    LIVE_STATUS_WORDS,
    StatusFacts,
    died_without_end,
    produced_result,
    status_for_session_dir,
    status_word,
    task_snippet,
)
from agent6.viewmodel.state import (  # re-export
    MAX_LOG_TAIL,
    apply_event,
    fold_session,
    initial_state,
    open_approval,
    open_approval_of,
    pending_questions,
    task_tree_views,
)
from agent6.viewmodel.state_types import (  # re-export
    ApprovalPrompt,
    BudgetView,
    ModelCallView,
    Question,
    QuestionPrompt,
    RunCommit,
    SessionState,
    TaskNodeView,
    ToolCallView,
    approval_parts,
)
from agent6.viewmodel.state_view import (  # re-export
    context_fill,
    session_state_as_dict,
    status_facts,
)
from agent6.viewmodel.tail import LogTail, tail_events, tail_lines  # re-export
from agent6.viewmodel.transcript import (  # re-export
    TranscriptFold,
    fold_transcript,
    operator_inputs,
    restate,
    worker_models,
)
from agent6.viewmodel.transcript_items import TranscriptItem  # re-export
from agent6.viewmodel.wire import (  # re-export
    UnknownCommitError,
    existing_run_branch,
    machine_snapshot,
    manifest_branches,
    manifest_header,
    session_snapshot,
)

__all__ = [
    "LIVE_STATUS_WORDS",
    "MAX_LOG_TAIL",
    "ApprovalPrompt",
    "BudgetView",
    "InstanceProbes",
    "LogScan",
    "LogTail",
    "MachineState",
    "MachineSummary",
    "ModelCallView",
    "NotificationView",
    "Question",
    "QuestionPrompt",
    "RunCommit",
    "SessionState",
    "SessionSummary",
    "StatusFacts",
    "TaskNodeView",
    "ToolCallView",
    "TranscriptFold",
    "TranscriptItem",
    "TransitionView",
    "UnknownCommitError",
    "apply_event",
    "approval_parts",
    "armed_wait",
    "context_fill",
    "died_without_end",
    "event_epoch",
    "existing_run_branch",
    "fold_machine",
    "fold_session",
    "fold_transcript",
    "format_log_line",
    "initial_state",
    "is_session_husk",
    "is_winner",
    "machine_files",
    "machine_instance_dirs",
    "machine_snapshot",
    "machine_state_as_dict",
    "machine_status_word",
    "machine_verb_refusal",
    "machine_verb_refusals",
    "machine_word_for_dir",
    "manifest_branches",
    "manifest_header",
    "newest_session_dir",
    "open_approval",
    "open_approval_of",
    "operator_inputs",
    "pending_questions",
    "probe_instance",
    "produced_result",
    "restate",
    "scan_session_log",
    "session_compare",
    "session_dirs",
    "session_is_live",
    "session_mtime",
    "session_policy",
    "session_snapshot",
    "session_state_as_dict",
    "status_facts",
    "status_for_session_dir",
    "status_word",
    "summarize_machine_dir",
    "summarize_session_dir",
    "summary_row",
    "tail_events",
    "tail_lines",
    "task_snippet",
    "task_tree_views",
    "verb_answer",
    "worker_models",
]
