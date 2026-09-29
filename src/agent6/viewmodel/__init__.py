# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Fold the JSONL event stream into the render-ready state every front-end paints.

The CLI, the TUI, the web UI and ACP read the same `<run-dir>/logs.jsonl`, fold it
through the same pure functions here, and differ only in how they paint the result.
The folds do no I/O and hold no async: frozen dataclasses and pure functions, so a
viewer in any language mirrors `SessionState` and `MachineState` field for field.

Modules:
    events.py            typed read model of the event families the fold consumes.
    state.py             event fold: list[event] -> SessionState.
    machine_state.py     journal fold: machine journal -> MachineState, plus the watch cursor.
    tail.py              stdlib JSONL file tailer, the event source.
    transcript.py        event fold: logs.jsonl -> live conversation TranscriptItems.
    transcript_style.py  one styled-line renderer for a folded TranscriptItem.
    transcript_render.py fold and Markdown render of the per-call provider transcripts.
    listing.py           run-dir scan -> SessionSummary rows for listings and pickers.
    log_line.py          one-line renderings of an event for the log views.
    wire.py              the one-object wire snapshots of a session and of a machine.
    policy.py            a session's policy facts, folded from its dir.
    format.py            shared glyphs and cost/status formatting.
    config_view.py       effective-config tree -> the `config show` view.
"""

from __future__ import annotations

from agent6.viewmodel.events import event_epoch  # noqa: ICN003  # re-export
from agent6.viewmodel.listing import (  # noqa: ICN003  # re-export
    LIVE_STATUS_WORDS,
    LogScan,
    SessionSummary,
    StatusFacts,
    died_without_end,
    is_session_husk,
    is_winner,
    newest_session_dir,
    produced_result,
    scan_session_log,
    session_compare,
    session_dirs,
    session_is_live,
    session_mtime,
    status_for_session_dir,
    status_word,
    summarize_session_dir,
    summary_row,
    task_snippet,
)
from agent6.viewmodel.log_line import format_log_line  # noqa: ICN003  # re-export
from agent6.viewmodel.machine_state import (  # noqa: ICN003  # re-export
    AgentExecution,
    InstanceProbes,
    MachineState,
    MachineSummary,
    MachineWatchCursor,
    NewestExecutionFold,
    NotificationView,
    TransitionView,
    armed_wait,
    execution_of,
    fold_machine,
    machine_files,
    machine_instance_dirs,
    machine_spend,
    machine_state_as_dict,
    machine_status_word,
    machine_verb_refusal,
    machine_verb_refusals,
    machine_word_for_dir,
    newest_agent_execution,
    newest_state_log,
    notification_key,
    probe_instance,
    read_complete_lines,
    summarize_machine_dir,
    verb_answer,
)
from agent6.viewmodel.policy import session_policy  # noqa: ICN003  # re-export
from agent6.viewmodel.state import (  # noqa: ICN003  # re-export
    MAX_LOG_TAIL,
    ApprovalPrompt,
    BudgetView,
    QuestionPrompt,
    RoleCall,
    SessionState,
    TaskNodeView,
    ToolCallView,
    apply_event,
    approval_parts,
    fold_session,
    initial_state,
    open_approval,
    open_approval_of,
    open_question,
    session_state_as_dict,
    status_facts,
    task_tree_views,
)
from agent6.viewmodel.tail import LogTail, tail_events  # noqa: ICN003  # re-export
from agent6.viewmodel.transcript import (  # noqa: ICN003  # re-export
    TranscriptFold,
    TranscriptItem,
    fold_transcript,
    operator_inputs,
    restate,
    salient_arg,
    worker_models,
)
from agent6.viewmodel.wire import (  # noqa: ICN003  # re-export
    UnknownStepError,
    existing_run_branch,
    machine_snapshot,
    manifest_branches,
    manifest_header,
    session_snapshot,
)

__all__ = [
    "LIVE_STATUS_WORDS",
    "MAX_LOG_TAIL",
    "AgentExecution",
    "ApprovalPrompt",
    "BudgetView",
    "InstanceProbes",
    "LogScan",
    "LogTail",
    "MachineState",
    "MachineSummary",
    "MachineWatchCursor",
    "NewestExecutionFold",
    "NotificationView",
    "QuestionPrompt",
    "RoleCall",
    "SessionState",
    "SessionSummary",
    "StatusFacts",
    "TaskNodeView",
    "ToolCallView",
    "TranscriptFold",
    "TranscriptItem",
    "TransitionView",
    "UnknownStepError",
    "apply_event",
    "approval_parts",
    "armed_wait",
    "died_without_end",
    "event_epoch",
    "execution_of",
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
    "machine_spend",
    "machine_state_as_dict",
    "machine_status_word",
    "machine_verb_refusal",
    "machine_verb_refusals",
    "machine_word_for_dir",
    "manifest_branches",
    "manifest_header",
    "newest_agent_execution",
    "newest_session_dir",
    "newest_state_log",
    "notification_key",
    "open_approval",
    "open_approval_of",
    "open_question",
    "operator_inputs",
    "probe_instance",
    "produced_result",
    "read_complete_lines",
    "restate",
    "salient_arg",
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
    "task_snippet",
    "task_tree_views",
    "verb_answer",
    "worker_models",
]
