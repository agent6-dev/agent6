# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""State machines: declarative, replayable mini-agents.

Load and validate a `.asm.toml` (`spec`, `_semantics`), render it as a diagram (`graph`),
and drive it deterministically (`engine`) over an append-only `journal` with crash recovery
and offline replay. An `agent` state runs an agent6 loop through an injected runner.
"""

from __future__ import annotations

from agent6.machine._semantics import (  # noqa: ICN003  # re-export
    fixture_problems,
    load_machine,
    validate_record_payload,
    validate_semantics,
)
from agent6.machine.authoring import build_authoring_prompt  # noqa: ICN003  # re-export
from agent6.machine.dryrun import BranchCheck, DryRunReport, StateCheck, dry_run  # noqa: ICN003  # re-export
from agent6.machine.engine import (  # noqa: ICN003  # re-export
    AgentExecResult,
    AgentRequest,
    EngineError,
    LiveWorld,
    MachineResult,
    ToolExecResult,
    ToolPolicyFactory,
    WaitWake,
    World,
    drive,
)
from agent6.machine.graph import render_dot, render_mermaid  # noqa: ICN003  # re-export
from agent6.machine.journal import (  # noqa: ICN003  # re-export
    AgentFact,
    AttemptSpend,
    JournalError,
    MachineBegin,
    MachineEnd,
    MachineJournal,
    MachineNotify,
    PendingWait,
    Snapshot,
    StepEvent,
    WaitFact,
    bundle_drift,
    clear_stop_request,
    machine_lock,
    read_source,
    stop_requested,
    write_bundle,
    write_source,
    write_stop_request,
)
from agent6.machine.schema import (  # noqa: ICN003  # re-export
    PROTECTED_OVERLAY_LEAVES,
    PROTECTED_OVERLAY_TABLES,
    AgentState,
    FieldSpec,
    MachineError,
    MachineSpec,
    StateSpec,
    ToolState,
    protected_overlay_error,
    protected_overlay_key_error,
)
from agent6.prompts.machine import MACHINE_AUTHOR_GUIDE  # noqa: ICN003  # re-export

__all__ = [
    "MACHINE_AUTHOR_GUIDE",
    "PROTECTED_OVERLAY_LEAVES",
    "PROTECTED_OVERLAY_TABLES",
    "AgentExecResult",
    "AgentFact",
    "AgentRequest",
    "AgentState",
    "AttemptSpend",
    "BranchCheck",
    "DryRunReport",
    "EngineError",
    "FieldSpec",
    "JournalError",
    "LiveWorld",
    "MachineBegin",
    "MachineEnd",
    "MachineError",
    "MachineJournal",
    "MachineNotify",
    "MachineResult",
    "MachineSpec",
    "PendingWait",
    "Snapshot",
    "StateCheck",
    "StateSpec",
    "StepEvent",
    "ToolExecResult",
    "ToolPolicyFactory",
    "ToolState",
    "WaitFact",
    "WaitWake",
    "World",
    "build_authoring_prompt",
    "bundle_drift",
    "clear_stop_request",
    "drive",
    "dry_run",
    "fixture_problems",
    "load_machine",
    "machine_lock",
    "protected_overlay_error",
    "protected_overlay_key_error",
    "read_source",
    "render_dot",
    "render_mermaid",
    "stop_requested",
    "validate_record_payload",
    "validate_semantics",
    "write_bundle",
    "write_source",
    "write_stop_request",
]
