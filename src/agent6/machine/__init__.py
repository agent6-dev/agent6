# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""agent6 state machines, declarative, replayable mini-agents.

Phase 1: load + validate a `.asm.toml` file (`model`) and render it as a
diagram (`graph`). Phase 2: execute it deterministically (`engine`) over an
append-only `journal`, with crash recovery and offline replay. Phase 3 wires
the `agent` state kind into a normal agent6 loop through an injected runner.
Phase 4 adds 24/7 ergonomics: `machine status`/`poke` and an external-scheduler
`--exit-on-wait` mode that persists the next wake instead of blocking.
"""

from __future__ import annotations

from agent6.machine._semantics import load_machine, validate_semantics
from agent6.machine.authoring import (
    MACHINE_AUTHOR_GUIDE,
    SCRIPTS_PAYLOAD_KEY,
    TOML_PAYLOAD_KEY,
    build_authoring_prompt,
    extract_scripts,
    extract_toml,
)
from agent6.machine.dryrun import BranchCheck, DryRunReport, StateCheck, dry_run
from agent6.machine.engine import (
    AgentExecResult,
    AgentRequest,
    EngineError,
    LiveWorld,
    MachineResult,
    ToolExecResult,
    WaitWake,
    World,
    drive,
)
from agent6.machine.graph import GraphFormat, render, render_dot, render_mermaid
from agent6.machine.journal import (
    AgentFact,
    JournalError,
    MachineBegin,
    MachineEnd,
    MachineJournal,
    PendingWait,
    Snapshot,
    StepEvent,
    WaitFact,
    machine_lock,
    read_source,
    write_source,
)
from agent6.machine.model import (
    AgentState,
    MachineError,
    MachineSpec,
    ToolState,
)

__all__ = [
    "MACHINE_AUTHOR_GUIDE",
    "SCRIPTS_PAYLOAD_KEY",
    "TOML_PAYLOAD_KEY",
    "AgentExecResult",
    "AgentFact",
    "AgentRequest",
    "AgentState",
    "BranchCheck",
    "DryRunReport",
    "EngineError",
    "GraphFormat",
    "JournalError",
    "LiveWorld",
    "MachineBegin",
    "MachineEnd",
    "MachineError",
    "MachineJournal",
    "MachineResult",
    "MachineSpec",
    "PendingWait",
    "Snapshot",
    "StateCheck",
    "StepEvent",
    "ToolExecResult",
    "ToolState",
    "WaitFact",
    "WaitWake",
    "World",
    "build_authoring_prompt",
    "drive",
    "dry_run",
    "extract_scripts",
    "extract_toml",
    "load_machine",
    "machine_lock",
    "read_source",
    "render",
    "render_dot",
    "render_mermaid",
    "validate_semantics",
    "write_source",
]
