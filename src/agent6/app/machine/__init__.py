# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Compose the machine engine behind `agent6 machine run` and `create`.

Part of `app`: it imports `agent6.machine`, `harness` and the layers below, never `ui`.
It resolves one config per state and runs the sandbox, egress and budget preflight.
It spawns the runner of each `agent` state.
`ui/cli/machine_cmds.py` keeps argv adaptation and console rendering.
Main modules: `run`, `create`, `agent`, `check`, `state_configs`, `scriptcheck`.
"""

from __future__ import annotations

from agent6.app.machine.bundle import (  # re-export
    MachineFileSummary,
    summarize_machine_file,
    validate_bundle,
)
from agent6.app.machine.crash_spend import book_crashed_attempt  # re-export
from agent6.app.machine.host_preflight import (  # re-export
    build_machine_notify_hook,
    machine_pass_env_refusal,
    machine_protect_paths,
)
from agent6.app.machine.scriptcheck import (  # re-export
    OfflineTestOutcome,
    available_tools,
    lint_and_typecheck,
    run_offline_tests,
)
from agent6.app.machine.state_configs import (  # re-export
    ConfigWidenedError,
    MachineConfigs,
    StateConfig,
    config_widening,
    resolve_configs,
)

# `create` and `run` are not re-exported: both import `agent`, which imports `crash_spend`.
__all__ = [
    "ConfigWidenedError",
    "MachineConfigs",
    "MachineFileSummary",
    "OfflineTestOutcome",
    "StateConfig",
    "available_tools",
    "book_crashed_attempt",
    "build_machine_notify_hook",
    "config_widening",
    "lint_and_typecheck",
    "machine_pass_env_refusal",
    "machine_protect_paths",
    "resolve_configs",
    "run_offline_tests",
    "summarize_machine_file",
    "validate_bundle",
]
