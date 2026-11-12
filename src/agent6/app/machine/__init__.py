# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Compose the machine engine behind `agent6 machine run` and `create`.

`ui/cli/machine_cmds.py` keeps argv adaptation and console rendering; this package composes
`agent6.machine`, resolves one config per state and the sandbox, egress and budget preflight,
and spawns the per-`agent`-state runner.
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
    MachineConfigs,
    SandboxWidenedError,
    StateConfig,
    resolve_configs,
    sandbox_widening,
)

# `create` and `run` are not re-exported: both import `agent`, which imports `crash_spend`.
__all__ = [
    "MachineConfigs",
    "MachineFileSummary",
    "OfflineTestOutcome",
    "SandboxWidenedError",
    "StateConfig",
    "available_tools",
    "book_crashed_attempt",
    "build_machine_notify_hook",
    "lint_and_typecheck",
    "machine_pass_env_refusal",
    "machine_protect_paths",
    "resolve_configs",
    "run_offline_tests",
    "sandbox_widening",
    "summarize_machine_file",
    "validate_bundle",
]
