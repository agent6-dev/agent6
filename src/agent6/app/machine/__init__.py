# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Compose the machine engine behind `agent6 machine run` and `create`.

`ui/cli/machine_cmds.py` keeps argv adaptation and console rendering; this package composes
`agent6.machine`, resolves one config per state and the sandbox, egress and budget preflight,
and spawns the per-`agent`-state runner.
"""

from __future__ import annotations

from agent6.app.machine._bundle import MachineFileSummary, summarize_machine_file, validate_bundle  # noqa: ICN003  # re-export
from agent6.app.machine._preflight import (  # noqa: ICN003  # re-export
    build_machine_notify_hook,
    machine_pass_env_refusal,
    machine_protect_paths,
)
from agent6.app.machine._scriptcheck import (  # noqa: ICN003  # re-export
    OfflineTestOutcome,
    available_tools,
    lint_and_typecheck,
    run_offline_tests,
)
from agent6.app.machine._spend import book_crashed_attempt  # noqa: ICN003  # re-export
from agent6.app.machine._states import (  # noqa: ICN003  # re-export
    MachineConfigs,
    SandboxWidenedError,
    StateConfig,
    resolve_configs,
    sandbox_widening,
)
from agent6.viewmodel.machine_state import Spend, machine_spend, read_budget_totals  # noqa: ICN003  # re-export

# `create` and `run` are not re-exported: both import `app.machine_agent`, which imports `_spend`.
__all__ = [
    "MachineConfigs",
    "MachineFileSummary",
    "OfflineTestOutcome",
    "SandboxWidenedError",
    "Spend",
    "StateConfig",
    "available_tools",
    "book_crashed_attempt",
    "build_machine_notify_hook",
    "lint_and_typecheck",
    "machine_pass_env_refusal",
    "machine_protect_paths",
    "machine_spend",
    "read_budget_totals",
    "resolve_configs",
    "run_offline_tests",
    "sandbox_widening",
    "summarize_machine_file",
    "validate_bundle",
]
