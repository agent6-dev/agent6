# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Machine command lifecycle: the engine composition behind `agent6 machine
run`/`create` and the preflight/validation helpers the read-only CLI commands
share.

`ui/cli/machine_cmds.py` keeps only argv adaptation + console rendering; the
composition lives here, behind the `MachineFrontend` seam (mirroring
`app.run.RunFrontend`). The machine ENGINE itself (`agent6.machine`) is
unchanged; this package composes it, resolves the sandbox/egress/budget
preflight, and spawns the per-`agent`-state runner.
"""

from __future__ import annotations

from agent6.app.machine._bundle import is_inside, validate_bundle
from agent6.app.machine._preflight import (
    build_machine_notify_hook,
    hard_usd_preflight_error,
    machine_network_refusal,
    machine_protect_paths,
)
from agent6.app.machine._spend import Spend, machine_spend, read_budget_totals

__all__ = [
    "Spend",
    "build_machine_notify_hook",
    "hard_usd_preflight_error",
    "is_inside",
    "machine_network_refusal",
    "machine_protect_paths",
    "machine_spend",
    "read_budget_totals",
    "validate_bundle",
]
