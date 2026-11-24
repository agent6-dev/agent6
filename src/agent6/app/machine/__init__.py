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
