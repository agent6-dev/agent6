# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The agent6 command-line interface.

The entry point that wires a run: it imports every other `ui` package, and none imports it.
`entry` parses argv and dispatches to one module per command family.
The console script and the tests enter through `entry.main` and `entry.cli_main`.
Main modules: `entry`, `run`, `sessions_cmds`, `machine_cmds`, `config_cmds`.
"""

from __future__ import annotations
