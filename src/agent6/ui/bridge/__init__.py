# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Front-end process-spawn + device-notification helpers.

Every front-end (CLI, TUI, web) drives a run by spawning the same `agent6` CLI
a user would run; `spawn.py` finds the exe and launches it detached (new work /
machines / background resume). `notify.py` fires a best-effort desktop
notification. The run-dir answer-file contract itself lives in
`agent6.runs.ipc` (the workflow process polls it), not here.

Nothing here imports a UI toolkit; it is pure stdlib + subprocess, so the CLI,
the Textual TUI, and the browser server all share one implementation.
"""

from __future__ import annotations
