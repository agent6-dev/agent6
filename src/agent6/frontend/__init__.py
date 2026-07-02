# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Shared, textual-free write bridge between a front-end and the workflow process.

Every front-end (CLI, TUI, web) drives a run the same way: spawn the same
`agent6` CLI a user would run, then answer its approval / question / steer
prompts by writing the small answer files the workflow process polls. This
package holds that contract so no front-end re-implements it.

Layout:
    approval.py  file-based bridges (approve / ask_user / steer) workflow<->front-end.
    spawn.py     find the agent6 exe and spawn it detached (new work / machines).

Nothing here imports a UI toolkit; it is pure filesystem + subprocess, so the
CLI, the Textual TUI, and the browser server all share one implementation.
"""

from __future__ import annotations

from agent6.frontend.approval import (
    APPROVAL_DIR_NAME,
    write_answer,
)

__all__ = [
    "APPROVAL_DIR_NAME",
    "write_answer",
]
