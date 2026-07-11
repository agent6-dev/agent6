# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""UI subtree for agent6, read-only viewers over the JSONL event stream.

Everything in this package is **optional, side-effect-free, and consumes
`<run-dir>/logs.jsonl` from disk**. Nothing in here is part of the
core agent loop; reviewers can skip this directory and still understand
how agent6 actually plans and edits code.

The render-ready state and the JSONL tailer live in `agent6.ui.viewmodel` (shared
with the CLI and the web client); this package is the textual painting of that
state. The file-based write bridge (approve / ask_user / steer, spawn) lives in
`agent6.ui.bridge`, shared with the CLI and web.

Layout:
    modals.py    textual modal screens (approve / steer / question).
    app.py       the run dashboard (Agent6TUI + run_tui).
    home.py      the `agent6 tui` hub: list runs + launch run/plan/ask.

Everything is launched out-of-process and only reads `logs.jsonl` + writes the
small answer files the workflow polls (via `agent6.ui.bridge`), so the core loop
is untouched and any other front-end (VS Code, web, desktop) mirrors the same
file contract.
"""

from __future__ import annotations
