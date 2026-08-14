# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""UI subtree for agent6, read-only viewers over the JSONL event stream.

Everything in this package is **optional and out of the core loop**: it
consumes `<run-dir>/logs.jsonl` from disk, and its only writes are the
answer files and its own ui.toml preferences. Reviewers can skip this
directory and still understand how agent6 actually plans and edits code.

The render-ready state and the JSONL tailer live in `agent6.viewmodel` (shared
with the CLI and the web client); this package is the textual painting of that
state. The file-based write side lives in `agent6.sessions.ipc` (approve / ask_user
/ steer) and `agent6.ui.spawn` (launch the CLI detached), shared with the CLI
and web.

Everything is launched out-of-process and only reads `logs.jsonl` + writes the
small answer files the workflow polls (via `agent6.sessions.ipc`), so the core loop
is untouched and any other front-end (VS Code, web, desktop) mirrors the same
file contract.
"""

from __future__ import annotations
