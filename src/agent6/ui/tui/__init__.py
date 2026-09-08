# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""UI subtree for agent6, read-only viewers over the JSONL event stream.

Everything in this package is optional and out of the core loop: it runs
out-of-process, consumes `<run-dir>/logs.jsonl` from disk, and its only writes
are the answer files and its own ui.toml preferences.

The render-ready state and the JSONL tailer live in `agent6.viewmodel` (shared
with the CLI and the web client); this package is the textual painting of that
state. The file-based write side lives in `agent6.sessions.ipc` (approve /
ask_user / steer) and `agent6.ui.spawn` (launch the CLI detached), shared with
the CLI, the web UI and ACP.
"""

from __future__ import annotations
