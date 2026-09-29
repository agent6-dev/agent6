# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The terminal UI: read-only viewers over a session's event stream.

The package runs out of process, reads `<run-dir>/logs.jsonl` from disk, and
writes only the answer files and its own `ui.toml` preferences. The render-ready
state and the tailer live in `agent6.viewmodel`; the write side lives in
`agent6.sessions.ipc` and `agent6.ui.spawn`, shared with every front-end.
"""

from __future__ import annotations
