# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The terminal UI: the hub and the run views over a session's event stream.

A front-end: it may import every layer and the shared `ui` helpers, never another front-end.
It runs outside the worker's process and reads `<session-dir>/logs.jsonl` from disk.
The render-ready state and the tailer live in `agent6.viewmodel`.
It answers through `agent6.sessions.ipc` and spawns through `agent6.ui.spawn`.
Main modules: `app`, `home`, `dashboard`, `conversation`, `composer`.
"""

from __future__ import annotations
