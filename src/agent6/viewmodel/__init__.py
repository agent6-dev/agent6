# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Fold the JSONL event stream into the render-ready state every front-end paints.

A shared package: any layer may import it, and it imports no layer.
The CLI, the TUI, the web UI and ACP fold the same `<session-dir>/logs.jsonl` through it.
The folds are pure functions over frozen dataclasses, with no I/O and no async.
A viewer in any language mirrors `SessionState` and `MachineState` field for field.
Main modules: `transcript`, `listing`, `review`, `receipt`, `wire`, `telemetry`, `tail`.
"""

from __future__ import annotations
