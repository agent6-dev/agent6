# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The agent6 UI layer: the three front-ends (cli, tui, web), the shared
read-model fold (viewmodel), and the two front-end write helpers (spawn: launch
the CLI detached; notify: desktop notification). Everything here is the top of
the dependency graph -- it may depend on the engine (workflows, tools, sandbox,
...); the engine never depends on it."""

from __future__ import annotations
