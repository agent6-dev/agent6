# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A group's child sessions: their launch files, spawning, relay, waiting, resume and import.

Part of `app`: it imports `harness` and the layers below, never `ui`.
The dispatcher the loop runs is `harness.parallel`.
Main modules: `children`, `launch`, `relay`, `wait`, `results`, `resume`.
"""

from __future__ import annotations
