# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The children a run dispatches: their dispatch, merge, ranking, panels and recovery.

Part of `harness`: it imports `tools` and the layers below, never `app` or `ui`.
`[harness.parallel]` configures it.
Main modules: `dispatch`, `merge`, `rank`, `panel`, `recovery`, `work`.
"""

from __future__ import annotations
