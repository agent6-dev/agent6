# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The run's task graph in the loop: the focus, the operator's tasks and the standing goal.

Part of `harness`: every module reads `graph`, and none imports `app` or `ui`.
Main modules: `focus`, `operator_tasks`, `standing`.
"""

from __future__ import annotations
