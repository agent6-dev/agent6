# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A `TurnContext` for advisor pins: a plain run with no gate, no budget and
no tasks unless a test says otherwise."""

from __future__ import annotations

from typing import Any

from agent6.config import WorkflowConfig
from agent6.workflows._guards import GuardSettings, TurnContext


def turn_context(**overrides: Any) -> TurnContext:
    facts: dict[str, Any] = {
        "mode": "run",
        "iteration": 1,
        "leg_start": 1,
        "workflow": WorkflowConfig(),
        "guards": GuardSettings(),
        "metric": False,
        "memory_wired": False,
        "gate_present": lambda: False,
        "verify_command": lambda: (),
        "tree_sha": lambda: "",
        "tree_green": lambda: None,
        "budget_remaining": lambda: None,
        "operator_wait_s": lambda: 0.0,
        "open_subtasks": list,
    }
    facts.update(overrides)
    return TurnContext(**facts)
