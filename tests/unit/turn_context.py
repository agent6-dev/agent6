# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A `TurnContext` for advisor pins: a plain run with no gate, no budget and
no tasks unless a test says otherwise."""

from __future__ import annotations

from typing import Any

from agent6.workflows._advice import GuardSettings, TurnContext
from agent6.workflows._loop_state import TurnState


def _never_rejected(_turn: TurnState, _ending: str) -> bool:
    return False


def _no_standing_task(_reason: str, _iteration: int) -> None:
    return None


def turn_context(**overrides: Any) -> TurnContext:
    facts: dict[str, Any] = {
        "mode": "run",
        "iteration": 1,
        "leg_start": 1,
        "guards": GuardSettings(),
        "verify_when": "finish",
        "verify_retries": 2,
        "finish_validator": None,
        "metric": False,
        "memory_wired": False,
        "gate_present": lambda: False,
        "verify_command": lambda: (),
        "tree_sha": lambda: "",
        "tree_green": lambda: None,
        "budget_remaining": lambda: None,
        "operator_wait_s": lambda: 0.0,
        "open_subtasks": list,
        "end_reviewed": _never_rejected,
        "standing_absorb": _no_standing_task,
    }
    facts.update(overrides)
    return TurnContext(**facts)
