# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 run`/`resume` process exit codes.

CONFIG.md documents a budget-exhausted run as exit 3 (resumable: raise the cap
and `agent6 resume`); everything else completed=False is exit 1, success is 0.
"""

from __future__ import annotations

from agent6.cli.run import _run_exit_code  # pyright: ignore[reportPrivateUsage]
from agent6.workflows.loop import RunResult


def _result(*, completed: bool, reason: str) -> RunResult:
    return RunResult(completed=completed, reason=reason, summary="", iterations=1, tool_calls=1)


def test_exit_code_success_is_zero() -> None:
    assert _run_exit_code(_result(completed=True, reason="finish_run")) == 0


def test_exit_code_budget_exhausted_is_three() -> None:
    # The documented "raise the cap and resume" signal.
    assert _run_exit_code(_result(completed=False, reason="budget_exhausted")) == 3


def test_exit_code_other_failures_are_one() -> None:
    for reason in ("provider_error", "max_iterations", "went_quiet", "steer_abort"):
        assert _run_exit_code(_result(completed=False, reason=reason)) == 1
