# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 run`/`resume` process exit codes.

CONFIG.md documents a budget-exhausted run as exit 3 (resumable: raise the cap
and `agent6 resume`) and a finish over a red verify as exit 4; everything else
completed=False is exit 1, a clean or ungated finish is 0.
"""

from __future__ import annotations

from agent6.app.finalize import session_exit_code
from agent6.workflows._session_state import SessionEndReason, Verification
from agent6.workflows.loop import SessionResult


def _result(
    *, completed: bool, reason: SessionEndReason, verified: Verification = "not_applicable"
) -> SessionResult:
    return SessionResult(
        completed=completed,
        reason=reason,
        summary="",
        iterations=1,
        tool_calls=1,
        verified=verified,
    )


def test_exit_code_success_is_zero() -> None:
    assert session_exit_code(_result(completed=True, reason="finish_session")) == 0


def test_exit_code_budget_exhausted_is_three() -> None:
    # The documented "raise the cap and resume" signal.
    assert session_exit_code(_result(completed=False, reason="budget_exhausted")) == 3


def test_exit_code_other_failures_are_one() -> None:
    for reason in ("provider_error", "max_iterations", "went_quiet", "steer_abort"):
        assert session_exit_code(_result(completed=False, reason=reason)) == 1


def test_exit_code_finish_over_a_red_verify_is_four() -> None:
    """`completed` means the agent stopped deliberately, not that the work
    verified: a finish_session over a red or stale gate exited 0 and read as
    success to every script. Its own code, distinct from a broken run (1)."""
    assert (
        session_exit_code(_result(completed=True, reason="finish_session", verified="failed")) == 4
    )
    assert session_exit_code(_result(completed=True, reason="settled", verified="failed")) == 4


def test_exit_code_verified_finish_is_zero() -> None:
    # Green, and gateless (nothing to verify) -- both are exit 0.
    assert (
        session_exit_code(_result(completed=True, reason="finish_session", verified="passed")) == 0
    )
    assert (
        session_exit_code(_result(completed=True, reason="settled", verified="not_applicable")) == 0
    )
