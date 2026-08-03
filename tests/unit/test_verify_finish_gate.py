# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The verify finish gate: finish_run can never report 'passed' over a red or
stale verify (honest default), and require_verify_to_finish turns that into an
opt-in hard gate. Both ground on _tree_is_verify_green."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from agent6.config import Config
from agent6.workflows.loop import (
    Workflow,
    _LoopState,  # pyright: ignore[reportPrivateUsage]
)


def _wf(*, verify: bool, mode: str = "run") -> Workflow:
    data: dict[str, Any] = {"workflow": {"verify_command": ["true"]}} if verify else {}
    return Workflow(
        root=Path("/tmp"),
        config=Config.model_validate(data),
        provider=MagicMock(),
        dispatcher=MagicMock(),
        logger=lambda _m: None,
        mode=mode,
    )


def _green(wf: Workflow, **state_kw: Any) -> bool | None:
    state = _LoopState(original_task="t", tool_calls=0, **state_kw)
    return wf._tree_is_verify_green(state)  # pyright: ignore[reportPrivateUsage]


def test_no_verify_command_is_not_gated() -> None:
    # Nothing to gate on -> None -> finish is always an honest pass.
    assert _green(_wf(verify=False), last_verify_ok=None) is None
    assert _green(_wf(verify=False), last_verify_ok=False) is None


def test_green_only_when_last_verify_passed_and_tree_unedited() -> None:
    wf = _wf(verify=True)
    assert _green(wf, last_verify_ok=True, edited_since_verify=False) is True
    # Never verified, or last verify failed -> not green.
    assert _green(wf, last_verify_ok=None) is False
    assert _green(wf, last_verify_ok=False) is False
    # A green verify that has since been edited over is stale -> not green.
    assert _green(wf, last_verify_ok=True, edited_since_verify=True) is False


def test_require_verify_to_finish_defaults_off() -> None:
    assert Config().workflow.require_verify_to_finish is False


def _verified(wf: Workflow, **state_kw: Any) -> str:
    state = _LoopState(original_task="t", tool_calls=0, **state_kw)
    return wf._verification(state)  # pyright: ignore[reportPrivateUsage]


def test_verification_carries_the_same_verdict_the_event_does() -> None:
    """RunResult.verified is the app layer's copy of run.end.all_passed's
    grounding, so exit code, auto-merge, and the notify hook read the verify
    truth instead of `completed` (true for any deliberate finish)."""
    assert _verified(_wf(verify=True), last_verify_ok=True, edited_since_verify=False) == "passed"
    assert _verified(_wf(verify=True), last_verify_ok=False) == "failed"
    # Green but edited since: stale, not verified.
    assert _verified(_wf(verify=True), last_verify_ok=True, edited_since_verify=True) == "failed"
    # Gateless: nothing ever gated this run, so there is no verdict to claim.
    assert _verified(_wf(verify=False), last_verify_ok=None) == "not_applicable"


def test_plan_and_ask_are_never_gated_on_verify() -> None:
    """plan/ask end clean whatever the tree looks like -- finish_planning and
    the ask answer both emit run.end all_passed=True -- so they have no verify
    verdict to report. Reporting one made `agent6 plan` exit 4 (preflight
    INFERS a verify command for plan, and plan never runs it, so the tree read
    as red) while its own journal and every listing said passed."""
    for mode in ("plan", "ask"):
        assert _verified(_wf(verify=True, mode=mode), last_verify_ok=None) == "not_applicable"
        assert _verified(_wf(verify=True, mode=mode), last_verify_ok=False) == "not_applicable"
