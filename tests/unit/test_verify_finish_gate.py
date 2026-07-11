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


def _wf(*, verify: bool) -> Workflow:
    data: dict[str, Any] = {"workflow": {"verify_command": ["true"]}} if verify else {}
    return Workflow(
        root=Path("/tmp"),
        config=Config.model_validate(data),
        provider=MagicMock(),
        dispatcher=MagicMock(),
        logger=lambda _m: None,
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
