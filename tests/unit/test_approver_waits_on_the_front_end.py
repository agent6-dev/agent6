# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A live front-end's answer is awaited for as long as the front-end lives.

An approval a front-end left unanswered for ten minutes fell through to the terminal prompt,
which under `run --tui` read the tty the dashboard owned.
"""

from __future__ import annotations

import os
import pathlib

import pytest

from agent6.sessions import ipc, process
from agent6.tools import operator_prompts
from agent6.ui.cli import _interact


def test_the_terminal_is_not_read_while_a_front_end_is_live(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_dir = tmp_path / "run"
    session_dir.mkdir()
    process.register_frontend(session_dir, os.getpid())

    class _Clock:
        """Eleven minutes pass per sleep; the front-end answers during the second one."""

        def __init__(self) -> None:
            self.now = 0.0
            self.sleeps = 0

        def monotonic(self) -> float:
            return self.now

        def sleep(self, seconds: float) -> None:
            self.now += 660.0
            self.sleeps += 1
            if self.sleeps == 2:
                ipc.write_answer(session_dir, "approval-1", "yes")

    def _terminal(*_a: object, **_k: object) -> str | None:
        pytest.fail("the terminal prompt ran while a front-end was live")

    monkeypatch.setattr(ipc, "time", _Clock())
    monkeypatch.setattr(_interact.portable, "has_controlling_tty", lambda: True)
    monkeypatch.setattr(_interact, "default_stdin_approver", _terminal)
    request = operator_prompts.ApprovalRequest(
        id="approval-1", prompt="Allow run_command: ls", scope="command", call_id=1
    )
    assert _interact.build_approver(session_dir)(request) == operator_prompts.ApprovalAnswer(
        True, "frontend"
    )
