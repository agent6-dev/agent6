# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Startup-warning helpers in app/preflight."""

from __future__ import annotations

import pytest

from agent6.app.preflight import headless_approval_refusal
from agent6.config import Config


def _ask_cfg() -> Config:
    return Config.model_validate({"sandbox": {"run_commands": "ask"}})


def test_a_run_that_cannot_be_asked_refuses_instead_of_hanging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`ask` needs someone to answer. With no terminal, no TUI and no away-mode
    the first command waits forever -- and since the verify gate is a command
    too, that is essentially every run, every /parallel lane included. It used
    to print a note and hang anyway."""
    refusal = headless_approval_refusal(_ask_cfg(), tui_enabled=False, away="", can_ask=False)
    assert refusal is not None
    assert "would wait forever" in refusal
    assert "--auto-approve" in refusal  # the fix is named


@pytest.mark.parametrize(
    ("tui", "away", "commands", "can_ask"),
    [
        (True, "", "ask", False),  # a TUI can answer
        (False, "wait", "ask", False),  # an away-mode says what an absent operator meant
        (False, "deny", "ask", False),  # ... including "deny", which a btw uses
        (False, "", "yes", False),  # nothing to approve
        (False, "", "no", False),  # commands withheld entirely
        (False, "", "ask", True),  # the front-end asks out of band (a terminal, ACP)
    ],
)
def test_answerable_runs_are_not_refused(
    tui: bool, away: str, commands: str, can_ask: bool
) -> None:
    cfg = Config.model_validate({"sandbox": {"run_commands": commands}})
    assert headless_approval_refusal(cfg, tui_enabled=tui, away=away, can_ask=can_ask) is None
