# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Startup-warning helpers in app/preflight."""

from __future__ import annotations

import io

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
    monkeypatch.setattr("sys.stdin", io.StringIO())  # isatty() -> False
    refusal = headless_approval_refusal(_ask_cfg(), tui_enabled=False, away="")
    assert refusal is not None
    assert "would wait forever" in refusal
    assert "--auto-approve" in refusal  # the fix is named


@pytest.mark.parametrize(
    ("tui", "away", "commands"),
    [
        (True, "", "ask"),  # a TUI can answer
        (False, "wait", "ask"),  # an away-mode says what an absent operator meant
        (False, "deny", "ask"),  # ... including "deny", which a btw uses
        (False, "", "yes"),  # nothing to approve
        (False, "", "no"),  # commands withheld entirely
    ],
)
def test_answerable_runs_are_not_refused(
    monkeypatch: pytest.MonkeyPatch, tui: bool, away: str, commands: str
) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO())
    cfg = Config.model_validate({"sandbox": {"run_commands": commands}})
    assert headless_approval_refusal(cfg, tui_enabled=tui, away=away) is None
