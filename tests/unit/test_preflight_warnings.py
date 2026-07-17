# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Startup-warning helpers in app/preflight."""

from __future__ import annotations

import io

import pytest

from agent6.app.preflight import warn_if_headless_ask
from agent6.config import Config


def _ask_cfg() -> Config:
    return Config.model_validate({"sandbox": {"run_commands": "ask"}})


def test_headless_ask_note_prints_when_no_approver_reachable(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # No terminal, no TUI, run_commands='ask': a run_command PAUSES with nothing
    # to approve it. The note fires for every mode this helper is called from
    # (run/plan/ask) -- all three expose run_command (see
    # test_tool_definitions_ask_mode_is_read_only_with_commands).
    monkeypatch.setattr("sys.stdin", io.StringIO())  # isatty() -> False
    warn_if_headless_ask(_ask_cfg(), tui_enabled=False)
    assert "run_command will PAUSE" in capsys.readouterr().err


def test_headless_ask_note_silent_with_approver_or_no_ask(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO())  # isatty() -> False
    # A TUI can answer the prompt: no note.
    warn_if_headless_ask(_ask_cfg(), tui_enabled=True)
    assert capsys.readouterr().err == ""
    # run_commands != 'ask' never prompts: no note.
    warn_if_headless_ask(
        Config.model_validate({"sandbox": {"run_commands": "no"}}), tui_enabled=False
    )
    assert capsys.readouterr().err == ""
