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


def test_headless_ask_note_prints_for_run_but_not_ask_mode(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # No terminal, no TUI, run_commands='ask': a real run's run_command would
    # PAUSE, so the note is warranted. In read-only `ask` mode there is no
    # run_command tool, so the note (about an unrelated config knob) is noise.
    monkeypatch.setattr("sys.stdin", io.StringIO())  # isatty() -> False
    cfg = _ask_cfg()

    warn_if_headless_ask(cfg, tui_enabled=False, mode="run")
    assert "run_command will PAUSE" in capsys.readouterr().err

    warn_if_headless_ask(cfg, tui_enabled=False, mode="ask")
    assert capsys.readouterr().err == ""
