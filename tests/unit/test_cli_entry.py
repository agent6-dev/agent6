# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The guarded console-script entry point `cli_main`."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.ui import cli
from agent6.ui.cli import cli_main


def test_cli_main_passes_through_return_code(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "main", lambda: 3)
    assert cli_main() == 3


def test_cli_main_converts_unexpected_exception_to_friendly_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _boom() -> int:
        raise RuntimeError("kaboom")

    monkeypatch.setattr(cli, "main", _boom)
    monkeypatch.delenv("AGENT6_DEBUG", raising=False)
    rc = cli_main()
    assert rc == 1
    err = capsys.readouterr().err
    assert "ERROR: unexpected RuntimeError: kaboom" in err
    # Points at a saved traceback that actually exists and contains the stack.
    tb_line = next(line for line in err.splitlines() if "full traceback:" in line)
    tb_path = Path(tb_line.split("full traceback:", 1)[1].strip())
    assert tb_path.is_file()
    assert "RuntimeError: kaboom" in tb_path.read_text(encoding="utf-8")
    tb_path.unlink()


def test_cli_main_reraises_under_debug(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom() -> int:
        raise RuntimeError("kaboom")

    monkeypatch.setattr(cli, "main", _boom)
    monkeypatch.setenv("AGENT6_DEBUG", "1")
    with pytest.raises(RuntimeError, match="kaboom"):
        cli_main()


def test_cli_main_handles_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _interrupt() -> int:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "main", _interrupt)
    assert cli_main() == 130
    assert "interrupted" in capsys.readouterr().err
