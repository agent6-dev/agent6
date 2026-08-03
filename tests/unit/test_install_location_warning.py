# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""An in-workspace agent6 install warns at run entry (never refuses)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

import agent6
from agent6.app import _session as session_mod
from agent6.config import Config


def test_detects_an_install_root_inside_the_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = tmp_path / "ws"
    pkg = ws / ".venv" / "lib" / "site-packages" / "agent6"
    pkg.mkdir(parents=True)
    monkeypatch.setattr(agent6, "__file__", str(pkg / "__init__.py"))
    assert session_mod.install_inside_workspace(ws) == pkg
    assert session_mod.install_inside_workspace(tmp_path / "elsewhere") is None


def test_run_entry_warns_once_and_never_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The warning is loud, names the install root and the remedy, and the
    session still starts (agent6 developing agent6 is exactly this shape)."""
    ws = tmp_path / "ws"
    pkg = ws / ".venv" / "agent6"
    pkg.mkdir(parents=True)
    monkeypatch.setattr(agent6, "__file__", str(pkg / "__init__.py"))
    reporter = MagicMock()
    session_mod.start_isolation(Config(), "strict", cwd=ws, reporter=reporter)
    warnings = [c.args[0] for c in reporter.err.call_args_list if "WARNING" in c.args[0]]
    assert any("installed inside" in w and "pipx" in w for w in warnings)


def test_no_warning_for_an_outside_install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    outside = tmp_path / "opt" / "agent6"
    outside.mkdir(parents=True)
    monkeypatch.setattr(agent6, "__file__", str(outside / "__init__.py"))
    ws = tmp_path / "ws"
    ws.mkdir()
    reporter = MagicMock()
    session_mod.start_isolation(Config(), "strict", cwd=ws, reporter=reporter)
    assert not [c for c in reporter.err.call_args_list if "installed inside" in c.args[0]]
