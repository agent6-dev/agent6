# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""An in-workspace agent6 install warns at run entry (never refuses)."""

from __future__ import annotations

from pathlib import Path

import pytest

import agent6
from agent6.app import _session as session_mod
from agent6.app.reporter import Reporter


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
    said: list[str] = []
    session_mod.warn_install_inside_workspace(
        ws, reporter=Reporter(out=said.append, err=said.append)
    )
    warnings = [line for line in said if "WARNING" in line]
    assert any("installed inside" in w and "pipx" in w for w in warnings)


def test_no_warning_for_an_outside_install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    outside = tmp_path / "opt" / "agent6"
    outside.mkdir(parents=True)
    monkeypatch.setattr(agent6, "__file__", str(outside / "__init__.py"))
    ws = tmp_path / "ws"
    ws.mkdir()
    said: list[str] = []
    session_mod.warn_install_inside_workspace(
        ws, reporter=Reporter(out=said.append, err=said.append)
    )
    assert not [line for line in said if "installed inside" in line]
