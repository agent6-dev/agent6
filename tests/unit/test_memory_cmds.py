# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 memory` CLI: add/list/show/rm over the file-per-fact store."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.memory import MemoryStoreError
from agent6.ui.cli.memory_cmds import (
    _cmd_memory_add,  # pyright: ignore[reportPrivateUsage]
    _cmd_memory_list,  # pyright: ignore[reportPrivateUsage]
    _cmd_memory_rm,  # pyright: ignore[reportPrivateUsage]
    _cmd_memory_show,  # pyright: ignore[reportPrivateUsage]
)


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("AGENT6_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_list_empty_is_actionable(env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _cmd_memory_list() == 0
    out = capsys.readouterr().out
    assert "no memories" in out
    assert "memory" in out  # names the dir


def test_add_list_show_rm_roundtrip(env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _cmd_memory_add("build-quirk", "Needs FOO=1.\nMore detail.") == 0
    assert _cmd_memory_list() == 0
    assert "build-quirk: Needs FOO=1." in capsys.readouterr().out
    assert _cmd_memory_show("build-quirk") == 0
    assert capsys.readouterr().out == "Needs FOO=1.\nMore detail.\n"
    assert _cmd_memory_rm("build-quirk") == 0
    capsys.readouterr()
    assert _cmd_memory_list() == 0
    assert "no memories" in capsys.readouterr().out


def test_bad_name_refuses_loud(env: Path) -> None:
    with pytest.raises(MemoryStoreError, match="bad memory name"):
        _cmd_memory_add("Bad Name", "x")
