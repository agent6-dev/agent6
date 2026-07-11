# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 memory` CLI: add/list/invalidate output."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.ui.cli.memory_cmds import (
    _cmd_memory_add,  # pyright: ignore[reportPrivateUsage]
    _cmd_memory_invalidate,  # pyright: ignore[reportPrivateUsage]
    _cmd_memory_list,  # pyright: ignore[reportPrivateUsage]
)


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("AGENT6_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_list_empty_is_actionable(env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _cmd_memory_list(None, include_invalidated=False) == 0
    out = capsys.readouterr().out
    assert "no memories yet" in out
    assert "agent6 memory add" in out


def test_list_groups_by_scope_and_shows_body_and_id(
    env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _cmd_memory_add("facts", "CI runs on the self-hosted runner") == 0
    assert _cmd_memory_add("preferences", "Prefers tabs over spaces") == 0
    capsys.readouterr()
    assert _cmd_memory_list(None, include_invalidated=False) == 0
    out = capsys.readouterr().out
    assert "facts" in out and "preferences" in out
    assert "CI runs on the self-hosted runner" in out
    assert "Prefers tabs over spaces" in out
    # the scope header prints once, not once per entry
    assert out.count("facts") == 1


def test_list_hides_invalidated_until_asked(env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _cmd_memory_add("facts", "stale note") == 0
    out = capsys.readouterr().out
    mem_id = out.split()[1]
    assert _cmd_memory_invalidate(mem_id, "outdated") == 0
    capsys.readouterr()
    assert _cmd_memory_list(None, include_invalidated=False) == 0
    assert "no active memories" in capsys.readouterr().out
    assert _cmd_memory_list(None, include_invalidated=True) == 0
    shown = capsys.readouterr().out
    assert "stale note" in shown
    assert "invalidated" in shown
