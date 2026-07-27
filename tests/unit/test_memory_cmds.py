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


def test_pin_unpin_roundtrip_and_list_marker(env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from agent6.memory import list_entries
    from agent6.ui.cli._common import _state_dir  # pyright: ignore[reportPrivateUsage]
    from agent6.ui.cli.memory_cmds import (
        _cmd_memory_pin,  # pyright: ignore[reportPrivateUsage]
        _cmd_memory_unpin,  # pyright: ignore[reportPrivateUsage]
    )

    assert _cmd_memory_add("decisions", "squash merges only") == 0
    mem_id = list_entries(_state_dir(Path.cwd()))[0].id
    capsys.readouterr()
    assert _cmd_memory_pin(mem_id) == 0
    assert "pinned" in capsys.readouterr().out
    assert _cmd_memory_list(None, include_invalidated=False) == 0
    assert "[pinned]" in capsys.readouterr().out
    assert _cmd_memory_unpin(mem_id) == 0
    assert "unpinned" in capsys.readouterr().out
    # errors are loud and non-zero
    assert _cmd_memory_unpin(mem_id) == 2
    assert "not pinned" in capsys.readouterr().err


def test_list_warns_when_pins_exceed_the_block_cap(
    env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from agent6.memory import list_entries
    from agent6.ui.cli._common import _state_dir  # pyright: ignore[reportPrivateUsage]
    from agent6.ui.cli.memory_cmds import (
        _cmd_memory_pin,  # pyright: ignore[reportPrivateUsage]
    )

    for i in range(12):
        assert _cmd_memory_add("facts", f"pin {i} " + "y" * 1150) == 0
    for e in list_entries(_state_dir(Path.cwd())):
        _cmd_memory_pin(e.id)
    capsys.readouterr()
    assert _cmd_memory_list(None, include_invalidated=False) == 0
    out = capsys.readouterr().out
    assert "exceed the memory block cap" in out
    assert "oldest pinned" in out
    # The cap is global: a scope-filtered listing must still warn.
    assert _cmd_memory_list("decisions", include_invalidated=False) == 0
    assert "exceed the memory block cap" in capsys.readouterr().out


def test_list_of_one_scope_reports_an_unreadable_other_scope(
    env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The over-cap warning re-reads EVERY scope, and that read sat outside the
    error guard: `memory list --scope facts` died with a raw traceback when an
    unrelated scope's file was unreadable, where it used to print the listing."""
    from agent6.ui.cli._common import _state_dir  # pyright: ignore[reportPrivateUsage]

    assert _cmd_memory_add("facts", "a fact") == 0
    bad = _state_dir(Path.cwd()) / "memories" / "decisions.md"
    bad.write_bytes(b"id: x\nscope: decisions\n\xff\xfe body\n")

    assert _cmd_memory_list("facts", include_invalidated=False) == 2
    assert "MEMORY ERROR" in capsys.readouterr().err
