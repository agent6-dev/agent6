# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 memory` CLI: add/list/show/rm over the file-per-fact store."""

from __future__ import annotations

import pathlib

import pytest

from agent6 import memory, paths
from agent6.ui.cli import memory_cmds


@pytest.fixture(autouse=True)
def _no_controlling_terminal(  # pyright: ignore[reportUnusedFunction]
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run as CI does, with no controlling terminal: `rm` without `--yes` asks on one."""
    monkeypatch.setattr("agent6.portable.has_controlling_tty", lambda: False)


@pytest.fixture
def env(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_list_empty_is_actionable(env: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert memory_cmds._cmd_memory_list() == 0
    out = capsys.readouterr().out
    assert "no memories" in out
    assert "memory" in out  # names the dir


def test_add_list_show_rm_roundtrip(env: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert memory_cmds._cmd_memory_add("build-quirk", "Needs FOO=1.\nMore detail.") == 0
    assert memory_cmds._cmd_memory_list() == 0
    assert "build-quirk: Needs FOO=1." in capsys.readouterr().out
    assert memory_cmds._cmd_memory_show("build-quirk") == 0
    assert capsys.readouterr().out == "Needs FOO=1.\nMore detail.\n"
    assert memory_cmds._cmd_memory_rm("build-quirk") == 0
    capsys.readouterr()
    assert memory_cmds._cmd_memory_list() == 0
    assert "no memories" in capsys.readouterr().out


def test_list_names_the_store_files_that_are_not_facts(
    env: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A store entry outside the fact rule (a stale `MEMORY.md`) is state the list shows."""
    memory_cmds._cmd_memory_add("a-fact", "A first.")
    store = memory.memory_dir(paths.state_dir(pathlib.Path.cwd()))
    (store / "MEMORY.md").write_text("- a-fact: An older line.\n", encoding="utf-8")
    (store / "notes.txt").write_text("x\n", encoding="utf-8")
    capsys.readouterr()
    assert memory_cmds._cmd_memory_list() == 0
    out = capsys.readouterr().out
    assert out.startswith("- a-fact: A first.\n    written ")
    assert out.endswith("\nignored (not a fact file): MEMORY.md, notes.txt\n")
    assert "An older line" not in out


def test_bad_name_refuses_loud(env: pathlib.Path) -> None:
    with pytest.raises(memory.MemoryStoreError, match="bad memory name"):
        memory_cmds._cmd_memory_add("Bad Name", "x")


@pytest.mark.parametrize("name", ["-fact", "a" * 65])
def test_add_rejects_names_outside_help_limits(env: pathlib.Path, name: str) -> None:
    """A leading dash or more than 64 characters is not a memory name."""
    with pytest.raises(memory.MemoryStoreError, match="bad memory name"):
        memory_cmds._cmd_memory_add(name, "A fact.")


@pytest.mark.parametrize("name", ["a", "0", "a" * 64, "0-fact", "fact-"])
def test_add_accepts_name_boundaries(env: pathlib.Path, name: str) -> None:
    """The documented endpoints and a digit prefix are accepted."""
    assert memory_cmds._cmd_memory_add(name, "A fact.") == 0


def test_decisions_prints_the_rulings_or_says_none(
    env: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert memory_cmds._cmd_memory_decisions() == 0
    assert "no rulings recorded" in capsys.readouterr().out
    memory.record_decision(
        paths.state_dir(pathlib.Path.cwd()), question="Q?", answer="A", session="s", when=0
    )
    assert memory_cmds._cmd_memory_decisions() == 0
    assert capsys.readouterr().out == "- 1970-01-01 00:00Z [s] Q: Q?\n  A: A\n"


def test_memory_commands_move_an_old_rulings_file_out_of_the_store_and_say_so(
    env: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A `memory/DECISIONS.md` left by an older agent6 moves to `decisions.md` once.

    The note goes to stderr, so a piped `memory decisions` or `memory show` stays clean; a
    stray one beside an existing `decisions.md` is reported the same way and never merged.
    """
    state = paths.state_dir(pathlib.Path.cwd())
    legacy = memory.memory_dir(state) / "DECISIONS.md"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("- old ruling\n", encoding="utf-8")
    assert memory_cmds._cmd_memory_decisions() == 0
    out, err = capsys.readouterr()
    assert out == "- old ruling\n"
    assert err == f"operator rulings moved from {legacy} to {state / 'decisions.md'}\n"
    assert memory_cmds._cmd_memory_list() == 0
    assert "moved" not in "".join(capsys.readouterr())
    legacy.write_text("- planted\n", encoding="utf-8")
    assert memory_cmds._cmd_memory_decisions() == 0
    out, err = capsys.readouterr()
    assert out == "- old ruling\n"
    assert err.startswith(f"ignored a stray {legacy}:")


def test_decisions_reads_an_empty_file_as_none_recorded(
    env: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The empty `decisions.md` a run start creates prints as no rulings, not as nothing."""
    memory.ensure_decisions_file(paths.state_dir(pathlib.Path.cwd()))
    assert memory_cmds._cmd_memory_decisions() == 0
    assert "no rulings recorded" in capsys.readouterr().out


def test_list_shows_who_wrote_and_read_each_fact(
    env: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The use record prints under its entry.

    An operator-added fact reads `written ... by operator, never read`; a run's reads follow.
    """
    assert memory_cmds._cmd_memory_add("build-quirk", "Needs FOO=1.") == 0
    state = paths.state_dir(pathlib.Path.cwd())
    memory.record_use(state, session="run-a", wrote=(), read={"build-quirk": 3}, when=86400.0)
    capsys.readouterr()
    assert memory_cmds._cmd_memory_list() == 0
    out = capsys.readouterr().out
    assert "- build-quirk: Needs FOO=1.\n" in out
    assert "    written " in out
    assert "by operator" in out
    assert "read 3 times, last 1970-01-02 by run-a" in out
    # A fact the record never saw (written by hand) still reads as never read.
    (memory.memory_dir(state) / "by-hand.md").write_text("By hand.\n", encoding="utf-8")
    assert memory_cmds._cmd_memory_list() == 0
    assert "- by-hand: By hand.\n    never read\n" in capsys.readouterr().out


def test_format_use_says_each_state_plainly() -> None:
    first, second = (
        memory.Touch("run-a", "2026-01-01 00:00Z"),
        memory.Touch("run-b", "2026-01-02 00:00Z"),
    )
    assert memory_cmds.format_use(memory.MemoryUse()) == "never read"
    assert memory_cmds.format_use(memory.MemoryUse(created=first, writes=(first,))) == (
        "written 2026-01-01 by run-a, never read"
    )
    assert memory_cmds.format_use(
        memory.MemoryUse(
            created=first,
            writes=(first, second),
            reads=1,
            last_read=memory.Touch("run-c", "2026-01-03 00:00Z"),
        )
    ) == (
        "written 2026-01-01 by run-a, edited 2026-01-02 by run-b,"
        " read once, last 2026-01-03 by run-c"
    )
    # A fact the record never saw created: its writes are edits.
    assert (
        memory_cmds.format_use(memory.MemoryUse(writes=(first, second)))
        == "edited 2026-01-02 by run-b, never read"
    )
    # A count without a last reader (a hand-edited record) still counts.
    assert memory_cmds.format_use(memory.MemoryUse(reads=2)) == "read 2 times"
