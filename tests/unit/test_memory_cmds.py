# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 memory` CLI: add/list/show/rm over the file-per-fact store."""

from __future__ import annotations

import pathlib

import pytest

from agent6 import memory, paths
from agent6.ui.cli import memory_cmds


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


def test_bad_name_refuses_loud(env: pathlib.Path) -> None:
    with pytest.raises(memory.MemoryStoreError, match="bad memory name"):
        memory_cmds._cmd_memory_add("Bad Name", "x")


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


def test_rm_keeps_the_index_bytes_it_does_not_touch(tmp_path: pathlib.Path) -> None:
    """`memory rm` keeps the index bytes it does not touch.

    An index rewrite read through the replacing decoder turns every byte that is not UTF-8 anywhere
    in the file into U+FFFD.
    """
    idx = memory.index_path(tmp_path)
    idx.parent.mkdir(parents=True, exist_ok=True)
    idx.write_bytes(b"# Memory index\n\n- one: first\n- two: second\nnote: caf\xe9 build\n")
    (idx.parent / "one.md").write_text("first\n", encoding="utf-8")
    memory.remove(tmp_path, "one")
    assert idx.read_bytes() == b"# Memory index\n\n- two: second\nnote: caf\xe9 build\n"
    assert not (idx.parent / "one.md").exists()


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
    # A line whose file never landed (a model's write refused after its index
    # edit) says so, in place of a use line for a fact that will not open.
    with (memory.memory_dir(state) / "MEMORY.md").open("a", encoding="utf-8") as fh:
        fh.write("- ghost: A fact with no file.\n")
    assert memory_cmds._cmd_memory_list() == 0
    assert "- ghost: A fact with no file.\n    no file:" in capsys.readouterr().out
    # A fact the record never saw (written by hand) still reads as never read.
    (memory.memory_dir(state) / "by-hand.md").write_text("By hand.\n", encoding="utf-8")
    with (memory.memory_dir(state) / "MEMORY.md").open("a", encoding="utf-8") as fh:
        fh.write("- by-hand: By hand.\n")
    assert memory_cmds._cmd_memory_list() == 0
    assert "- by-hand: By hand.\n    never read\n" in capsys.readouterr().out


def test_list_names_orphans_when_the_index_is_absent_or_blank(
    env: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The one orphan whose line went was the only entry.

    The list said "no memories" and named nothing to prune.
    """
    memory_cmds._cmd_memory_add("only", "The only fact.")
    state = paths.state_dir(pathlib.Path.cwd())
    memory.index_path(state).unlink()
    assert (memory.memory_dir(state) / "only.md").is_file()
    capsys.readouterr()
    assert memory_cmds._cmd_memory_list() == 0
    out = capsys.readouterr().out
    assert "no memories" in out
    assert "not in the index (no run sees them; `memory rm` deletes): only" in out


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


def test_list_names_the_files_the_index_no_longer_lists(
    env: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`memory list` names the files the index does not list.

    A run that drops a fact's index line leaves its file behind, invisible to every later run; the
    operator pruning the store needs to see it under the index.
    """
    memory_cmds._cmd_memory_add("kept", "A fact that stays.")
    memory_cmds._cmd_memory_add("dropped", "A fact whose line went.")
    state = paths.state_dir(pathlib.Path.cwd())
    lines = memory.index_path(state).read_text(encoding="utf-8").splitlines()
    memory.index_path(state).write_text(
        "\n".join(ln for ln in lines if not ln.startswith("- dropped:")) + "\n", encoding="utf-8"
    )
    assert (memory.memory_dir(state) / "dropped.md").is_file()
    capsys.readouterr()
    assert memory_cmds._cmd_memory_list() == 0
    out = capsys.readouterr().out
    assert "- kept: A fact that stays." in out
    assert "- dropped:" not in out
    assert "not in the index (no run sees them; `memory rm` deletes): dropped" in out
