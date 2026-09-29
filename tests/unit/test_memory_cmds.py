# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 memory` CLI: add/list/show/rm over the file-per-fact store."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.memory import MemoryStoreError
from agent6.paths import state_dir
from agent6.ui.cli.memory_cmds import (
    _cmd_memory_add,  # pyright: ignore[reportPrivateUsage]
    _cmd_memory_list,  # pyright: ignore[reportPrivateUsage]
    _cmd_memory_rm,  # pyright: ignore[reportPrivateUsage]
    _cmd_memory_show,  # pyright: ignore[reportPrivateUsage]
)


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
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


def test_decisions_prints_the_rulings_or_says_none(
    env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from agent6.memory import record_decision
    from agent6.ui.cli.memory_cmds import (
        _cmd_memory_decisions,  # pyright: ignore[reportPrivateUsage]
    )

    assert _cmd_memory_decisions() == 0
    assert "no rulings recorded" in capsys.readouterr().out
    record_decision(state_dir(Path.cwd()), question="Q?", answer="A", session="s", when=0)
    assert _cmd_memory_decisions() == 0
    assert capsys.readouterr().out == "- 1970-01-01 00:00Z [s] Q: Q?\n  A: A\n"


def test_rm_keeps_the_index_bytes_it_does_not_touch(tmp_path: Path) -> None:
    """The index rewrite read through the replacing decoder, so `memory rm`
    turned every byte that is not UTF-8 anywhere in the file into U+FFFD."""
    from agent6.memory import index_path, remove

    idx = index_path(tmp_path)
    idx.parent.mkdir(parents=True, exist_ok=True)
    idx.write_bytes(b"# Memory index\n\n- one: first\n- two: second\nnote: caf\xe9 build\n")
    (idx.parent / "one.md").write_text("first\n", encoding="utf-8")
    remove(tmp_path, "one")
    assert idx.read_bytes() == b"# Memory index\n\n- two: second\nnote: caf\xe9 build\n"
    assert not (idx.parent / "one.md").exists()


def test_list_shows_who_wrote_and_read_each_fact(
    env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The use record prints under its entry: an operator-added fact reads
    `written ... by operator, never read`; a run's reads follow."""
    from agent6.memory import memory_dir, record_use

    assert _cmd_memory_add("build-quirk", "Needs FOO=1.") == 0
    state = state_dir(Path.cwd())
    record_use(state, session="run-a", wrote=(), read={"build-quirk": 3}, when=86400.0)
    capsys.readouterr()
    assert _cmd_memory_list() == 0
    out = capsys.readouterr().out
    assert "- build-quirk: Needs FOO=1.\n" in out
    assert "    written " in out
    assert "by operator" in out
    assert "read 3 times, last 1970-01-02 by run-a" in out
    # A line whose file never landed (a model's write refused after its index
    # edit) says so, in place of a use line for a fact that will not open.
    with (memory_dir(state) / "MEMORY.md").open("a", encoding="utf-8") as fh:
        fh.write("- ghost: A fact with no file.\n")
    assert _cmd_memory_list() == 0
    assert "- ghost: A fact with no file.\n    no file:" in capsys.readouterr().out
    # A fact the record never saw (written by hand) still reads as never read.
    (memory_dir(state) / "by-hand.md").write_text("By hand.\n", encoding="utf-8")
    with (memory_dir(state) / "MEMORY.md").open("a", encoding="utf-8") as fh:
        fh.write("- by-hand: By hand.\n")
    assert _cmd_memory_list() == 0
    assert "- by-hand: By hand.\n    never read\n" in capsys.readouterr().out


def test_list_names_orphans_when_the_index_is_absent_or_blank(
    env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The one orphan whose line went was the only entry: the list said
    "no memories" and named nothing to prune."""
    from agent6.memory import index_path, memory_dir

    _cmd_memory_add("only", "The only fact.")
    state = state_dir(Path.cwd())
    index_path(state).unlink()
    assert (memory_dir(state) / "only.md").is_file()
    capsys.readouterr()
    assert _cmd_memory_list() == 0
    out = capsys.readouterr().out
    assert "no memories" in out
    assert "not in the index (no run sees them; `memory rm` deletes): only" in out


def test_format_use_says_each_state_plainly() -> None:
    from agent6.memory import MemoryUse, Touch
    from agent6.ui.cli.memory_cmds import format_use

    first, second = Touch("run-a", "2026-01-01 00:00Z"), Touch("run-b", "2026-01-02 00:00Z")
    assert format_use(MemoryUse()) == "never read"
    assert format_use(MemoryUse(created=first, writes=(first,))) == (
        "written 2026-01-01 by run-a, never read"
    )
    assert format_use(
        MemoryUse(
            created=first,
            writes=(first, second),
            reads=1,
            last_read=Touch("run-c", "2026-01-03 00:00Z"),
        )
    ) == (
        "written 2026-01-01 by run-a, edited 2026-01-02 by run-b,"
        " read once, last 2026-01-03 by run-c"
    )
    # A fact the record never saw created: its writes are edits.
    assert format_use(MemoryUse(writes=(first, second))) == "edited 2026-01-02 by run-b, never read"
    # A count without a last reader (a hand-edited record) still counts.
    assert format_use(MemoryUse(reads=2)) == "read 2 times"


def test_list_names_the_files_the_index_no_longer_lists(
    env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A run that drops a fact's index line leaves its file behind, invisible
    to every later run and to `memory list`; the operator pruning the store
    saw nothing to prune. The list names such files under the index."""
    from agent6.memory import index_path, memory_dir

    _cmd_memory_add("kept", "A fact that stays.")
    _cmd_memory_add("dropped", "A fact whose line went.")
    state = state_dir(Path.cwd())
    lines = index_path(state).read_text(encoding="utf-8").splitlines()
    index_path(state).write_text(
        "\n".join(ln for ln in lines if not ln.startswith("- dropped:")) + "\n", encoding="utf-8"
    )
    assert (memory_dir(state) / "dropped.md").is_file()
    capsys.readouterr()
    assert _cmd_memory_list() == 0
    out = capsys.readouterr().out
    assert "- kept: A fact that stays." in out
    assert "- dropped:" not in out
    assert "not in the index (no run sees them; `memory rm` deletes): dropped" in out
