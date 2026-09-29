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
    from agent6.memory import record_use

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
