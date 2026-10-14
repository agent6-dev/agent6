# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 memory rm` asks before it deletes on a terminal, and never without one."""

from __future__ import annotations

import os
import pathlib

import pytest

from agent6 import memory, paths, portable
from agent6.ui.cli import _steer, main
from tests.unit import test_sessions_rm_confirm as pty_cli

pytestmark = pytest.mark.filterwarnings(
    "ignore:This process.*is multi-threaded, use of fork:DeprecationWarning"
)


def _fact(repo: pathlib.Path) -> pathlib.Path:
    """Write one memory, and return its file."""
    return memory.add(paths.state_dir(repo), "build-quirk", "Needs FOO=1.\nMore detail.")


@pytest.mark.parametrize(("typed", "deleted"), [(b"n\n", False), (b"\n", False), (b"y\n", True)])
def test_memory_rm_on_a_terminal_names_the_fact_and_asks(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, typed: bytes, deleted: bool
) -> None:
    """`memory rm` on a terminal deleted the fact with no question.

    It names the fact and its first line, and deletes on `y` alone.
    """
    monkeypatch.chdir(tmp_path)
    fact = _fact(tmp_path)
    pid, master = pty_cli.cli_under_pty(["memory", "rm", "build-quirk"])
    try:
        asked = pty_cli.read_pty(master, b"[y/N]: ")
        assert asked.endswith("[y/N]: "), asked
        assert "memory rm will delete build-quirk" in asked and "Needs FOO=1." in asked, asked
        os.write(master, typed)
        after = pty_cli.read_pty(master, b"\x00", seconds=10.0)  # drain to the child's exit
        assert pty_cli.reap(pid) == 0
    finally:
        os.close(master)
    assert fact.exists() is not deleted
    assert ("removed build-quirk" in after) is deleted
    assert ("nothing was deleted" in after) is not deleted


@pytest.mark.parametrize("flag", ["--yes", "-y"])
def test_memory_rm_yes_deletes_on_a_terminal_without_asking(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, flag: str
) -> None:
    """`--yes` and `-y` skip the question."""
    monkeypatch.chdir(tmp_path)
    fact = _fact(tmp_path)
    pid, master = pty_cli.cli_under_pty(["memory", "rm", flag, "build-quirk"])
    try:
        out = pty_cli.read_pty(master, b"\x00", seconds=10.0)
        assert pty_cli.reap(pid) == 0
    finally:
        os.close(master)
    assert "[y/N]" not in out and "removed build-quirk" in out
    assert not fact.exists()


def test_memory_rm_without_a_terminal_deletes_without_asking(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """With no controlling terminal `memory rm` never asks, so a script keeps working."""
    monkeypatch.chdir(tmp_path)
    fact = _fact(tmp_path)
    monkeypatch.setattr(portable, "has_controlling_tty", lambda: False)

    def never(*_args: object, **_kwargs: object) -> str | None:
        raise AssertionError("memory rm asked with no terminal")

    monkeypatch.setattr(_steer, "tty_prompt", never)
    assert main(["memory", "rm", "build-quirk"]) == 0
    assert "removed build-quirk" in capsys.readouterr().out
    assert not fact.exists()


def test_memory_rm_help_states_the_question(capsys: pytest.CaptureFixture[str]) -> None:
    """`memory rm --help` says when it asks and names `--yes`."""
    with pytest.raises(SystemExit):
        main(["memory", "rm", "--help"])
    out = " ".join(capsys.readouterr().out.split())
    assert "asks [y/N] first" in out and "--yes" in out and "with no terminal" in out
