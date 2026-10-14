# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 sessions rm` asks before it deletes on a terminal, and never without one."""

from __future__ import annotations

import json
import os
import pathlib
import pty
import select
import signal
import subprocess
import sys
import time

import pytest

from agent6 import git_ops, paths
from agent6.sessions import layout as sessions_layout
from agent6.ui.cli import _steer, main

_ID = "dead-run-BBBB22"


def _git(repo: pathlib.Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _ended_unmerged_run(repo: pathlib.Path) -> sessions_layout.SessionLayout:
    """Seed one ended run whose chain holds a commit its base lacks, with no visible branch."""
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-q", "-m", "init")
    base = _git(repo, "rev-parse", "HEAD")
    work = _git(repo, "commit-tree", "HEAD^{tree}", "-p", "HEAD", "-m", "the run's work")
    _git(repo, "update-ref", git_ops.chain_ref_for(_ID), work)
    layout = sessions_layout.SessionLayout(state_dir=paths.state_dir(repo), session_id=_ID)
    layout.ensure()
    layout.manifest_path.write_text(
        json.dumps(
            {
                "version": 2,
                "session_id": _ID,
                "mode": "run",
                "base_sha": base,
                "base_branch": "main",
                "user_task": "Fix the parser bug",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    layout.logs_path.write_text(
        '{"type": "session.start", "mode": "run"}\n'
        '{"type": "session.end", "reason": "finish_session", "all_passed": true}\n',
        encoding="utf-8",
    )
    return layout


def _rm_under_pty(argv: list[str]) -> tuple[int, int]:
    """Fork a child under a fresh pty that runs `agent6 sessions rm` on that terminal."""
    pid, master = pty.fork()
    if pid == 0:  # pragma: no cover - child process
        code = 1
        try:
            # The pty, in place of the streams pytest captured.
            sys.stdin = open(0, closefd=False)  # noqa: SIM115  # held until the child exits
            sys.stdout = open(1, "w", closefd=False)  # noqa: SIM115  # held until the child exits
            sys.stderr = open(2, "w", closefd=False)  # noqa: SIM115  # held until the child exits
            code = main(["sessions", "rm", *argv])
            sys.stdout.flush()
        finally:
            os._exit(code)
    return pid, master


def _read_pty(master: int, until: bytes, seconds: float = 15.0) -> str:
    """Return the pty's output up to `until`, or what arrived before the deadline or EOF."""
    buf = b""
    deadline = time.monotonic() + seconds
    while until not in buf and time.monotonic() < deadline:
        if select.select([master], [], [], 0.2)[0]:
            try:
                chunk = os.read(master, 4096)
            except OSError:  # the child exited and the pty closed
                break
            if not chunk:
                break
            buf += chunk
    return buf.decode(errors="replace")


def _reap(pid: int) -> int:
    """Return the child's exit code, killing it when it is still at its prompt after 5 s."""
    deadline = time.monotonic() + 5.0
    done, status = os.waitpid(pid, os.WNOHANG)
    while done == 0 and time.monotonic() < deadline:
        time.sleep(0.05)
        done, status = os.waitpid(pid, os.WNOHANG)
    if done == 0:
        os.kill(pid, signal.SIGKILL)
        _, status = os.waitpid(pid, 0)
    return os.waitstatus_to_exitcode(status)


@pytest.mark.parametrize(("typed", "deleted"), [(b"n\n", False), (b"\n", False), (b"y\n", True)])
def test_a_bare_rm_on_a_terminal_names_the_session_and_asks(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, typed: bytes, deleted: bool
) -> None:
    """A bare `sessions rm` on a terminal deleted the newest session with no question.

    It names the session it resolved to, its task, its state and its unmerged commits, and
    deletes on `y` alone.
    """
    monkeypatch.chdir(tmp_path)
    layout = _ended_unmerged_run(tmp_path)
    pid, master = _rm_under_pty([])
    try:
        asked = _read_pty(master, b"[y/N]: ")
        assert asked.endswith("[y/N]: "), asked
        for shown in (_ID, "Fix the parser bug", "passed", "not merged"):
            assert shown in asked, asked
        os.write(master, typed)
        after = _read_pty(master, b"\x00", seconds=10.0)  # drain to the child's exit
        assert _reap(pid) == 0
    finally:
        os.close(master)
    assert layout.session_dir.exists() is not deleted
    assert bool(git_ops.chain_tip(tmp_path, git_ops.chain_ref_for(_ID))) is not deleted
    assert (f"removed {_ID}" in after) is deleted
    assert ("nothing was deleted" in after) is not deleted


@pytest.mark.parametrize("flag", ["--yes", "-y"])
def test_rm_yes_deletes_on_a_terminal_without_asking(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, flag: str
) -> None:
    """`--yes` and `-y` skip the question."""
    monkeypatch.chdir(tmp_path)
    layout = _ended_unmerged_run(tmp_path)
    pid, master = _rm_under_pty([flag, _ID])
    try:
        out = _read_pty(master, b"\x00", seconds=10.0)
        assert _reap(pid) == 0
    finally:
        os.close(master)
    assert "[y/N]" not in out
    assert f"removed {_ID}" in out
    assert not layout.session_dir.exists()


def test_rm_asks_on_a_terminal_before_clearing_the_asks(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`sessions rm --asks` on a terminal names how many asks go and where from, then asks."""
    monkeypatch.chdir(tmp_path)
    asks = paths.state_dir(tmp_path) / "sessions" / "asks"
    for name in ("ask-one", "ask-two"):
        (asks / name).mkdir(parents=True)
    pid, master = _rm_under_pty(["--asks"])
    try:
        asked = _read_pty(master, b"[y/N]: ")
        assert "2 asks" in asked and str(tmp_path) in asked, asked
        os.write(master, b"n\n")
        _read_pty(master, b"\x00", seconds=10.0)
        assert _reap(pid) == 0
    finally:
        os.close(master)
    assert asks.is_dir()


def test_rm_without_a_terminal_deletes_without_asking(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """With no terminal on stdin and stdout `rm` never asks, so a script keeps working."""
    monkeypatch.chdir(tmp_path)
    layout = _ended_unmerged_run(tmp_path)

    def never(*_args: object, **_kwargs: object) -> str | None:
        raise AssertionError("rm asked with no terminal")

    monkeypatch.setattr(_steer, "tty_prompt", never)
    assert main(["sessions", "rm"]) == 0
    assert f"removed {_ID}" in capsys.readouterr().out
    assert not layout.session_dir.exists()
