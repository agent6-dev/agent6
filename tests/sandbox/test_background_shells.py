# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Background commands must never lie about being alive.

The failure these pin is the one that makes a background feature useless: a
command dies, nothing says so, and the agent either waits forever or keeps
reporting a shell that is already gone.
"""

from __future__ import annotations

import os
import shutil
import signal
import threading
import time
from pathlib import Path

import pytest

from agent6.sandbox.jail import locate_jail_binary, run_in_jail
from agent6.tools.background import BackgroundError, BackgroundShells
from agent6.types import IsolationLevel, JailPolicy

pytestmark = pytest.mark.needs_namespaces


@pytest.fixture
def shells(tmp_path: Path) -> BackgroundShells:
    if locate_jail_binary() is None:
        pytest.skip("no agent6-jail binary")
    return BackgroundShells(tmp_path / "shells")


def _policy_for(cwd: Path, isolation: IsolationLevel = "hardened"):
    def build(argv: tuple[str, ...], rw: tuple[Path, ...]) -> JailPolicy:
        return JailPolicy(
            cwd=cwd, argv=argv, isolation=isolation, extra_rw_paths=rw, timeout_s=60.0
        )

    return build


def _wait_state(shells: BackgroundShells, shell_id: str, state: str, timeout: float = 15.0) -> str:
    """Poll until *shell_id* reaches *state*. Returns the state actually seen."""
    deadline = time.monotonic() + timeout
    seen = ""
    while time.monotonic() < deadline:
        seen = next(v.state for v in shells.roster() if v.id == shell_id)
        if seen == state:
            return seen
        time.sleep(0.05)
    return seen


def test_a_command_that_exits_on_its_own_reports_its_code(
    shells: BackgroundShells, tmp_path: Path
) -> None:
    """The core lie: a background command ends and the roster still says
    "running". State comes from the process, the code from the launcher."""
    view = shells.start(("/bin/sh", "-c", "echo bye; exit 7"), _policy_for(tmp_path))
    assert view.state == "running"
    assert _wait_state(shells, view.id, "exited") == "exited"
    after, output = shells.read(view.id, tail_lines=50)
    assert after.returncode == 7
    assert "bye" in output


def test_a_command_killed_from_outside_is_never_reported_running(
    shells: BackgroundShells, tmp_path: Path
) -> None:
    """A crash agent6 did not ask for (an OOM kill, an operator's kill -9)
    still has to read as over."""
    view = shells.start(("/bin/sh", "-c", "echo up; sleep 300"), _policy_for(tmp_path))
    deadline = time.monotonic() + 15.0
    while "up" not in shells.read(view.id, tail_lines=10)[1]:
        assert time.monotonic() < deadline, "the command never started"
        time.sleep(0.05)
    pid = next(iter(_launcher_pids(shells, view.id)))
    os.killpg(os.getpgid(pid), signal.SIGKILL)
    assert _wait_state(shells, view.id, "died") in {"died", "exited"}
    assert not any(v.state == "running" for v in shells.roster())


def test_reading_a_live_command_never_blocks(shells: BackgroundShells, tmp_path: Path) -> None:
    """There is no wait: a read of a command that will run for five minutes
    returns immediately, so the agent cannot get stuck on it."""
    view = shells.start(("/bin/sh", "-c", "sleep 300"), _policy_for(tmp_path))
    start = time.monotonic()
    for _ in range(5):
        shells.read(view.id, tail_lines=10)
        shells.roster()
    assert time.monotonic() - start < 2.0
    shells.stop(view.id)


def test_stopped_and_died_are_different_words(shells: BackgroundShells, tmp_path: Path) -> None:
    """ "I killed it" and "it died on me" must not read the same, or an
    unexplained disappearance looks like a deliberate stop."""
    stopped = shells.start(("/bin/sh", "-c", "sleep 300"), _policy_for(tmp_path))
    failed = shells.start(("/bin/sh", "-c", "exit 3"), _policy_for(tmp_path))
    assert shells.stop(stopped.id).state == "stopped"
    assert _wait_state(shells, failed.id, "exited") == "exited"
    words = {v.id: v.state for v in shells.roster()}
    assert words == {stopped.id: "stopped", failed.id: "exited"}


def test_the_roster_rides_on_every_answer(shells: BackgroundShells, tmp_path: Path) -> None:
    """Reading one command reports them all, so a second one dying is seen
    without having to ask about it."""
    quiet = shells.start(("/bin/sh", "-c", "sleep 300"), _policy_for(tmp_path))
    doomed = shells.start(("/bin/sh", "-c", "exit 1"), _policy_for(tmp_path))
    assert _wait_state(shells, doomed.id, "exited") == "exited"
    roster = shells.read(quiet.id, tail_lines=5)[0]
    assert roster.state == "running"
    assert [v.state for v in shells.roster() if v.id == doomed.id] == ["exited"]
    shells.stop_all()


def test_stop_all_kills_everything_and_the_processes_are_gone(
    shells: BackgroundShells, tmp_path: Path
) -> None:
    """Run-end teardown: nothing a run started may outlive it."""
    views = [
        shells.start(("/bin/sh", "-c", f"echo {i}; sleep 300"), _policy_for(tmp_path))
        for i in range(3)
    ]
    pids = {p for v in views for p in _launcher_pids(shells, v.id)}
    assert shells.stop_all()
    for pid in pids:
        deadline = time.monotonic() + 10.0
        while _alive(pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not _alive(pid), f"launcher {pid} survived teardown"
    assert not any(v.state == "running" for v in shells.roster())
    assert shells.stop_all() == []  # idempotent


def test_an_unknown_id_names_what_exists(shells: BackgroundShells, tmp_path: Path) -> None:
    shells.start(("/bin/true",), _policy_for(tmp_path))
    with pytest.raises(BackgroundError, match="bg99"):
        shells.read("bg99", tail_lines=5)


def test_output_survives_the_command(shells: BackgroundShells, tmp_path: Path) -> None:
    """The log is a file, so what a command printed is still readable after it
    is gone -- the point of not streaming through a pipe."""
    view = shells.start(("/bin/sh", "-c", "echo first; echo second; exit 0"), _policy_for(tmp_path))
    assert _wait_state(shells, view.id, "exited") == "exited"
    _after, output = shells.read(view.id, tail_lines=50)
    assert "first" in output and "second" in output


@pytest.mark.skipif(shutil.which("unshare") is None, reason="needs userns for strict")
def test_strict_confines_a_background_command_too(shells: BackgroundShells, tmp_path: Path) -> None:
    """A detached command is the same jail as a foreground one: no weaker
    isolation just because nobody is waiting on it."""
    view = shells.start(
        ("/bin/sh", "-c", "echo escaped > /etc/agent6-bg-escape"), _policy_for(tmp_path, "strict")
    )
    assert _wait_state(shells, view.id, "exited") == "exited"
    assert not Path("/etc/agent6-bg-escape").exists()
    assert next(v for v in shells.roster() if v.id == view.id).returncode != 0


def test_a_foreground_commands_sweep_spares_a_background_one(
    shells: BackgroundShells, tmp_path: Path
) -> None:
    """The escapee sweep kills whatever a jailed command leaves behind, and a
    background command is precisely that shape -- a live jail nobody is
    waiting on. It is a deliberate child, so it must survive every later
    command."""
    view = shells.start(("/bin/sh", "-c", "echo up; sleep 300"), _policy_for(tmp_path))
    deadline = time.monotonic() + 15.0
    while "up" not in shells.read(view.id, tail_lines=10)[1]:
        assert time.monotonic() < deadline, "the command never started"
        time.sleep(0.05)
    for _ in range(3):
        res = run_in_jail(
            JailPolicy(cwd=tmp_path, argv=("/bin/true",), isolation="hardened", timeout_s=10.0)
        )
        assert res.returncode == 0
    assert next(v for v in shells.roster() if v.id == view.id).state == "running"
    shells.stop_all()


def test_a_command_started_mid_sweep_window_is_spared(
    shells: BackgroundShells, tmp_path: Path
) -> None:
    """Started WHILE a foreground command is in flight, a background command is
    not in that command's before-snapshot, so only its registration as a live
    launcher keeps the sweep off it."""
    started: list[str] = []

    def start_midway() -> None:
        time.sleep(0.4)
        started.append(shells.start(("/bin/sh", "-c", "sleep 300"), _policy_for(tmp_path)).id)

    thread = threading.Thread(target=start_midway)
    thread.start()
    res = run_in_jail(
        JailPolicy(
            cwd=tmp_path, argv=("/bin/sh", "-c", "sleep 2"), isolation="hardened", timeout_s=20.0
        )
    )
    thread.join()
    assert res.returncode == 0
    assert next(v for v in shells.roster() if v.id == started[0]).state == "running"
    shells.stop_all()


def _launcher_pids(shells: BackgroundShells, shell_id: str) -> set[int]:
    shell = shells._get(shell_id)  # pyright: ignore[reportPrivateUsage]
    return {shell.job.pid}


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
