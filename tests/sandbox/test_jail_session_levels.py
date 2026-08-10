# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""One serving launcher at EVERY isolation level, `none` included.

A session used to be strict-only, so hardened paid Landlock + seccomp setup on
every command and `none` never reached the launcher at all: three execution
paths for one lifecycle. Only strict has the PID namespace, so the two bounds
it provided for free are explicit elsewhere -- the launcher sweeps what it
backgrounded when its request channel closes, and the Python side sweeps each
command's escapees.
"""

from __future__ import annotations

import os
import signal
import time
from pathlib import Path

import pytest

from agent6.sandbox.jail import JailSession
from agent6.types import IsolationLevel, JailPolicy

# hardened and none need no namespaces; the strict case is marked per-test.
_NO_NAMESPACE_LEVELS: tuple[IsolationLevel, ...] = ("hardened", "none")


def _session(cwd: Path, isolation: IsolationLevel) -> JailSession:
    return JailSession.open(
        JailPolicy(cwd=cwd, argv=("true",), isolation=isolation, network="host", timeout_s=30.0)
    )


def _running(pid: int) -> bool:
    """Live, not merely present. `os.kill(pid, 0)` succeeds for a ZOMBIE, and
    the agent is a subreaper, so a swept grandchild lingers unreaped and reads
    as alive to a signal probe."""
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()[0]
    except (OSError, IndexError):
        return False
    return state != "Z"


@pytest.mark.parametrize("isolation", _NO_NAMESPACE_LEVELS)
def test_a_session_serves_commands_without_namespaces(
    tmp_path: Path, isolation: IsolationLevel
) -> None:
    session = _session(tmp_path, isolation)
    try:
        res = session.run(("/bin/sh", "-c", "echo out; echo err >&2; exit 3"))
    finally:
        session.close()
    assert res.returncode == 3
    assert res.stdout.strip() == "out"
    assert "err" in res.stderr


@pytest.mark.parametrize("isolation", _NO_NAMESPACE_LEVELS)
def test_backgrounding_works_without_a_pid_namespace(
    tmp_path: Path, isolation: IsolationLevel
) -> None:
    """The launcher used to refuse a background request outside strict, which
    is why those levels needed a launcher of their own per command."""
    session = _session(tmp_path, isolation)
    try:
        pid = session.start_background(("/bin/sh", "-c", "sleep 60"))
        assert session.status_background(pid).running
        session.stop_background(pid)
        assert not session.status_background(pid).running
    finally:
        session.close()


@pytest.mark.parametrize("isolation", _NO_NAMESPACE_LEVELS)
def test_a_backgrounded_command_dies_with_the_session(
    tmp_path: Path, isolation: IsolationLevel
) -> None:
    """strict's PID namespace does this by construction; without one the
    launcher sweeps the pids it started when its request channel closes, or a
    server would outlive the run that started it."""
    session = _session(tmp_path, isolation)
    pid = session.start_background(("/bin/sh", "-c", "sleep 60"))
    assert _running(pid)
    session.close()
    time.sleep(1.0)
    still = _running(pid)
    if still:  # never leave one behind, even on failure
        os.kill(pid, signal.SIGKILL)
    assert not still


@pytest.mark.parametrize("isolation", _NO_NAMESPACE_LEVELS)
def test_a_setsid_escapee_does_not_outlive_its_command(
    tmp_path: Path, isolation: IsolationLevel
) -> None:
    """A `setsid` child leaves the command's process group, so the launcher's
    killpg misses it. Per-command launchers used to sweep it; the session does
    the same on the levels with no PID namespace to do it for them."""
    session = _session(tmp_path, isolation)
    marker = tmp_path / "escapee.pid"
    try:
        session.run(("/bin/sh", "-c", f"setsid sh -c 'echo $$ > {marker}; sleep 60' & sleep 0.4"))
        assert marker.exists(), "the escapee never started; the test proves nothing"
        pid = int(marker.read_text().strip())
        still = _running(pid)
        if still:
            os.kill(pid, signal.SIGKILL)
        assert not still
    finally:
        session.close()


def test_the_unconfined_level_says_so_on_startup(tmp_path: Path) -> None:
    """`none` reaches the launcher now, so "the launcher ran" no longer implies
    "confinement was applied". It is loud instead: the caller surfaces this as
    `jail.degraded`."""
    session = _session(tmp_path, "none")
    try:
        assert "UNCONFINED" in session.startup_stderr
    finally:
        session.close()


def test_a_confined_level_stays_silent_on_startup(tmp_path: Path) -> None:
    """The unconfined warning must not fire for a level that does confine."""
    session = _session(tmp_path, "hardened")
    try:
        assert "UNCONFINED" not in session.startup_stderr
    finally:
        session.close()
