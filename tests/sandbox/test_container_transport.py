# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
from __future__ import annotations

import os
import pathlib
import subprocess
import time

import pytest

from agent6 import kinds
from agent6.sandbox import container, jail, jail_session
from tests.sandbox import conftest


def _alive(pid: int) -> bool:
    try:
        state = pathlib.Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()[0]
    except (OSError, IndexError):
        return False
    return state != "Z"


def _wait_gone(pid: int, seconds: float = 5.0) -> bool:
    deadline = time.monotonic() + seconds
    while _alive(pid):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)
    return True


def _pid_of(session: jail_session.JailSession, marker: pathlib.Path, seconds: float = 5.0) -> int:
    """Wait for a command inside to write its pid to a workspace file the host reads."""
    deadline = time.monotonic() + seconds
    while not marker.exists() or not marker.read_text().strip():
        assert time.monotonic() < deadline, "the command never wrote its pid"
        time.sleep(0.05)
    return int(marker.read_text().strip())


def test_a_backgrounded_command_dies_with_the_prepared_session(target: conftest.Target) -> None:
    """Closing the session ends its jobs through the supervisor, with no host pid scan."""
    marker = target.discovery.target.host_root / "bg.pid"
    with conftest.opened(target) as session:
        handle = session.start_background(("/bin/sh", "-c", f'echo $$ > "{marker}"; exec sleep 60'))
        pid = _pid_of(session, marker)
        assert _alive(pid)
        assert session.status_background(handle).running
    assert _wait_gone(pid), "the backgrounded command outlived its session"


def test_a_setsid_escapee_does_not_outlive_its_command(target: conftest.Target) -> None:
    """A daemonizing child leaves the process group; the job's subreaper still ends it."""
    marker = target.discovery.target.host_root / "escapee.pid"
    with conftest.opened(target) as session:
        result = session.run(
            ("/bin/sh", "-c", f"setsid sh -c 'echo $$ > \"{marker}\"; exec sleep 60' & sleep 0.3")
        )
        assert isinstance(result, kinds.CommandResult) and result.returncode == 0
        pid = _pid_of(session, marker)
        assert _wait_gone(pid), "the setsid escapee outlived the command that spawned it"


def test_a_command_past_its_checkin_is_handed_back_then_stopped(target: conftest.Target) -> None:
    """The check-in hands a long command back as a job; stop ends it and keeps the record."""
    with conftest.opened(target) as session:
        outcome = session.run(("/bin/sleep", "30"), checkin_s=0.2, timeout_s=0.0)
        assert isinstance(outcome, kinds.BackgroundHandoff)
        assert session.status_background(outcome.handle).running
        stopped = session.stop_background(outcome.handle)
        assert stopped.error == "" and not stopped.survivors
        assert not session.status_background(outcome.handle).running


def test_an_immediate_stop_ends_a_foreground_command_quickly(target: conftest.Target) -> None:
    """The operator's stop reaches the command through the channel within a few polls."""
    with conftest.opened(target) as session:
        start = time.monotonic()
        result = session.run(("/bin/sleep", "30"), interrupted=lambda: True)
        elapsed = time.monotonic() - start
    assert isinstance(result, kinds.CommandResult)
    assert elapsed < 10, f"the stop took {elapsed:.1f}s"
    assert result.returncode != 0


def test_output_flood_is_capped_and_the_session_survives(target: conftest.Target) -> None:
    """Twenty megabytes of output come back capped; the next command still runs."""
    with conftest.opened(target) as session:
        result = session.run(("/bin/sh", "-c", "head -c 20000000 /dev/zero | tr '\\0' x"))
        assert isinstance(result, kinds.CommandResult) and result.returncode == 0
        assert len(result.stdout) < 20_000_000 and "omitted" in result.stdout
        again = session.run(("/bin/echo", "still here"))
        assert isinstance(again, kinds.CommandResult) and again.stdout == "still here\n"


def test_losing_the_supervisor_ends_its_jobs_and_reads_unknown(target: conftest.Target) -> None:
    """The agent side dies: the jobs inside end with their supervisor, the record says unknown."""
    marker = target.discovery.target.host_root / "lost.pid"
    session = container.prepare(target.discovery)
    try:
        handle = session.start_background(("/bin/sh", "-c", f'echo $$ > "{marker}"; exec sleep 60'))
        pid = _pid_of(session, marker)
        session._proc.kill()  # pyright: ignore[reportPrivateUsage]
        session._proc.wait(timeout=10)  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(jail.JailUnavailableError, match="protocol failed"):
            session.status_background(handle)
        assert _wait_gone(pid), "a job outlived its dead supervisor"
    finally:
        assert "unknown" in session.close()


def test_a_closed_channel_ends_the_jobs_as_a_dead_agent_would(target: conftest.Target) -> None:
    """The agent side goes away with its pipe: the launcher exits and takes its jobs with it."""
    marker = target.discovery.target.host_root / "eof.pid"
    session = container.prepare(target.discovery)
    try:
        session.start_background(("/bin/sh", "-c", f'echo $$ > "{marker}"; exec sleep 60'))
        pid = _pid_of(session, marker)
        proc = session._proc  # pyright: ignore[reportPrivateUsage]
        assert proc.stdin is not None
        proc.stdin.close()
        assert proc.wait(timeout=10) == 0, "the launcher outlived its channel"
        assert _wait_gone(pid), "a job outlived the launcher whose channel closed"
    finally:
        assert session.close() == ""


def test_a_close_spares_other_sessions_and_unrelated_processes(target: conftest.Target) -> None:
    """Cleanup reaches a session's own jobs only: a sibling session and a stranger live on."""
    root = target.discovery.target.host_root
    stranger = subprocess.Popen(
        ("/bin/sleep", "60"), env={**os.environ, "HOME": str(target.home)}, cwd=root
    )
    try:
        first = container.prepare(target.discovery)
        second = container.prepare(target.discovery)
        try:
            marker = root / "first.pid"
            first.start_background(("/bin/sh", "-c", f'echo $$ > "{marker}"; exec sleep 60'))
            job = _pid_of(first, marker)
            assert not first.close()
            assert _wait_gone(job), "the closed session's job survived"
            assert _alive(stranger.pid), "an unrelated process died with another session"
            result = second.run(("/bin/echo", "still here"))
            assert isinstance(result, kinds.CommandResult) and result.stdout == "still here\n"
        finally:
            second.close()
        assert _alive(stranger.pid), "an unrelated process died with the last session"
    finally:
        stranger.kill()
        stranger.wait(timeout=10)
