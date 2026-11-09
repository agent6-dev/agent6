# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Session teardown tolerates closed pipes and reaps a launcher that ignored EOF."""

from __future__ import annotations

import io
import json
import pathlib
import select
import subprocess
import time
from typing import IO, Any

import pytest

from agent6 import kinds
from agent6.sandbox import jail, jail_jobs, jail_session


class _FakeProc:
    """A launcher Popen stand-in with a bounded wait."""

    def __init__(self, *, wait_raises: BaseException | None, alive_after: bool) -> None:
        self.pid = 424242
        self.stdin: IO[bytes] | None = None
        self.stdout = None
        self.stderr = None
        self._wait_raises = wait_raises
        self._alive_after = alive_after
        self.waited = False

    def wait(self, timeout: float | None = None) -> int:
        self.waited = True
        if self._wait_raises is not None:
            raise self._wait_raises
        return 0

    def poll(self) -> int | None:
        return None if self._alive_after else 0


def _session(proc: Any, *, pid_namespaced: bool = True) -> jail_session.JailSession:
    return jail_session.JailSession(
        _proc=proc,
        _binary=pathlib.Path("/nonexistent"),
        _pid_namespaced=pid_namespaced,
        _memory_limit_mib=0,
    )


def test_close_tolerates_an_already_closed_stdin() -> None:
    proc = _FakeProc(wait_raises=None, alive_after=False)
    proc.stdin = io.BytesIO()
    proc.stdin.close()
    _session(proc).close()  # must not raise
    assert proc.waited


def test_close_kills_a_launcher_that_ignored_eof(monkeypatch: pytest.MonkeyPatch) -> None:
    """close() SIGKILLs the group of a launcher left alive by a timed-out wait().

    The kill reaches the pid, and close() does not raise when the killpg of an already-exited
    process errors.
    """
    killed: list[int] = []

    def _fake_killpg(pid: int, sig: int) -> None:
        killed.append(pid)

    monkeypatch.setattr("agent6.sandbox.jail.os.killpg", _fake_killpg)
    proc = _FakeProc(
        wait_raises=subprocess.TimeoutExpired(cmd="jail", timeout=10.0),
        alive_after=True,
    )
    _session(proc).close()  # must not raise
    assert killed == [proc.pid]


@pytest.mark.parametrize(
    "isolation", ["none", "partial", pytest.param("full", marks=pytest.mark.needs_namespaces)]
)
def test_close_drains_a_response_interrupted_before_it_was_read(
    tmp_path: pathlib.Path, isolation: kinds.IsolationLevel
) -> None:
    """An unread response larger than the pipe must not force a healthy launcher to be killed."""
    session = jail_session.JailSession.open(
        kinds.JailPolicy(cwd=tmp_path, argv=(), isolation=isolation, network="host")
    )
    proc = session._proc  # pyright: ignore[reportPrivateUsage]

    def interrupt_response() -> bool:
        assert proc.stdout is not None
        assert select.select([proc.stdout], [], [], 5)[0], "no response became readable"
        assert proc.poll() is None
        raise KeyboardInterrupt

    try:
        with pytest.raises(KeyboardInterrupt):
            session.run(
                ("python3", "-c", "print('x' * (1024 * 1024))"),
                interrupted=interrupt_response,
            )
    finally:
        started = time.monotonic()
        survivors = session.close()
        elapsed = time.monotonic() - started
    assert elapsed < 3, f"close took {elapsed:.2f}s; launcher exit={proc.returncode}"
    assert proc.returncode == 0
    assert not survivors


def test_a_survivor_of_the_sweep_is_named_by_every_stop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """A stop that could not kill an escapee says so on every path.

    `JailedProcess.close`, `LocalJob.stop`, `BackgroundJob.stop` and `SessionJob.stop` report
    the sweep's survivors, so the MCP-server path and `stop_background` under `none` and
    `partial` never answer "stopped" over a process the sweep could not kill.
    """

    def sweep(exclude: frozenset[int]) -> frozenset[int]:
        return frozenset({4321})

    def no_children() -> dict[int, int]:
        return {}

    monkeypatch.setattr("agent6.sandbox.jail.kill_escapees", sweep)
    monkeypatch.setattr("agent6.sandbox.jail.own_children", no_children)

    def sleeper() -> subprocess.Popen[bytes]:
        return subprocess.Popen(["sleep", "60"], start_new_session=True)

    proc = sleeper()
    assert "4321" in jail_jobs.LocalJob(proc, tmp_path / "local").stop() and proc.poll() is not None
    proc = sleeper()
    (tmp_path / "bg").mkdir()  # the outcome dir the shell registry creates
    assert (
        "4321" in jail_jobs.BackgroundJob(proc, tmp_path / "bg").stop() and proc.poll() is not None
    )
    # The launcher is gone, so its ending is recorded even though the sweep failed.
    assert json.loads((tmp_path / "bg" / "result.json").read_text())["stopped"] is True
    proc = sleeper()
    assert jail.JailedProcess(proc).close() == frozenset({4321}) and proc.poll() is not None

    def stopped(
        self: jail_session.JailSession, request: dict[str, object], *, interrupted: object = None
    ) -> dict[str, object]:
        return {"stopped": True, "returncode": 0}

    monkeypatch.setattr(jail_session.JailSession, "_request", stopped)
    session = _session(_FakeProc(wait_raises=None, alive_after=False), pid_namespaced=False)
    assert (
        "4321"
        in jail_jobs.SessionJob(
            session, 7, tmp_path / "job", before=session.child_snapshot()
        ).stop()
    )
    # A job the registry already settled (its command exited) still sweeps on stop.
    settled = jail_jobs.SessionJob(
        session, 8, tmp_path / "job-settled", before=session.child_snapshot()
    )
    settled._final = jail_session.BackgroundStatus(running=False, returncode=0, error="")  # pyright: ignore[reportPrivateUsage]
    assert "4321" in settled.stop()
    namespaced = _session(_FakeProc(wait_raises=None, alive_after=False))
    job2 = jail_jobs.SessionJob(
        namespaced, 7, tmp_path / "job2", before=namespaced.child_snapshot()
    )
    assert job2.stop() == ""  # the namespace bounds them


def test_a_survivor_of_the_sweep_fails_the_command_and_comes_back_from_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The session path fails a command whose escapee the sweep could not kill, like `run_in_jail`.

    `run` and `close` keep the sweep's answer, so no process outlives the run in silence.
    """

    def sweep(exclude: frozenset[int]) -> frozenset[int]:
        return frozenset({4321})

    def no_children() -> dict[int, int]:
        return {}

    def answered(
        self: jail_session.JailSession, request: dict[str, object], *, interrupted: object = None
    ) -> dict[str, object]:
        return {"returncode": 0, "stdout": "", "stderr": ""}

    monkeypatch.setattr("agent6.sandbox.jail.kill_escapees", sweep)
    monkeypatch.setattr("agent6.sandbox.jail.own_children", no_children)
    monkeypatch.setattr(jail_session.JailSession, "_request", answered)
    session = _session(_FakeProc(wait_raises=None, alive_after=False), pid_namespaced=False)
    with pytest.raises(jail.JailUnavailableError, match="4321"):
        session.run(("true",))
    assert session.close() == frozenset({4321})
