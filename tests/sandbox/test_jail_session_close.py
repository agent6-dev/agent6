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
from agent6.tools import background


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
        self.killed = False

    def wait(self, timeout: float | None = None) -> int:
        self.waited = True
        if self._wait_raises is not None:
            raise self._wait_raises
        return 0

    def kill(self) -> None:
        self.killed = True

    def poll(self) -> int | None:
        return None if self._alive_after else 0


def _session(proc: Any) -> jail_session.JailSession:
    return jail_session.JailSession(
        _proc=proc,
        _memory_limit_mib=0,
    )


def test_close_tolerates_an_already_closed_stdin() -> None:
    proc = _FakeProc(wait_raises=None, alive_after=False)
    proc.stdin = io.BytesIO()
    proc.stdin.close()
    _session(proc).close()  # must not raise
    assert proc.waited


def test_close_kills_a_launcher_that_ignored_eof() -> None:
    """A timed-out close kills the transport, not its job subreapers."""
    proc = _FakeProc(
        wait_raises=subprocess.TimeoutExpired(cmd="jail", timeout=10.0),
        alive_after=True,
    )
    _session(proc).close()  # must not raise
    assert proc.killed


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


@pytest.mark.parametrize("status_first", [False, True])
@pytest.mark.parametrize(
    ("reply", "detail"),
    [
        ({"stopped": False, "returncode": 23, "survivors": [4321]}, "4321"),
        (
            {"stopped": False, "returncode": 23, "error": "job supervisor closed"},
            "job supervisor closed",
        ),
    ],
)
def test_failed_stop_preserves_received_exit_and_cleanup_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    status_first: bool,
    reply: dict[str, object],
    detail: str,
) -> None:
    """A stop's cleanup failure cannot discard the exit code in the same response."""

    def answered(
        self: jail_session.JailSession, request: dict[str, object], *, interrupted: object = None
    ) -> dict[str, object]:
        if request["kind"] == "status":
            return {"running": False, "returncode": 23}
        assert request["kind"] == "stop"
        return reply

    monkeypatch.setattr(jail_session.JailSession, "_request", answered)
    session = _session(_FakeProc(wait_raises=None, alive_after=False))
    outcome = tmp_path / "shells" / "bg1"
    outcome.mkdir(parents=True)
    (outcome / "meta.json").write_text('{"command":"exit 23"}')
    job = jail_jobs.SessionJob(session, "instance:7", outcome)
    if status_first:
        assert job.status().returncode == 23
    for _ in range(2):
        assert detail in job.stop()
        assert job.status().returncode == 23
        record = json.loads((outcome / "result.json").read_text())
        assert record["returncode"] == 23
        assert detail in record["error"]
        rendered = background.roster_from_dir(outcome.parent)[0]
        assert "exited 23" in rendered and detail in rendered


def test_supervisor_survivors_fail_a_foreground_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def answered(
        self: jail_session.JailSession, request: dict[str, object], *, interrupted: object = None
    ) -> dict[str, object]:
        return {"returncode": 0, "stdout": "", "stderr": "", "survivors": [4321]}

    monkeypatch.setattr(jail_session.JailSession, "_request", answered)
    session = _session(_FakeProc(wait_raises=None, alive_after=False))
    with pytest.raises(jail.JailUnavailableError, match="4321"):
        session.run(("true",))
