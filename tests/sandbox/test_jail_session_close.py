# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`JailSession.close()` must never propagate an exception at teardown.

Found via the wheel CI-mirror leg on Python 3.12: close() used to `stdin.close()`
then `communicate()`, whose flush re-hits the now-closed pipe and raises
`ValueError: flush of closed file`. Python 3.14 (the dev interpreter) tolerates
that, so no gate caught it -- but AGENTS.md supports 3.12+, where it was an
unhandled crash in `ToolDispatcher.close()`. These tests use a fake launcher
proc, so they need no namespaces and run on every interpreter.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from agent6.sandbox.jail import JailSession


class _FakeProc:
    """A launcher Popen stand-in whose communicate() raises like 3.12's does."""

    def __init__(self, *, communicate_raises: BaseException | None, alive_after: bool) -> None:
        self.pid = 424242
        self.stdin = None  # close() must not depend on stdin being present
        self._communicate_raises = communicate_raises
        self._alive_after = alive_after
        self.communicated = False

    def communicate(self, timeout: float | None = None) -> tuple[bytes, bytes]:
        self.communicated = True
        if self._communicate_raises is not None:
            raise self._communicate_raises
        return (b"", b"")

    def poll(self) -> int | None:
        return None if self._alive_after else 0


def _session(proc: Any) -> JailSession:
    return JailSession(_proc=proc, _binary=Path("/nonexistent"), _memory_limit_mb=0)


def test_close_swallows_the_closed_stdin_flush_valueerror() -> None:
    proc = _FakeProc(communicate_raises=ValueError("flush of closed file"), alive_after=False)
    _session(proc).close()  # must not raise
    assert proc.communicated


def test_close_kills_a_launcher_that_outlived_the_drain(monkeypatch: pytest.MonkeyPatch) -> None:
    """A communicate() that times out leaves the launcher alive; close() then
    SIGKILLs its group. The kill must reach the pid, and close() must not raise
    even if the (already-exited) killpg errors."""
    killed: list[int] = []

    def _fake_killpg(pid: int, sig: int) -> None:
        killed.append(pid)

    monkeypatch.setattr("agent6.sandbox.jail.os.killpg", _fake_killpg)
    proc = _FakeProc(
        communicate_raises=subprocess.TimeoutExpired(cmd="jail", timeout=10.0),
        alive_after=True,
    )
    _session(proc).close()  # must not raise
    assert killed == [proc.pid]
