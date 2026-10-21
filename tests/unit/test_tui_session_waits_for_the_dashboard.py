# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`run --tui` proceeds only once the dashboard holds the terminal.

An approval asked before the dashboard registered as a front-end fell to the terminal
prompt, which read the same tty as the dashboard: the gate's baseline run asks at once, so
every keystroke raced between the two readers and the row answered nothing.
"""

from __future__ import annotations

import os
import pathlib
import threading
import time

import pytest

from agent6.sessions import process
from agent6.ui.cli import _live


class _Dashboard:
    """A spawned dashboard that registers as a front-end a little after it starts."""

    def __init__(self, session_dir: pathlib.Path) -> None:
        self.returncode: int | None = None
        self._session_dir = session_dir
        threading.Timer(0.3, self._register).start()

    def _register(self) -> None:
        process.register_frontend(self._session_dir, os.getpid())

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.returncode = 0
        return 0

    def send_signal(self, _sig: int) -> None:
        pass

    def terminate(self) -> None:
        pass

    def kill(self) -> None:
        pass


def test_the_run_starts_once_the_dashboard_is_a_live_front_end(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_dir = tmp_path / "run"
    session_dir.mkdir()

    def _popen(*_a: object, **_k: object) -> _Dashboard:
        return _Dashboard(session_dir)

    monkeypatch.setattr(_live.subprocess, "Popen", _popen)
    started = time.monotonic()
    with _live.tui_session(session_dir, enabled=True):
        assert process.frontend_is_live(session_dir), "the block ran before the dashboard was up"
    assert time.monotonic() - started < 4.0, "the wait ends at the registration, not a deadline"


def test_a_dashboard_that_never_registers_does_not_hold_the_run(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dashboard that exits at once degrades to a run without it, as a spawn failure does."""
    session_dir = tmp_path / "run"
    session_dir.mkdir()

    class _Gone(_Dashboard):
        def __init__(self, _session_dir: pathlib.Path) -> None:
            self.returncode = 1

        def _register(self) -> None:
            pass

    def _popen(*_a: object, **_k: object) -> _Gone:
        return _Gone(session_dir)

    monkeypatch.setattr(_live.subprocess, "Popen", _popen)
    started = time.monotonic()
    with _live.tui_session(session_dir, enabled=True):
        pass
    assert time.monotonic() - started < 2.0


def test_a_dashboard_slower_than_five_seconds_is_still_waited_for(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dashboard registering after 5 s was left behind.

    The run started without it, and its first approval prompt read the terminal the
    dashboard owned.
    """
    session_dir = tmp_path / "run"
    session_dir.mkdir()

    class _Clock:
        """Ten seconds pass per reading, so any deadline is behind the first check."""

        def __init__(self) -> None:
            self.now = 0.0

        def monotonic(self) -> float:
            self.now += 10.0
            return self.now

        def sleep(self, seconds: float) -> None:
            time.sleep(seconds)

    def _popen(*_a: object, **_k: object) -> _Dashboard:
        return _Dashboard(session_dir)

    monkeypatch.setattr(_live.subprocess, "Popen", _popen)
    monkeypatch.setattr(_live, "time", _Clock())
    with _live.tui_session(session_dir, enabled=True):
        assert process.frontend_is_live(session_dir), "the block ran before the dashboard was up"
