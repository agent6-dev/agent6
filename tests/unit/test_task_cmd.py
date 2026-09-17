# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 task`: queue work into a live run's graph from anywhere.

The sibling of `agent6 steer` for work that is for later: it writes the queue
the loop drains, never the steer channel, so the turn in flight is untouched.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent6.paths import state_dir
from agent6.sessions.ipc import (
    drain_queued_tasks,
    steer_request_pending,
    write_worker_pid,
)
from agent6.ui.cli import main


def _run_session(tmp_path: Path, session_id: str) -> Path:
    d = state_dir(tmp_path) / "sessions" / "runs" / session_id
    d.mkdir(parents=True)
    (d / "logs.jsonl").write_text("", encoding="utf-8")
    return d


def test_task_queues_for_a_live_run_without_steering_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / ".state"))
    monkeypatch.chdir(tmp_path)
    d = _run_session(tmp_path, "tiny-run-AAAA11")
    write_worker_pid(d, os.getpid())

    assert main(["task", "tiny-run", "Add a --json flag to the stats report"]) == 0

    assert "task queued for tiny-run-AAAA11" in capsys.readouterr().out
    assert drain_queued_tasks(d) == ["Add a --json flag to the stats report"]
    assert not steer_request_pending(d)  # the run is not interrupted


def test_task_refuses_a_session_that_is_not_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Nothing would drain the queue, so the refusal names the verb that does
    take work to a stopped session."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / ".state"))
    monkeypatch.chdir(tmp_path)
    d = _run_session(tmp_path, "tiny-run-BBBB22")

    assert main(["task", "tiny-run", "later work"]) == 2

    assert "resume tiny-run-BBBB22 --steer" in capsys.readouterr().err
    assert drain_queued_tasks(d) == []


def test_task_refuses_empty_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / ".state"))
    monkeypatch.chdir(tmp_path)
    d = _run_session(tmp_path, "tiny-run-CCCC33")
    write_worker_pid(d, os.getpid())

    assert main(["task", "tiny-run", "   "]) == 2

    assert "a task needs text" in capsys.readouterr().err
    assert drain_queued_tasks(d) == []
