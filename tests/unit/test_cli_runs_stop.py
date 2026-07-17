# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 runs stop` drops the graceful stop marker for a running run."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent6.config.layer import resolved_state_dir
from agent6.runs.ipc import stop_request_pending, write_worker_pid
from agent6.runs.layout import RunLayout
from agent6.ui.cli import main


def _run_dir(repo: Path, run_id: str) -> Path:
    layout = RunLayout(state_dir=resolved_state_dir(repo), run_id=run_id)
    layout.ensure()
    layout.manifest_path.write_text('{"version": 2}', encoding="utf-8")
    (layout.run_dir / "logs.jsonl").write_text("", encoding="utf-8")
    return layout.run_dir


def test_runs_stop_requests_stop_for_a_live_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    rd = _run_dir(tmp_path, "live-run-AAA111")
    write_worker_pid(rd, os.getpid())  # this process is alive -> the run reads as running
    assert not stop_request_pending(rd)

    assert main(["runs", "stop", "live-run-AAA111"]) == 0
    assert stop_request_pending(rd)  # the marker the worker honors at the next step
    assert "requested stop" in capsys.readouterr().out


def test_runs_stop_on_a_dead_run_is_a_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    rd = _run_dir(tmp_path, "dead-run-BBB222")
    # A pid that is not alive: no worker.
    write_worker_pid(rd, 2**31 - 1)
    assert main(["runs", "stop", "dead-run-BBB222"]) == 0
    assert not stop_request_pending(rd)
    assert "not running" in capsys.readouterr().err
