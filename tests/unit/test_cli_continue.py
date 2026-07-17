# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for `agent6 run --continue` routing helpers."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent6.config.layer import resolved_state_dir
from agent6.ui.cli import main
from agent6.ui.cli.plan_watch import (
    _most_recent_plan_run_id,  # pyright: ignore[reportPrivateUsage]
)
from agent6.viewmodel import most_recent_run_id as _most_recent_run_id


def test_most_recent_run_id_none_outside_workspace(tmp_path: Path) -> None:
    assert _most_recent_run_id(tmp_path / "missing") is None


def test_most_recent_run_id_none_when_empty(tmp_path: Path) -> None:
    (resolved_state_dir(tmp_path) / "runs").mkdir(parents=True)
    assert _most_recent_run_id(resolved_state_dir(tmp_path) / "runs") is None


def test_most_recent_run_id_uses_log_activity_not_frontend_dir_touch(tmp_path: Path) -> None:
    runs = resolved_state_dir(tmp_path) / "runs"
    runs.mkdir(parents=True)
    older = runs / "alpha-bravo-charlie"
    newer = runs / "delta-echo-foxtrot"
    older.mkdir()
    newer.mkdir()
    (older / "logs.jsonl").write_text('{"type":"run.start"}\n', encoding="utf-8")
    (newer / "logs.jsonl").write_text('{"type":"run.start"}\n', encoding="utf-8")
    os.utime(older / "logs.jsonl", (100, 100))
    os.utime(newer / "logs.jsonl", (1000, 1000))
    (older / "frontend.pid").write_text("12345", encoding="utf-8")
    assert _most_recent_run_id(runs) == "delta-echo-foxtrot"


def test_most_recent_plan_run_id_uses_log_activity_not_frontend_dir_touch(tmp_path: Path) -> None:
    runs = resolved_state_dir(tmp_path) / "runs"
    runs.mkdir(parents=True)
    older = runs / "older-plan"
    newer = runs / "newer-plan"
    older.mkdir()
    newer.mkdir()
    for run_dir in (older, newer):
        (run_dir / "plan.md").write_text("# Plan\n", encoding="utf-8")
        (run_dir / "logs.jsonl").write_text('{"type":"run.start"}\n', encoding="utf-8")
    os.utime(older / "logs.jsonl", (100, 100))
    os.utime(newer / "logs.jsonl", (1000, 1000))
    (older / "frontend.pid").write_text("12345", encoding="utf-8")
    assert _most_recent_plan_run_id(runs) == "newer-plan"


def test_continue_with_task_argument_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "agent6.toml").write_text("# placeholder\n", encoding="utf-8")
    rc = main(["run", "do a thing", "--continue"])
    assert rc == 2
    assert "either a task OR --continue" in capsys.readouterr().err


def test_continue_with_explicit_run_id_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "agent6.toml").write_text("# placeholder\n", encoding="utf-8")
    rc = main(["run", "--continue", "--run-id", "x"])
    assert rc == 2
    assert "--run-id is incompatible with --continue" in capsys.readouterr().err


def test_continue_without_any_runs_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "agent6.toml").write_text("# placeholder\n", encoding="utf-8")
    rc = main(["run", "--continue"])
    assert rc == 2
    assert "no prior runs" in capsys.readouterr().err


def test_run_without_task_or_continue_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "agent6.toml").write_text("# placeholder\n", encoding="utf-8")
    rc = main(["run"])
    assert rc == 2
    # With no task AND no prior plan to fall back to, `run` still errors.
    assert "needs a task" in capsys.readouterr().err


def test_run_no_task_points_at_most_recent_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # No task given but a prior plan exists: non-interactively (pytest stdin is
    # not a TTY) refuse, but point the user at the plan + the --from-plan form.
    monkeypatch.chdir(tmp_path)
    run_dir = resolved_state_dir(tmp_path) / "runs" / "tidy-otter-AB12CD"
    run_dir.mkdir(parents=True)
    (run_dir / "plan.md").write_text("# Plan: wire up the thing\n", encoding="utf-8")
    rc = main(["run"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "tidy-otter-AB12CD" in err
    assert "--from-plan" in err
