# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for `agent6 run --continue` routing helpers."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent6.cli import (
    _most_recent_run_id,  # pyright: ignore[reportPrivateUsage]
    main,
)


def test_most_recent_run_id_none_outside_workspace(tmp_path: Path) -> None:
    assert _most_recent_run_id(tmp_path / "missing") is None


def test_most_recent_run_id_none_when_empty(tmp_path: Path) -> None:
    (tmp_path / ".agent6" / "runs").mkdir(parents=True)
    assert _most_recent_run_id(tmp_path / ".agent6" / "runs") is None


def test_most_recent_run_id_picks_newest_mtime(tmp_path: Path) -> None:
    runs = tmp_path / ".agent6" / "runs"
    runs.mkdir(parents=True)
    older = runs / "alpha-bravo-charlie"
    newer = runs / "delta-echo-foxtrot"
    older.mkdir()
    newer.mkdir()
    os.utime(older, (1, 1))
    os.utime(newer, (1000, 1000))
    assert _most_recent_run_id(runs) == "delta-echo-foxtrot"


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
    assert "needs a task argument" in capsys.readouterr().err
