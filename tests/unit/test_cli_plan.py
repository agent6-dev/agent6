# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 plan --show/--edit` and `agent6 run --from-plan` CLI smoke."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.cli import main


def _seed_plan(tmp_path: Path, run_id: str, body: str) -> Path:
    plan_dir = tmp_path / ".agent6" / "runs" / run_id
    plan_dir.mkdir(parents=True)
    plan = plan_dir / "plan.md"
    plan.write_text(body, encoding="utf-8")
    return plan


def test_plan_show_prints_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    _seed_plan(tmp_path, "happy-tree-abcd", "# Plan: foo\n\nbody\n")
    rc = main(["plan", "--show", "happy-tree-abcd"])
    assert rc == 0
    assert "# Plan: foo" in capsys.readouterr().out


def test_plan_show_resolves_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    _seed_plan(tmp_path, "happy-tree-abcd", "# Plan: foo\n")
    rc = main(["plan", "--show", "happy"])
    assert rc == 0
    assert "# Plan: foo" in capsys.readouterr().out


def test_plan_show_missing_run_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".agent6" / "runs").mkdir(parents=True)
    rc = main(["plan", "--show", "nonexistent"])
    assert rc == 2
    assert "ERROR" in capsys.readouterr().err


def test_plan_show_no_plan_md_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".agent6" / "runs" / "happy-tree-abcd").mkdir(parents=True)
    rc = main(["plan", "--show", "happy-tree-abcd"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "plan.md" in err


def test_plan_show_and_edit_mutually_exclusive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    rc = main(["plan", "--show", "x", "--edit", "y"])
    assert rc == 2


def test_plan_requires_task_or_show(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    rc = main(["plan"])
    assert rc == 2
    assert "ERROR" in capsys.readouterr().err


def test_plan_edit_invokes_editor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    plan = _seed_plan(tmp_path, "happy-tree-abcd", "original\n")
    marker = tmp_path / "editor_ran"
    script = tmp_path / "fake_editor.sh"
    script.write_text(f"#!/bin/sh\necho edited >> $1\ntouch {marker}\n", encoding="utf-8")
    script.chmod(0o755)
    monkeypatch.setenv("EDITOR", str(script))
    rc = main(["plan", "--edit", "happy-tree-abcd"])
    assert rc == 0
    assert marker.exists()
    assert "edited" in plan.read_text(encoding="utf-8")


def test_run_from_plan_and_task_mutually_exclusive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    _seed_plan(tmp_path, "happy-tree-abcd", "x\n")
    rc = main(["run", "do thing", "--from-plan", "happy-tree-abcd"])
    assert rc == 2
    assert "mutually exclusive" in capsys.readouterr().err
