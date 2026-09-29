# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 run` with no task: the newest-run/plan fallbacks and refusals."""

from __future__ import annotations

import os
import pathlib

import pytest

from agent6 import paths
from agent6.ui.cli import _common, main, plan_watch
from agent6.viewmodel import newest_session_dir


def test_newest_run_dir_none_for_missing_bucket(tmp_path: pathlib.Path) -> None:
    assert newest_session_dir([tmp_path / "missing"]) is None


def test_newest_run_dir_none_when_empty(tmp_path: pathlib.Path) -> None:
    runs = paths.state_dir(tmp_path) / "sessions" / "runs"
    runs.mkdir(parents=True)
    assert newest_session_dir([runs]) is None


def test_newest_run_dir_uses_log_activity_not_frontend_dir_touch(tmp_path: pathlib.Path) -> None:
    runs = paths.state_dir(tmp_path) / "sessions" / "runs"
    runs.mkdir(parents=True)
    older = runs / "alpha-bravo-charlie"
    newer = runs / "delta-echo-foxtrot"
    older.mkdir()
    newer.mkdir()
    (older / "logs.jsonl").write_text('{"type":"session.start"}\n', encoding="utf-8")
    (newer / "logs.jsonl").write_text('{"type":"session.start"}\n', encoding="utf-8")
    os.utime(older / "logs.jsonl", (100, 100))
    os.utime(newer / "logs.jsonl", (1000, 1000))
    (older / "frontend.pid").write_text("12345", encoding="utf-8")
    newest = newest_session_dir([runs])
    assert newest is not None
    assert newest.name == "delta-echo-foxtrot"


def test_most_recent_plan_run_id_uses_log_activity_not_frontend_dir_touch(
    tmp_path: pathlib.Path,
) -> None:
    plans = paths.state_dir(tmp_path) / "sessions" / "plans"
    plans.mkdir(parents=True)
    older = plans / "older-plan"
    newer = plans / "newer-plan"
    older.mkdir()
    newer.mkdir()
    for session_dir in (older, newer):
        (session_dir / "plan.md").write_text("# Plan\n", encoding="utf-8")
        (session_dir / "logs.jsonl").write_text('{"type":"session.start"}\n', encoding="utf-8")
    os.utime(older / "logs.jsonl", (100, 100))
    os.utime(newer / "logs.jsonl", (1000, 1000))
    (older / "frontend.pid").write_text("12345", encoding="utf-8")
    assert plan_watch._most_recent_plan_session_id(plans) == "newer-plan"


def test_run_without_task_errors(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "agent6.toml").write_text("# placeholder\n", encoding="utf-8")
    rc = main(["run"])
    assert rc == 2
    # With no task AND no prior plan to fall back to, name both ways to start.
    err = capsys.readouterr().err
    assert 'agent6 run "TASK"' in err
    assert 'agent6 plan "TASK"' in err


def test_run_no_task_points_at_most_recent_plan(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # No task but a prior plan, non-interactively: refuse and point at the plan and the --from form.
    monkeypatch.chdir(tmp_path)
    session_dir = paths.state_dir(tmp_path) / "sessions" / "plans" / "tidy-otter-AB12CD"
    session_dir.mkdir(parents=True)
    (session_dir / "plan.md").write_text("# Plan: wire up the thing\n", encoding="utf-8")
    rc = main(["run"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "tidy-otter-AB12CD" in err
    assert "--from" in err


def test_run_no_task_terminal_offer_names_the_newest_updated_plan(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:

    monkeypatch.chdir(tmp_path)
    plans = paths.state_dir(tmp_path) / "sessions" / "plans"
    old = plans / "zebra-plan-OLD111"
    new = plans / "alpha-plan-NEW222"
    for session_dir, title in ((old, "old work"), (new, "new work")):
        session_dir.mkdir(parents=True)
        (session_dir / "plan.md").write_text(f"# Plan: {title}\n", encoding="utf-8")
        (session_dir / "logs.jsonl").write_text("{}\n", encoding="utf-8")
    os.utime(old / "logs.jsonl", (100, 100))
    os.utime(new / "logs.jsonl", (1000, 1000))
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)

    def decline(_prompt: str) -> str:
        return "n"

    monkeypatch.setattr(_common, "safe_input", decline)

    assert main(["run"]) == 0

    out = capsys.readouterr().out
    assert "alpha-plan-NEW222  (new work)" in out
    assert "zebra-plan-OLD111" not in out


def test_run_no_task_at_a_terminal_executes_the_plan_on_enter(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`Execute it now? [Y/n]` defaults to yes: Enter runs the plan, `n` and an EOF abort."""
    from agent6.ui.cli import run as run_mod

    monkeypatch.chdir(tmp_path)
    session_dir = paths.state_dir(tmp_path) / "sessions" / "plans" / "tidy-otter-AB12CD"
    session_dir.mkdir(parents=True)
    (session_dir / "plan.md").write_text("# Plan: wire up the thing\n", encoding="utf-8")
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    ran: list[str] = []

    def fake_run(config: object, task: str, **kwargs: object) -> int:
        ran.append(task)
        return 0

    monkeypatch.setattr(run_mod, "_cmd_run", fake_run)
    for typed, executes in (("", True), ("y", True), ("n", False), (None, False)):

        def answer(prompt: str, typed: str | None = typed) -> str | None:
            return typed

        monkeypatch.setattr(_common, "safe_input", answer)
        ran.clear()
        assert main(["run"]) == 0
        assert (len(ran) == 1) is executes, typed
        out = capsys.readouterr().out
        assert ("Aborted" in out) is not executes, typed
    assert "wire up the thing" in ran[0] if ran else True


def test_run_continue_flag_is_gone(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # `run --continue` was a subset of `resume`; argparse refuses it like any unknown flag.
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as exc:
        main(["run", "--continue"])
    assert exc.value.code == 2


def test_parallel_refuses_an_explicit_run_id(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Each lane mints its own run id, so --session-id is refused like -i and --tui.
    monkeypatch.chdir(tmp_path)
    rc = main(["run", "--parallel", "2", "--session-id", "myid", "task"])
    assert rc == 2
    assert "--session-id" in capsys.readouterr().err


def test_parallel_refuses_a_standing_goal(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`--parallel` refuses `--standing`, which reaches no lane, as it refuses --session-id."""
    monkeypatch.chdir(tmp_path)
    rc = main(["run", "--parallel", "2", "--standing", "keep tests green", "task"])
    assert rc == 2
    assert "--standing" in capsys.readouterr().err
