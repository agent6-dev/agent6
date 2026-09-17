# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`/task` writes the queue, never the steer channel, wherever it is typed.

One owner parses a composer line (`ui.directives`), so the web composer, the
pause menu and `agent6 steer` all act on the same words the same way. The TUI
composer is covered by tests/tui/test_tui_task_directive.py, which needs a
running app.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent6.directive import LIVE_RUN_COMMANDS, STEER_COMMANDS, parse_task
from agent6.paths import state_dir
from agent6.sessions.ipc import drain_queued_tasks, steer_request_pending, write_worker_pid
from agent6.ui.cli import main
from agent6.ui.web import actions


def _live_run(tmp_path: Path) -> Path:
    d = state_dir(tmp_path) / "sessions" / "runs" / "live-one-AAAAAA"
    d.mkdir(parents=True)
    (d / "logs.jsonl").write_text('{"type": "session.start", "mode": "run"}\n', encoding="utf-8")
    write_worker_pid(d, os.getpid())
    return d


def test_parse_task_takes_the_text_and_leaves_other_lines_alone() -> None:
    assert parse_task("/task add a --json flag") == "add a --json flag"
    assert parse_task("/task") == ""  # a bare directive: the caller says so
    assert parse_task("/taskfoo bar") is None
    assert parse_task("mention /task in a sentence") is None
    assert parse_task("/task first line\nsecond line") == "first line\nsecond line"


def test_the_directive_is_offered_only_on_a_live_run() -> None:
    """Nothing drains the queue on a finished run, so the resume composer
    withholds it, as it does for the other live-run directives."""
    assert "/task" in STEER_COMMANDS
    assert "/task" in LIVE_RUN_COMMANDS


def test_the_web_composer_queues_instead_of_steering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    d = _live_run(tmp_path)

    ok, msg = actions.steer(tmp_path, "live-one-AAAAAA", "/task add a --json flag")

    assert ok and msg.startswith("task queued")
    assert drain_queued_tasks(d) == ["add a --json flag"]
    assert not steer_request_pending(d)


def test_the_web_composer_refuses_a_bare_directive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    d = _live_run(tmp_path)

    ok, msg = actions.steer(tmp_path, "live-one-AAAAAA", "/task")

    assert not ok
    assert "/task needs the work" in msg
    assert drain_queued_tasks(d) == []


def test_the_pause_menu_queues_and_re_prompts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An info command: it prints and hands the prompt back, so the operator
    can queue several without ending the pause."""
    from agent6.ui.cli._steer_menu import _run_info_command  # pyright: ignore[reportPrivateUsage]

    _run_info_command("/task ship the changelog", tmp_path)

    assert "task queued" in capsys.readouterr().out
    assert drain_queued_tasks(tmp_path) == ["ship the changelog"]


def test_the_pause_menu_refuses_a_bare_directive(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from agent6.ui.cli._steer_menu import _run_info_command  # pyright: ignore[reportPrivateUsage]

    _run_info_command("/task", tmp_path)

    assert "/task needs the work" in capsys.readouterr().out
    assert drain_queued_tasks(tmp_path) == []


def test_agent6_steer_acts_on_the_directive_instead_of_sending_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`agent6 steer ID "/task ..."` used to send the literal text to the model
    as an instruction: the CLI parsed `/btw` and `/compact` but not `/task`."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / ".state"))
    monkeypatch.chdir(tmp_path)
    d = state_dir(tmp_path) / "sessions" / "runs" / "tiny-run-AAAA11"
    d.mkdir(parents=True)
    (d / "logs.jsonl").write_text("", encoding="utf-8")
    write_worker_pid(d, os.getpid())

    assert main(["steer", "tiny-run", "/task add a --json flag"]) == 0

    assert "task queued" in capsys.readouterr().out
    assert drain_queued_tasks(d) == ["add a --json flag"]
    assert not steer_request_pending(d)


def test_agent6_steer_takes_now_as_the_composers_spell_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`/now <text>` is what `--now` spells, so both reach the same marker."""
    from agent6.sessions.ipc import steer_interrupt_pending, take_steer_answer

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / ".state"))
    monkeypatch.chdir(tmp_path)
    d = state_dir(tmp_path) / "sessions" / "runs" / "tiny-run-BBBB22"
    d.mkdir(parents=True)
    (d / "logs.jsonl").write_text("", encoding="utf-8")
    write_worker_pid(d, os.getpid())

    assert main(["steer", "tiny-run", "/now wrap up"]) == 0

    assert steer_interrupt_pending(d)
    assert take_steer_answer(d) == "wrap up"
