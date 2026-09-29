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
from collections.abc import Callable
from pathlib import Path

import pytest

from agent6.directive import LIVE_RUN_COMMANDS, STEER_COMMANDS, parse_task
from agent6.paths import state_dir
from agent6.sessions.ipc import drain_requests, steer_request_pending, write_worker_pid
from agent6.ui.cli import main
from agent6.ui.web import actions


def _live_run(tmp_path: Path) -> Path:
    d = state_dir(tmp_path) / "sessions" / "runs" / "live-one-AAAAAA"
    d.mkdir(parents=True, exist_ok=True)
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
    """The resume composer withholds `/task` on a finished run, where nothing drains the queue."""
    assert "/task" in STEER_COMMANDS
    assert "/task" in LIVE_RUN_COMMANDS


def test_the_web_composer_queues_instead_of_steering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    d = _live_run(tmp_path)

    ok, msg = actions.steer(tmp_path, "live-one-AAAAAA", "/task add a --json flag")

    assert ok and msg.startswith("task queued")
    assert [r.text for r in drain_requests(d)] == ["add a --json flag"]
    assert not steer_request_pending(d)


def test_the_web_composer_refuses_a_bare_directive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    d = _live_run(tmp_path)

    ok, msg = actions.steer(tmp_path, "live-one-AAAAAA", "/task")

    assert not ok
    assert "/task needs the work" in msg
    assert [r.text for r in drain_requests(d)] == []


def _paused(tmp_path: Path) -> Path:
    """A run dir mid-pause: the menu reads its state off disk."""
    (tmp_path / "logs.jsonl").write_text(
        '{"type": "session.start", "user_task": "t", "mode": "run"}\n', encoding="utf-8"
    )
    write_worker_pid(tmp_path, os.getpid())
    return tmp_path


def _feed(lines: list[str]) -> Callable[[str], str]:
    it = iter(lines)
    return lambda _prompt: next(it)


def test_the_pause_menu_queues_and_re_prompts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`/task <text>` through the real prompt reaches the queue, not the model as steer text.

    `/task` shares a prefix with `/tasks` and the argument branch listed its own commands.
    """
    from agent6.ui.cli._steer_menu import pause_menu

    d = _paused(tmp_path)

    assert pause_menu(d, input_fn=_feed(["/task ship the changelog", "/continue"])) == ""

    assert "task queued" in capsys.readouterr().out
    assert [r.text for r in drain_requests(d)] == ["ship the changelog"]


def test_the_pause_menu_refuses_a_bare_directive(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from agent6.ui.cli._steer_menu import pause_menu

    d = _paused(tmp_path)

    assert pause_menu(d, input_fn=_feed(["/task", "/continue"])) == ""

    assert "/task needs the work" in capsys.readouterr().out
    assert [r.text for r in drain_requests(d)] == []


def test_a_partial_command_never_fires(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A prefix drives Tab completion, never an action.

    `/stat` fired `/status` until `/standing` made it ambiguous.
    """
    from agent6.ui.cli._steer_menu import pause_menu

    d = _paused(tmp_path)

    assert pause_menu(d, input_fn=_feed(["/stat", "/continue"])) == ""

    out = capsys.readouterr().out
    assert "unknown command '/stat'" in out and "/status" in out


def test_agent6_steer_acts_on_the_directive_instead_of_sending_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`agent6 steer ID "/task ..."` queues the task instead of sending the text to the model."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / ".state"))
    monkeypatch.chdir(tmp_path)
    d = state_dir(tmp_path) / "sessions" / "runs" / "tiny-run-AAAA11"
    d.mkdir(parents=True)
    (d / "logs.jsonl").write_text("", encoding="utf-8")
    write_worker_pid(d, os.getpid())

    assert main(["steer", "tiny-run", "/task add a --json flag"]) == 0

    assert "task queued" in capsys.readouterr().out
    assert [r.text for r in drain_requests(d)] == ["add a --json flag"]
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


def test_now_acts_the_same_wherever_it_is_typed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One submitter carries `/now`'s urgency on every surface, and `--now` says the same thing."""
    from agent6.sessions.ipc import steer_interrupt_pending, take_steer_answer
    from agent6.ui.directives import submit_composer_line

    monkeypatch.chdir(tmp_path)
    for text, flag in (("/now wrap up", False), ("wrap up", True)):
        d = _live_run(tmp_path / text.replace("/", "").replace(" ", "-"))

        did, said = submit_composer_line(d, text, now=flag)

        assert did and "interrupting the call in flight" in said
        assert steer_interrupt_pending(d)
        assert take_steer_answer(d) == "wrap up"


def test_a_bare_now_refuses_everywhere(tmp_path: Path) -> None:
    from agent6.ui.directives import submit_composer_line

    did, said = submit_composer_line(tmp_path, "/now")

    assert not did and "/now needs the instruction" in said


def test_a_directive_typed_mid_line_is_named_in_the_reply(tmp_path: Path) -> None:
    """A directive is only a leading token; a later one travels as text, and the hint says so."""
    from agent6.sessions.ipc import take_steer_answer
    from agent6.ui.directives import submit_composer_line

    d = _live_run(tmp_path)

    did, said = submit_composer_line(d, "carry on, then /task add a --json flag")

    assert did and "`/task` mid-line is text" in said
    assert take_steer_answer(d) == "carry on, then /task add a --json flag"


def test_a_directive_behind_leading_space_is_not_called_text(tmp_path: Path) -> None:
    """Every directive acts through leading whitespace, and the hint reads the text sent."""
    from agent6.sessions.ipc import steer_interrupt_pending, take_steer_answer
    from agent6.ui.directives import submit_composer_line

    d = _live_run(tmp_path)

    did, said = submit_composer_line(d, "  /now hurry")

    assert did and "mid-line" not in said
    assert steer_interrupt_pending(d)
    assert take_steer_answer(d) == "hurry"


def test_an_ordinary_line_and_a_real_directive_say_nothing_extra(tmp_path: Path) -> None:
    from agent6.ui.directives import submit_composer_line

    d = _live_run(tmp_path / "plain")
    assert submit_composer_line(d, "focus on the parser") == (True, "steering")
    # `/parallel` separates tasks by design, so a later one is meant.
    assert submit_composer_line(d, "/parallel a /parallel b")[1] == "steering"


def test_a_composer_command_cannot_be_a_fresh_runs_task(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A run's first prompt refuses `/task`, for `agent6 run` and the ACP bridge alike.

    It is not a composer, so the text would have become the literal task.
    """
    import time
    from unittest.mock import MagicMock

    from agent6.app.run import run_task
    from agent6.config.layer import load_effective

    rc = run_task(
        load_effective(Path.cwd(), None).config,
        "/task fix the parser",
        started_at=time.time(),
        frontend=MagicMock(),
        session_id="run-REFUSED-AAAAAA",
        mode="run",
    )

    assert rc == 2
    assert "composer command" in capsys.readouterr().err
