# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 answer`: the headless seat at an `ask_user` question.

`agent6 steer` refuses a run blocked on a question and every other way in needs a terminal.
"""

from __future__ import annotations

import json
import os
import pathlib

import pytest

from agent6 import paths
from agent6.sessions import ipc, process
from agent6.sessions import layout as sessions_layout
from agent6.ui.cli import answer_cmd as cli_answer_cmd  # pyright: ignore[reportPrivateUsage]


def _run_with_question(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, *, questions: list[dict[str, object]]
) -> sessions_layout.SessionLayout:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)

    layout = sessions_layout.SessionLayout(
        state_dir=paths.state_dir(repo), session_id="curious-fox-AAAA11"
    )
    layout.ensure()
    events: list[dict[str, object]] = [
        {"type": "session.start", "mode": "run", "user_task": "t"},
        {"type": "question.prompt", "id": "question-1", "questions": questions},
    ]
    layout.logs_path.write_text(
        "".join(json.dumps(e) + "\n" for e in events),
        encoding="utf-8",
    )
    # this process = a live worker
    process.write_worker_pid(layout.session_dir, os.getpid())
    # A detached run left on "wait" is the seat this verb exists for.
    ipc.set_away_mode(layout.session_dir, "wait")
    return layout


def test_answer_writes_the_file_the_run_is_waiting_on(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    layout = _run_with_question(
        tmp_path,
        monkeypatch,
        questions=[{"question": "Which port?", "options": ["8080", "9090"]}],
    )

    assert cli_answer_cmd._cmd_answer("curious-fox", ("9090",)) == 0

    assert ipc.read_question_answers(layout.session_dir, "question-1", timeout_s=1.0) == ("9090",)
    assert "answered curious-fox-AAAA11" in capsys.readouterr().out


def test_answer_with_no_text_prints_the_question_and_its_options(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A bare call prints the question and its options."""
    _run_with_question(
        tmp_path,
        monkeypatch,
        questions=[{"question": "Which port?", "options": ["8080", "9090"]}],
    )

    assert cli_answer_cmd._cmd_answer("curious-fox", ()) == 0

    out = capsys.readouterr().out
    assert "Which port?" in out
    assert "options: 8080, 9090" in out


def test_a_short_answer_list_is_refused_rather_than_misaligned(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A short answer list is refused rather than misaligned.

    Answers align to the prompt's questions by index.
    """
    layout = _run_with_question(
        tmp_path,
        monkeypatch,
        questions=[{"question": "Which port?"}, {"question": "Which host?"}],
    )

    assert cli_answer_cmd._cmd_answer("curious-fox", ("9090",)) == 2

    err = capsys.readouterr().err
    assert "2 question(s); 1 answer(s) given" in err
    assert not (layout.session_dir / "questions" / "question-1.answer").exists()


def test_answer_refuses_a_run_that_is_not_waiting(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)

    layout = sessions_layout.SessionLayout(
        state_dir=paths.state_dir(repo), session_id="curious-fox-AAAA11"
    )
    layout.ensure()
    layout.logs_path.write_text(
        json.dumps({"type": "session.start", "mode": "run", "user_task": "t"}) + "\n",
        encoding="utf-8",
    )
    process.write_worker_pid(layout.session_dir, os.getpid())
    ipc.set_away_mode(layout.session_dir, "wait")

    assert cli_answer_cmd._cmd_answer("curious-fox", ("yes",)) == 2

    assert "no pending question" in capsys.readouterr().err


def test_answer_on_an_ended_run_saves_it_for_the_resume(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An ended run's question stays answerable; the answer waits for the resume that reads it."""
    layout = _run_with_question(tmp_path, monkeypatch, questions=[{"question": "Which port?"}])
    process.write_worker_pid(layout.session_dir, 999_999_999)

    assert cli_answer_cmd._cmd_answer("curious-fox", ("9090",)) == 0

    assert ipc.read_question_answers(layout.session_dir, "question-1", timeout_s=1.0) == ("9090",)
    out = capsys.readouterr().out
    assert "saved" in out and "agent6 resume curious-fox-AAAA11" in out


def test_answer_refuses_a_question_another_surface_answered(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The first answer stands: a second, from this verb, is refused and named."""
    layout = _run_with_question(tmp_path, monkeypatch, questions=[{"question": "Which port?"}])
    assert ipc.write_question_answers(layout.session_dir, "question-1", ["8080"])

    assert cli_answer_cmd._cmd_answer("curious-fox", ("9090",)) == 2

    assert "already answered from another surface" in capsys.readouterr().err
    assert ipc.read_question_answers(layout.session_dir, "question-1", timeout_s=1.0) == ("8080",)


def test_answer_reaches_a_run_waiting_at_its_own_terminal(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Answer reaches a run waiting at its own terminal.

    A foreground run's terminal prompt reads the answer file while it waits, so the verb takes a
    run with no away-mode and no front-end too.
    """
    layout = _run_with_question(tmp_path, monkeypatch, questions=[{"question": "Which port?"}])
    (layout.session_dir / "approvals" / "away.mode").unlink()  # no away-mode, no front-end

    assert cli_answer_cmd._cmd_answer("curious-fox", ("9090",)) == 0

    assert ipc.read_question_answers(layout.session_dir, "question-1", timeout_s=1.0) == ("9090",)
    assert "answered curious-fox-AAAA11" in capsys.readouterr().out


def test_a_bare_call_prints_the_question_even_on_a_terminal_bound_run(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Reading the question needs no delivery channel."""
    layout = _run_with_question(
        tmp_path, monkeypatch, questions=[{"question": "Which port?", "options": ["8080"]}]
    )
    (layout.session_dir / "approvals" / "away.mode").unlink()  # terminal-bound

    assert cli_answer_cmd._cmd_answer("curious-fox", ()) == 0

    assert "Which port?" in capsys.readouterr().out


def test_a_live_run_with_no_question_says_so(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A live run with no question says so.

    The refusal names the state the run is in, not one it is not.
    """
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)

    layout = sessions_layout.SessionLayout(
        state_dir=paths.state_dir(repo), session_id="curious-fox-AAAA11"
    )
    layout.ensure()
    layout.logs_path.write_text(
        json.dumps({"type": "session.start", "mode": "run", "user_task": "t"}) + "\n",
        encoding="utf-8",
    )
    # live, no away-mode, no front-end
    process.write_worker_pid(layout.session_dir, os.getpid())

    assert cli_answer_cmd._cmd_answer("curious-fox", ("yes",)) == 2

    assert "has no pending question" in capsys.readouterr().err


def test_answer_asks_the_affordance_question_the_other_verbs_ask(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The command words the run's state through the one liveness predicate the verbs share.

    `answer`, `steer` and the web's approve read the same status word; a run with nothing
    pending is refused for that, whether it runs or not.
    """
    import inspect

    from agent6.ui.cli import answer_cmd

    assert "session_is_live" in inspect.getsource(answer_cmd)
    assert "worker_is_alive" not in inspect.getsource(answer_cmd)

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / ".state"))
    monkeypatch.chdir(tmp_path)
    d = paths.state_dir(tmp_path) / "sessions" / "runs" / "gone-run-AAAA11"
    d.mkdir(parents=True)
    (d / "logs.jsonl").write_text("", encoding="utf-8")

    assert cli_answer_cmd._cmd_answer("gone-run", ("yes",)) == 2

    assert "gone-run-AAAA11 has no pending question" in capsys.readouterr().err


def _run_with_events(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, events: list[dict[str, object]]
) -> sessions_layout.SessionLayout:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    layout = sessions_layout.SessionLayout(
        state_dir=paths.state_dir(repo), session_id="curious-fox-AAAA11"
    )
    layout.ensure()
    layout.logs_path.write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    process.write_worker_pid(layout.session_dir, os.getpid())
    return layout


def test_answer_lists_every_pending_set_and_picks_one_by_id(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """With several sets pending, a bare call lists them and an answer names its set."""
    layout = _run_with_events(
        tmp_path,
        monkeypatch,
        [
            {"type": "session.start", "mode": "run", "user_task": "t"},
            {
                "type": "question.prompt",
                "id": "question-1",
                "questions": [{"question": "Which port?"}],
                "wait": False,
            },
            {
                "type": "question.prompt",
                "id": "question-2",
                "questions": [{"question": "Keep the cache?"}, {"question": "Which DB?"}],
                "wait": False,
            },
        ],
    )
    assert cli_answer_cmd._cmd_answer("curious-fox", ()) == 0
    listed = capsys.readouterr().out
    assert "question-1" in listed and "question-2" in listed and "Which DB?" in listed
    assert "--question question-2 TEXT TEXT" in listed

    assert cli_answer_cmd._cmd_answer("curious-fox", ("yes", "pg")) == 2
    assert "--question" in capsys.readouterr().err

    assert cli_answer_cmd._cmd_answer("curious-fox", ("yes", "pg"), question_id="question-2") == 0
    assert ipc.read_question_answers(layout.session_dir, "question-2", timeout_s=1.0) == (
        "yes",
        "pg",
    )
    assert "question-2" in capsys.readouterr().out
    assert not ipc.question_answers_written(layout.session_dir, "question-1")

    assert cli_answer_cmd._cmd_answer("curious-fox", ("x",), question_id="question-9") == 2
    assert "no pending question 'question-9'" in capsys.readouterr().err


def test_an_answer_to_a_queued_question_names_the_next_turn(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A live run reads a queued question's answer at its next turn, and the command says so."""
    _run_with_events(
        tmp_path,
        monkeypatch,
        [
            {"type": "session.start", "mode": "run", "user_task": "t"},
            {
                "type": "question.prompt",
                "id": "question-1",
                "questions": [{"question": "Which port?"}],
                "wait": False,
            },
        ],
    )
    assert cli_answer_cmd._cmd_answer("curious-fox", ("9090",)) == 0
    assert "next turn" in capsys.readouterr().out


def test_the_printed_answer_form_parses() -> None:
    """`agent6 answer RUN --question ID TEXT...`, the form the listing prints, parses as printed."""
    from agent6.ui.cli import parser as cli_parser

    ns = cli_parser.build_parser().parse_args(
        ["answer", "curious-fox", "--question", "question-2", "yes", "pg"]
    )
    assert (ns.target, ns.question, ns.answers) == ("curious-fox", "question-2", ["yes", "pg"])
    bare = cli_parser.build_parser().parse_args(["answer", "curious-fox"])
    assert (bare.question, bare.answers) == (None, [])
