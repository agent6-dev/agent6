# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 steer`: the cron-friendly wrapper over the one steer channel."""

from __future__ import annotations

import os
import pathlib

import pytest

from agent6 import paths
from agent6.app import stop as app_stop
from agent6.sessions import ipc
from agent6.ui.cli import main


def _run_session(tmp_path: pathlib.Path, session_id: str) -> pathlib.Path:
    d = paths.state_dir(tmp_path) / "sessions" / "runs" / session_id
    d.mkdir(parents=True)
    (d / "logs.jsonl").write_text("", encoding="utf-8")
    return d


def test_steer_queues_for_a_live_run(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / ".state"))
    monkeypatch.chdir(tmp_path)
    d = _run_session(tmp_path, "tiny-run-AAAA11")
    ipc.write_worker_pid(d, os.getpid())

    assert main(["steer", "tiny-run", "land your best patch now"]) == 0
    out = capsys.readouterr().out
    assert "steering for tiny-run-AAAA11" in out
    # The one shared channel: request marker + answer, exactly what the
    # composers write and the loop consumes. A plain steer never carries the
    # interrupt urgency; --now writes it into the marker.

    assert ipc.steer_request_pending(d)
    assert not ipc.steer_interrupt_pending(d)
    assert ipc.take_steer_answer(d) == "land your best patch now"

    assert main(["steer", "tiny-run", "wrap up", "--now"]) == 0
    out = capsys.readouterr().out
    assert "interrupting the call in flight" in out
    assert ipc.steer_interrupt_pending(d)
    assert ipc.take_steer_answer(d) == "wrap up"


def test_steer_reports_a_failed_marker_write(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A queued message is only true when the request marker landed."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / ".state"))
    monkeypatch.chdir(tmp_path)
    session = _run_session(tmp_path, "tiny-run-EEEE55")
    ipc.write_worker_pid(session, os.getpid())
    (session / "steer.request").mkdir()

    assert main(["steer", "tiny-run-EEEE55", "hello"]) == 1
    captured = capsys.readouterr()
    assert "could not write the steer request" in captured.err
    assert "steer queued" not in captured.out
    assert ipc.take_steer_answer(session) is None


def test_steer_refuses_a_session_that_is_not_running(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A dead session's steer is refused with the `resume --steer` remedy, never parked silently."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / ".state"))
    monkeypatch.chdir(tmp_path)
    d = _run_session(tmp_path, "tiny-run-BBBB22")
    ipc.write_worker_pid(d, 10**9)  # a pid that is never alive

    assert main(["steer", "tiny-run-BBBB22", "hello"]) == 2
    err = capsys.readouterr().err
    assert "not running" in err
    assert "agent6 resume tiny-run-BBBB22 --steer" in err
    assert not ipc.steer_request_pending(d)


def test_steer_refuses_a_finished_run_even_with_a_stale_live_pid(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A session end outranks a stale or reused worker pid."""
    import json

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / ".state"))
    monkeypatch.chdir(tmp_path)
    d = _run_session(tmp_path, "done-run-FFFF66")
    (d / "logs.jsonl").write_text(
        json.dumps({"type": "session.start", "user_task": "t"})
        + "\n"
        + json.dumps({"type": "session.end", "reason": "finish_session", "all_passed": True})
        + "\n",
        encoding="utf-8",
    )
    ipc.write_worker_pid(d, os.getpid())

    assert main(["steer", "done-run-FFFF66", "more work"]) == 2
    err = capsys.readouterr().err
    assert "not running" in err
    assert "agent6 resume done-run-FFFF66 --steer" in err
    assert not ipc.steer_request_pending(d)


def test_steer_compact_writes_the_compaction_marker(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / ".state"))
    monkeypatch.chdir(tmp_path)
    d = _run_session(tmp_path, "tiny-run-GGGG77")
    ipc.write_worker_pid(d, os.getpid())

    assert main(["steer", "tiny-run-GGGG77", "/compact focus on test failures"]) == 0
    out = capsys.readouterr().out
    assert "compaction requested" in out
    assert ipc.read_compact_request(d) == "focus on test failures"
    assert not ipc.steer_request_pending(d)
    assert ipc.take_steer_answer(d) is None


def test_steer_btw_opens_a_side_ask_instead_of_queuing_the_directive(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / ".state"))
    monkeypatch.chdir(tmp_path)
    d = _run_session(tmp_path, "tiny-run-HHHH88")
    ipc.write_worker_pid(d, os.getpid())
    opened: list[tuple[pathlib.Path, str]] = []

    def _open_btw(session_dir: pathlib.Path, question: str) -> tuple[bool, str]:
        opened.append((session_dir, question))
        return True, "[agent6] btw side-ask-IIII99 opened"

    monkeypatch.setattr("agent6.ui.btw.open_btw", _open_btw, raising=False)

    assert main(["steer", "tiny-run-HHHH88", "/btw is the migration safe?"]) == 0
    assert opened == [(d, "is the migration safe?")]
    assert "btw side-ask-IIII99 opened" in capsys.readouterr().out
    assert not ipc.steer_request_pending(d)
    assert ipc.take_steer_answer(d) is None


def test_steer_stop_is_the_one_stop(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`agent6 steer ID /stop` is `agent6 stop ID`, not a steer text for the model."""
    from agent6.app import stop

    calls: list[pathlib.Path] = []

    def _fake(session_dir: pathlib.Path, *, after_step: bool = False) -> stop.StopOutcome:
        calls.append(session_dir)
        return stop.StopOutcome(session_dir.name, True, "stopped", f"{session_dir.name} stopped")

    monkeypatch.setattr(app_stop, "stop_session", _fake)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / ".state"))
    monkeypatch.chdir(tmp_path)
    d = _run_session(tmp_path, "tiny-run-BBBB22")
    ipc.write_worker_pid(d, os.getpid())
    assert main(["steer", "tiny-run", "/stop"]) == 0
    assert calls == [d] and "tiny-run-BBBB22 stopped" in capsys.readouterr().out
    assert not ipc.steer_request_pending(d)


def test_steer_reports_an_unknown_id(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / ".state"))
    monkeypatch.chdir(tmp_path)
    assert main(["steer", "nonesuch", "hello"]) == 2
    assert "ERROR" in capsys.readouterr().err


def test_steer_notes_an_unanswered_prompt_park(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A steer to a run parked on an approval says so instead of implying delivery."""
    import json

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / ".state"))
    monkeypatch.chdir(tmp_path)
    d = _run_session(tmp_path, "tiny-run-CCCC33")
    ipc.write_worker_pid(d, os.getpid())
    events = [
        {"type": "session.start", "mode": "run", "user_task": "t"},
        {
            "type": "approval.prompt",
            "id": "approval-1",
            "prompt": "Allow fetch: x",
            "ts": "2026-08-24T00:00:00+00:00",
        },
    ]
    (d / "logs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")

    assert main(["steer", "tiny-run-CCCC33", "hello"]) == 0
    out = capsys.readouterr().out
    assert "the run is waiting (approval" in out
    assert "agent6 attach tiny-run-CCCC33" in out


def test_steer_names_the_answer_verb_for_a_question(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`agent6 answer` takes a question in whichever seat the run waits.

    Its terminal prompt reads the answer file too; naming it for an approval sent the operator
    straight to a refusal.
    """
    import json

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / ".state"))
    monkeypatch.chdir(tmp_path)
    d = _run_session(tmp_path, "tiny-run-DDDD44")
    ipc.write_worker_pid(d, os.getpid())
    events = [
        {"type": "session.start", "mode": "run", "user_task": "t"},
        {
            "type": "question.prompt",
            "id": "question-1",
            "questions": [{"question": "Which port?"}],
            "ts": "2026-08-24T00:00:00+00:00",
        },
    ]
    (d / "logs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")

    # No away-mode and no front-end: the run's own terminal prompt reads the
    # answer file, so the verb delivers there too.
    assert main(["steer", "tiny-run-DDDD44", "hello"]) == 0
    out = capsys.readouterr().out
    assert "the run is waiting (question" in out
    assert "agent6 answer tiny-run-DDDD44" in out

    # Detached on "wait": the same verb.
    ipc.set_away_mode(d, "wait")
    assert main(["steer", "tiny-run-DDDD44", "hello"]) == 0
    assert "agent6 answer tiny-run-DDDD44" in capsys.readouterr().out


def test_steer_refuses_a_view_only_word(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`agent6 steer ID /shells` is refused: a view acts on it, and a script has no view."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / ".state"))
    monkeypatch.chdir(tmp_path)
    d = _run_session(tmp_path, "tiny-run-AAAA11")
    ipc.write_worker_pid(d, os.getpid())

    assert main(["steer", "tiny-run", "/shells"]) == 2
    assert "acts on a view of the run" in capsys.readouterr().err
    assert not ipc.steer_request_pending(d)
