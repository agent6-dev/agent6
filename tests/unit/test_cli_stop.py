# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 stop`: the one stop verb, at the top level with the other verbs that
act on a live run."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from agent6.app import stop as stop_mod
from agent6.paths import state_dir
from agent6.sessions.ipc import STEER_ANSWER_FILE, STOP_REQUEST_FILE, write_worker_pid
from agent6.ui.cli import main
from agent6.ui.cli.parser import build_parser

_ANSWERING_WORKER = """
import json, sys, time
from pathlib import Path
d = Path(sys.argv[1])
while not (d / "steer.answer").exists():
    time.sleep(0.05)
end = {"type": "session.end", "reason": "steer_abort", "all_passed": False}
with (d / "logs.jsonl").open("a") as fh:
    fh.write(json.dumps(end) + "\\n")
"""


def _run(repo: Path, name: str, *, finished: bool = False) -> Path:
    d = state_dir(repo) / "sessions" / "runs" / name
    d.mkdir(parents=True)
    events: list[dict[str, object]] = [
        {"type": "session.start", "session_id": name, "mode": "run", "user_task": "t"}
    ]
    if finished:
        events.append({"type": "session.end", "reason": "finish_session", "all_passed": True})
    (d / "logs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    (d / "manifest.json").write_text(
        json.dumps({"version": 3, "session_id": name, "mode": "run", "user_task": "t"}),
        encoding="utf-8",
    )
    return d


def test_stop_is_a_top_level_verb_and_sessions_stop_is_gone() -> None:
    """The verbs that act on a live run sit at the top level (attach, steer,
    answer, exec, forward); stop hid under the record verbs."""
    parser = build_parser()
    args = parser.parse_args(["stop", "some-run", "--after-step", "--all"])
    assert (args.command, args.session_id, args.after_step, args.all) == (
        "stop",
        "some-run",
        True,
        True,
    )
    with pytest.raises(SystemExit):
        parser.parse_args(["sessions", "stop", "some-run"])


def test_after_step_writes_the_marker_and_names_the_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    d = _run(tmp_path, "step-run-AAAAAA")
    write_worker_pid(d, os.getpid())
    assert main(["stop", "step-run", "--after-step"]) == 0
    out = capsys.readouterr().out
    assert "step-run-AAAAAA stops after its current step" in out
    assert "resume with:  agent6 resume step-run-AAAAAA" in out
    assert (d / STOP_REQUEST_FILE).exists() and not (d / STEER_ANSWER_FILE).exists()


def test_all_stops_every_live_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    live = [_run(tmp_path, "live-one-AAAAAA"), _run(tmp_path, "live-two-BBBBBB")]
    _run(tmp_path, "done-run-CCCCCC", finished=True)
    workers = [
        subprocess.Popen([sys.executable, "-c", _ANSWERING_WORKER, str(d)], start_new_session=True)
        for d in live
    ]
    for d, proc in zip(live, workers, strict=True):
        write_worker_pid(d, proc.pid)
    try:
        assert main(["stop", "--all"]) == 0
        out = capsys.readouterr().out
        assert "live-one-AAAAAA stopped" in out and "live-two-BBBBBB stopped" in out
        assert "done-run-CCCCCC" not in out
        for proc in workers:
            proc.wait(timeout=5)
    finally:
        for proc in workers:
            proc.kill()


def test_all_with_nothing_live_is_a_successful_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _run(tmp_path, "done-run-AAAAAA", finished=True)
    assert main(["stop", "--all"]) == 0
    captured = capsys.readouterr()
    assert "no live session to stop" in captured.err and not captured.out


def test_a_fanout_stops_only_its_live_lanes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    fan = _run(tmp_path, "fan-run-AAAAAA")
    live = _run(tmp_path, "fan-run-AAAAAA-l1")
    ended = _run(tmp_path, "fan-run-AAAAAA-l2", finished=True)
    (fan / "manifest.json").write_text(
        json.dumps(
            {
                "version": 3,
                "session_id": fan.name,
                "mode": "run",
                "user_task": "t",
                "fanout": {"lanes": 2, "spec": "2"},
            }
        ),
        encoding="utf-8",
    )
    for lane, number in ((live, 1), (ended, 2)):
        (lane / "manifest.json").write_text(
            json.dumps(
                {
                    "version": 3,
                    "session_id": lane.name,
                    "mode": "run",
                    "user_task": "t",
                    "parallel": {"group": fan.name, "lane": number, "coordinator": fan.name},
                }
            ),
            encoding="utf-8",
        )
    workers = [
        subprocess.Popen([sys.executable, "-c", _ANSWERING_WORKER, str(d)], start_new_session=True)
        for d in (fan, live)
    ]
    for d, proc in zip((fan, live), workers, strict=True):
        write_worker_pid(d, proc.pid)
    try:
        assert main(["stop", "fan-run-AAAAAA"]) == 0
        out = capsys.readouterr().out
        assert "fan-run-AAAAAA stopped with its 1 live lane; what they landed" in out
        assert "fan-run-AAAAAA-l2" not in out
        assert "resume with:" not in out
        for proc in workers:
            proc.wait(timeout=5.0)
    finally:
        for proc in workers:
            proc.kill()


_DRAINING_COORDINATOR = """
import json, sys, time
from pathlib import Path
from agent6.viewmodel import session_is_live
d, lane = Path(sys.argv[1]), Path(sys.argv[2])
while session_is_live(lane):
    time.sleep(0.05)
end = {"type": "session.end", "reason": "steer_abort", "all_passed": False}
with (d / "logs.jsonl").open("a") as fh:
    fh.write(json.dumps(end) + "\\n")
"""


def _fanout(tmp_path: Path, fan: str, lane: str) -> tuple[Path, Path]:
    fan_dir, lane_dir = _run(tmp_path, fan), _run(tmp_path, lane)
    (fan_dir / "manifest.json").write_text(
        json.dumps(
            {
                "version": 3,
                "session_id": fan,
                "mode": "run",
                "user_task": "t",
                "fanout": {"lanes": 1, "spec": "1"},
            }
        ),
        encoding="utf-8",
    )
    (lane_dir / "manifest.json").write_text(
        json.dumps(
            {
                "version": 3,
                "session_id": lane,
                "mode": "run",
                "user_task": "t",
                "parallel": {"group": fan, "lane": 1, "coordinator": fan},
            }
        ),
        encoding="utf-8",
    )
    return fan_dir, lane_dir


def test_a_fanouts_lanes_end_before_its_coordinator_drains(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A coordinator ends only after its lanes: it drains them, imports what
    they landed and ranks it. Stopped first, with a run's wait, it was killed
    mid-drain and nothing was imported."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(stop_mod, "FANOUT_WAIT_S", 3.0)
    fan, lane = _fanout(tmp_path, "drain-run-AAAAAA", "drain-run-AAAAAA-l1")
    coordinator = subprocess.Popen(
        [sys.executable, "-c", _DRAINING_COORDINATOR, str(fan), str(lane)],
        start_new_session=True,
    )
    worker = subprocess.Popen(
        [sys.executable, "-c", _ANSWERING_WORKER, str(lane)], start_new_session=True
    )
    write_worker_pid(fan, coordinator.pid)
    write_worker_pid(lane, worker.pid)
    try:
        assert main(["stop", "drain-run-AAAAAA"]) == 0
        out = capsys.readouterr().out
        assert "drain-run-AAAAAA stopped with its 1 live lane" in out
        assert coordinator.wait(timeout=5.0) == 0, "the coordinator was killed, not drained"
        assert worker.wait(timeout=5.0) == 0
    finally:
        for proc in (coordinator, worker):
            proc.kill()


def test_a_session_that_is_not_live_gets_a_note_and_exit_0(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A stop that finds nothing running has done what was asked."""
    monkeypatch.chdir(tmp_path)
    _run(tmp_path, "done-run-AAAAAA", finished=True)
    assert main(["stop", "done-run"]) == 0
    err = capsys.readouterr().err
    assert "done-run-AAAAAA is already passed; nothing to stop" in err


def test_an_unknown_session_is_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _run(tmp_path, "some-run-AAAAAA")
    assert main(["stop", "nope"]) == 2
    assert capsys.readouterr().err


def _session_dir(repo: Path, session_id: str) -> Path:
    from agent6.sessions.layout import SessionLayout

    layout = SessionLayout(state_dir=state_dir(repo), session_id=session_id)
    layout.ensure()
    layout.manifest_path.write_text('{"version": 2}', encoding="utf-8")
    (layout.session_dir / "logs.jsonl").write_text("", encoding="utf-8")
    return layout.session_dir


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes through a read-only dir")
def test_a_marker_that_cannot_be_written_is_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The write error was swallowed and the command announced a stop over a
    marker that never landed."""
    from agent6.sessions.ipc import stop_request_pending

    monkeypatch.chdir(tmp_path)
    rd = _session_dir(tmp_path, "live-run-RO1111")
    write_worker_pid(rd, os.getpid())
    rd.chmod(0o555)
    try:
        assert main(["stop", "live-run-RO1111", "--after-step"]) == 1
        assert not stop_request_pending(rd)
    finally:
        rd.chmod(0o700)
    captured = capsys.readouterr()
    assert "could not write the stop request" in captured.err
    assert "stops after" not in captured.out


def test_a_finished_run_with_a_lingering_pid_is_already_over(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """session.end lands before teardown clears worker.pid; in that window the
    loop has exited, so a promised stop would be one nobody keeps. The gate is
    the liveness owner, not the pid."""
    from agent6.sessions.ipc import stop_request_pending

    monkeypatch.chdir(tmp_path)
    rd = _session_dir(tmp_path, "done-run-CCC333")
    (rd / "logs.jsonl").write_text(
        '{"type": "session.start", "mode": "run"}\n'
        '{"type": "session.end", "all_passed": true, "reason": "finish_session"}\n',
        encoding="utf-8",
    )
    write_worker_pid(rd, os.getpid())  # teardown not finished yet
    assert main(["stop", "done-run-CCC333"]) == 0
    assert not stop_request_pending(rd)
    err = capsys.readouterr().err
    assert "already passed" in err and "not running" not in err


def test_a_parked_run_says_it_has_not_started(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from agent6.sessions.ipc import stop_request_pending

    monkeypatch.chdir(tmp_path)
    rd = _session_dir(tmp_path, "parked-run-DDD444")
    (rd / "manifest.json").write_text(
        '{"version": 3, "session_id": "parked-run-DDD444", "mode": "run",'
        ' "parked_task": "wait for the checkout"}',
        encoding="utf-8",
    )
    assert main(["stop", "parked-run-DDD444"]) == 0
    assert not stop_request_pending(rd)
    err = capsys.readouterr().err
    assert "is parked" in err and "has not started" in err


def test_a_dead_run_is_not_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from agent6.sessions.ipc import stop_request_pending

    monkeypatch.chdir(tmp_path)
    rd = _session_dir(tmp_path, "dead-run-BBB222")
    write_worker_pid(rd, 2**31 - 1)  # no such process
    assert main(["stop", "dead-run-BBB222"]) == 0
    assert not stop_request_pending(rd)
    assert "not running" in capsys.readouterr().err


def test_a_fan_out_gets_no_resume_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A fan-out coordinator has no loop to resume: the stop reaches it (and
    its live lanes) and the message promises no resume."""
    from agent6.sessions.ipc import stop_request_pending

    monkeypatch.chdir(tmp_path)
    rd = _session_dir(tmp_path, "fan-AAAA11")
    (rd / "manifest.json").write_text(
        '{"version": 3, "mode": "run", "fanout": {"lanes": 2, "spec": "2"}}', encoding="utf-8"
    )
    write_worker_pid(rd, os.getpid())
    assert main(["stop", "fan-AAAA11", "--after-step"]) == 0
    assert stop_request_pending(rd)
    out = capsys.readouterr().out
    assert "fan-AAAA11 stops after its current step" in out and "resume" not in out
