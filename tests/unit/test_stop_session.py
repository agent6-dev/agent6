# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`stop_session`: the one stop behind every surface."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from agent6.app.stop import stop_session
from agent6.sessions.ipc import (
    STEER_ANSWER_FILE,
    STOP_REQUEST_FILE,
    steer_request_pending,
    write_worker_pid,
)

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

_DEAF_WORKER = "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"


def _live_run(tmp_path: Path, name: str) -> Path:
    d = tmp_path / "sessions" / "runs" / name
    d.mkdir(parents=True)
    (d / "logs.jsonl").write_text(
        json.dumps({"type": "session.start", "session_id": name, "user_task": "t"}) + "\n",
        encoding="utf-8",
    )
    return d


def _spawn(script: str, *args: str) -> subprocess.Popen[bytes]:
    return subprocess.Popen([sys.executable, "-c", script, *args], start_new_session=True)


def _gone(proc: subprocess.Popen[bytes], within_s: float = 3.0) -> bool:
    try:
        proc.wait(timeout=within_s)
    except subprocess.TimeoutExpired:
        return False
    return True


def test_after_step_writes_the_marker_alone(tmp_path: Path) -> None:
    d = _live_run(tmp_path, "step-run-AAAAAA")
    write_worker_pid(d, os.getpid())
    out = stop_session(d, after_step=True)
    assert (out.ok, out.how) == (True, "after_step") and "after its current step" in out.message
    assert (d / STOP_REQUEST_FILE).exists() and not (d / STEER_ANSWER_FILE).exists()


def test_a_run_that_answers_stops_and_is_not_killed(tmp_path: Path) -> None:
    """The cooperative path: both bridges land, the worker reads the abort and
    ends the run itself; the verb reports "stopped" once the log says so."""
    d = _live_run(tmp_path, "kind-run-AAAAAA")
    proc = _spawn(_ANSWERING_WORKER, str(d))
    write_worker_pid(d, proc.pid)
    try:
        out = stop_session(d, wait_s=5.0, grace_s=0.5)
        assert (out.ok, out.how) == (True, "stopped"), out
        assert (d / STEER_ANSWER_FILE).read_text(encoding="utf-8").strip() == "abort"
        assert steer_request_pending(d) and (d / STOP_REQUEST_FILE).exists()
        assert _gone(proc)
    finally:
        proc.kill()


def test_a_worker_that_does_not_answer_is_killed_with_its_host_commands(tmp_path: Path) -> None:
    """Nothing read the requests (a wedged stream, a parked prompt): after the
    wait the worker and its host-side background commands get SIGTERM, and
    SIGKILL after the grace for one that ignores SIGTERM."""
    d = _live_run(tmp_path, "deaf-run-AAAAAA")
    worker = _spawn("import time; time.sleep(60)")
    write_worker_pid(d, worker.pid)
    polite = _spawn("import time; time.sleep(60)")
    deaf = _spawn(_DEAF_WORKER)
    for name, proc in (("bg1", polite), ("bg2", deaf)):
        shell = d / "shells" / name
        shell.mkdir(parents=True)
        (shell / "meta.json").write_text(
            json.dumps({"id": name, "command": "sleep 60", "pid": proc.pid}), encoding="utf-8"
        )
    started = time.monotonic()
    try:
        out = stop_session(d, wait_s=0.3, grace_s=0.5)
        assert (out.ok, out.how) == (True, "killed"), out
        assert "2 background commands" in out.message and "resume continues it" in out.message
        assert _gone(worker) and _gone(polite) and _gone(deaf)
        assert time.monotonic() - started < 5.0
    finally:
        for proc in (worker, polite, deaf):
            proc.kill()


def test_a_session_that_is_not_live_is_refused_with_its_state(tmp_path: Path) -> None:
    d = _live_run(tmp_path, "done-run-AAAAAA")
    with (d / "logs.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps({"type": "session.end", "reason": "finish_session", "all_passed": True})
            + "\n"
        )
    out = stop_session(d)
    assert (out.ok, out.how) == (False, "not_live")
    assert "is already passed" in out.message and "nothing to stop" in out.message
    assert not (d / STOP_REQUEST_FILE).exists() and not (d / STEER_ANSWER_FILE).exists()
