# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`stop_session`: the one stop behind every surface."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from agent6.app.stop import StopOutcome, stop_session
from agent6.sessions.ipc import (
    STEER_ANSWER_FILE,
    STOP_REQUEST_FILE,
    pid_record,
    process_identity,
    process_is_alive,
    steer_request_pending,
    write_worker_pid,
)
from agent6.tools.background import BackgroundShells, shell_host_processes
from agent6.types import JailPolicy

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

_GROUP_STOPPER = """
import sys
from pathlib import Path
from agent6.app.stop import stop_session
out = stop_session(Path(sys.argv[1]), wait_s=0.0, grace_s=0.1)
Path(sys.argv[2]).write_text(out.how)
"""

_GROUP_WORKER = """
import os, subprocess, sys, time
from pathlib import Path
from agent6.sessions.ipc import write_worker_pid
d = Path(sys.argv[1])
write_worker_pid(d, os.getpid())
subprocess.Popen([sys.executable, "-c", sys.argv[3], str(d), sys.argv[2]])
while True:
    time.sleep(1)
"""


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


def test_a_run_parked_on_an_approval_reads_the_stop(tmp_path: Path) -> None:
    d = _live_run(tmp_path, "approval-run-AAAAAA")
    with (d / "logs.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "type": "approval.prompt",
                    "prompt_id": "approval-1",
                    "tool": "run_command",
                    "ts": time.time(),
                }
            )
            + "\n"
        )
    worker = _spawn(_ANSWERING_WORKER, str(d))
    write_worker_pid(d, worker.pid)
    try:
        out = stop_session(d, wait_s=5.0, grace_s=0.1)
        assert (out.ok, out.how) == (True, "stopped")
        assert _gone(worker)
    finally:
        worker.kill()


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
            json.dumps(
                {
                    "id": name,
                    "command": "sleep 60",
                    "pid": proc.pid,
                    "pid_start": pid_record(proc.pid).partition(" ")[2],
                }
            ),
            encoding="utf-8",
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


def test_a_started_host_shell_records_its_process_identity(tmp_path: Path) -> None:
    shells = BackgroundShells(tmp_path / "shells")

    def policy_for(argv: tuple[str, ...], rw: tuple[Path, ...]) -> JailPolicy:
        return JailPolicy(
            cwd=tmp_path,
            argv=argv,
            isolation="none",
            extra_rw_paths=rw,
            timeout_s=60.0,
        )

    try:
        view = shells.start(("/bin/sh", "-c", "sleep 60"), policy_for)
        meta = json.loads((tmp_path / "shells" / view.id / "meta.json").read_text(encoding="utf-8"))
        identity = (meta["pid"], meta["pid_start"])
        assert process_is_alive(identity)
        assert shell_host_processes(tmp_path / "shells") == [identity]
    finally:
        shells.stop_all()


def test_a_gone_shell_pid_is_ignored(tmp_path: Path) -> None:
    proc = _spawn("import time; time.sleep(60)")
    identity = process_identity(proc.pid)
    shell = tmp_path / "shells" / "bg1"
    shell.mkdir(parents=True)
    (shell / "meta.json").write_text(
        json.dumps(
            {"id": "bg1", "command": "sleep 60", "pid": identity[0], "pid_start": identity[1]}
        ),
        encoding="utf-8",
    )
    proc.terminate()
    proc.wait(timeout=3.0)
    assert shell_host_processes(tmp_path / "shells") == []


def test_a_namespace_local_shell_has_no_host_target(tmp_path: Path) -> None:
    shell = tmp_path / "shells" / "bg1"
    shell.mkdir(parents=True)
    (shell / "meta.json").write_text(
        json.dumps({"id": "bg1", "command": "sleep 60", "pid": None, "pid_start": None}),
        encoding="utf-8",
    )
    assert shell_host_processes(tmp_path / "shells") == []


def test_a_stop_does_not_kill_a_worker_from_a_new_resume(tmp_path: Path) -> None:
    """The worker that received the request can end before the wait polls it.
    A new resume is a different process even though it owns the same run dir."""
    from agent6.app import stop as stop_mod

    d = _live_run(tmp_path, "resumed-run-AAAAAA")
    first = _spawn("import time; time.sleep(60)")
    replacement = _spawn("import time; time.sleep(60)")
    write_worker_pid(d, first.pid)
    answer: list[StopOutcome] = []
    original_poll = stop_mod._POLL_S  # pyright: ignore[reportPrivateUsage]
    stop_mod._POLL_S = 0.3  # pyright: ignore[reportPrivateUsage]
    stopper = threading.Thread(
        target=lambda: answer.append(stop_session(d, wait_s=0.5, grace_s=0.1))
    )
    try:
        stopper.start()
        deadline = time.monotonic() + 3.0
        while not (d / STOP_REQUEST_FILE).exists():
            assert time.monotonic() < deadline, "the stop request did not land"
            time.sleep(0.01)
        first.terminate()
        first.wait(timeout=3.0)
        write_worker_pid(d, replacement.pid)
        stopper.join(timeout=3.0)
        assert not stopper.is_alive()
        assert replacement.poll() is None, "the stop killed the replacement worker"
        assert answer and answer[0].how == "stopped"
    finally:
        stop_mod._POLL_S = original_poll  # pyright: ignore[reportPrivateUsage]
        for proc in (first, replacement):
            proc.kill()
            proc.wait()


def test_a_recycled_worker_pid_is_not_signalled(tmp_path: Path) -> None:
    d = _live_run(tmp_path, "recycled-worker-run-AAAAAA")
    unrelated = _spawn("import time; time.sleep(60)")
    write_worker_pid(d, unrelated.pid)
    recorded = pid_record(unrelated.pid).partition(" ")[2]
    (d / "worker.pid").write_text(f"{unrelated.pid} old:{recorded}", encoding="utf-8")
    try:
        out = stop_session(d, wait_s=0.0, grace_s=0.1)
        assert (out.ok, out.how) == (False, "not_live")
        assert unrelated.poll() is None
        assert not (d / STOP_REQUEST_FILE).exists()
    finally:
        unrelated.kill()
        unrelated.wait()


def test_a_recycled_shell_pid_is_not_signalled(tmp_path: Path) -> None:
    """A shell record identifies the process that the run started, not a later
    process that inherited its pid."""
    d = _live_run(tmp_path, "recycled-shell-run-AAAAAA")
    worker = _spawn(_DEAF_WORKER)
    unrelated = _spawn("import time; time.sleep(60)")
    original_start = f"old:{pid_record(unrelated.pid).partition(' ')[2]}"
    write_worker_pid(d, worker.pid)
    shell = d / "shells" / "bg1"
    shell.mkdir(parents=True)
    (shell / "meta.json").write_text(
        json.dumps(
            {
                "id": "bg1",
                "command": "sleep 60",
                "pid": unrelated.pid,
                "pid_start": original_start,
            }
        ),
        encoding="utf-8",
    )
    try:
        out = stop_session(d, wait_s=0.0, grace_s=0.1)
        assert out.how == "killed"
        assert unrelated.poll() is None, "the stop signalled a process that did not start the shell"
        assert "background command" not in out.message
    finally:
        for proc in (worker, unrelated):
            proc.kill()
            proc.wait()


def test_a_stop_never_signals_its_own_process_group(tmp_path: Path) -> None:
    """A stop can share a process group with the worker it targets. The worker
    must die without the group signal taking the stopping process with it."""
    d = _live_run(tmp_path, "shared-group-run-AAAAAA")
    result = tmp_path / "stop.out"
    worker = _spawn(_GROUP_WORKER, str(d), str(result), _GROUP_STOPPER)
    try:
        deadline = time.monotonic() + 5.0
        while not result.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert result.read_text(encoding="utf-8") == "killed"
        assert worker.wait(timeout=3.0) == -15
    finally:
        worker.kill()
        worker.wait()


def test_a_stop_that_finds_nothing_to_signal_says_so(tmp_path: Path) -> None:
    """The recorded worker is this process (a front-end that is the worker), so
    the kill signals nothing: the outcome says the run did not answer and
    nothing was left to signal, never that a worker was killed."""
    d = _live_run(tmp_path, "self-run-AAAAAA")
    write_worker_pid(d, os.getpid())
    out = stop_session(d, wait_s=0.0, grace_s=0.1)
    assert (out.ok, out.how) == (True, "stale"), out
    assert "nothing of it is left to signal" in out.message and "killed" not in out.message


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
