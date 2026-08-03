# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the file-based approval bridge."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import pytest

from agent6.sessions.ipc import (
    frontend_is_live,
    read_answer,
    read_question_answers,
    register_frontend,
    unregister_frontend,
    write_answer,
    write_question_answers,
    write_steer_answer,
)


def test_no_tui_pid_means_not_live(tmp_path: Path) -> None:
    assert frontend_is_live(tmp_path) is False


def test_dead_pid_is_not_live(tmp_path: Path) -> None:
    # PID 1 is init; signal-0 to it from a non-root process raises PermissionError
    # which we treat as "not us" -> dead. PID 0 is invalid -> ProcessLookupError.
    register_frontend(tmp_path, 999999999)  # almost certainly not allocated
    assert frontend_is_live(tmp_path) is False


def test_own_pid_is_live(tmp_path: Path) -> None:
    register_frontend(tmp_path, os.getpid())
    assert frontend_is_live(tmp_path) is True
    unregister_frontend(tmp_path, os.getpid())
    assert frontend_is_live(tmp_path) is False


def test_read_answer_returns_none_when_no_tui_and_no_answer(tmp_path: Path) -> None:
    # tui not live -> short-circuit immediately
    assert read_answer(tmp_path, "abc", timeout_s=2.0, poll_s=0.05) is None


def test_read_answer_picks_up_written_answer(tmp_path: Path) -> None:
    register_frontend(tmp_path, os.getpid())

    def writer() -> None:
        time.sleep(0.2)
        write_answer(tmp_path, "abc", approved=True)

    t = threading.Thread(target=writer, daemon=True)
    t.start()
    result = read_answer(tmp_path, "abc", timeout_s=2.0, poll_s=0.05)
    t.join(timeout=1)
    assert result is True


def test_write_answer_no_round_trips(tmp_path: Path) -> None:
    register_frontend(tmp_path, os.getpid())
    write_answer(tmp_path, "x", approved=False)
    assert read_answer(tmp_path, "x", timeout_s=1.0) is False


# --- liveness grace: a transient front-end drop must not deny the prompt ------


def test_read_answer_survives_transient_frontend_drop(tmp_path: Path) -> None:
    # The front-end is dead at poll time (an SSE drop / page reload) but comes
    # back within the grace window and answers: the answer must be returned, not
    # an instant headless None.
    register_frontend(tmp_path, 999999999)  # dead pid: the gate reads not-live

    def revive_and_answer() -> None:
        time.sleep(0.2)
        register_frontend(tmp_path, os.getpid())
        write_answer(tmp_path, "g1", approved=True)

    t = threading.Thread(target=revive_and_answer, daemon=True)
    t.start()
    result = read_answer(tmp_path, "g1", timeout_s=5.0, poll_s=0.05, dead_grace_s=2.0)
    t.join(timeout=2)
    assert result is True


def test_read_answer_falls_back_after_grace_expires(tmp_path: Path) -> None:
    # A front-end that stays dead past the grace window falls back headless
    # (None) well before the answer timeout.
    register_frontend(tmp_path, 999999999)
    start = time.monotonic()
    result = read_answer(tmp_path, "g2", timeout_s=30.0, poll_s=0.05, dead_grace_s=0.3)
    elapsed = time.monotonic() - start
    assert result is None
    assert 0.3 <= elapsed < 5.0  # grace elapsed, timeout not


def test_read_question_answer_survives_transient_frontend_drop(tmp_path: Path) -> None:
    register_frontend(tmp_path, 999999999)

    def revive_and_answer() -> None:
        time.sleep(0.2)
        register_frontend(tmp_path, os.getpid())
        write_question_answers(tmp_path, "q1", ["picked"])

    t = threading.Thread(target=revive_and_answer, daemon=True)
    t.start()
    result = read_question_answers(tmp_path, "q1", timeout_s=5.0, poll_s=0.05, dead_grace_s=2.0)
    t.join(timeout=2)
    assert result == ("picked",)


# --- atomic answer writes: the 0.2s poll must never consume a torn file -------


def test_answer_writes_leave_no_tmp_and_are_never_torn(tmp_path: Path) -> None:
    # write_* goes tmp+fsync+rename: a poller keyed on existence can only ever
    # read the complete text (a plain write_text exposes an empty file first,
    # which read_answer would consume as deny / "").
    payload = "y" * 65536
    # The answer file holds json.dumps([payload]) now, so the complete on-disk
    # content the poller must only ever see is that JSON list, not the bare string.
    expected = json.dumps([payload])
    target = tmp_path / "questions" / "q9.answer"
    stop = threading.Event()
    torn: list[str] = []

    def poller() -> None:
        while not stop.is_set():
            try:
                txt = target.read_text(encoding="utf-8")
            except FileNotFoundError:
                continue
            if txt != expected:
                torn.append(txt)
                return

    t = threading.Thread(target=poller, daemon=True)
    t.start()
    for _ in range(100):
        write_question_answers(tmp_path, "q9", [payload])
        target.unlink(missing_ok=True)
    stop.set()
    t.join(timeout=5)
    assert torn == []
    assert not list((tmp_path / "questions").glob("*.tmp"))
    write_answer(tmp_path, "a9", approved=True)
    assert not list((tmp_path / "approvals").glob("*.tmp"))
    write_steer_answer(tmp_path, "steer text")
    assert not list(tmp_path.glob("*.tmp"))
    assert (tmp_path / "steer.answer").read_text(encoding="utf-8") == "steer text"


def test_answer_landing_during_dead_verdict_is_consumed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An answer written between the round's read and the frontend-dead verdict
    was ignored: read_answer returned None (deny) while the completed answer
    file stayed on disk. The final consume honors it."""
    from agent6.sessions import ipc

    def write_then_dead(_live: Path) -> bool:
        write_answer(tmp_path, "abc", approved=True)
        return False

    monkeypatch.setattr(ipc, "frontend_is_live", write_then_dead)
    assert ipc.read_answer(tmp_path, "abc", timeout_s=5.0, poll_s=0.01, dead_grace_s=0.0) is True
    assert not (tmp_path / "approvals" / "abc.answer").exists()


def test_answer_landing_at_deadline_is_consumed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same race on the timeout exit: the answer lands during the final sleep,
    after the last read; the deadline then expires. Honored, not dropped."""
    from agent6.sessions import ipc

    register_frontend(tmp_path, os.getpid())

    class _Clock:
        def __init__(self) -> None:
            self.now = 0.0

        def monotonic(self) -> float:
            return self.now

        def sleep(self, s: float) -> None:
            self.now += s
            write_answer(tmp_path, "abc", approved=True)

    monkeypatch.setattr(ipc, "time", _Clock())
    assert ipc.read_answer(tmp_path, "abc", timeout_s=0.05, poll_s=0.1) is True
    assert not (tmp_path / "approvals" / "abc.answer").exists()


@pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="needs /proc (Linux)")
def test_worker_pid_recycled_pid_reads_dead(tmp_path: Path) -> None:
    """worker.pid proves identity, not just 'some same-user process owns this
    number': after a SIGKILL'd worker left the file behind, a recycled pid made
    the dead run read running forever -- blocking resume and hanging the
    /parallel lane await. The recorded kernel start time disambiguates."""
    from agent6.sessions import ipc

    ipc.write_worker_pid(tmp_path, os.getpid())
    assert ipc.worker_is_alive(tmp_path) is True
    pid_s, start = (tmp_path / "worker.pid").read_text(encoding="utf-8").split()
    # Same pid, different process: simulate the recycle by shifting the
    # recorded start time.
    (tmp_path / "worker.pid").write_text(f"{pid_s} {int(start) - 7}", encoding="utf-8")
    assert ipc.read_worker_pid(tmp_path) == os.getpid()
    assert ipc.worker_is_alive(tmp_path) is False


def test_worker_pid_without_start_time_probes_pid_only(tmp_path: Path) -> None:
    """A record with no start time (written on a host without /proc) degrades
    to the plain pid probe."""
    from agent6.sessions import ipc

    (tmp_path / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")
    assert ipc.worker_is_alive(tmp_path) is True
    (tmp_path / "worker.pid").write_text("999999999", encoding="utf-8")
    assert ipc.worker_is_alive(tmp_path) is False


def test_ps_start_time_reports_self_and_rejects_dead() -> None:
    from agent6.sessions import ipc

    assert ipc._ps_start_time(os.getpid())  # pyright: ignore[reportPrivateUsage]
    assert ipc._ps_start_time(999999999) == ""  # pyright: ignore[reportPrivateUsage]


def test_worker_pid_identity_via_ps_where_proc_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """macOS has no /proc, so the first fix degraded to the plain kill-0 probe
    there and pid reuse still misread a dead run as running. `ps -o lstart=`
    supplies the identity; its value contains spaces, so the record splits
    once only."""
    from agent6.sessions import ipc

    monkeypatch.setattr(ipc, "_HAS_PROC", False)
    ipc.write_worker_pid(tmp_path, os.getpid())
    record = (tmp_path / "worker.pid").read_text(encoding="utf-8")
    assert len(record.split(maxsplit=1)) == 2  # pid + a spaced lstart identity
    assert ipc.read_worker_pid(tmp_path) == os.getpid()
    assert ipc.worker_is_alive(tmp_path) is True
    # Same pid, different start time = a recycled pid: reads dead.
    (tmp_path / "worker.pid").write_text(f"{os.getpid()} Sun Jan  4 00:00:00 1970")
    assert ipc.worker_is_alive(tmp_path) is False
