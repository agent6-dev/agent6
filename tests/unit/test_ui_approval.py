# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the file-based approval bridge."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

from agent6.runs.bridge import (
    clear_frontend_pid,
    frontend_is_live,
    read_answer,
    read_question_answers,
    write_answer,
    write_frontend_pid,
    write_question_answers,
    write_steer_answer,
)


def test_no_tui_pid_means_not_live(tmp_path: Path) -> None:
    assert frontend_is_live(tmp_path) is False


def test_dead_pid_is_not_live(tmp_path: Path) -> None:
    # PID 1 is init; signal-0 to it from a non-root process raises PermissionError
    # which we treat as "not us" -> dead. PID 0 is invalid -> ProcessLookupError.
    write_frontend_pid(tmp_path, 999999999)  # almost certainly not allocated
    assert frontend_is_live(tmp_path) is False


def test_own_pid_is_live(tmp_path: Path) -> None:
    write_frontend_pid(tmp_path, os.getpid())
    assert frontend_is_live(tmp_path) is True
    clear_frontend_pid(tmp_path)
    assert frontend_is_live(tmp_path) is False


def test_read_answer_returns_none_when_no_tui_and_no_answer(tmp_path: Path) -> None:
    # tui not live -> short-circuit immediately
    assert read_answer(tmp_path, "abc", timeout_s=2.0, poll_s=0.05) is None


def test_read_answer_picks_up_written_answer(tmp_path: Path) -> None:
    write_frontend_pid(tmp_path, os.getpid())

    def writer() -> None:
        time.sleep(0.2)
        write_answer(tmp_path, "abc", approved=True)

    t = threading.Thread(target=writer, daemon=True)
    t.start()
    result = read_answer(tmp_path, "abc", timeout_s=2.0, poll_s=0.05)
    t.join(timeout=1)
    assert result is True


def test_write_answer_no_round_trips(tmp_path: Path) -> None:
    write_frontend_pid(tmp_path, os.getpid())
    write_answer(tmp_path, "x", approved=False)
    assert read_answer(tmp_path, "x", timeout_s=1.0) is False


# --- liveness grace: a transient front-end drop must not deny the prompt ------


def test_read_answer_survives_transient_frontend_drop(tmp_path: Path) -> None:
    # The front-end is dead at poll time (an SSE drop / page reload) but comes
    # back within the grace window and answers: the answer must be returned, not
    # an instant headless None.
    write_frontend_pid(tmp_path, 999999999)  # dead pid: the gate reads not-live

    def revive_and_answer() -> None:
        time.sleep(0.2)
        write_frontend_pid(tmp_path, os.getpid())
        write_answer(tmp_path, "g1", approved=True)

    t = threading.Thread(target=revive_and_answer, daemon=True)
    t.start()
    result = read_answer(tmp_path, "g1", timeout_s=5.0, poll_s=0.05, dead_grace_s=2.0)
    t.join(timeout=2)
    assert result is True


def test_read_answer_falls_back_after_grace_expires(tmp_path: Path) -> None:
    # A front-end that stays dead past the grace window falls back headless
    # (None) well before the answer timeout.
    write_frontend_pid(tmp_path, 999999999)
    start = time.monotonic()
    result = read_answer(tmp_path, "g2", timeout_s=30.0, poll_s=0.05, dead_grace_s=0.3)
    elapsed = time.monotonic() - start
    assert result is None
    assert 0.3 <= elapsed < 5.0  # grace elapsed, timeout not


def test_read_question_answer_survives_transient_frontend_drop(tmp_path: Path) -> None:
    write_frontend_pid(tmp_path, 999999999)

    def revive_and_answer() -> None:
        time.sleep(0.2)
        write_frontend_pid(tmp_path, os.getpid())
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
