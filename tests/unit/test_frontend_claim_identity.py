# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A front-end claim names a process; only a start time proves it is the SAME one.

`worker_is_alive` already refuses a recycled pid by comparing the recorded
start time. A front-end claim carried no such record, so any live process of
ours satisfied it: a front-end that died and had its pid reused read as live
forever, and `_await_answer` then waited out its whole timeout instead of the
dead-grace -- exactly the stall away-mode exists to avoid on an unattended run.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from agent6.sessions.ipc import (
    FRONTENDS_DIR,
    _answer_path,  # pyright: ignore[reportPrivateUsage]
    _await_answer,  # pyright: ignore[reportPrivateUsage]
    approvals_dir,
    frontend_is_live,
    register_frontend,
)


def _session(tmp_path: Path) -> Path:
    (tmp_path / FRONTENDS_DIR).mkdir(parents=True, exist_ok=True)
    approvals_dir(tmp_path).mkdir(parents=True, exist_ok=True)
    return tmp_path


def test_a_live_front_end_reads_live(tmp_path: Path) -> None:
    """The negative control: the identity check must not reject a real one."""
    session = _session(tmp_path)
    register_frontend(session, os.getpid())
    assert frontend_is_live(session) is True


def test_a_recycled_pid_does_not_read_as_a_front_end(tmp_path: Path) -> None:
    """A claim naming a live pid that is NOT the process which registered it."""
    session = _session(tmp_path)
    victim = subprocess.Popen(["sleep", "60"])
    try:
        claim = session / FRONTENDS_DIR / str(victim.pid)
        # Alive and ours, but started at a different time than recorded.
        claim.write_text("999999", encoding="utf-8")
        assert frontend_is_live(session) is False
        assert not claim.exists(), "a stale claim must be pruned, not left to block later polls"
    finally:
        victim.kill()
        victim.wait()


def test_an_answer_wait_gives_up_on_a_recycled_front_end(tmp_path: Path) -> None:
    """The consequence being fixed: without the identity check this waited the
    whole timeout for an answer nobody would give."""
    session = _session(tmp_path)
    victim = subprocess.Popen(["sleep", "60"])
    try:
        (session / FRONTENDS_DIR / str(victim.pid)).write_text("999999", encoding="utf-8")
        started = time.monotonic()
        answer = _await_answer(
            _answer_path(approvals_dir(session), "p1"),
            session,
            timeout_s=6.0,
            poll_s=0.1,
            dead_grace_s=1.0,
        )
        waited = time.monotonic() - started
    finally:
        victim.kill()
        victim.wait()
    assert answer is None
    assert waited < 3.0, f"waited {waited:.1f}s; the dead-grace was 1.0s, the timeout 6.0s"


def test_a_claim_with_no_recorded_start_is_trusted(tmp_path: Path) -> None:
    """Same tolerance the worker record has: no start time recorded means the
    liveness check alone decides, rather than refusing every claim."""
    session = _session(tmp_path)
    (session / FRONTENDS_DIR / str(os.getpid())).write_text("", encoding="utf-8")
    assert frontend_is_live(session) is True


def test_a_recycled_netns_holder_is_not_joinable(tmp_path: Path) -> None:
    """Same class, worse consequence: `agent6 exec` / `agent6 forward` open
    /proc/<pid>/ns/{user,net} on this number, so a pid the kernel handed to
    someone else put the operator's command inside an unrelated process's
    namespaces while reporting it as the run's."""
    from agent6.sessions.ipc import (
        NETNS_PID_FILE,
        read_session_netns_pid,
        write_session_netns_pid,
    )

    write_session_netns_pid(tmp_path, os.getpid())
    assert read_session_netns_pid(tmp_path) == os.getpid()

    victim = subprocess.Popen(["sleep", "60"])
    try:
        # Alive, /proc/<pid>/ns/net exists, but not the process that published.
        (tmp_path / NETNS_PID_FILE).write_text(f"{victim.pid} 999999", encoding="utf-8")
        assert read_session_netns_pid(tmp_path) is None
    finally:
        victim.kill()
        victim.wait()

    # No identity recorded: trusted, as every sibling record is.
    (tmp_path / NETNS_PID_FILE).write_text(str(os.getpid()), encoding="utf-8")
    assert read_session_netns_pid(tmp_path) == os.getpid()
