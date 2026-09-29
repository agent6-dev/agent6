# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A front-end claim names a process; only a start time proves it is the same one.

Without it a front-end that died and had its pid reused reads as live forever, and `_await_answer`
waits out its whole timeout instead of the dead-grace.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import time

import pytest

from agent6 import portable
from agent6.sessions import ipc as sessions_ipc


def _session(tmp_path: pathlib.Path) -> pathlib.Path:
    (tmp_path / sessions_ipc.FRONTENDS_DIR).mkdir(parents=True, exist_ok=True)
    sessions_ipc.approvals_dir(tmp_path).mkdir(parents=True, exist_ok=True)
    return tmp_path


def test_a_live_front_end_reads_live(tmp_path: pathlib.Path) -> None:
    """The negative control: the identity check must not reject a real one."""
    session = _session(tmp_path)
    sessions_ipc.register_frontend(session, os.getpid())
    assert sessions_ipc.frontend_is_live(session) is True


def test_a_frontend_claim_is_published_atomically(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A liveness probe must never prune a partially written start identity."""
    writes: list[pathlib.Path] = []
    real = portable.atomic_write

    def spy(path: pathlib.Path, text: str) -> None:
        writes.append(path)
        real(path, text)

    monkeypatch.setattr(portable, "atomic_write", spy)
    sessions_ipc.register_frontend(tmp_path, os.getpid())

    assert writes == [tmp_path / sessions_ipc.FRONTENDS_DIR / str(os.getpid())]
    assert sessions_ipc.frontend_is_live(tmp_path) is True


def test_a_liveness_probe_does_not_delete_an_inflight_claim(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The claim scan must leave atomic_write's visible sibling temp alone."""

    def publish_with_probe(path: pathlib.Path, text: str) -> None:
        temp = path.with_name(f".{path.name}.race.tmp")
        temp.write_text(text, encoding="utf-8")
        assert sessions_ipc.frontend_is_live(tmp_path) is False
        temp.replace(path)

    monkeypatch.setattr(portable, "atomic_write", publish_with_probe)
    sessions_ipc.register_frontend(tmp_path, os.getpid())

    assert sessions_ipc.frontend_is_live(tmp_path) is True


def test_a_recycled_pid_does_not_read_as_a_front_end(tmp_path: pathlib.Path) -> None:
    """A claim naming a live pid that is NOT the process which registered it."""
    session = _session(tmp_path)
    victim = subprocess.Popen(["sleep", "60"])
    try:
        claim = session / sessions_ipc.FRONTENDS_DIR / str(victim.pid)
        # Alive and ours, but started at a different time than recorded.
        claim.write_text("999999", encoding="utf-8")
        assert sessions_ipc.frontend_is_live(session) is False
        assert not claim.exists(), "a stale claim must be pruned, not left to block later polls"
    finally:
        victim.kill()
        victim.wait()


def test_an_answer_wait_gives_up_on_a_recycled_front_end(tmp_path: pathlib.Path) -> None:
    """Without the identity check the wait lasts the whole timeout for an answer nobody gives."""
    session = _session(tmp_path)
    victim = subprocess.Popen(["sleep", "60"])
    try:
        (session / sessions_ipc.FRONTENDS_DIR / str(victim.pid)).write_text(
            "999999", encoding="utf-8"
        )
        started = time.monotonic()
        answer = sessions_ipc._await_answer(
            sessions_ipc._answer_path(sessions_ipc.approvals_dir(session), "p1"),
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


def test_a_claim_with_no_recorded_start_is_trusted(tmp_path: pathlib.Path) -> None:
    """No start time recorded means the liveness check alone decides, as for the worker record."""
    session = _session(tmp_path)
    (session / sessions_ipc.FRONTENDS_DIR / str(os.getpid())).write_text("", encoding="utf-8")
    assert sessions_ipc.frontend_is_live(session) is True


def test_a_recycled_netns_holder_is_not_joinable(tmp_path: pathlib.Path) -> None:
    """`exec` and `forward` open /proc/<pid>/ns on the claim, so a recycled pid must be refused."""
    sessions_ipc.write_session_netns_pid(tmp_path, os.getpid())
    assert sessions_ipc.read_session_netns_pid(tmp_path) == os.getpid()

    victim = subprocess.Popen(["sleep", "60"])
    try:
        # Alive, /proc/<pid>/ns/net exists, but not the process that published.
        (tmp_path / sessions_ipc.NETNS_PID_FILE).write_text(
            f"{victim.pid} 999999", encoding="utf-8"
        )
        assert sessions_ipc.read_session_netns_pid(tmp_path) is None
    finally:
        victim.kill()
        victim.wait()

    # No identity recorded: trusted, as every sibling record is.
    (tmp_path / sessions_ipc.NETNS_PID_FILE).write_text(str(os.getpid()), encoding="utf-8")
    assert sessions_ipc.read_session_netns_pid(tmp_path) == os.getpid()
