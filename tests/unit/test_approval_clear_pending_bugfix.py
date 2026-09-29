# SPDX-License-Identifier: Apache-2.0
"""Regression tests for clear_pending_answers.

A leftover `steer.request` from a prior session is dropped at start, and `frontend.pid` is
cleared only when no live front-end owns it.
"""

from __future__ import annotations

import os
import pathlib
import time

from agent6.sessions import ipc


def test_clear_pending_drops_leftover_steer_request(tmp_path: pathlib.Path) -> None:
    session_dir = tmp_path / "run"
    session_dir.mkdir()
    ipc.request_steer(session_dir)
    assert ipc.steer_request_pending(session_dir)

    ipc.clear_pending_answers(session_dir, started_at=time.time() + 60)

    # The phantom steer marker must be gone so resume doesn't stall.
    assert not ipc.steer_request_pending(session_dir)


def test_clear_pending_preserves_live_frontend_claims(tmp_path: pathlib.Path) -> None:
    session_dir = tmp_path / "run"
    session_dir.mkdir()
    # Our own pid is a live process => a live foreign watcher.
    ipc.register_frontend(session_dir, os.getpid())
    assert ipc.frontend_is_live(session_dir)

    ipc.clear_pending_answers(session_dir, started_at=time.time() + 60)

    # A live watcher's claim must survive so its modals stay wired up.
    assert ipc.frontend_is_live(session_dir)


def test_dead_frontend_claims_are_pruned_by_the_liveness_probe(tmp_path: pathlib.Path) -> None:
    session_dir = tmp_path / "run"
    session_dir.mkdir()
    dead_pid = _find_dead_pid()
    ipc.register_frontend(session_dir, dead_pid)
    # A hard-killed front-end's claim reads not-live and is pruned, so the poll never blocks on it.
    assert not ipc.frontend_is_live(session_dir)
    assert not (session_dir / "frontends" / str(dead_pid)).exists()


def test_concurrent_frontends_do_not_deregister_each_other(tmp_path: pathlib.Path) -> None:
    """Concurrent front-ends do not deregister each other.

    One claim file per front-end: each removes only its own.
    """
    session_dir = tmp_path / "run"
    session_dir.mkdir()
    attach_pid = os.getpid()
    web_pid = _find_dead_pid()  # stands in for a second front-end's pid slot
    ipc.register_frontend(session_dir, attach_pid)
    ipc.register_frontend(session_dir, web_pid)
    ipc.unregister_frontend(session_dir, web_pid)  # the browser closes
    assert ipc.frontend_is_live(session_dir)  # the attach watcher keeps bridging
    ipc.unregister_frontend(session_dir, attach_pid)
    assert not ipc.frontend_is_live(session_dir)
    # Unregistering an absent claim is a no-op.
    ipc.unregister_frontend(session_dir, attach_pid)


def _find_dead_pid() -> int:
    for candidate in range(2_000_000, 2_000_100):
        try:
            os.kill(candidate, 0)
        except ProcessLookupError:
            return candidate
        except PermissionError:
            continue
    # Fallback: very unlikely to be reached.
    return 2_000_000


def test_clear_pending_keeps_a_file_written_within_the_tick_of_the_executions_start(
    tmp_path: pathlib.Path,
) -> None:
    """The sweep keeps a file written within a scheduler tick of the execution's start.

    File timestamps run up to a tick behind `time.time()`.
    """
    session_dir = tmp_path / "run"
    session_dir.mkdir()
    started_at = 1_700_000_000.0
    ipc.write_answer(session_dir, "approval-1", "yes")
    old = session_dir / "approvals" / "approval-1.answer"
    assert old.is_file()
    os.utime(old, (started_at - 10, started_at - 10))
    ipc.request_stop(session_dir)
    one_tick = 0.004  # HZ=250
    written_at = started_at - one_tick
    os.utime(session_dir / "stop.request", (written_at, written_at))

    ipc.clear_pending_answers(session_dir, started_at=started_at)

    assert not old.exists()
    assert ipc.stop_request_pending(session_dir)
    ipc.clear_pending_answers(session_dir, started_at=time.time() + 60)
    assert not ipc.stop_request_pending(session_dir)
