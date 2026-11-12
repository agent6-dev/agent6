# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Watch started lanes until they land, and say what they wait on.

The single-lane await and drain behind `run_lane_to_completion`, and where a lane's record
lives.
"""

from __future__ import annotations

import contextlib
import os
import pathlib
import signal
import threading
import time
from collections.abc import Callable

from agent6 import kinds, lane_records
from agent6.app import stop
from agent6.sandbox import jail
from agent6.sessions import layout, process
from agent6.viewmodel import summarize_session_dir

# How often the await loop polls lane liveness.
POLL_INTERVAL_S = 2.0

# How long a stop waits for a lane to finish its in-flight step before giving up on it.
STOP_GRACE_S = 30.0


def lane_terminal(
    session_dir: pathlib.Path, status: str, worker_is_alive: Callable[[pathlib.Path], bool]
) -> bool:
    """Return whether an awaited lane is terminal: the fold left "running" and the worker is gone.

    `session.stop` lands before the teardown clears `worker.pid`, so status alone races it. A
    lane that dies without a `session.stop` cannot hang the gate: the fold reads it "stale".

    Args:
        session_dir: The lane's run dir.
        status: The lane's status as the fold reads it.
        worker_is_alive: The liveness check for the run dir's worker.

    Returns:
        True when the status is not "running" and the worker is gone.
    """
    return status != "running" and not worker_is_alive(session_dir)


def await_lane(
    res: lane_records.LaneResult,
    *,
    poll_interval_s: float = POLL_INTERVAL_S,
    should_stop: Callable[[], bool] | None = None,
    deadline_at: float | None = None,
    clock: Callable[[], float] = time.time,
    on_poll: Callable[[str], bool] | None = None,
) -> bool:
    """Block until the lane is terminal, or until a stop or the deadline ends the wait.

    Args:
        res: The started lane, awaited on its real run dir.
        poll_interval_s: How often to poll.
        should_stop: The coordinator's abort channel, read between polls.
        deadline_at: The wall-clock second the wait ends at; None for no deadline.
        clock: The wall clock.
        on_poll: Called on each poll that finds the lane running, with its status word: the
            parent counts the lane's spend and answers its prompt there. True ends the wait
            as a stop does.

    Returns:
        True when the lane ended, False when a stop or the deadline ended the wait.
    """
    while True:
        summary = summarize_session_dir(res.session_dir)
        if lane_terminal(res.session_dir, summary.status, process.worker_is_alive):
            return True
        if should_stop is not None and should_stop():
            return False
        if deadline_at is not None and clock() >= deadline_at:
            return False
        if on_poll is not None and on_poll(summary.status):
            return False
        time.sleep(poll_interval_s)


def stop_child(pid: int, *, grace_s: float = stop.KILL_GRACE_S) -> str:
    """Stop a lane process the coordinator spawned: TERM its group, KILL after the grace, reap.

    Args:
        pid: The child's pid; 0 for none.
        grace_s: SIGTERM to SIGKILL.

    Returns:
        `gone` (nothing alive to stop), `stopped` (ended on TERM) or `killed`.
    """
    if pid <= 0 or not process.pid_alive(pid):
        _reap(pid)
        return "gone"
    jail.signal_group(pid, signal.SIGTERM)
    if _ended(pid, grace_s):
        return "stopped"
    jail.signal_group(pid, signal.SIGKILL)
    _ended(pid, stop.KILL_GRACE_S)
    return "killed"


def _ended(pid: int, within_s: float) -> bool:
    """Reap and poll the pid until it is gone or the wait runs out.

    Args:
        pid: The child.
        within_s: The wait.

    Returns:
        True when the process is gone.
    """
    deadline = time.monotonic() + within_s
    while True:
        _reap(pid)
        if not process.pid_alive(pid):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


def _reap(pid: int) -> None:
    """Collect a finished direct child, so a stopped lane never lingers as a zombie."""
    if pid <= 0:
        return
    with contextlib.suppress(ChildProcessError, OSError):
        os.waitpid(pid, os.WNOHANG)


def drain_lane(
    res: lane_records.LaneResult, *, poll_interval_s: float, hard_stop: threading.Event | None
) -> bool:
    """Wait a bounded grace after a stop for the lane to land, so its work still imports.

    Args:
        res: The stopped lane.
        poll_interval_s: How often to poll.
        hard_stop: A process teardown, which skips the wait.

    Returns:
        True when the lane landed in time, False to leave it running un-imported.
    """
    deadline = time.monotonic() + STOP_GRACE_S
    while time.monotonic() < deadline:
        if hard_stop is not None and hard_stop.is_set():
            return False
        summary = summarize_session_dir(res.session_dir)
        if lane_terminal(res.session_dir, summary.status, process.worker_is_alive):
            return True
        if hard_stop is not None:
            if hard_stop.wait(poll_interval_s):
                return False
        else:
            time.sleep(poll_interval_s)
    return False


def lane_dir(record_state: pathlib.Path, session_id: str, *, mode: str) -> pathlib.Path:
    """Return a lane's session dir: the bucket for its mode under the state its records live in.

    Args:
        record_state: The state dir of the repository whose run dispatched the lane.
        session_id: The lane's session id.
        mode: The lane's session mode.

    Returns:
        The lane's dir.
    """
    return layout.bucket_dir(record_state, kinds.session_bucket(mode)) / session_id
