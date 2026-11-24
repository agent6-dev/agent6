# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Watch started children until they end, and say what they wait on.

The single-child await and drain behind `run_child_to_completion`, and where a child's record
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

from agent6 import child_records, errors, kinds
from agent6.app import stop
from agent6.sandbox import jail
from agent6.sessions import child_dispatch, layout, manifest, process
from agent6.vcs.git import read as git_read
from agent6.viewmodel import listing

# How often the await loop polls child liveness.
POLL_INTERVAL_S = 2.0

# How long a stop waits for a child to finish its in-flight step before giving up on it.
STOP_GRACE_S = 30.0


def observe_commits(res: child_records.ChildResult) -> None:
    """Measure the work child's own commit line before import or clone cleanup."""
    if res.spec.mode != "run":
        return
    try:
        saved = manifest.read_manifest(res.session_dir)
        if not saved.base_sha:
            return
        tree = res.spec.worktree
        tip = (
            "HEAD"
            if saved.git_control == "model"
            else (git_read.chain_tip(tree, git_read.chain_ref_for(res.spec.session_id)) or "HEAD")
        )
        count = git_read.commit_count(tree, saved.base_sha, tip)
        if count is not None:
            child_dispatch.write_commit_count(res.session_dir, count)
    except (errors.GitError, manifest.ManifestError, OSError):
        # A missing observation must not stop the child or affect its merge.
        return


def child_terminal(
    session_dir: pathlib.Path, status: str, worker_is_alive: Callable[[pathlib.Path], bool]
) -> bool:
    """Return whether an awaited child is terminal: the fold left "running" and the worker is gone.

    `session.stop` is written before the teardown clears `worker.pid`, so status alone races it. A
    child that dies without a `session.stop` cannot hang the gate: the fold reads it "stale".

    Args:
        session_dir: The child's session dir.
        status: The child's status as the fold reads it.
        worker_is_alive: The liveness check for the session dir's worker.

    Returns:
        True when the status is not "running" and the worker is gone.
    """
    return status != "running" and not worker_is_alive(session_dir)


def await_child(
    res: child_records.ChildResult,
    *,
    poll_interval_s: float = POLL_INTERVAL_S,
    should_stop: Callable[[], bool] | None = None,
    deadline_at: float | None = None,
    clock: Callable[[], float] = time.time,
    on_poll: Callable[[str], bool] | None = None,
) -> bool:
    """Block until the child is terminal, or until a stop or the deadline ends the wait.

    Args:
        res: The started child, awaited on its real session dir.
        poll_interval_s: How often to poll.
        should_stop: The parent's abort channel, read between polls.
        deadline_at: The wall-clock second the wait ends at; None for no deadline.
        clock: The wall clock.
        on_poll: Called on each poll that finds the child running, with its status word: the
            parent counts the child's spend and answers its prompt there. True ends the wait
            as a stop does.

    Returns:
        True when the child ended, False when a stop or the deadline ended the wait.
    """
    while True:
        summary = listing.summarize_session_dir(res.session_dir)
        observe_commits(res)
        if child_terminal(res.session_dir, summary.status, process.worker_is_alive):
            return True
        if should_stop is not None and should_stop():
            return False
        if deadline_at is not None and clock() >= deadline_at:
            return False
        if on_poll is not None and on_poll(summary.status):
            return False
        time.sleep(poll_interval_s)


def stop_child(pid: int, *, grace_s: float = stop.KILL_GRACE_S) -> str:
    """Stop a child process the parent spawned: TERM its group, KILL after the grace, reap.

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
    """Collect a finished direct child, so a stopped child never lingers as a zombie."""
    if pid <= 0:
        return
    with contextlib.suppress(ChildProcessError, OSError):
        os.waitpid(pid, os.WNOHANG)


def drain_child(
    res: child_records.ChildResult, *, poll_interval_s: float, hard_stop: threading.Event | None
) -> bool:
    """Wait a bounded grace after a stop for the child to end, so its work still imports.

    Args:
        res: The stopped child.
        poll_interval_s: How often to poll.
        hard_stop: A process teardown, which skips the wait.

    Returns:
        True when the worker is gone, False when its work is not safe to import.
    """
    deadline = time.monotonic() + STOP_GRACE_S
    while True:
        if hard_stop is not None and hard_stop.is_set():
            return False
        summary = listing.summarize_session_dir(res.session_dir)
        if child_terminal(res.session_dir, summary.status, process.worker_is_alive):
            observe_commits(res)
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        interval = min(poll_interval_s, remaining)
        if hard_stop is not None:
            if hard_stop.wait(interval):
                return False
        else:
            time.sleep(interval)


def child_dir(record_state: pathlib.Path, session_id: str, *, mode: str) -> pathlib.Path:
    """Return a child's session dir: the bucket for its mode under the state its records live in.

    Args:
        record_state: The state dir of the repository whose run dispatched the child.
        session_id: The child's session id.
        mode: The child's session mode.

    Returns:
        The child's dir.
    """
    return layout.bucket_dir(record_state, kinds.session_bucket(mode)) / session_id
