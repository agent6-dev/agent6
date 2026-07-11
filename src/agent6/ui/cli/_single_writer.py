# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""One authoritative writer per run dir: the `worker.lock` flock.

`agent6 run`/`resume`/`fork` drive one run's shared state (loop_state.json,
checkpoints, the curator DAG, the run branch). The run-level flock (the
analogue of ``machine_lock``) refuses a second concurrent writer.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

from agent6.portable import lock_exclusive, unlock


def acquire_single_writer(run_dir: Path) -> int | None:
    """Take a non-blocking exclusive lock on ``<run-dir>/worker.lock``.

    One run's shared state (``loop_state.json``, ``checkpoints/``, the curator
    DAG, the run branch) has exactly one authoritative writer. A second
    ``agent6 run``/``resume``/``fork`` targeting the SAME run dir would spawn a
    second curator whose independent in-memory cache silently clobbers the
    first's parent->child links (a lost update), and would interleave commits on
    the run branch. This is the run-level analogue of ``machine_lock``.

    Returns the held fd on success (the caller keeps the process alive to hold
    it, and passes it to ``release_single_writer`` at teardown), or ``None``
    when another live process holds it (the caller refuses). A crashed writer
    leaves no lock -- flock releases on process death -- so resume-after-crash is
    never blocked by a stale lock.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    fd = os.open(run_dir / "worker.lock", os.O_CREAT | os.O_RDWR, 0o644)
    try:
        lock_exclusive(fd, blocking=False)
    except OSError:
        os.close(fd)
        return None
    return fd


def release_single_writer(fd: int | None) -> None:
    """Release + close a lock fd from ``acquire_single_writer`` (no-op on None).

    Explicit close matters: the fd is a raw int (``os.open``), so it does not
    self-close on GC. A leaked fd would keep the flock held and wrongly refuse a
    later same-dir run in the same process (tests, embedding)."""
    if fd is None:
        return
    with contextlib.suppress(OSError):
        unlock(fd)
    with contextlib.suppress(OSError):
        os.close(fd)


SINGLE_WRITER_BUSY = (
    "REFUSING: run {rid!r} is already being driven by another agent6 process "
    "(its worker.lock is held). Concurrent run/resume of the same run would "
    "corrupt its state (a second curator clobbers the task graph, and commits "
    "interleave on the run branch). Wait for that process to finish; a crashed "
    "one releases the lock automatically."
)
