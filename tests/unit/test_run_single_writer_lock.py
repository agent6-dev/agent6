# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""One authoritative writer per run dir.

`agent6 run`/`resume`/`fork` drive one run's shared state (loop_state,
checkpoints, the curator DAG, the run branch). A second process on the same run
dir would spawn a second curator whose stale in-memory cache clobbers the
first's parent->child links, and would interleave commits. The run-level flock
(the analogue of `machine_lock`) refuses the second writer.
"""

from __future__ import annotations

import multiprocessing
import time
from multiprocessing.synchronize import Event as EventType
from pathlib import Path

from agent6.cli.run import (
    _acquire_single_writer,  # pyright: ignore[reportPrivateUsage]
    _release_single_writer,  # pyright: ignore[reportPrivateUsage]
)


def test_second_acquire_on_same_dir_is_refused(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "R"
    run_dir.mkdir(parents=True)
    fd = _acquire_single_writer(run_dir)
    assert fd is not None
    try:
        # A concurrent writer on the SAME dir (a second `agent6 resume R`) refuses.
        assert _acquire_single_writer(run_dir) is None
    finally:
        _release_single_writer(fd)


def test_distinct_run_dirs_are_independent(tmp_path: Path) -> None:
    a = tmp_path / "runs" / "A"
    b = tmp_path / "runs" / "B"
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    fd_a = _acquire_single_writer(a)
    fd_b = _acquire_single_writer(b)
    try:
        assert fd_a is not None and fd_b is not None  # different runs never contend
    finally:
        _release_single_writer(fd_a)
        _release_single_writer(fd_b)


def test_reacquire_after_release_succeeds(tmp_path: Path) -> None:
    # Sequential resume-after-exit: once the first writer releases, the next
    # acquires cleanly (the lock must not stay stuck).
    run_dir = tmp_path / "runs" / "R"
    run_dir.mkdir(parents=True)
    fd1 = _acquire_single_writer(run_dir)
    assert fd1 is not None
    _release_single_writer(fd1)
    fd2 = _acquire_single_writer(run_dir)
    assert fd2 is not None
    _release_single_writer(fd2)


def test_release_none_is_noop() -> None:
    _release_single_writer(None)  # the refusal path passes None; must not raise


def _hold_lock(run_dir: str, ready: EventType, done: EventType) -> None:
    fd = _acquire_single_writer(Path(run_dir))
    if fd is None:  # pragma: no cover - defensive
        return
    ready.set()
    done.wait(timeout=5.0)
    _release_single_writer(fd)


def test_cross_process_contention_and_release_on_death(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "R"
    run_dir.mkdir(parents=True)
    ctx = multiprocessing.get_context("spawn")
    ready = ctx.Event()
    done = ctx.Event()
    holder = ctx.Process(target=_hold_lock, args=(str(run_dir), ready, done))
    holder.start()
    try:
        assert ready.wait(timeout=10.0)
        # Another process cannot acquire while the holder is alive.
        assert _acquire_single_writer(run_dir) is None
    finally:
        done.set()
        holder.join(timeout=10.0)
    # flock releases on the holder's exit, so the dir is acquirable again.
    deadline = time.monotonic() + 5.0
    fd = None
    while fd is None and time.monotonic() < deadline:
        fd = _acquire_single_writer(run_dir)
        if fd is None:
            time.sleep(0.05)
    assert fd is not None
    _release_single_writer(fd)
