# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for agent6.portable cross-platform primitives.

These run on every platform: the lock helper guards the machine + graph
single-writer invariants on Windows/macOS/Linux alike.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent6.portable import fsync_dir, lock_exclusive, unlock


def test_exclusive_lock_blocks_second_holder(tmp_path: Path) -> None:
    lock_path = tmp_path / "x.lock"
    fd1 = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    fd2 = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        lock_exclusive(fd1, blocking=False)
        with pytest.raises(OSError):
            lock_exclusive(fd2, blocking=False)
        unlock(fd1)
        # Now the second holder can take it.
        lock_exclusive(fd2, blocking=False)
        unlock(fd2)
    finally:
        os.close(fd1)
        os.close(fd2)


def test_lock_then_unlock_is_reusable(tmp_path: Path) -> None:
    lock_path = tmp_path / "y.lock"
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        lock_exclusive(fd, blocking=True)
        unlock(fd)
        lock_exclusive(fd, blocking=True)
        unlock(fd)
    finally:
        os.close(fd)


def test_fsync_dir_does_not_raise(tmp_path: Path) -> None:
    # Should be a durable no-op-or-fsync regardless of platform.
    fsync_dir(tmp_path)
