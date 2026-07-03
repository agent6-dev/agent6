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

from agent6 import portable
from agent6.portable import atomic_write, fsync_dir, lock_exclusive, unlock


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


def test_atomic_write_fsyncs_file_and_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fsynced_files: list[int] = []
    fsynced_dirs: list[Path] = []

    def record_file_fsync(fd: int) -> None:
        fsynced_files.append(fd)

    def record_dir_fsync(path: Path) -> None:
        fsynced_dirs.append(path)

    monkeypatch.setattr(portable.os, "fsync", record_file_fsync)
    monkeypatch.setattr(portable, "fsync_dir", record_dir_fsync)

    target = tmp_path / "state.json"
    atomic_write(target, '{"ok": true}')

    assert target.read_text(encoding="utf-8") == '{"ok": true}'
    assert not (tmp_path / "state.json.tmp").exists()
    assert fsynced_files, "temp file must be fsync'd before replace"
    assert fsynced_dirs == [tmp_path], "parent dir must be fsync'd after replace"


def test_atomic_write_fsyncs_new_parent_dirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fsynced_dirs: list[Path] = []

    def record_dir_fsync(path: Path) -> None:
        fsynced_dirs.append(path)

    monkeypatch.setattr(portable, "fsync_dir", record_dir_fsync)

    target = tmp_path / "new" / "nested" / "state.json"
    atomic_write(target, "ok")

    assert target.read_text(encoding="utf-8") == "ok"
    assert fsynced_dirs == [tmp_path, tmp_path / "new", tmp_path / "new" / "nested"]


def test_atomic_write_concurrent_writers_do_not_share_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    writers = 8
    target = tmp_path / "state.json"
    barrier = threading.Barrier(writers)
    errors: list[BaseException] = []

    def wait_at_file_fsync(_fd: int) -> None:
        barrier.wait(timeout=5.0)

    def noop_fsync_dir(_path: Path) -> None:
        return None

    def write_payload(n: int) -> None:
        try:
            atomic_write(target, f"payload-{n}")
        except Exception as exc:  # pragma: no cover - assertion reports it
            errors.append(exc)

    monkeypatch.setattr(portable.os, "fsync", wait_at_file_fsync)
    monkeypatch.setattr(portable, "fsync_dir", noop_fsync_dir)

    threads = [threading.Thread(target=write_payload, args=(n,)) for n in range(writers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert target.read_text(encoding="utf-8") in {f"payload-{n}" for n in range(writers)}


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_atomic_write_new_file_is_owner_only_under_restrictive_umask(tmp_path: Path) -> None:
    import stat

    # A new state file must not be widened to a hardcoded 0o644 (bypassing the
    # umask): these are per-user run/machine state, owner-only by default.
    old = os.umask(0o077)
    try:
        target = tmp_path / "state.json"
        atomic_write(target, "{}")
        assert stat.S_IMODE(target.stat().st_mode) == 0o600
    finally:
        os.umask(old)


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_atomic_write_preserves_existing_mode(tmp_path: Path) -> None:
    import stat

    target = tmp_path / "state.json"
    atomic_write(target, "first")
    target.chmod(0o640)
    atomic_write(target, "second")  # a re-publish must keep the file's own mode
    assert stat.S_IMODE(target.stat().st_mode) == 0o640
