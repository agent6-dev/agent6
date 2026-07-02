# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the file-based approval bridge."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

from agent6.frontend.approval import (
    clear_frontend_pid,
    frontend_is_live,
    read_answer,
    write_answer,
    write_frontend_pid,
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
