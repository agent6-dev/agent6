# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The launcher's exec mode: the command's terminal and its exit code."""

from __future__ import annotations

import os
import pathlib
import pty
import re
import select
import signal
import subprocess
import time

import pytest

from agent6 import kinds
from agent6.sandbox import jail

_LEVELS = [pytest.param("full", marks=pytest.mark.needs_namespaces), "partial"]


def _policy(cwd: pathlib.Path, isolation: kinds.IsolationLevel, script: str) -> kinds.JailPolicy:
    return kinds.JailPolicy(
        cwd=cwd, argv=("/bin/sh", "-c", script), isolation=isolation, network="host", timeout_s=0.0
    )


def _read_until(fd: int, pattern: bytes, timeout_s: float = 20.0) -> bytes:
    """Return the pty's output up to the first match of the pattern, failing at the timeout."""
    seen = b""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        ready, _, _ = select.select([fd], [], [], 0.1)
        if not ready:
            continue
        try:
            chunk = os.read(fd, 65536)
        except OSError:
            break
        seen += chunk
        if re.search(pattern, seen):
            return seen
    pytest.fail(f"never saw {pattern!r} in {seen!r}")


@pytest.mark.parametrize("isolation", _LEVELS)
def test_the_command_runs_in_the_foreground_of_its_controlling_terminal(
    tmp_path: pathlib.Path, isolation: kinds.IsolationLevel
) -> None:
    """The command reads its terminal, survives Ctrl-Z, and Ctrl-C ends it with 130.

    Left in a background group of the terminal, the command stops at its first read and the
    interrupt ends the launcher. A stop nothing resumes leaves the command stopped for good.
    """
    master, slave = pty.openpty()
    try:
        proc = jail.spawn_in_jail(
            _policy(tmp_path, isolation, "read line; echo GOT-$line; exec sleep 30"),
            stdin=slave,
            stdout=slave,
            stderr=slave,
            controlling_terminal=True,
        )
    finally:
        os.close(slave)
    try:
        os.write(master, b"typed\n")
        _read_until(master, rb"GOT-typed")
        os.write(master, b"\x1a")
        time.sleep(0.5)
        os.write(master, b"\x03")
        assert proc.popen.wait(timeout=20) == 128 + signal.SIGINT
    finally:
        left = proc.close()
        os.close(master)
    assert not left


@pytest.mark.parametrize("isolation", _LEVELS)
def test_a_command_that_lost_the_foreground_reads_its_terminal_again(
    tmp_path: pathlib.Path, isolation: kinds.IsolationLevel
) -> None:
    """A command whose child took the terminal's foreground and died still reads the terminal.

    The launcher answered the terminal's stop with SIGCONT alone. Still in a background group,
    the command stopped again at the same read, and the two looped until the command was killed.
    """
    taker = (
        "import os, signal; signal.signal(signal.SIGTTOU, signal.SIG_IGN); "
        "os.setpgid(0, 0); os.tcsetpgrp(0, os.getpgrp())"
    )
    script = f'python3 -c "{taker}"; echo TAKEN; read line; echo GOT-$line'
    master, slave = pty.openpty()
    try:
        proc = jail.spawn_in_jail(
            _policy(tmp_path, isolation, script),
            stdin=slave,
            stdout=slave,
            stderr=slave,
            controlling_terminal=True,
        )
    finally:
        os.close(slave)
    try:
        _read_until(master, rb"TAKEN")
        os.write(master, b"typed\n")
        _read_until(master, rb"GOT-typed", timeout_s=10.0)
        assert proc.popen.wait(timeout=20) == 0
    finally:
        left = proc.close()
        os.close(master)
    assert not left


@pytest.mark.parametrize("isolation", _LEVELS)
def test_a_command_a_signal_ended_exits_128_plus_the_signal(
    tmp_path: pathlib.Path, isolation: kinds.IsolationLevel
) -> None:
    """A command SIGTERM ended reports 143; every signal read as 137."""
    proc = jail.spawn_in_jail(
        _policy(tmp_path, isolation, "kill -TERM $$"),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        assert proc.popen.wait(timeout=20) == 128 + signal.SIGTERM
    finally:
        proc.close()
