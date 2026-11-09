# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""JailSession.open on a launcher that dies during setup.

The failure is a JailUnavailableError with the launcher's stderr, the child is reaped, and every
pipe is closed at the failure site; an abandoned Popen leaves BrokenPipeError noise and a zombie.
"""

from __future__ import annotations

import pathlib
import subprocess
from collections.abc import Buffer
from typing import Any, cast

import pytest

from agent6 import kinds
from agent6.sandbox import jail, jail_protocol, jail_session


def _fake_binary(tmp_path: pathlib.Path, script: str) -> pathlib.Path:
    fake = tmp_path / "agent6-jail"
    fake.write_text(f"#!/bin/sh\n{script}\n")
    fake.chmod(0o755)
    return fake


def _recording_popen(monkeypatch: pytest.MonkeyPatch) -> list[subprocess.Popen[bytes]]:
    seen: list[subprocess.Popen[bytes]] = []
    real = subprocess.Popen

    def rec(*args: Any, **kwargs: Any) -> subprocess.Popen[bytes]:
        p = cast("subprocess.Popen[bytes]", real(*args, **kwargs))
        seen.append(p)
        return p

    monkeypatch.setattr(jail.subprocess, "Popen", rec)
    return seen


def test_a_launcher_dead_at_setup_is_reaped_with_its_pipes_closed(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(jail, "require_jail_binary", lambda: _fake_binary(tmp_path, "exit 7"))
    seen = _recording_popen(monkeypatch)
    with pytest.raises(jail.JailUnavailableError, match="died during setup"):
        jail_session.JailSession.open(kinds.JailPolicy(cwd=tmp_path, argv=("/usr/bin/true",)))
    (proc,) = seen
    assert proc.returncode is not None  # reaped: no zombie outlives the failure
    assert proc.stdin is not None and proc.stdin.closed
    assert proc.stdout is not None and proc.stdout.closed
    assert proc.stderr is not None and proc.stderr.closed


def test_a_spec_write_failure_is_the_same_setup_death(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """EPIPE at the spec write is JailUnavailableError like the EOF case; the child is reaped."""
    monkeypatch.setattr(jail, "require_jail_binary", lambda: _fake_binary(tmp_path, "read policy"))
    seen = _recording_popen(monkeypatch)
    real_write = jail_protocol.os.write

    def write(fd: int, data: Buffer) -> int:
        if seen and seen[0].stdin is not None and fd == seen[0].stdin.fileno():
            raise BrokenPipeError(32, "Broken pipe")
        return real_write(fd, data)

    monkeypatch.setattr(jail_protocol.os, "write", write)
    with pytest.raises(jail.JailUnavailableError, match="died during setup"):
        jail_session.JailSession.open(kinds.JailPolicy(cwd=tmp_path, argv=("/usr/bin/true",)))
    (proc,) = seen
    assert proc.returncode is not None
    assert proc.stdin is not None and proc.stdin.closed
    assert proc.stdout is not None and proc.stdout.closed
    assert proc.stderr is not None and proc.stderr.closed


def test_the_launchers_setup_error_is_the_error_both_transports_raise(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The launcher's sentence for a failed setup reaches the error a surface prints.

    A grant the launcher cannot bind is named on its stderr only; a transport that dropped
    it would leave the operator a bare exit code.
    """
    said = "agent6-jail: rootfs setup failed: write grant /data: bind: ENOENT"
    fake = _fake_binary(tmp_path, f'echo "{said}" >&2; exit 2')
    monkeypatch.setattr(jail, "require_jail_binary", lambda: fake)
    policy = kinds.JailPolicy(cwd=tmp_path, argv=("/usr/bin/true",), isolation="partial")
    with pytest.raises(jail.JailUnavailableError) as one_shot:
        jail.run_in_jail(policy)
    with pytest.raises(jail.JailUnavailableError) as served:
        jail_session.JailSession.open(policy)
    assert str(one_shot.value) == f"agent6-jail launcher exited 2: {said}"
    assert str(served.value) == f"jail session died during setup: {said}"
