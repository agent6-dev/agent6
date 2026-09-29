# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The `none` isolation runs the command as a plain subprocess on any platform."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from agent6.kinds import JailPolicy
from agent6.sandbox.jail import run_in_jail


def test_none_profile_runs_plain_subprocess(tmp_path: Path) -> None:
    res = run_in_jail(
        JailPolicy(
            cwd=tmp_path,
            argv=(sys.executable, "-c", "print('hello-unsandboxed')"),
            isolation="none",
            timeout_s=30.0,
        )
    )
    assert res.returncode == 0
    assert "hello-unsandboxed" in res.stdout


def test_none_profile_reports_nonzero_exit(tmp_path: Path) -> None:
    res = run_in_jail(
        JailPolicy(
            cwd=tmp_path,
            argv=(sys.executable, "-c", "import sys; sys.exit(7)"),
            isolation="none",
            timeout_s=30.0,
        )
    )
    assert res.returncode == 7
    assert res.ok is False


def test_none_profile_runs_in_cwd(tmp_path: Path) -> None:
    res = run_in_jail(
        JailPolicy(
            cwd=tmp_path,
            argv=(sys.executable, "-c", "import os; print(os.getcwd())"),
            isolation="none",
            timeout_s=30.0,
        )
    )
    assert res.returncode == 0
    assert str(tmp_path.resolve()) in res.stdout.strip()


def test_none_profile_overlays_policy_env(tmp_path: Path) -> None:
    res = run_in_jail(
        JailPolicy(
            cwd=tmp_path,
            argv=(sys.executable, "-c", "import os; print(os.environ.get('AGENT6_TEST_VAR'))"),
            isolation="none",
            env=(("AGENT6_TEST_VAR", "set-by-policy"),),
            timeout_s=30.0,
        )
    )
    assert res.returncode == 0
    assert "set-by-policy" in res.stdout


def test_none_profile_preserves_non_utf8_output_lossily(tmp_path: Path) -> None:
    # Child output is not guaranteed UTF-8; the contract is a lossy decode, never a raised error.
    res = run_in_jail(
        JailPolicy(
            cwd=tmp_path,
            argv=(
                sys.executable,
                "-c",
                "import sys;"
                " sys.stdout.buffer.write(b'caf\\xe9 out');"
                " sys.stderr.buffer.write(b'caf\\xe9 err')",
            ),
            isolation="none",
            timeout_s=30.0,
        )
    )
    assert res.returncode == 0
    assert res.stdout == "caf� out"
    assert res.stderr == "caf� err"


def test_none_profile_timeout_returns_124_not_exception(tmp_path: Path) -> None:
    # The jailed levels surface a timeout as rc=124; the `none` path must match.
    res = run_in_jail(
        JailPolicy(
            cwd=tmp_path,
            argv=(sys.executable, "-c", "import time; time.sleep(10)"),
            isolation="none",
            timeout_s=0.5,
        )
    )
    assert res.returncode == 124


def test_closing_one_unconfined_server_spares_a_later_sibling(tmp_path: Path) -> None:
    """`spawn_in_jail(isolation="none")` registers its pid like the jailed path does.

    Unregistered, closing server A escapee-sweeps a sibling spawned after it.
    """
    import subprocess

    from agent6.sandbox.jail import JailedProcess, spawn_in_jail

    def _spawn() -> JailedProcess:
        return spawn_in_jail(
            JailPolicy(
                cwd=tmp_path,
                argv=(sys.executable, "-c", "import time; time.sleep(30)"),
                isolation="none",
                timeout_s=60.0,
            ),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    a = _spawn()
    b = _spawn()
    try:
        a.close()
        assert b.popen.poll() is None, "closing A killed sibling B"
    finally:
        b.close()
    assert b.popen.poll() is not None


def test_child_exec_failure_is_command_error_not_jail_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bad argv path is a shell-style 127, not "jail unavailable": the jail worked."""
    import subprocess

    from agent6.sandbox import jail as jail_mod

    monkeypatch.setattr(jail_mod, "locate_jail_binary", lambda: Path("/fake/agent6-jail"))

    # A clean exec failure (launcher rc=2) maps to a command error 127, not JailUnavailableError.
    class FakePopen:
        def __init__(self, *a: object, **k: object) -> None:
            self.pid = 424242
            self.returncode = 2

        def communicate(self, input: object = None, timeout: object = None) -> tuple[str, str]:
            return (
                "",
                "agent6-jail: child execution failed: No such file or directory (os error 2)",
            )

        def poll(self) -> int:
            return self.returncode  # already exited, so the escapee sweep leaves it alone

    monkeypatch.setattr(subprocess, "Popen", FakePopen)
    res = run_in_jail(
        JailPolicy(cwd=tmp_path, argv=("/usr/local/go/bin/go", "test"), isolation="hardened")
    )
    assert res.returncode == 127
    assert "not found or not executable" in res.stderr
