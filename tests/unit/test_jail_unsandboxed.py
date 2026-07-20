# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the `none` (unsandboxed) jail profile used on non-Linux hosts.

These run on any platform and need no namespaces: the `none` profile runs the
command as a plain subprocess instead of invoking the Rust launcher.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from agent6.sandbox.jail import run_in_jail
from agent6.types import JailPolicy


def test_none_profile_runs_plain_subprocess(tmp_path: Path) -> None:
    res = run_in_jail(
        JailPolicy(
            cwd=tmp_path,
            argv=(sys.executable, "-c", "print('hello-unsandboxed')"),
            profile="none",
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
            profile="none",
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
            profile="none",
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
            profile="none",
            env=(("AGENT6_TEST_VAR", "set-by-policy"),),
            timeout_s=30.0,
        )
    )
    assert res.returncode == 0
    assert "set-by-policy" in res.stdout


def test_none_profile_preserves_non_utf8_output_lossily(tmp_path: Path) -> None:
    # Child output is not guaranteed UTF-8 (grep over a binary, cat of a
    # latin-1 file). The contract is a returned CommandResult with a lossy
    # decode, never a UnicodeDecodeError escaping communicate().
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
            profile="none",
            timeout_s=30.0,
        )
    )
    assert res.returncode == 0
    assert res.stdout == "caf� out"
    assert res.stderr == "caf� err"


def test_none_profile_timeout_returns_124_not_exception(tmp_path: Path) -> None:
    # The jailed profiles surface a timeout as rc=124; the `none` path used to
    # leak subprocess.TimeoutExpired instead. It must match the contract.
    res = run_in_jail(
        JailPolicy(
            cwd=tmp_path,
            argv=(sys.executable, "-c", "import time; time.sleep(10)"),
            profile="none",
            timeout_s=0.5,
        )
    )
    assert res.returncode == 124


def test_child_exec_failure_is_command_error_not_jail_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bad argv path (model guessed /usr/local/go/bin/go) means the JAIL
    worked and the COMMAND failed. Reporting it as 'jail unavailable' tells
    the model the sandbox is broken; report a shell-style 127 instead."""
    import subprocess

    from agent6.sandbox import jail as jail_mod

    monkeypatch.setattr(jail_mod, "locate_jail_binary", lambda: Path("/fake/agent6-jail"))

    # run_in_jail now uses Popen (it needs the pid to group-kill on timeout), so
    # fake the launcher there: a clean exec failure -> launcher rc=2 + the child
    # failure on stderr, which must map to a command error (127), not a raised
    # JailUnavailableError.
    class FakePopen:
        def __init__(self, *a: object, **k: object) -> None:
            self.pid = 424242
            self.returncode = 2

        def communicate(self, input: object = None, timeout: object = None) -> tuple[str, str]:
            return (
                "",
                "agent6-jail: child execution failed: No such file or directory (os error 2)",
            )

    monkeypatch.setattr(subprocess, "Popen", FakePopen)
    res = run_in_jail(
        JailPolicy(cwd=tmp_path, argv=("/usr/local/go/bin/go", "test"), profile="hardened")
    )
    assert res.returncode == 127
    assert "not found or not executable" in res.stderr
