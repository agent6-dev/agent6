# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the `none` (unsandboxed) jail profile used on non-Linux hosts.

These run on any platform and need no namespaces: the `none` profile runs the
command as a plain subprocess instead of invoking the Rust launcher.
"""

from __future__ import annotations

import sys
from pathlib import Path

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
