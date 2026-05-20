# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Smoke tests for the Rust jail binary.

These tests are marked `needs_namespaces` and skipped unless unprivileged user
namespaces are available on the host. Building the jail is also opt-in via the
AGENT6_BUILD_JAIL env var so CI can choose when to pay the cost.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from agent6.sandbox.jail import JailUnavailableError, run_in_jail
from agent6.types import JailPolicy


def _userns_available() -> bool:
    res = subprocess.run(
        ["unshare", "-U", "-r", "true"],
        capture_output=True,
        check=False,
    )
    return res.returncode == 0


def _jail_binary() -> Path | None:
    env = os.environ.get("AGENT6_JAIL_BIN")
    if env and Path(env).is_file():
        return Path(env)
    p = Path(__file__).resolve().parents[2] / "jail" / "target" / "release" / "agent6-jail"
    if p.is_file():
        return p
    p_dbg = Path(__file__).resolve().parents[2] / "jail" / "target" / "debug" / "agent6-jail"
    if p_dbg.is_file():
        return p_dbg
    return None


pytestmark = pytest.mark.needs_namespaces


@pytest.fixture(scope="module")
def jail_bin() -> Path:
    if not _userns_available():
        pytest.skip("unprivileged user namespaces not available")
    bin_path = _jail_binary()
    if bin_path is None:
        if not os.environ.get("AGENT6_BUILD_JAIL"):
            pytest.skip("agent6-jail binary not built; set AGENT6_BUILD_JAIL=1 to build")
        cargo = shutil.which("cargo")
        if cargo is None:
            pytest.skip("cargo not available")
        repo_root = Path(__file__).resolve().parents[2]
        manifest = str(repo_root / "jail" / "Cargo.toml")
        subprocess.run(
            [cargo, "build", "--release", "--manifest-path", manifest],
            check=True,
        )
        bin_path = _jail_binary()
        assert bin_path is not None
    os.environ["AGENT6_JAIL_BIN"] = str(bin_path)
    return bin_path


def test_jail_runs_true(jail_bin: Path, tmp_path: Path) -> None:
    res = run_in_jail(JailPolicy(cwd=tmp_path, argv=("/usr/bin/true",), timeout_s=10.0))
    assert res.returncode == 0


def test_jail_blocks_network_when_disallowed(jail_bin: Path, tmp_path: Path) -> None:
    res = run_in_jail(
        JailPolicy(
            cwd=tmp_path,
            argv=("/usr/bin/getent", "hosts", "example.com"),
            allow_network=False,
            timeout_s=10.0,
        )
    )
    # nonzero return means resolution failed, which is what we want.
    assert res.returncode != 0


def test_jail_blocks_write_outside_workspace(jail_bin: Path, tmp_path: Path) -> None:
    marker = Path("/tmp/agent6-jail-host-escape-marker")
    if marker.exists():
        marker.unlink()
    try:
        run_in_jail(
            JailPolicy(
                cwd=tmp_path,
                argv=("/bin/sh", "-c", f"echo escape > {marker} || true"),
                timeout_s=10.0,
            )
        )
    except JailUnavailableError:
        pytest.skip("jail unavailable")
    # The file system inside the jail is bind-mounted onto /tmp; the host /tmp marker
    # should NOT exist because the in-jail /tmp is a fresh tmpfs.
    assert not marker.exists()
