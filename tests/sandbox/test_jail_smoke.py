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
    p = Path(__file__).resolve().parents[2] / "src" / "agent6" / "jail" / "target"
    release = p / "release" / "agent6-jail"
    if release.is_file():
        return release
    debug = p / "debug" / "agent6-jail"
    if debug.is_file():
        return debug
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
        manifest = str(repo_root / "src" / "agent6" / "jail" / "Cargo.toml")
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


def test_jail_dev_null_is_writable(jail_bin: Path, tmp_path: Path) -> None:
    """Writes to /dev/null and friends must succeed under both profiles.

    Regression test for the click-short-help bench task INTERNALERROR:
    pytest's logging plugin opens /dev/null O_WRONLY|O_APPEND when a
    `log_file` is configured (click's conftest does this), and the previous
    Landlock rules granted only read+execute on /dev — surfacing as
    PermissionError before any test could run.
    """
    for profile in ("strict", "hardened"):
        res = run_in_jail(
            JailPolicy(
                cwd=tmp_path,
                argv=("/bin/sh", "-c", "echo x > /dev/null && echo OK"),
                profile=profile,
                timeout_s=10.0,
            )
        )
        assert res.returncode == 0, f"{profile} stderr: {res.stderr!r}"
        assert "OK" in res.stdout, f"{profile} stdout: {res.stdout!r}"


def test_jail_protect_paths_block_writes_to_subdir(jail_bin: Path, tmp_path: Path) -> None:
    """extra_protect_paths must make a sub-directory of cwd read-only."""
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    # First confirm without protection the write succeeds inside the jail.
    res_unprotected = run_in_jail(
        JailPolicy(
            cwd=tmp_path,
            argv=("/bin/sh", "-c", "echo pwned > .git/HEAD && cat .git/HEAD"),
            timeout_s=10.0,
        )
    )
    assert res_unprotected.returncode == 0
    assert "pwned" in res_unprotected.stdout
    # Reset and protect.
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    res_protected = run_in_jail(
        JailPolicy(
            cwd=tmp_path,
            argv=("/bin/sh", "-c", "echo pwned > .git/HEAD; cat .git/HEAD"),
            extra_protect_paths=(git_dir,),
            timeout_s=10.0,
        )
    )
    # The shell write fails (EROFS) but `cat` still runs; HEAD is unchanged.
    assert "pwned" not in res_protected.stdout
    assert (git_dir / "HEAD").read_text(encoding="utf-8") == "ref: refs/heads/main\n"


def test_jail_protect_paths_block_writes_to_file(jail_bin: Path, tmp_path: Path) -> None:
    """extra_protect_paths must also protect individual files (not just dirs)."""
    cfg = tmp_path / "protected.txt"
    cfg.write_text("original\n", encoding="utf-8")
    res = run_in_jail(
        JailPolicy(
            cwd=tmp_path,
            argv=("/bin/sh", "-c", "echo pwned > protected.txt; cat protected.txt"),
            extra_protect_paths=(cfg,),
            timeout_s=10.0,
        )
    )
    assert "pwned" not in res.stdout
    assert cfg.read_text(encoding="utf-8") == "original\n"


def test_jail_hardened_protect_paths_block_writes(jail_bin: Path, tmp_path: Path) -> None:
    """Hardened profile blocks writes to protect_paths via Landlock carve-out.

    Hardened has no mount namespace so it cannot bind-remount RO; instead the
    launcher switches its Landlock rules from `RW on cwd` to `R on cwd + RW
    on every top-level entry except the protect set`. End result for paths
    that exist at jail-launch time is the same: writes are denied.
    """
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    # Make a sibling that the worker IS allowed to write to, to prove we
    # didn't accidentally lock down the whole cwd.
    (tmp_path / "src").mkdir()
    res = run_in_jail(
        JailPolicy(
            cwd=tmp_path,
            argv=(
                "/bin/sh",
                "-c",
                "echo ok > src/x.txt && echo pwned > .git/HEAD; cat src/x.txt; cat .git/HEAD",
            ),
            profile="hardened",
            extra_protect_paths=(git_dir,),
            timeout_s=10.0,
        )
    )
    assert "ok" in res.stdout  # sibling write succeeded
    assert "pwned" not in res.stdout  # protected write rejected
    assert (git_dir / "HEAD").read_text(encoding="utf-8") == "ref: refs/heads/main\n"
