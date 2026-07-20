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


def test_jail_memory_limit_caps_child_allocation(jail_bin: Path, tmp_path: Path) -> None:
    """memory_limit_mb turns a runaway allocation into a plain failed command.

    A child allocating 200 MiB under a 64 MiB cap must die with MemoryError
    (RLIMIT_DATA, applied in run_child and shared by both profiles) while the
    host never approaches the OOM killer; the same allocation with the 0
    opt-out succeeds.
    """
    alloc = (
        "import sys\n"
        "try:\n"
        "    bytearray(200 * 1024 * 1024)\n"
        "except MemoryError:\n"
        "    sys.exit(9)\n"
        "print('ALLOC-OK')\n"
    )
    capped = run_in_jail(
        JailPolicy(
            cwd=tmp_path,
            argv=("/usr/bin/python3", "-c", alloc),
            memory_limit_mb=64,
            timeout_s=30.0,
        )
    )
    assert capped.returncode == 9, f"stderr: {capped.stderr!r}"
    uncapped = run_in_jail(
        JailPolicy(
            cwd=tmp_path,
            argv=("/usr/bin/python3", "-c", alloc),
            memory_limit_mb=0,
            timeout_s=30.0,
        )
    )
    assert uncapped.returncode == 0, f"stderr: {uncapped.stderr!r}"
    assert "ALLOC-OK" in uncapped.stdout


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


def test_jail_hardened_symlink_escaping_cwd_gets_no_rw(jail_bin: Path, tmp_path: Path) -> None:
    """A top-level symlink whose target escapes cwd must not receive RW.

    Under hardened the per-top-level-entry RW carve-out used PathFd::new (which
    follows symlinks), so a symlink like ``./escape -> /outside`` got a
    recursive RW Landlock rule on the *outside* inode, letting the child write
    beyond the workspace. The target is placed under ``/dev/shm`` -- outside cwd
    and NOT under ``/tmp`` (which the jail grants RW), so /dev (read+exec only)
    is the governing rule unless the symlink wrongly widens it.
    """
    import shutil as _shutil
    import uuid as _uuid

    shm = Path("/dev/shm")
    if not shm.is_dir() or not os.access(shm, os.W_OK):
        pytest.skip("/dev/shm not usable for the out-of-cwd target")
    outside = shm / f"agent6-jail-escape-{_uuid.uuid4().hex}"
    outside.mkdir()
    try:
        (tmp_path / ".git").mkdir()  # a protect path so the carve-out loop runs
        (tmp_path / "src").mkdir()
        (tmp_path / "escape").symlink_to(outside, target_is_directory=True)
        res = run_in_jail(
            JailPolicy(
                cwd=tmp_path,
                argv=("/bin/sh", "-c", "echo ok > src/x.txt; echo pwned > escape/sentinel; true"),
                profile="hardened",
                extra_protect_paths=(tmp_path / ".git",),
                timeout_s=10.0,
            )
        )
        # In-cwd sibling write still works; the escaping write is denied.
        assert (tmp_path / "src" / "x.txt").read_text(encoding="utf-8").strip() == "ok"
        escaped = (outside / "sentinel").exists()
        assert not escaped, f"escaped write succeeded; stderr={res.stderr!r}"
    finally:
        _shutil.rmtree(outside, ignore_errors=True)


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


def test_jail_tool_paths_make_a_nonworkspace_binary_reachable(
    jail_bin: Path, tmp_path: Path
) -> None:
    # A tool dir OUTSIDE the workspace (like ~/.local/bin or a pipx /opt target) is
    # unreachable by default; passing it as tool_paths bind-mounts it RO+exec at its
    # real path so PATH resolves it. This is what makes an operator's uv reachable.
    work = tmp_path / "work"
    work.mkdir()
    tools = tmp_path / "tools"  # sibling of cwd -> not under the workspace mount
    tools.mkdir()
    script = tools / "mytool"
    script.write_text("#!/bin/sh\necho tool-ran\n")
    script.chmod(0o755)
    env = (("PATH", f"/usr/bin:/bin:{tools}"),)

    with_mount = run_in_jail(
        JailPolicy(
            cwd=work,
            argv=("/bin/sh", "-c", "mytool"),
            env=env,
            tool_paths=(tools,),
            timeout_s=10.0,
        )
    )
    assert with_mount.returncode == 0, with_mount.stderr
    assert "tool-ran" in with_mount.stdout

    # Same PATH but no tool_paths: the dir is not mounted, so exec fails (guards
    # against the mount silently becoming a no-op).
    without_mount = run_in_jail(
        JailPolicy(cwd=work, argv=("/bin/sh", "-c", "mytool"), env=env, timeout_s=10.0)
    )
    assert without_mount.returncode != 0


def test_jail_extra_ro_paths_mount_at_their_real_location(jail_bin: Path, tmp_path: Path) -> None:
    # The documented contract: a granted toolchain (a conda env, a shared data
    # dir) is usable via its own absolute paths and shebangs. The grant used to
    # remap under an undocumented /ro<src>, where nothing could find it.
    work = tmp_path / "work"
    work.mkdir()
    toolchain = tmp_path / "toolchain"  # outside the workspace mount
    toolchain.mkdir()
    script = toolchain / "hello.sh"
    script.write_text("#!/bin/sh\necho reached-real-path\n")
    script.chmod(0o755)

    granted = run_in_jail(
        JailPolicy(cwd=work, argv=(str(script),), extra_ro_paths=(toolchain,), timeout_s=10.0)
    )
    assert granted.returncode == 0, granted.stderr
    assert "reached-real-path" in granted.stdout

    # Read-only: a write inside the grant is refused.
    ro = run_in_jail(
        JailPolicy(
            cwd=work,
            argv=("/bin/sh", "-c", f"echo x > {toolchain}/marker"),
            extra_ro_paths=(toolchain,),
            timeout_s=10.0,
        )
    )
    assert ro.returncode != 0
    assert not (toolchain / "marker").exists()

    # Without the grant the path does not exist inside the jail at all.
    ungranted = run_in_jail(JailPolicy(cwd=work, argv=(str(script),), timeout_s=10.0))
    assert ungranted.returncode != 0


def test_jail_preserves_non_utf8_output(jail_bin: Path, tmp_path: Path) -> None:
    """A command emitting non-UTF-8 bytes must return a lossy-decoded result,
    not a silently empty stdout. read_to_string dropped the whole stream to ""
    on the first invalid byte (grep over a binary, cat of a latin-1 file)."""
    for profile in ("strict", "hardened"):
        res = run_in_jail(
            JailPolicy(
                cwd=tmp_path,
                argv=("/bin/sh", "-c", "printf 'caf'; printf '\\351'; printf 'x'"),
                profile=profile,
                timeout_s=10.0,
            )
        )
        assert res.returncode == 0, f"{profile} stderr: {res.stderr!r}"
        # 0xe9 decodes to the replacement char; the surrounding bytes survive.
        assert res.stdout.startswith("caf"), f"{profile} stdout: {res.stdout!r}"
        assert res.stdout.endswith("x"), f"{profile} stdout: {res.stdout!r}"
        assert "�" in res.stdout, f"{profile} stdout: {res.stdout!r}"


def test_jail_backgrounded_pipe_holder_does_not_hang(jail_bin: Path, tmp_path: Path) -> None:
    """A command that backgrounds a process inheriting stdout, then exits 0,
    must return promptly with rc=0 -- not block on the reader join until the
    (30s-sleeping) grandchild dies and then report a false rc=124 timeout.
    The process-group teardown runs on the normal-exit path, not only on
    timeout. Hardened has no PID namespace, so it is the exposed profile."""
    import time

    start = time.monotonic()
    res = run_in_jail(
        JailPolicy(
            cwd=tmp_path,
            argv=("/bin/sh", "-c", "sleep 30 & echo done; exit 0"),
            profile="hardened",
            timeout_s=10.0,
        )
    )
    elapsed = time.monotonic() - start
    assert res.returncode == 0, f"stderr: {res.stderr!r}"
    assert "done" in res.stdout
    assert elapsed < 8.0, f"launcher blocked on the backgrounded fd-holder ({elapsed:.1f}s)"


def test_jail_strict_seccomp_blocks_modern_mount_api(jail_bin: Path, tmp_path: Path) -> None:
    """A strict jailed child is userns-root over its own mount ns; without the
    modern mount API in the seccomp deny-list it could mount_setattr(2) away the
    RO flag on the .git protect bind and defeat protect_git. The syscall must
    return EPERM. Uses ctypes so no extra tooling is needed."""
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    prog = (
        "import ctypes, ctypes.util, os\n"
        "libc = ctypes.CDLL(ctypes.util.find_library('c'), use_errno=True)\n"
        # mount_setattr(dirfd=AT_FDCWD, path, flags=0, attr=NULL, size=0):
        # NULL attr makes it a pure permission probe -- EPERM (seccomp) vs
        # EFAULT/EINVAL (syscall reached the kernel) is what we assert on.
        "r = libc.syscall(442, -100, b'/workspace/.git', 0, 0, 0)\n"
        "e = ctypes.get_errno()\n"
        "print('EPERM' if (r == -1 and e == 1) else f'REACHED:{e}')\n"
    )
    res = run_in_jail(
        JailPolicy(
            cwd=tmp_path,
            argv=("/usr/bin/python3", "-c", prog),
            profile="strict",
            extra_protect_paths=(git_dir,),
            timeout_s=10.0,
        )
    )
    assert res.returncode == 0, f"stderr: {res.stderr!r}"
    assert "EPERM" in res.stdout, f"mount_setattr not blocked: {res.stdout!r}"


def test_jail_extra_rw_paths_mount_at_their_real_location(jail_bin: Path, tmp_path: Path) -> None:
    # extra_rw (the machine data dir) is writable AT the host abspath, so
    # $AGENT6_MACHINE_DATA_DIR is the same string in every profile.
    work = tmp_path / "work"
    work.mkdir()
    data = tmp_path / "data"  # outside the workspace mount
    data.mkdir()
    res = run_in_jail(
        JailPolicy(
            cwd=work,
            argv=("/bin/sh", "-c", f"echo persisted > {data}/out"),
            extra_rw_paths=(data,),
            timeout_s=10.0,
        )
    )
    assert res.returncode == 0, res.stderr
    assert (data / "out").read_text().strip() == "persisted"
