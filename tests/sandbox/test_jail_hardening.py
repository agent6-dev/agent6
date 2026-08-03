# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""What a jailed command must not be able to do to the host."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.needs_namespaces


@pytest.mark.parametrize("level", ["hardened", "strict"])
def test_a_jailed_command_cannot_create_a_device_node(tmp_path: Path, level: str) -> None:
    """Under `sudo agent6` on a profile with no user namespace the child holds
    real CAP_MKNOD and no MS_NODEV bind: a block device for the host disk,
    created in its own workspace, reads and writes raw sectors past every path
    rule. Denied twice -- Landlock handles MakeChar/MakeBlock without granting
    them, and seccomp refuses mknod/mknodat."""
    from agent6.config import Config
    from agent6.sandbox.jail import run_in_jail
    from agent6.tools.dispatch import jail_policy

    res = run_in_jail(
        jail_policy(
            tmp_path,
            Config(),
            level,  # pyright: ignore[reportArgumentType]
            ("sh", "-c", "mknod disk b 8 0; mknod tty c 5 0; ls disk tty 2>&1"),
        )
    )
    assert "No such file" in res.stdout or "cannot access" in res.stdout
    assert not (tmp_path / "disk").exists() and not (tmp_path / "tty").exists()


def test_a_fifo_is_still_a_thing_a_build_can_make(tmp_path: Path) -> None:
    """Device nodes are blocked by MODE, not by denying the syscall: `mkfifo`
    and socket nodes go through mknodat too, and builds legitimately use them.
    Denying it outright would have broken them for a threat that is only about
    character and block devices.

    `strict` only: on `hardened` with protect_git, no NEW top-level entry in
    cwd can be created at all (the carve-out grants RW on cwd's existing
    children, never on cwd itself), so a fifo there fails for an unrelated
    reason -- identically before and after this filter.
    """
    level = "strict"
    from agent6.config import Config
    from agent6.sandbox.jail import run_in_jail
    from agent6.tools.dispatch import jail_policy

    res = run_in_jail(
        jail_policy(
            tmp_path,
            Config(),
            level,  # pyright: ignore[reportArgumentType]
            ("sh", "-c", f"mkfifo {level}.pipe && test -p {level}.pipe && echo fifo-ok"),
        )
    )
    assert "fifo-ok" in res.stdout, res.stdout + res.stderr


def test_every_mount_carries_the_nosuid_nodev_floor(tmp_path: Path) -> None:
    """EVERY mount, read from the jail's own mountinfo -- not a list of paths
    someone remembered to add.

    Three separate audits each found another mount missing the floor the
    comments call unconditional: the system binds, the writable /tmp, then the
    root tmpfs. Checking the paths those audits named would have passed each
    time; enumerating what is actually mounted is what closes the class.

    The /dev nodes are the one exception, and by necessity: `nodev` means "do
    not interpret device special files", so a device node mounted nodev is
    unusable. They carry nosuid and noexec instead.
    """
    from agent6.sandbox.jail import run_in_jail
    from agent6.types import JailPolicy

    probe = (
        "for l in open('/proc/self/mountinfo'):\n"
        "    f = l.split(' - ')[0].split()\n"
        "    print(f[4], f[5])\n"
    )
    res = run_in_jail(
        JailPolicy(cwd=tmp_path, argv=("python3", "-c", probe), isolation="strict", timeout_s=20.0)
    )
    rows = [ln.split() for ln in (res.stdout or "").strip().splitlines() if ln.split()]
    if not rows:
        pytest.skip(f"probe did not run: {res.stderr[:200]}")

    for mountpoint, flags in rows:
        assert "nosuid" in flags, f"{mountpoint} lacks nosuid: {flags}"
        if mountpoint.startswith("/dev/"):
            continue  # a device node mounted nodev cannot be used as one
        assert "nodev" in flags, f"{mountpoint} lacks nodev: {flags}"


def test_the_legacy_umount_syscall_is_blocked_too(tmp_path: Path) -> None:
    """Denying only umount2 left syscall 22 (`umount`, still live on x86_64)
    reaching the same teardown: a jailed child could unmount /proc, /dev or the
    workspace bind from inside its own namespace. Probed ALLOWED against a build
    without it.
    """
    from agent6.sandbox.jail import run_in_jail
    from agent6.types import JailPolicy

    probe = (
        "import ctypes\n"
        "libc = ctypes.CDLL(None, use_errno=True)\n"
        "r = libc.syscall(ctypes.c_long(22), b'/proc')\n"
        "print('blocked' if r == -1 else 'ALLOWED')\n"
    )
    res = run_in_jail(
        JailPolicy(cwd=tmp_path, argv=("python3", "-c", probe), isolation="strict", timeout_s=20.0)
    )
    if "blocked" not in (res.stdout or "") and "ALLOWED" not in (res.stdout or ""):
        pytest.skip(f"probe did not run: {res.stderr[:200]}")
    assert "ALLOWED" not in (res.stdout or ""), "the legacy umount syscall reached the jail"
