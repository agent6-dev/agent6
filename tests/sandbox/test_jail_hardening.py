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

    A bind inherits its SOURCE mount's flags, so on a host whose /tmp is already
    nosuid this cannot distinguish an explicit floor from an inherited one. The
    launcher sets the flags explicitly for that reason; probed on ext4, the
    tool_paths mount came back `ro,relatime` without them.
    """
    from agent6.sandbox.jail import run_in_jail
    from agent6.types import JailPolicy

    probe = (
        "for l in open('/proc/self/mountinfo'):\n"
        "    f = l.split(' - ')[0].split()\n"
        "    print(f[4], f[5])\n"
    )
    # WITH the operator grants: a bare policy mounts none of them, and checking
    # only what a bare policy mounts is how tool_paths and extra_ro_paths kept
    # their gap through the sweep that was meant to close this class.
    tool_dir, ro_dir = tmp_path / "tools", tmp_path / "ro"
    tool_dir.mkdir()
    ro_dir.mkdir()
    (ro_dir / "f.txt").write_text("x", encoding="utf-8")
    res = run_in_jail(
        JailPolicy(
            cwd=tmp_path,
            argv=("python3", "-c", probe),
            isolation="strict",
            tool_paths=(tool_dir,),
            extra_ro_paths=(ro_dir,),
            timeout_s=20.0,
        )
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


def test_the_jail_launcher_does_not_carry_the_agent_env_into_the_jail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A jailed command must not be able to read the operator's provider key.

    The launcher becomes PID 1 of the jail's own PID namespace, and strict
    mounts a fresh /proc -- so /proc/1/environ IS the launcher's environment.
    Spawned without an explicit env it inherited the agent's, and a jailed
    command could read `OPENROUTER_API_KEY=...` straight out of it. Probed and
    reproduced before the fix.

    docs/security.md says secrets never reach the jail; they were not mounted,
    they were inherited. The launcher reads nothing from its environment (the
    policy arrives on stdin), so it gets none.
    """
    from agent6.sandbox.jail import run_in_jail
    from agent6.types import JailPolicy

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-CANARY-must-not-leak")
    probe = (
        "import glob\n"
        "for p in sorted(glob.glob('/proc/[0-9]*/environ')):\n"
        "    try: d = open(p, 'rb').read()\n"
        "    except Exception: continue\n"
        "    if b'CANARY' in d: print('LEAK ' + p)\n"
    )
    res = run_in_jail(
        JailPolicy(cwd=tmp_path, argv=("python3", "-c", probe), isolation="strict", timeout_s=20.0)
    )
    assert "LEAK" not in (res.stdout or ""), f"the agent's env reached the jail: {res.stdout}"


def test_a_fully_populated_policy_holds_every_invariant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every jail invariant at once, against a policy with EVERY field set.

    The tests around this one each exercise a default-shaped policy, and that is
    how tool_paths and extra_ro_paths kept a missing mount floor through a sweep
    meant to close that class: a bare policy mounts neither, so enumerating
    "every mount" enumerated everything except them. This one populates the
    whole surface -- ro/rw/protect grants, tool paths, a child env, a memory cap
    -- and asserts the properties together.

    Honest limit on the GAPS half: a bind inherits its SOURCE mount's flags, and
    pytest's tmp_path is usually on a tmpfs that already carries nosuid,nodev --
    so on such a host this cannot tell an explicit floor from an inherited one
    (red-verified: reverting the tool_paths floor leaves this passing). It bites
    where tmp is ext4, and the ext4 case was probed by hand. The LEAK and
    PROTECT halves are deterministic everywhere.
    """
    from agent6.sandbox.jail import run_in_jail
    from agent6.types import JailPolicy

    ws, ro, rw, tools = (tmp_path / n for n in ("ws", "ro", "rw", "tools"))
    for d in (ws, ro, rw, tools):
        d.mkdir()
    (ws / ".git").mkdir()
    (ws / ".git" / "config").write_text("[core]\n", encoding="utf-8")
    (ro / "f.txt").write_text("ro", encoding="utf-8")
    monkeypatch.setenv("AGENT_ONLY_CANARY", "sk-PARENT-SECRET")

    probe = (
        "import os, glob\n"
        "me = os.getpid()\n"
        "gaps = []\n"
        "for l in open('/proc/self/mountinfo'):\n"
        "    f = l.split(' - ')[0].split()\n"
        "    if f[4].startswith('/dev/'): continue\n"
        "    if 'nosuid' not in f[5] or 'nodev' not in f[5]: gaps.append(f[4])\n"
        "print('GAPS', gaps)\n"
        "leak = [p for p in glob.glob('/proc/[0-9]*/environ')\n"
        "        if int(p.split('/')[2]) != me and b'PARENT-SECRET' in open(p,'rb').read()]\n"
        "print('LEAK', leak)\n"
        "try:\n"
        "    open('/workspace/.git/config','w').write('x'); print('PROTECT writable')\n"
        "except OSError: print('PROTECT refused')\n"
    )
    res = run_in_jail(
        JailPolicy(
            cwd=ws,
            argv=("python3", "-c", probe),
            isolation="strict",
            env=(("CHILD_VAR", "child-only"),),
            extra_ro_paths=(ro,),
            extra_rw_paths=(rw,),
            extra_protect_paths=(ws / ".git",),
            tool_paths=(tools,),
            timeout_s=30.0,
            memory_limit_mb=512,
        )
    )
    out = res.stdout or ""
    if "GAPS" not in out:
        pytest.skip(f"probe did not run: {res.stderr[:200]}")
    assert "GAPS []" in out, f"a mount in a fully-populated policy lacks the floor: {out}"
    assert "LEAK []" in out, f"the agent's environment reached the jail: {out}"
    assert "PROTECT refused" in out, f"a protect path was writable: {out}"
