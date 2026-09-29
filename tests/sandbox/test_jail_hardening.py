# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""What a jailed command must not be able to do to the host."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.needs_namespaces


@pytest.mark.parametrize("level", ["hardened", "strict"])
def test_a_jailed_command_cannot_create_a_device_node(tmp_path: Path, level: str) -> None:
    """A jailed command cannot mknod a block device for the host disk.

    Under `sudo agent6` on a profile with no user namespace the child holds real CAP_MKNOD
    and no MS_NODEV bind, so such a node in its own workspace would read and write raw
    sectors past every path. Denied by the seccomp rule, which refuses mknod/mknodat by
    device type, and by Landlock, which grants MakeChar/MakeBlock nowhere; this pin cannot
    tell the two apart.
    """
    from agent6.config import Config
    from agent6.sandbox.jail import run_in_jail
    from agent6.tools.dispatch import jail_policy

    res = run_in_jail(
        jail_policy(
            tmp_path,
            Config(),
            level,  # pyright: ignore[reportArgumentType]
            ("sh", "-c", "mknod disk b 8 0; mknod tty c 5 0; ls disk tty 2>&1"),
            network="none",
        )
    )
    assert "No such file" in res.stdout or "cannot access" in res.stdout
    assert not (tmp_path / "disk").exists() and not (tmp_path / "tty").exists()


def test_a_fifo_is_still_a_thing_a_build_can_make(tmp_path: Path) -> None:
    """Device nodes are blocked by mode, not by denying mknodat: a fifo still works.

    `mkfifo` and socket nodes go through mknodat too, and builds use them. `strict` only: on
    `hardened` with protect_git no new top-level entry in cwd can be created at all (the
    carve-out grants RW on cwd's existing children, never cwd itself), so a fifo there fails
    for an unrelated reason.
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
            network="none",
        )
    )
    assert "fifo-ok" in res.stdout, res.stdout + res.stderr


def test_every_mount_carries_the_nosuid_nodev_floor(tmp_path: Path) -> None:
    """Every mount in the jail's own mountinfo carries the nosuid/nodev/noexec floor.

    Enumerating what is mounted closes the class; a list of remembered paths misses the
    next one. The /dev nodes are the one exception, by necessity: a device node mounted
    nodev is unusable, so they carry nosuid and noexec only. A bind inherits its source
    mount's flags, so on a host whose /tmp is already nosuid this cannot distinguish an
    explicit floor from an inherited one; the launcher sets the flags explicitly for that
    reason (probed on ext4, the tool_paths mount came back `ro,relatime` without them).
    """
    from agent6.kinds import JailPolicy
    from agent6.sandbox.jail import run_in_jail

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


def test_a_submount_inside_a_grant_carries_the_floor_too(tmp_path: Path) -> None:
    """The floor reaches the submounts a recursive bind carries in, not just each grant's top.

    `MS_REC` is silently ignored on `MS_REMOUNT`; recursive attribute changes need
    `mount_setattr(AT_RECURSIVE)`. Otherwise a mount nested inside a grant arrives with its
    source flags: a tmpfs under a read-only grant comes in `rw,relatime`, and a jailed command
    writes a file that is still on the host afterwards. The sibling mountinfo test cannot
    catch this on a host with no nested mount under any grant. Creating the submount needs a
    mount namespace, so the probe runs under `unshare`; the jail's own userns nests inside it.
    """
    import shutil
    import subprocess
    import sys
    import textwrap

    if shutil.which("unshare") is None:
        pytest.skip("needs unshare to nest a mount under a grant")

    ws, ro, tools = (tmp_path / n for n in ("ws", "ro", "tools"))
    for d in (ws, ws / "vendor", ro / "sub", tools / "sub"):
        d.mkdir(parents=True)

    inner = textwrap.dedent(
        """
        import ctypes, sys
        from pathlib import Path
        from agent6.sandbox.jail import run_in_jail
        from agent6.kinds import JailPolicy

        libc = ctypes.CDLL(None, use_errno=True)
        ws, ro, tools = (Path(p) for p in sys.argv[1:4])
        for sub in (ws / "vendor", ro / "sub", tools / "sub"):
            if libc.mount(b"tmpfs", str(sub).encode(), b"tmpfs", 0, None) != 0:
                print("MOUNT_FAILED", ctypes.get_errno())
                raise SystemExit(0)

        probe = (
            "for l in open('/proc/self/mountinfo'):\\n"
            "    f = l.split(' - ')[0].split()\\n"
            "    print(f[4], f[5])\\n"
        )
        res = run_in_jail(
            JailPolicy(
                cwd=ws,
                argv=("python3", "-c", probe),
                isolation="strict",
                tool_paths=(tools,),
                extra_ro_paths=(ro,),
                timeout_s=20.0,
            )
        )
        print(res.stdout)
        print("STDERR", res.stderr[:300])
        """
    )
    proc = subprocess.run(
        [
            "unshare",
            "--map-root-user",
            "--mount",
            "--propagation",
            "private",  # the unshare(1) flag value, not our network vocabulary
            sys.executable,
            "-c",
            inner,
            str(ws),
            str(ro),
            str(tools),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    out = proc.stdout
    if "MOUNT_FAILED" in out or proc.returncode != 0:
        pytest.skip(f"could not nest a mount: {out[:200]} {proc.stderr[:200]}")
    rows = [ln.split() for ln in out.strip().splitlines() if ln.startswith("/")]
    if not rows:
        pytest.skip(f"probe did not run: {out[:200]} {proc.stderr[:300]}")

    nested = [(mp, fl) for mp, fl in rows if mp.endswith(("/vendor", "/sub"))]
    assert len(nested) == 3, f"the submounts did not reach the jail: {rows}"
    for mountpoint, flags in nested:
        assert "nosuid" in flags and "nodev" in flags, f"{mountpoint} lacks the floor: {flags}"
        # Under a read-only grant the submount must be read-only as well: the
        # grant is what the operator made read-only, not its top mount.
        if "/ro/" in mountpoint or "/tools/" in mountpoint:
            assert flags.startswith("ro"), f"{mountpoint} is writable inside a RO grant: {flags}"


def test_a_protect_path_with_its_own_submount_still_jails(tmp_path: Path) -> None:
    """A mount nested under a protect path stays visible and read-only under the protect bind.

    `.git/objects` on its own bind is carried in by the recursive workspace bind and then
    covered by the protect bind; remounting its stale mountinfo line, which is not a mount
    point any more, is EINVAL and would refuse every jailed command.
    """
    import shutil
    import subprocess
    import sys
    import textwrap

    if shutil.which("unshare") is None:
        pytest.skip("needs unshare to nest a mount under a protect path")

    ws = tmp_path / "ws"
    (ws / ".git" / "objects").mkdir(parents=True)

    probe = (
        "echo run-ok; "
        "cat .git/objects/seed; "
        "{ echo x > .git/objects/tamper; } 2>/dev/null"
        " && echo OBJECTS-WRITABLE || echo objects-protected; "
        "{ echo x > .git/tamper; } 2>/dev/null"
        " && echo GIT-WRITABLE || echo git-protected; "
        "echo w > note.txt && echo ws-writable"
    )
    inner = textwrap.dedent(
        f"""
        import ctypes, sys
        from pathlib import Path
        from agent6.sandbox.jail import run_in_jail
        from agent6.kinds import JailPolicy

        libc = ctypes.CDLL(None, use_errno=True)
        ws = Path(sys.argv[1])
        sub = ws / ".git" / "objects"
        if libc.mount(b"tmpfs", str(sub).encode(), b"tmpfs", 0, None) != 0:
            print("MOUNT_FAILED", ctypes.get_errno())
            raise SystemExit(0)
        (sub / "seed").write_text("seeded-content")
        try:
            res = run_in_jail(
                JailPolicy(
                    cwd=ws,
                    argv=("sh", "-c", {probe!r}),
                    isolation="strict",
                    extra_protect_paths=(ws / ".git",),
                    timeout_s=20.0,
                )
            )
        except Exception as exc:
            print("JAIL-REFUSED:", exc)
        else:
            print(res.stdout)
            print("STDERR", res.stderr[:300])
        """
    )
    proc = subprocess.run(
        [
            "unshare",
            "--map-root-user",
            "--mount",
            "--propagation",
            "private",  # the unshare(1) flag value, not our network vocabulary
            sys.executable,
            "-c",
            inner,
            str(ws),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    out = proc.stdout
    if "MOUNT_FAILED" in out or proc.returncode != 0:
        pytest.skip(f"could not nest a mount: {out[:200]} {proc.stderr[:200]}")
    assert "run-ok" in out, f"the jail refused outright: {out[:400]}"
    assert "seeded-content" in out, f"the nested mount's content is hidden: {out[:400]}"
    assert "objects-protected" in out and "OBJECTS-WRITABLE" not in out, out[:400]
    assert "git-protected" in out and "GIT-WRITABLE" not in out, out[:400]
    assert "ws-writable" in out, out[:400]


def test_a_locked_flag_on_a_system_bind_source_is_carried_not_cleared(tmp_path: Path) -> None:
    """A system bind whose source carries a locked flag is remounted read-only with those flags.

    /etc/alternatives on a noexec tmpfs is a hardened host's shape; clearing a locked flag
    in a user namespace is refused EPERM, so the jail would fail closed on exactly the hosts
    hardened the way its own floor recommends.
    """
    import shutil
    import subprocess
    import sys
    import textwrap

    if shutil.which("unshare") is None:
        pytest.skip("needs unshare to overmount a system bind source")
    if not Path("/etc/alternatives").is_dir():
        pytest.skip("no /etc/alternatives on this host")

    ws = tmp_path / "ws"
    ws.mkdir()

    inner = textwrap.dedent(
        """
        import ctypes, sys
        from pathlib import Path
        from agent6.sandbox.jail import run_in_jail
        from agent6.kinds import JailPolicy

        MS_NOSUID, MS_NODEV, MS_NOEXEC = 2, 4, 8
        libc = ctypes.CDLL(None, use_errno=True)
        flags = MS_NOSUID | MS_NODEV | MS_NOEXEC
        if libc.mount(b"tmpfs", b"/etc/alternatives", b"tmpfs", flags, None) != 0:
            print("MOUNT_FAILED", ctypes.get_errno())
            raise SystemExit(0)
        try:
            res = run_in_jail(
                JailPolicy(
                    cwd=Path(sys.argv[1]),
                    argv=("sh", "-c", "echo probe-ok"),
                    isolation="strict",
                    timeout_s=20.0,
                )
            )
        except Exception as exc:
            print("JAIL-REFUSED:", exc)
        else:
            print(res.stdout)
            print("STDERR", res.stderr[:300])
        """
    )
    proc = subprocess.run(
        [
            "unshare",
            "--map-root-user",
            "--mount",
            "--propagation",
            "private",  # the unshare(1) flag value, not our network vocabulary
            sys.executable,
            "-c",
            inner,
            str(ws),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    out = proc.stdout
    if "MOUNT_FAILED" in out or proc.returncode != 0:
        pytest.skip(f"could not overmount: {out[:200]} {proc.stderr[:200]}")
    assert "probe-ok" in out, f"the jail refused on a hardened system mount: {out[:400]}"


def test_the_teardown_call_is_denied_and_pipe_is_not(tmp_path: Path) -> None:
    """umount2 is denied and syscall 22 (`pipe(2)` on x86_64) is left alone.

    The 64-bit table has no legacy umount; number 22 is the i386 table's, unreachable either
    way since seccompiler's arch prologue kills a foreign-arch caller. A probe that read
    22's success as "legacy umount allowed" would deny pipe(2) under a comment about
    unmounting, unnoticed because glibc routes pipe() through pipe2(). Both halves matter:
    the jail denies the teardown and leaves an ordinary syscall alone.
    """
    import platform

    from agent6.kinds import JailPolicy
    from agent6.sandbox.jail import run_in_jail

    # Numbers, not names: the point is which number the arch assigns to what.
    by_arch = {"x86_64": (166, 22), "aarch64": (39, None)}  # (umount2, pipe or none)
    if platform.machine() not in by_arch:
        pytest.skip(f"no syscall numbers pinned for {platform.machine()}")
    umount2_nr, pipe_nr = by_arch[platform.machine()]

    probe = (
        "import ctypes\n"
        "libc = ctypes.CDLL(None, use_errno=True)\n"
        "buf = (ctypes.c_int * 2)()\n"
        f"print('umount2', libc.syscall(ctypes.c_long({umount2_nr}), b'/proc', 0),"
        " ctypes.get_errno())\n"
        + (
            f"print('pipe', libc.syscall(ctypes.c_long({pipe_nr}), ctypes.byref(buf)),"
            " ctypes.get_errno())\n"
            if pipe_nr
            else ""
        )
    )
    res = run_in_jail(
        JailPolicy(cwd=tmp_path, argv=("python3", "-c", probe), isolation="strict", timeout_s=20.0)
    )
    out = res.stdout or ""
    if "umount2" not in out:
        pytest.skip(f"probe did not run: {res.stderr[:200]}")
    umount2_line = next(ln for ln in out.splitlines() if ln.startswith("umount2"))
    assert umount2_line.split()[1] == "-1", f"the unmount call reached the jail: {umount2_line}"
    assert umount2_line.split()[2] == "1", f"denied by something other than the filter: {out}"
    if pipe_nr:
        pipe_line = next(ln for ln in out.splitlines() if ln.startswith("pipe"))
        assert pipe_line.split()[1] == "0", f"the jail denies pipe(2): {pipe_line}"


def test_the_jail_launcher_does_not_carry_the_agent_env_into_the_jail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A jailed command cannot read the operator's provider key.

    The launcher is PID 1 of the jail's own PID namespace and strict mounts a fresh /proc,
    so /proc/1/environ is the launcher's environment; spawned with the agent's, a jailed
    command could read `OPENROUTER_API_KEY=...` straight out of it. docs/security.md says
    secrets never reach the jail: the launcher reads nothing from its environment (the policy
    arrives on stdin), so it gets none.
    """
    from agent6.kinds import JailPolicy
    from agent6.sandbox.jail import run_in_jail

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
    """Every jail invariant at once, against a policy with every field set.

    The tests around this one each exercise a default-shaped policy, which mounts neither
    tool_paths nor extra_ro_paths, so enumerating "every mount" there misses them. This one
    populates the whole surface (ro/rw/protect grants, tool paths, a child env, a memory
    cap) and asserts the properties together. Limit on the gaps half: a bind inherits its
    source mount's flags, and pytest's tmp_path is usually on a tmpfs already carrying
    nosuid,nodev, so on such a host this cannot tell an explicit floor from an inherited one;
    it bites where tmp is ext4. The leak and protect halves are deterministic everywhere.
    """
    from agent6.kinds import JailPolicy
    from agent6.sandbox.jail import run_in_jail

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
        "def env(p):\n"
        # Denied beats readable-and-empty: PID 1 is non-dumpable and the child
        # holds no capability, so its /proc entry cannot be opened at all.
        "    try: return open(p,'rb').read()\n"
        "    except OSError: return b''\n"
        "leak = [p for p in glob.glob('/proc/[0-9]*/environ')\n"
        "        if int(p.split('/')[2]) != me and b'PARENT-SECRET' in env(p)]\n"
        "print('LEAK', leak)\n"
        "try:\n"
        "    open('.git/config','w').write('x'); print('PROTECT writable')\n"
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


def test_the_jail_root_is_per_uid_and_named_in_the_refusal(tmp_path: Path) -> None:
    """The jail root carries the uid, and an unusable one names itself.

    A shared /tmp/agent6-jail-root is a cross-user denial of service: any local user can
    create it or plant a symlink, and every other user's jail then fails.
    """
    import os
    import re

    crate_main = Path(__file__).resolve().parents[2] / "src" / "agent6" / "jail" / "src" / "main.rs"
    src = crate_main.read_text(encoding="utf-8")
    assert '"/tmp/agent6-jail-root"' not in src, "the jail root must not be a shared path"
    assert re.search(r"agent6-jail-root-\{", src), "the jail root must carry the uid"

    from agent6.kinds import JailPolicy
    from agent6.sandbox.jail import run_in_jail

    res = run_in_jail(
        JailPolicy(
            cwd=tmp_path,
            argv=("sh", "-c", "pwd; ls /tmp | head -5"),
            isolation="strict",
            timeout_s=20.0,
        )
    )
    # The child lands in its cwd at that cwd's REAL path: every mount in the
    # jail is where it is outside, so nothing has to be translated.
    assert str(tmp_path) in res.stdout, res.stdout + res.stderr
    # The root the run actually created carries the CALLER's uid, not the 0 the
    # user namespace maps it to -- otherwise every user collides on -0 again.
    assert Path(f"/tmp/agent6-jail-root-{os.getuid()}").exists()


@pytest.mark.needs_namespaces
def test_launchers_starting_at_once_do_not_wipe_each_others_root(tmp_path: Path) -> None:
    """Two launchers starting together on one uid both come up, each in its own mount namespace.

    The jail root is shared per uid and setup clears nothing: each launcher mounts its own
    tmpfs over the shared mount point in its own namespace, so neither can remove the tree
    the other is building. Reachable from /parallel lanes (separate agent6 processes, one
    uid) and from any command run while an MCP server's launcher is alive. Asserted with
    real concurrency, and on the mount namespaces: a per-child namespace keeps one child's
    grants out of another's view.
    """
    import threading

    from agent6.kinds import JailPolicy
    from agent6.sandbox.jail import run_in_jail

    probe = (
        "import os, time\nprint(os.readlink('/proc/self/ns/mnt'), flush=True)\ntime.sleep(1.5)\n"
    )
    results: dict[int, str] = {}
    failures: dict[int, str] = {}

    def one(i: int) -> None:
        try:
            res = run_in_jail(
                JailPolicy(
                    cwd=tmp_path,
                    argv=("/usr/bin/python3", "-c", probe),
                    isolation="strict",
                    timeout_s=30.0,
                )
            )
            results[i] = res.stdout.strip() or f"rc={res.returncode} {res.stderr.strip()[:60]}"
        except Exception as exc:
            failures[i] = f"{type(exc).__name__}: {exc}"

    threads = [threading.Thread(target=one, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not failures, f"concurrent launchers collided: {failures}"
    namespaces = [v for v in results.values() if v.startswith("mnt:")]
    assert len(namespaces) == 4, results
    assert len(set(namespaces)) == 4, f"children shared a mount namespace: {namespaces}"


def test_dev_shm_is_the_jails_own_and_writable(tmp_path: Path) -> None:
    """/dev/shm is mounted and granted, as the jail's own tmpfs.

    POSIX shared memory is ordinary for real toolchains (a headless chromium aborts without
    it); a mount without a grant does nothing. What a command writes there is invisible to
    the host and gone when the jail exits.
    """
    from agent6.config import Config
    from agent6.sandbox.jail import run_in_jail
    from agent6.tools.policy import jail_policy

    marker = "agent6-shm-probe"
    res = run_in_jail(
        jail_policy(
            tmp_path,
            Config(),
            "strict",
            (
                "sh",
                "-c",
                f"stat -f -c %T /dev/shm && echo x > /dev/shm/{marker} && ls /dev/shm",
            ),
            network="none",
        )
    )
    assert res.returncode == 0, res.stderr[-400:]
    assert "tmpfs" in res.stdout, res.stdout
    assert marker in res.stdout, res.stdout
    assert not Path(f"/dev/shm/{marker}").exists(), "the jail wrote the HOST's /dev/shm"


@pytest.mark.parametrize("level", ["hardened", "strict"])
def test_pidfd_getfd_is_denied_and_pidfd_open_is_not(tmp_path: Path, level: str) -> None:
    """pidfd_getfd is EPERM at both levels; pidfd_open, the harmless handle, is not denied.

    pidfd_getfd steals an open fd out of another process's table, gated only by
    ptrace_may_access, the same check that gates the denied process_vm_readv/writev/kcmp;
    under `hardened` (no user namespace) that check plus the host's yama tunable is the only
    barrier to lifting a live fd out of the agent. Probed by number so a wrapper's own
    failure cannot read as a deny: 438 must be EPERM; 434 fails EINVAL/ESRCH on a bogus pid,
    never EPERM from the filter.
    """
    import platform

    from agent6.kinds import JailPolicy
    from agent6.sandbox.jail import run_in_jail

    if platform.machine() not in ("x86_64", "aarch64"):
        pytest.skip(f"pidfd syscall numbers not pinned for {platform.machine()}")
    probe = (
        "import ctypes\n"
        "libc = ctypes.CDLL(None, use_errno=True)\n"
        "def call(nr, *a):\n"
        "    ctypes.set_errno(0)\n"
        "    rc = libc.syscall(ctypes.c_long(nr), *[ctypes.c_long(x) for x in a])\n"
        "    return rc, ctypes.get_errno()\n"
        # pidfd_getfd(-1, -1, 0): a filtered call EPERMs before the kernel
        # validates the bogus args; unfiltered it would be EBADF.
        "print('getfd', *call(438, -1, -1, 0))\n"
        # pidfd_open(-1, 0): EINVAL for the bad pid if allowed, EPERM if filtered.
        "print('open', *call(434, -1, 0))\n"
    )
    res = run_in_jail(
        JailPolicy(cwd=tmp_path, argv=("python3", "-c", probe), isolation=level, timeout_s=20.0)  # pyright: ignore[reportArgumentType]
    )
    out = res.stdout or ""
    if "getfd" not in out:
        pytest.skip(f"probe did not run: {res.stderr[:200]}")
    getfd = next(ln for ln in out.splitlines() if ln.startswith("getfd"))
    assert getfd.split()[1] == "-1", f"pidfd_getfd reached the kernel: {getfd}"
    assert getfd.split()[2] == "1", f"pidfd_getfd denied by something other than seccomp: {getfd}"
    open_line = next(ln for ln in out.splitlines() if ln.startswith("open"))
    assert open_line.split()[2] != "1", (
        f"the jail denies pidfd_open, the harmless handle: {open_line}"
    )


@pytest.mark.parametrize("level", ["hardened", "strict"])
def test_io_uring_and_userfaultfd_are_denied(tmp_path: Path, level: str) -> None:
    """io_uring_setup and userfaultfd are EPERM at both levels; memfd_create stays allowed.

    io_uring's ops run in kernel worker threads seccomp never sees, so denying setup is the
    complete block (enter/register need a ring only setup creates); userfaultfd is the
    race-window primitive kernel UAF exploits lean on; memfd_create is ordinary anonymous
    memory real toolchains use. Probed by number so a wrapper's own failure cannot read as a
    deny.
    """
    import platform

    from agent6.kinds import JailPolicy
    from agent6.sandbox.jail import run_in_jail

    # (io_uring_setup, userfaultfd, memfd_create) per arch.
    by_arch = {"x86_64": (425, 323, 319), "aarch64": (425, 282, 279)}
    if platform.machine() not in by_arch:
        pytest.skip(f"syscall numbers not pinned for {platform.machine()}")
    iouring, uffd, memfd = by_arch[platform.machine()]
    probe = (
        "import ctypes\n"
        "libc = ctypes.CDLL(None, use_errno=True)\n"
        "def call(nr, *a):\n"
        "    ctypes.set_errno(0)\n"
        "    libc.syscall(ctypes.c_long(nr), *[ctypes.c_long(x) for x in a])\n"
        "    return ctypes.get_errno()\n"
        f"print('iouring', call({iouring}, 0, 0))\n"  # setup(0 entries, NULL params)
        f"print('uffd', call({uffd}, 0))\n"
        f"print('memfd', call({memfd}, 0, 0))\n"  # bad name ptr -> EFAULT if allowed
    )
    res = run_in_jail(
        JailPolicy(cwd=tmp_path, argv=("python3", "-c", probe), isolation=level, timeout_s=20.0)  # pyright: ignore[reportArgumentType]
    )
    out = res.stdout or ""
    if "iouring" not in out:
        pytest.skip(f"probe did not run: {res.stderr[:200]}")
    errs = dict(ln.split() for ln in out.splitlines() if ln and not ln.startswith("Traceback"))
    assert errs["iouring"] == "1", f"io_uring_setup reached the kernel (errno {errs['iouring']})"
    assert errs["uffd"] == "1", f"userfaultfd reached the kernel (errno {errs['uffd']})"
    assert errs["memfd"] != "1", f"the jail denies memfd_create (errno {errs['memfd']})"


def test_serve_launcher_refuses_a_request_with_an_unknown_field(tmp_path: Path) -> None:
    """The serve-mode ChildRequest refuses an unknown field, like Policy.

    A field this binary does not know is version skew with the Python side; dropping it
    silently could drop a confinement the caller meant to set.
    """
    import json
    import subprocess

    from agent6.config import Config
    from agent6.sandbox.jail import (
        _policy_spec,  # pyright: ignore[reportPrivateUsage]
        _require_jail_binary,  # pyright: ignore[reportPrivateUsage]
    )
    from agent6.tools.policy import jail_policy

    spec = _policy_spec(jail_policy(tmp_path, Config(), "strict", ("/bin/true",), network="none"))
    spec["mode"] = "serve"
    req = {"kind": "background", "argv": ["/bin/true"], "a_field_from_a_newer_agent6": 1}
    proc = subprocess.run(
        [str(_require_jail_binary())],
        input=(json.dumps(spec) + "\n" + json.dumps(req) + "\n").encode(),
        capture_output=True,
        timeout=30,
        check=False,
    )
    combined = (proc.stdout + proc.stderr).decode(errors="replace")
    assert "unknown field" in combined.lower(), combined[:300]
    assert "a_field_from_a_newer_agent6" in combined
