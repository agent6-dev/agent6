# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Python-side launcher for the `agent6-jail` Rust binary.

Serializes a JailPolicy to JSON on stdin and reads child stdout/stderr/return code
from the launcher's output. If the launcher is not available, falls back to a
plain (un-sandboxed) subprocess invocation only when the policy explicitly
opts in via `cwd-only-mode`, otherwise raises JailUnavailableError. This keeps
"silently weaker" failure modes out of the system.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import functools
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from agent6.types import CommandResult, JailPolicy

# --- operator tool reachability ----------------------------------------------
# The jail's baseline PATH is /usr/bin:/bin and it bind-mounts only the system
# roots below. Operator tools (uv, node, ruff, ...) installed elsewhere are
# otherwise unreachable, so a jailed command dies 127. We add the standard bin
# dirs that exist to PATH, and for those outside the system roots (or whose
# symlinks resolve out to one, a pipx `uv` at /usr/local/bin -> /opt/pipx/...)
# pass the real dirs as tool_paths for a real-location RO+exec mount. Read+exec
# only; the jail still confines writes and network, so containment is
# unchanged. Owned here so run_command and verify (tools.dispatch), machine
# tool states (machine.engine), and the host-side probe (`machine check`)
# resolve tools identically.
_JAIL_BASE_PATH_DIRS = ("/usr/bin", "/bin")
_SYSTEM_ROOTS = (
    Path("/usr"),
    Path("/bin"),
    Path("/sbin"),
    Path("/lib"),
    Path("/lib64"),
    Path("/etc"),
    Path("/dev"),
)


def _under_system_root(p: Path) -> bool:
    return any(p.is_relative_to(r) for r in _SYSTEM_ROOTS)


def operator_tool_paths() -> tuple[str, tuple[Path, ...]]:
    """Return (PATH string, real-location mount dirs) so operator-installed tools
    resolve in the jail. Recomputed per call so a tool the operator (or model)
    just installed is picked up (dirs under a mounted system root only join PATH;
    dirs outside it, and the real dirs symlinks resolve out to, also need the
    RO+exec mount)."""
    home = Path.home()
    candidates = (
        Path("/usr/local/bin"),
        Path("/usr/local/sbin"),
        home / ".local/bin",
        home / ".cargo/bin",
        Path("/opt/homebrew/bin"),
        Path("/snap/bin"),
    )
    path_dirs: list[str] = list(_JAIL_BASE_PATH_DIRS)
    mounts: set[Path] = set()
    for d in candidates:
        if not d.is_dir():
            continue
        path_dirs.append(str(d))
        if not _under_system_root(d):
            mounts.add(d)  # real binaries in a non-system dir need the dir itself
        try:
            entries = list(d.iterdir())
        except OSError:
            continue
        for entry in entries:
            if not entry.is_symlink():
                continue  # real files are covered by the dir / the /usr mount
            try:
                real = entry.resolve()
            except OSError:
                continue
            if real.is_file() and not _under_system_root(real):
                mounts.add(real.parent)  # e.g. /opt/pipx/venvs/uv/bin
    # Interpreter toolchains a repo venv's python may symlink to: uv-managed
    # CPython lives under XDG data, not any bin dir. Without this mount the
    # jail sees such a venv "linked to a non-existent interpreter" and an
    # in-jail `uv run` deletes and recreates the operator's .venv.
    # Mount-only, never a PATH entry.
    data_home = Path(os.environ.get("XDG_DATA_HOME") or home / ".local/share")
    uv_pythons = data_home / "uv" / "python"
    if uv_pythons.is_dir():
        mounts.add(uv_pythons)
    return ":".join(path_dirs), tuple(sorted(mounts))


def jail_search_path() -> str:
    """The PATH a jailed command resolves against, for host-side reachability
    probes (`machine check`): the jail baseline plus the standard bin dirs that
    exist right now. Advisory only; the jail recomputes its own per call."""
    return operator_tool_paths()[0]


class JailUnavailableError(Exception):
    """`agent6-jail` could not be located or refused to set up the namespace."""


def _lossy_text(v: object) -> str:
    """Decode child/launcher output for surfaces: one decode policy for this
    module. Command output is not guaranteed UTF-8 (grep over a binary, a
    latin-1 file), so bytes decode with errors="replace" -- a lossy result
    beats a crash or a dropped stream. str passes through; anything else
    (None from a drained pipe) is ""."""
    if isinstance(v, bytes):
        return v.decode(errors="replace")
    return v if isinstance(v, str) else ""


# Override for tests; checked first.
_ENV_VAR = "AGENT6_JAIL_BIN"


def locate_jail_binary() -> Path | None:
    """The launcher binary: an explicit override, else the one the build hook
    bundled into the installed package, else one on PATH.

    No source-tree fallback. The build hook compiles the crate into
    ``sandbox/_bin/`` on every install, editable ones included, so a checkout
    with cargo already has it there: rebuild and reinstall to pick a change up,
    or point ``AGENT6_JAIL_BIN`` at a ``cargo build`` output while iterating on
    the crate itself.
    """
    override = os.environ.get(_ENV_VAR)
    if override:
        p = Path(override)
        return p if p.is_file() else None
    # Bundled inside the installed package (the wheel ships the binary
    # under agent6/sandbox/_bin/agent6-jail; see hatch_build.py).
    bundled = Path(__file__).resolve().parent / "_bin" / "agent6-jail"
    if bundled.is_file():
        return bundled
    # Look in PATH
    found = shutil.which("agent6-jail")
    return Path(found) if found else None


def _policy_to_json(policy: JailPolicy) -> str:
    return json.dumps(
        {
            "isolation": policy.isolation,
            "cwd": str(policy.cwd),
            "argv": list(policy.argv),
            "env": [list(pair) for pair in policy.env],
            "allow_network": policy.allow_network,
            "extra_ro_paths": [str(p) for p in policy.extra_ro_paths],
            "extra_rw_paths": [str(p) for p in policy.extra_rw_paths],
            "extra_protect_paths": [str(p) for p in policy.extra_protect_paths],
            "tool_paths": [str(p) for p in policy.tool_paths],
            "timeout_s": policy.timeout_s,
            "memory_limit_mb": policy.memory_limit_mb,
        }
    )


def _run_unsandboxed(policy: JailPolicy) -> CommandResult:
    """Run `policy.argv` as a plain subprocess (no confinement).

    Used only for the `none` isolation on non-Linux hosts. Inherits the parent
    environment (so `PATH` etc. resolve normally) overlaid with `policy.env`;
    runs in `policy.cwd`. The sandbox-only knobs (network, ro/rw/protect paths,
    memory_limit_mb) have no effect here, there is no kernel mechanism to
    enforce them.
    """
    env = {**os.environ, **{k: v for k, v in policy.env}}
    start = time.monotonic()
    # Unsandboxed escape hatch (non-Linux only); see run_in_jail docstring.
    # Output is captured as bytes and decoded lossily: the strict text=True
    # decode raised UnicodeDecodeError out of communicate() on any non-UTF-8
    # byte, breaking the return-a-result contract.
    try:
        proc = subprocess.run(
            list(policy.argv),
            cwd=str(policy.cwd),
            env=env,
            capture_output=True,
            check=False,
            timeout=policy.timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        # Match the jailed contract: a timeout is an rc=124 result,
        # not a raised exception the caller would have to special-case.
        return CommandResult(
            argv=tuple(policy.argv),
            returncode=124,
            stdout=_lossy_text(exc.stdout),
            stderr=_lossy_text(exc.stderr),
            duration_s=time.monotonic() - start,
        )
    duration = time.monotonic() - start
    return CommandResult(
        argv=tuple(policy.argv),
        returncode=int(proc.returncode),
        stdout=_lossy_text(proc.stdout),
        stderr=_lossy_text(proc.stderr),
        duration_s=duration,
    )


@functools.lru_cache(maxsize=1)
def strict_namespaces_work() -> bool:
    """Return True iff the jail binary can actually set up a `strict` namespace.

    The cheap ``unshare -U -r true`` probe in ``detect.probe_userns_supported``
    under-reports on an AppArmor-restricted host (Ubuntu 24.04+ with
    ``kernel.apparmor_restrict_unprivileged_userns=1``) where a isolation grants
    the *agent6-jail* binary userns but not ``/usr/bin/unshare``. This runs the
    real jail binary with a trivial `strict` policy to get the authoritative
    answer. Cached for the process lifetime; the kernel/isolation state does not
    change mid-run. Returns False if the jail binary is missing.
    """
    if not Path("/usr/bin/true").exists():
        return False
    probe_cwd = Path(tempfile.gettempdir())
    try:
        res = run_in_jail(
            JailPolicy(
                cwd=probe_cwd,
                argv=("/usr/bin/true",),
                isolation="strict",
                allow_network=False,
                timeout_s=10.0,
            )
        )
    except JailUnavailableError:
        return False
    return res.returncode == 0


# --- escapee reaping ---------------------------------------------------------
# `strict` confines the child in a PID namespace, so nothing outlives it. In
# `hardened` there is no namespace: a child that calls setsid() leaves the
# launcher's process group, survives the launcher's killpg, and reparents to
# init. PR_SET_CHILD_SUBREAPER makes it reparent to the agent instead, so
# run_in_jail can find and kill it once the command returns.
#
# Security review note: this closes a break of the "no persistence after the
# run" guarantee (docs/security.md) on `hardened`. The launcher cannot do it
# itself, and must not: its own Landlock ruleset denies /proc, and granting
# /proc there would hand every jailed child the agent's environ (API keys).
#
# The sweep kills every child that appears while a command is in flight, which
# is sound because nothing else spawns during one: a process that runs jailed
# commands does so from one thread, and the egress broker, host spawner and
# `/parallel` lanes all predate the call, so they are in `before`. Deliberate
# children must keep predating the calls that follow them.
_PR_SET_CHILD_SUBREAPER = 36
_SWEEP_ROUNDS = 5
_sweep_lock = threading.Lock()
_live_launchers: set[int] = set()


@functools.cache
def _become_subreaper() -> None:
    if sys.platform != "linux":
        return
    libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
    libc.prctl.restype = ctypes.c_int
    if libc.prctl(_PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
        err = ctypes.get_errno()
        raise JailUnavailableError(
            f"prctl(PR_SET_CHILD_SUBREAPER) failed: {os.strerror(err)}."
            " Without it a sandboxed command could leave a process running after it returns."
        )


def _own_children() -> frozenset[int]:
    """Pids the kernel currently reports as children of this process."""
    me = str(os.getpid())
    found: set[int] = set()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
        except OSError:
            continue  # exited mid-scan
        # Field 4 is the ppid; the comm field before it may itself contain spaces.
        fields = stat[stat.rfind(")") + 1 :].split()
        if len(fields) > 1 and fields[1] == me:
            found.add(int(entry.name))
    return frozenset(found)


def _kill_escapees(before: frozenset[int]) -> None:
    """Kill anything the command left behind that has reparented to us."""
    own_group = os.getpgrp()
    with _sweep_lock:
        # Killing one layer orphans the next, which lands here in turn; a
        # daemon that re-daemonises as its parent dies needs another round.
        for _ in range(_SWEEP_ROUNDS):
            escapees = _own_children() - before - _live_launchers
            if not escapees:
                return
            for pid in escapees:
                try:
                    group = os.getpgid(pid)
                    # setsid() left it leading its own group: take the group so
                    # its children go too. Never our own group.
                    if group != own_group:
                        os.killpg(group, signal.SIGKILL)
                    else:
                        os.kill(pid, signal.SIGKILL)
                    os.waitpid(pid, 0)
                except OSError:
                    continue


def run_in_jail(policy: JailPolicy) -> CommandResult:
    """Run `policy.argv` inside the sandbox.

    Raises JailUnavailableError if the launcher binary is missing or setup fails.

    The `none` isolation is the unsandboxed path: the command runs as a plain
    subprocess with no kernel confinement. `auto` selects it only on non-Linux
    hosts; an explicit `isolation = "none"`, `--dangerously-disable-sandbox`, or
    `AGENT6_DANGEROUSLY_DISABLE_SANDBOX=1` selects it on any host. The CLI prints a
    prominent warning before any such run.

    Security review note: this is the single place where an
    LLM-influenced argv runs without the jail. It exists solely so agent6 is
    usable on platforms (macOS) where the Landlock/seccomp/namespace sandbox
    does not exist. Both real isolation levels still go through the Rust
    launcher; nothing here weakens the Linux boundary.
    """
    if policy.isolation == "none":
        return _run_unsandboxed(policy)
    binary = locate_jail_binary()
    if binary is None:
        raise JailUnavailableError(
            "agent6-jail binary not found. Install agent6 from a built wheel"
            " (which bundles the binary), or build from source with"
            " `cargo build --release --locked --manifest-path src/agent6/jail/Cargo.toml`,"
            f" or set {_ENV_VAR}=/path/to/agent6-jail."
        )
    spec = _policy_to_json(policy)
    start = time.monotonic()
    _become_subreaper()
    # Snapshot first: anything that is our child afterwards but was not before
    # escaped the command. A concurrent caller's launcher is excluded by pid.
    before = _own_children()
    with _sweep_lock:
        # Launch the launcher in its own session (group leader) so that, if it ever
        # hangs — e.g. a backgrounded grandchild holds the stdout pipe open past the
        # timeout — we can kill its whole process group and reap any orphaned
        # pidns-init/grandchild, not just the launcher itself. Use Popen (not
        # subprocess.run) so we keep the pid to target os.killpg.
        launcher = subprocess.Popen(
            [str(binary)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        _live_launchers.add(launcher.pid)
    try:
        return _launcher_result(launcher, policy, spec, start, binary)
    finally:
        with _sweep_lock:
            _live_launchers.discard(launcher.pid)
        _kill_escapees(before)


def _launcher_result(
    launcher: subprocess.Popen[bytes],
    policy: JailPolicy,
    spec: str,
    start: float,
    binary: Path,
) -> CommandResult:
    try:
        raw_out, raw_err = launcher.communicate(input=spec.encode(), timeout=policy.timeout_s + 5.0)
    except subprocess.TimeoutExpired as exc:
        # Kill the whole group, then drain whatever output was produced. Mirror
        # _run_unsandboxed: surface a timeout as the documented rc=124 result, not
        # a raised exception the caller would have to special-case.
        try:
            os.killpg(os.getpgid(launcher.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            launcher.kill()
        try:
            raw_out, raw_err = launcher.communicate(timeout=5.0)
        except subprocess.TimeoutExpired:
            raw_out, raw_err = b"", b""
        return CommandResult(
            argv=tuple(policy.argv),
            returncode=124,
            stdout=_lossy_text(raw_out) or _lossy_text(exc.stdout),
            stderr=_lossy_text(raw_err) or _lossy_text(exc.stderr),
            duration_s=time.monotonic() - start,
        )
    proc = subprocess.CompletedProcess(
        args=[str(binary)],
        returncode=launcher.returncode,
        stdout=_lossy_text(raw_out),
        stderr=_lossy_text(raw_err),
    )
    duration = time.monotonic() - start
    # The launcher prints a single JSON line on stdout describing the child's result,
    # then exits 0 itself. Anything else means setup failed, with one exception:
    # a child that could not be EXECUTED at all (bad path, missing interpreter)
    # also surfaces as a launcher error, but the jail itself worked fine. Report
    # that as an ordinary failed command (shell-style 127) so the model fixes
    # its argv instead of concluding the sandbox is broken.
    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        if "child execution failed" in stderr:
            return CommandResult(
                argv=tuple(policy.argv),
                returncode=127,
                stdout="",
                stderr=f"{policy.argv[0]}: command not found or not executable ({stderr})",
                duration_s=duration,
                exec_failed=True,
            )
        raise JailUnavailableError(f"agent6-jail launcher exited {proc.returncode}: {stderr}")
    try:
        result_json = json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError) as exc:
        raise JailUnavailableError(
            f"agent6-jail produced unparseable output: {proc.stdout!r}"
        ) from exc
    return CommandResult(
        argv=tuple(policy.argv),
        returncode=int(result_json["returncode"]),
        stdout=str(result_json.get("stdout", "")),
        stderr=str(result_json.get("stderr", "")),
        duration_s=duration,
    )
