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

import contextlib
import ctypes
import functools
import json
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from agent6.paths import cache_dir, data_dir, global_config_dir, state_base
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


def _is_agent6_private(p: Path) -> bool:
    """agent6's own config/state/data/cache dirs, which never belong in a jail
    mount however a tool symlink resolves into them.

    ``operator_tool_paths`` mounts ``real.parent`` for every symlink in a bin
    dir, so one resolving into the config dir mounted ``secrets.toml`` -- the
    provider API keys -- read-only into the jail, and one into the state dir
    mounted notes, memories and transcripts. Denied by identity rather than by
    inspecting what the file is. Read per call: the XDG vars are per-process.
    """
    private = (global_config_dir(), state_base(), data_dir(), cache_dir())
    return any(p == d or p.is_relative_to(d) for d in private)


# The launcher's OWN environment. It becomes PID 1 of the jail's PID namespace
# and strict mounts a fresh /proc, so /proc/1/environ is readable by the jailed
# command -- inheriting the agent's env put the operator's provider key there.
# The launcher reads nothing from the environment (its policy arrives on stdin
# and the child's env is passed explicitly in it), so it gets none.
_LAUNCHER_ENV: dict[str, str] = {}


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
        if not _under_system_root(d) and not _is_agent6_private(d):
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
            if (
                real.is_file()
                and not _under_system_root(real)
                and not _is_agent6_private(real.parent)
            ):
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
    """`agent6-jail` could not be located, refused to set up the namespace, or
    could not guarantee the command left nothing running."""


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
    ``kernel.apparmor_restrict_unprivileged_userns=1``) where an AppArmor profile grants
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


def _require_jail_binary() -> Path:
    binary = locate_jail_binary()
    if binary is None:
        raise JailUnavailableError(
            "agent6-jail binary not found. Install agent6 from a built wheel"
            " (which bundles the binary), or build from source with"
            " `cargo build --release --locked --manifest-path src/agent6/jail/Cargo.toml`,"
            f" or set {_ENV_VAR}=/path/to/agent6-jail."
        )
    return binary


# --- escapee reaping ---------------------------------------------------------
# `strict` confines the child in a PID namespace, so nothing outlives it.
# `hardened` has none: a child that calls setsid() leaves the launcher's process
# group, survives the launcher's killpg, and reparents to init. The agent makes
# itself a subreaper so escapees land on it instead, and kills them once the
# command returns.
#
# The launcher cannot do this itself, and must not: its own Landlock ruleset
# denies /proc, and granting it there would hand every jailed child the agent's
# environ.
#
# A process is the command's only if it appeared during the call AND sits
# outside the agent's session. The launcher runs in its own session, so every
# jailed descendant is outside ours (setsid creates sessions, setpgid cannot
# cross one), while a deliberate same-session child -- git, notify-send -- can
# never be swept, whatever thread spawns it.
_PR_SET_CHILD_SUBREAPER = 36
_SWEEP_DEADLINE_S = 5.0
_sweep_lock = threading.Lock()
_live_launchers: set[int] = set()
# Children agent6 started ON PURPOSE in their own session: a `/btw` ask, a
# `/parallel` lane. They look exactly like an escapee -- our child, different
# session -- so without this the first background command's teardown SIGKILLs
# them, destroying model work the operator has already paid for. Every detached
# spawn from this process must register here; `agent6.ui.spawn` is the one
# place that does it.
_own_detached: set[int] = set()


@functools.cache
def _become_subreaper() -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.restype = ctypes.c_int
    if libc.prctl(_PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
        err = ctypes.get_errno()
        raise JailUnavailableError(
            f"prctl(PR_SET_CHILD_SUBREAPER) failed: {os.strerror(err)}."
            " Without it a sandboxed command could leave a process running after it returns."
        )


def keep_out_of_the_sweep(pid: int) -> None:
    """Mark *pid* as a child agent6 started deliberately, not an escapee.

    Called right after a detached spawn. Never cleared: a pid this process
    started stays ours for its lifetime, and the set is bounded by how many
    sessions one run opens.
    """
    _own_detached.add(pid)


def _own_children() -> dict[int, int]:
    """``{pid: session id}`` for this process's children, right now.

    Read as bytes: comm is whatever a process named itself, so the line need
    not be valid UTF-8 and one hostile name must not break the scan.
    """
    me = str(os.getpid()).encode()
    found: dict[int, int] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_bytes()
        except OSError:
            continue  # exited mid-scan
        # comm can hold spaces and parens, so fields are taken after its closing
        # one: state, ppid, pgrp, session.
        fields = stat[stat.rfind(b")") + 1 :].split()
        if len(fields) > 3 and fields[1] == me:
            found[int(entry.name)] = int(fields[3])
    return found


def _kill_group_of(pid: int) -> None:
    """SIGKILL *pid*'s process group, or *pid* alone when it does not lead one.

    A pgid is a leader's pid, and it is only reusable once that leader is
    reaped. We hold every one of these as an unreaped child, so a pgid equal to
    the pid we looked up cannot have been recycled underneath us. A pgid that
    is NOT the pid belongs to a leader we do not hold, and under sudo signalling
    a recycled one would kill an unrelated group as root.
    """
    with contextlib.suppress(OSError):
        if os.getpgid(pid) == pid:
            os.killpg(pid, signal.SIGKILL)
        else:
            os.kill(pid, signal.SIGKILL)


def _kill_escapees(exclude: frozenset[int]) -> frozenset[int]:
    """Kill what the command left behind. Returns whatever is still alive."""
    our_session = os.getsid(0)
    deadline = time.monotonic() + _SWEEP_DEADLINE_S
    with _sweep_lock:
        while True:
            escapees = {
                pid
                for pid, session in _own_children().items()
                if session != our_session
                and pid not in exclude
                and pid not in _live_launchers
                and pid not in _own_detached
            }
            if not escapees:
                return frozenset()
            for pid in escapees:
                _kill_group_of(pid)
                with contextlib.suppress(OSError):
                    # WNOHANG: a child wedged in uninterruptible sleep must not
                    # hang every later command behind the sweep lock.
                    os.waitpid(pid, os.WNOHANG)
            if time.monotonic() >= deadline:
                return frozenset(escapees)
            time.sleep(0.01)  # killing one layer orphans the next onto us


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
    binary = _require_jail_binary()
    spec = _policy_to_json(policy)
    start = time.monotonic()
    _become_subreaper()
    # Snapshot first: anything that is our child afterwards but was not before
    # escaped the command. A concurrent caller's launcher is excluded by pid.
    before = frozenset(_own_children())
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
            env=_LAUNCHER_ENV,
        )
        _live_launchers.add(launcher.pid)
    survivors: frozenset[int] = frozenset()
    try:
        result = _launcher_result(launcher, policy, spec, start, binary)
    finally:
        if launcher.poll() is None:
            # Abandoned mid-command (an interrupt raised through communicate()):
            # take the launcher's group down so the jailed tree goes with it.
            # start_new_session made it its own group leader, and Popen holds
            # it unreaped, so its pgid is its pid and cannot have been recycled.
            try:
                os.killpg(launcher.pid, signal.SIGKILL)
            except OSError:
                launcher.kill()
            with contextlib.suppress(subprocess.TimeoutExpired):
                launcher.communicate(timeout=5.0)
        with _sweep_lock:
            _live_launchers.discard(launcher.pid)
        survivors = _kill_escapees(before | {launcher.pid})
    if survivors:
        raise JailUnavailableError(
            f"could not kill everything the command left running (pids {sorted(survivors)});"
            " a process would have outlived this run."
        )
    return result


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
            os.killpg(launcher.pid, signal.SIGKILL)
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
    return _result_from_json(result_json, tuple(policy.argv), duration)


def _result_from_json(
    result_json: dict[str, object], argv: tuple[str, ...], duration: float
) -> CommandResult:
    """The launcher's result object as a CommandResult."""
    return CommandResult(
        argv=argv,
        returncode=int(str(result_json["returncode"])),
        stdout=str(result_json.get("stdout", "")),
        stderr=str(result_json.get("stderr", "")),
        duration_s=duration,
    )


def _result_from_line(line: str, argv: tuple[str, ...], start: float) -> CommandResult:
    """One serve-mode reply line as a CommandResult."""
    try:
        parsed = json.loads(line)
    except ValueError as exc:
        raise JailUnavailableError(f"jail session produced unparseable output: {line!r}") from exc
    return _result_from_json(parsed, argv, time.monotonic() - start)


# --- detached commands -------------------------------------------------------
# A background command keeps running after the call that started it, so it is
# the one jailed child the escapee sweep must NOT kill: its launcher stays
# registered live until `stop`. Its own output is not captured here (the caller
# redirects it in argv); only the launcher's result JSON is, so the exit code
# survives the turn that started the command.
_RESULT_NAME = "result.json"
_LAUNCHER_ERR_NAME = "launcher.err"


@dataclass(frozen=True, slots=True)
class BackgroundStatus:
    """What a detached command is doing, right now.

    ``running`` is the live process, never an inference from output or age. A
    launcher that exited without a result reports ``error``: a command whose
    fate is unknown is never reported as still running.
    """

    running: bool
    returncode: int | None
    error: str


class BackgroundJob:
    """A jailed command detached from the call that started it."""

    def __init__(self, proc: subprocess.Popen[bytes], outcome_dir: Path | None) -> None:
        self._proc = proc
        self._outcome_dir = outcome_dir
        # Everything that was already ours when this command started: its own
        # escapees are whatever appears beyond this set.
        self._descendants = frozenset(_own_children())

    @property
    def pid(self) -> int:
        return self._proc.pid

    def status(self) -> BackgroundStatus:
        if self._proc.poll() is None:
            return BackgroundStatus(running=True, returncode=None, error="")
        if self._outcome_dir is None:  # unsandboxed: the child IS the process
            return BackgroundStatus(running=False, returncode=self._proc.returncode, error="")
        self._unregister()
        raw = ""
        with contextlib.suppress(OSError):
            raw = (self._outcome_dir / _RESULT_NAME).read_text(errors="replace")
        with contextlib.suppress(ValueError, IndexError, KeyError):
            return BackgroundStatus(
                running=False,
                returncode=int(json.loads(raw.strip().splitlines()[-1])["returncode"]),
                error="",
            )
        err = ""
        with contextlib.suppress(OSError):
            err = (self._outcome_dir / _LAUNCHER_ERR_NAME).read_text(errors="replace").strip()
        return BackgroundStatus(
            running=False,
            returncode=None,
            error=err or f"the sandbox launcher exited {self._proc.returncode} without a result",
        )

    def stop(self) -> None:
        """Kill the command and everything it started. Idempotent.

        killpg only reaches the launcher's group, so a child that called
        setsid() is missed exactly as it is for a foreground command -- and
        `run_in_jail`'s sweep can never catch it either, because by then it is
        not NEW. Unregistering first, then sweeping, makes this the moment its
        escapees stop being spared.
        """
        if self._proc.poll() is None:
            with contextlib.suppress(OSError):
                os.killpg(self._proc.pid, signal.SIGKILL)
            with contextlib.suppress(subprocess.TimeoutExpired):
                self._proc.wait(timeout=5.0)
        self._unregister()
        _kill_escapees(self._descendants)

    def _unregister(self) -> None:
        with _sweep_lock:
            _live_launchers.discard(self._proc.pid)


def start_in_jail(policy: JailPolicy, *, outcome_dir: Path) -> BackgroundJob:
    """Spawn `policy.argv` in the sandbox and return WITHOUT waiting for it.

    The caller owns the command's own output: nothing is captured here, so
    `policy.argv` must redirect it somewhere both sides can read. Only the
    launcher's result JSON and stderr land in *outcome_dir*, which is what lets
    the exit code outlive the turn that started the command.

    Security review note: same policy, same launcher, same confinement as
    `run_in_jail` -- the only difference is that this call does not wait. The
    escapee sweep spares it while it lives (it is a deliberate child, not
    something a command left behind) and `stop` takes its whole group down.
    """
    outcome_dir.mkdir(parents=True, exist_ok=True)
    if policy.isolation == "none":
        # Unsandboxed escape hatch (non-Linux only); see run_in_jail's note.
        proc = subprocess.Popen(
            list(policy.argv),
            cwd=str(policy.cwd),
            env={**os.environ, **dict(policy.env)},
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return BackgroundJob(proc, None)
    binary = _require_jail_binary()
    spec = _policy_to_json(policy)
    _become_subreaper()
    result = (outcome_dir / _RESULT_NAME).open("wb")
    errors = (outcome_dir / _LAUNCHER_ERR_NAME).open("wb")
    try:
        with _sweep_lock:
            launcher = subprocess.Popen(
                [str(binary)],
                stdin=subprocess.PIPE,
                stdout=result,
                stderr=errors,
                start_new_session=True,
                env=_LAUNCHER_ENV,
            )
            _live_launchers.add(launcher.pid)
    finally:
        result.close()
        errors.close()
    assert launcher.stdin is not None
    with contextlib.suppress(OSError):
        launcher.stdin.write(spec.encode())
    launcher.stdin.close()
    return BackgroundJob(launcher, outcome_dir)


@dataclass(slots=True)
class JailSession:
    """One long-lived launcher, serving every command of one run.

    The launcher establishes its namespaces, rootfs, Landlock and seccomp once
    and then reads one request per line, so the run's commands share a netns, a
    PID namespace and a /tmp: a server one command starts is reachable by the
    next, which per-command launchers cannot offer. Closing the session shuts
    stdin, and the PID namespace takes everything inside it down.

    Not thread-safe: one loop drives it, one command at a time.
    """

    _proc: subprocess.Popen[bytes]
    _binary: Path

    @classmethod
    def open(cls, policy: JailPolicy) -> JailSession:
        """Start a serving launcher confined by *policy* (its argv is ignored;
        each command arrives as a request)."""
        binary = _require_jail_binary()
        _become_subreaper()
        with _sweep_lock:
            proc = subprocess.Popen(
                [str(binary)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
                env=_LAUNCHER_ENV,
            )
            _live_launchers.add(proc.pid)
        spec = json.loads(_policy_to_json(policy))
        spec["mode"] = "serve"
        assert proc.stdin is not None
        proc.stdin.write((json.dumps(spec) + "\n").encode())
        proc.stdin.flush()
        return cls(_proc=proc, _binary=binary)

    def run(
        self,
        argv: tuple[str, ...],
        *,
        env: tuple[tuple[str, str], ...] = (),
        timeout_s: float = 600.0,
        background: bool = False,
    ) -> CommandResult:
        """Run one command in this session's namespaces.

        ``background`` answers as soon as the command starts and leaves it
        running for later commands to reach. Strict only: the PID namespace is
        what keeps it from outliving the run, and hardened has none.
        """
        assert self._proc.stdin is not None and self._proc.stdout is not None
        start = time.monotonic()
        request = {
            "argv": list(argv),
            "env": [list(p) for p in env],
            "timeout_s": timeout_s,
            "background": background,
        }
        self._proc.stdin.write((json.dumps(request) + "\n").encode())
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        if not line:
            raise JailUnavailableError("jail session ended before answering")
        return _result_from_line(line.decode(errors="replace"), argv, start)

    def close(self) -> None:
        """Shut the request channel; the PID namespace takes the rest down."""
        with contextlib.suppress(OSError):
            if self._proc.stdin is not None:
                self._proc.stdin.close()
        with contextlib.suppress(subprocess.TimeoutExpired):
            self._proc.communicate(timeout=10.0)
        if self._proc.poll() is None:
            with contextlib.suppress(OSError):
                os.killpg(self._proc.pid, signal.SIGKILL)
        with _sweep_lock:
            _live_launchers.discard(self._proc.pid)
