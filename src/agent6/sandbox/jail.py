# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Launch the `agent6-jail` binary and read the child's result back.

A `JailPolicy` goes to the launcher as JSON; the child's stdout, stderr and exit
code come back in its output. A missing launcher raises `JailUnavailableError`;
only a policy whose isolation is `none` runs as a plain subprocess.
"""

from __future__ import annotations

import contextlib
import ctypes
import dataclasses
import errno
import functools
import json
import os
import pathlib
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Sequence
from typing import IO, Any, NoReturn, cast

from agent6 import child_env, kinds, paths

# Loaded at import, never between fork and exec: a post-fork dlopen can deadlock on malloc.
_LIBC: ctypes.CDLL | None = ctypes.CDLL(None, use_errno=True) if sys.platform == "linux" else None


def die_with_parent(parent_pid: int, sig: int = signal.SIGTERM) -> Callable[[], None]:
    """Return a `preexec_fn` tying the child's life to its parent through PDEATHSIG.

    The kernel delivers the signal on any parent death, SIGKILL included; the
    re-check closes the window where the parent died before the prctl landed. A
    no-op off Linux. Launcher spawns tie with SIGKILL, machine children with SIGTERM.

    Args:
        parent_pid: The parent whose death ends the child.
        sig: The signal the kernel delivers.

    Returns:
        The function to run in the child before exec.
    """

    def _setup() -> None:
        if _LIBC is None:
            return
        _LIBC.prctl(1, sig)  # PR_SET_PDEATHSIG
        if os.getppid() != parent_pid:
            os._exit(128 + sig)

    return _setup


# The answer wait's poll interval: an operator Stop reads as immediate at this cost.
_ANSWER_POLL_S = 0.2

# The launcher is PID 1 of the jail and strict mounts a fresh /proc, so /proc/1/environ is
# readable by the jailed command: an inherited env would expose the operator's provider key.
# The launcher reads nothing from its environment; the policy arrives on stdin.
_LAUNCHER_ENV: dict[str, str] = {}


class JailUnavailableError(Exception):
    """The launcher is missing, refused to set up, or left something running."""


class JailBinaryError(JailUnavailableError):
    """The launcher binary is missing or the kernel refuses to execute it.

    Says nothing about the host's namespaces.
    """


def _lossy_text(v: object) -> str:
    """Return child or launcher output as text, decoding bytes lossily.

    Command output is not guaranteed UTF-8; a lossy result beats a dropped stream.

    Args:
        v: Bytes, a str, or None from a drained pipe.

    Returns:
        The text; "" for anything but bytes or str.
    """
    if isinstance(v, bytes):
        return v.decode(errors="replace")
    return v if isinstance(v, str) else ""


# The operator's override, checked before the bundled binary.
_ENV_VAR = "AGENT6_JAIL_BIN"


def locate_jail_binary() -> pathlib.Path | None:
    """Return the launcher binary: the override, else the bundled one, else PATH.

    No source-tree fallback: the build hook compiles the crate into `sandbox/_bin/`
    on every install, editable ones included. Point `AGENT6_JAIL_BIN` at a
    `cargo build` output while iterating on the crate.

    Returns:
        The binary's path, or None when none is found.
    """
    override = os.environ.get(_ENV_VAR)
    if override:
        p = pathlib.Path(override)
        return p if p.is_file() else None
    bundled = pathlib.Path(__file__).resolve().parent / "_bin" / "agent6-jail"
    if bundled.is_file():
        return bundled
    found = shutil.which("agent6-jail")
    return pathlib.Path(found) if found else None


# The holder unshares two namespaces and brings up loopback; slower is a launcher without the flag.
_HOLDER_READY_TIMEOUT_S = 10.0


def _read_available(pipe: IO[bytes] | None, budget_s: float = 0.5) -> bytes:
    """Return whatever the pipe already holds, never waiting for EOF.

    Reading a killed launcher's stderr to EOF hangs while a child holds the write end.

    Args:
        pipe: The pipe to drain, or None.
        budget_s: How long to keep reading.

    Returns:
        The bytes read.
    """
    if pipe is None:
        return b""
    chunks: list[bytes] = []
    deadline = time.monotonic() + budget_s
    while time.monotonic() < deadline:
        readable, _, _ = select.select([pipe], [], [], 0.05)
        if not readable:
            break
        chunk = os.read(pipe.fileno(), 4096)
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks)


@dataclasses.dataclass(frozen=True, slots=True)
class SessionNetwork:
    """The run's session network, held open by two file descriptors.

    One per run, created before anything that might join it. Every jailed child
    whose policy says `network = "session"` joins these, so the run's commands and
    its private MCP servers share one loopback with no route off the box. The user
    namespace travels with the network one because entering a netns needs
    CAP_SYS_ADMIN in the namespace that owns it (the launcher's `join_network`).

    Attributes:
        userns_fd: The user namespace, open; the descriptor keeps it alive.
        netns_fd: The network namespace, open.
        holder_pid: The holder process, alive for the run: `/proc/<pid>/ns/*` is the
            only way `agent6 exec` and `agent6 forward` can name the namespaces.
    """

    userns_fd: int
    netns_fd: int
    holder_pid: int
    _holder: subprocess.Popen[bytes] | None = None

    @classmethod
    def open(cls) -> SessionNetwork:
        """Start the holder and open its namespaces.

        Returns:
            The network, with the holder waiting on its stdin.

        Raises:
            JailUnavailableError: The holder never reported ready or its namespaces
                could not be opened.
            JailBinaryError: The kernel refused to execute the launcher.
        """
        binary = _require_jail_binary()
        # Not `_spawn_launcher`: nothing sweeps the holder, and `close` ends it through stdin.
        try:
            proc = subprocess.Popen(
                [str(binary), "--hold-netns"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=_LAUNCHER_ENV,
                preexec_fn=die_with_parent(os.getpid(), sig=signal.SIGKILL),  # noqa: PLW1509
            )
        except OSError as exc:
            _raise_for_exec_failure(binary, exc)
        fds: list[int] = []
        try:
            assert proc.stdout is not None
            # A launcher without --hold-netns reads a policy from stdin and blocks forever.
            ready, _, _ = select.select([proc.stdout], [], [], _HOLDER_READY_TIMEOUT_S)
            if not ready or proc.stdout.readline().strip() != b"ready":
                proc.kill()
                err = _read_available(proc.stderr)
                raise JailUnavailableError(
                    "the session network could not be created: "
                    + (
                        err.decode(errors="replace")[-400:]
                        if err.strip()
                        else f"the launcher said nothing in {_HOLDER_READY_TIMEOUT_S:.0f}s"
                        " (a stale AGENT6_JAIL_BIN cannot hold one)"
                    )
                )
            # /proc/<pid> is gone the moment the holder exits.
            for kind in ("user", "net"):
                fds.append(os.open(f"/proc/{proc.pid}/ns/{kind}", os.O_RDONLY))
        except OSError as exc:
            for fd in fds:
                with contextlib.suppress(OSError):
                    os.close(fd)
            proc.kill()
            raise JailUnavailableError(f"the session network could not be held: {exc}") from exc
        except BaseException:
            proc.kill()
            raise
        # The holder says "ready" once; a long-lived web or hub process would accumulate these.
        for pipe in (proc.stdout, proc.stderr):
            if pipe is not None:
                with contextlib.suppress(OSError):
                    pipe.close()
        return cls(userns_fd=fds[0], netns_fd=fds[1], holder_pid=proc.pid, _holder=proc)

    def args(self) -> list[str]:
        """Return the launcher flags that join these namespaces."""
        return ["--userns-fd", str(self.userns_fd), "--netns-fd", str(self.netns_fd)]

    def fds(self) -> tuple[int, int]:
        """Return the descriptors a launcher inherits to join."""
        return (self.userns_fd, self.netns_fd)

    def close(self) -> None:
        """Drop the run's session network; the kernel reclaims it after the last member.

        Closing the holder's stdin ends it, so a run that dies before reaching here
        still releases the namespace.
        """
        if self._holder is not None:
            if self._holder.stdin is not None:
                with contextlib.suppress(OSError):
                    self._holder.stdin.close()
            try:  # unreaped, it is a zombie holding a /proc entry
                self._holder.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._holder.kill()
                self._holder.wait(timeout=5)
        for fd in self.fds():
            with contextlib.suppress(OSError):
                os.close(fd)


def _join_args(
    policy: kinds.JailPolicy, session_net: SessionNetwork | None
) -> tuple[list[str], tuple[int, ...]]:
    """Return the launcher flags and inherited fds for the policy's network.

    Args:
        policy: The jail policy.
        session_net: The run's session network, when one exists.

    Returns:
        The flags and the descriptors; both empty unless the network is `session`.

    Raises:
        JailUnavailableError: The policy says `session` and no network was wired; an
            isolated child would be a different sandbox than the operator asked for.
    """
    if policy.network != "session":
        return [], ()
    if session_net is None:
        raise JailUnavailableError(
            "network = 'session' needs the run's session network; none was wired"
        )
    return session_net.args(), session_net.fds()


def _policy_spec(policy: kinds.JailPolicy) -> dict[str, Any]:
    """Return the launcher's policy spec, JSON-shaped; the caller encodes it.

    An absent `mode` is the launcher's "once"; the exec and serve callers add theirs.
    """
    return {
        "isolation": policy.isolation,
        "cwd": str(policy.cwd),
        "argv": list(policy.argv),
        "env": [list(pair) for pair in policy.env],
        "network": policy.network,
        "extra_ro_paths": [str(p) for p in policy.extra_ro_paths],
        "extra_rw_paths": [str(p) for p in policy.extra_rw_paths],
        "extra_device_paths": [str(p) for p in policy.extra_device_paths],
        "extra_protect_paths": [str(p) for p in policy.extra_protect_paths],
        "tool_paths": [str(p) for p in policy.tool_paths],
        # The builtin private set is unioned at this one choke point, so no policy can omit it:
        # secrets and state never enter the jail even under a $HOME-wide grant.
        "hide_paths": sorted({str(p) for p in paths.hidden_paths(policy.hide_paths)}),
        "timeout_s": policy.timeout_s,
        "memory_limit_mb": policy.memory_limit_mb,
    }


def _run_unsandboxed(policy: kinds.JailPolicy) -> kinds.CommandResult:
    """Run the policy's argv as a plain subprocess, with no confinement.

    The `none` isolation. The parent environment, overlaid with the policy's,
    minus agent6's provider keys: a jailed command never sees one. The
    sandbox-only knobs (network, paths, memory) have no effect here.

    Args:
        policy: The policy; only argv, cwd, env and timeout apply.

    Returns:
        The command's result; a timeout is rc 124, as in the jail.
    """
    env = child_env.without_provider_keys({**os.environ, **{k: v for k, v in policy.env}})
    start = time.monotonic()
    # Bytes, decoded lossily: text=True would raise out of communicate() on a non-UTF-8 byte.
    try:
        proc = subprocess.run(
            list(policy.argv),
            cwd=str(policy.cwd),
            env=env,
            capture_output=True,
            check=False,
            preexec_fn=die_with_parent(os.getpid(), sig=signal.SIGKILL),
            # <= 0 is no wall-clock kill (`agent6 exec`, a command whose check-in replaces it).
            timeout=policy.timeout_s if policy.timeout_s > 0 else None,
        )
    except subprocess.TimeoutExpired as exc:
        return kinds.CommandResult(
            argv=tuple(policy.argv),
            returncode=124,
            stdout=_lossy_text(exc.stdout),
            stderr=_lossy_text(exc.stderr),
            duration_s=time.monotonic() - start,
        )
    duration = time.monotonic() - start
    return kinds.CommandResult(
        argv=tuple(policy.argv),
        returncode=int(proc.returncode),
        stdout=_lossy_text(proc.stdout),
        stderr=_lossy_text(proc.stderr),
        duration_s=duration,
    )


@functools.lru_cache(maxsize=1)
def strict_namespaces_work() -> bool:
    """Return whether the jail binary can set up a `strict` namespace.

    The authoritative probe: `detect.probe_userns_supported` under-reports on an
    AppArmor-restricted host whose profile grants userns to the jail binary but not
    to `/usr/bin/unshare`. Cached for the process lifetime.

    Returns:
        Whether a trivial `strict` policy ran.

    Raises:
        JailBinaryError: The kernel cannot execute the binary; that says nothing about
            namespaces, so the callers refuse with it.
    """
    if not pathlib.Path("/usr/bin/true").exists():
        return False
    probe_cwd = pathlib.Path(tempfile.gettempdir())
    try:
        res = run_in_jail(
            kinds.JailPolicy(
                cwd=probe_cwd,
                argv=("/usr/bin/true",),
                isolation="strict",
                network="none",
                timeout_s=10.0,
            )
        )
    except JailBinaryError:
        raise
    except JailUnavailableError:
        return False
    return res.returncode == 0


def _require_jail_binary() -> pathlib.Path:
    """Return the launcher binary.

    Raises:
        JailBinaryError: No binary was found; the message names the remedies.
    """
    binary = locate_jail_binary()
    if binary is None:
        raise JailBinaryError(
            "agent6-jail binary not found. Install agent6 from a built wheel"
            " (which bundles the binary), or build from source with"
            " `cargo build --release --locked --manifest-path src/agent6/jail/Cargo.toml`,"
            f" or set {_ENV_VAR}=/path/to/agent6-jail."
        )
    return binary


def _raise_for_exec_failure(binary: pathlib.Path, exc: OSError) -> NoReturn:
    """Re-raise the launcher's spawn failure, as the binary's refusal when it is one.

    Args:
        binary: The launcher that failed to start.
        exc: The spawn's OSError.

    Raises:
        JailBinaryError: The kernel refused to execute the binary (ENOEXEC: a build
            for another architecture; EACCES: no exec bit).
        OSError: Any other spawn failure (fork or descriptor pressure), unchanged.
    """
    if exc.errno not in (errno.ENOEXEC, errno.EACCES):
        raise exc
    raise JailBinaryError(
        f"agent6-jail at {binary} cannot be executed: {exc.strerror}."
        " Reinstall the bundled binary with `uv sync --reinstall-package agent6`,"
        f" or point {_ENV_VAR} at a build for this host."
    ) from exc


def _spawn_launcher(
    binary: pathlib.Path,
    args: Sequence[str],
    *,
    stdin: int | IO[bytes] | None,
    stdout: int | IO[bytes] | None,
    stderr: int | IO[bytes] | None,
    pass_fds: Sequence[int] = (),
    die_with_agent: bool = True,
) -> subprocess.Popen[bytes]:
    """Start the launcher in its own session and register it live for the escapee sweep.

    Its own session makes a hang one killpg away, pidns-init and grandchildren
    included; the sweep lock keeps the sweep from seeing it half-registered.

    Args:
        binary: The launcher.
        args: Its flags.
        stdin: The launcher's stdin.
        stdout: The launcher's stdout.
        stderr: The launcher's stderr.
        pass_fds: Descriptors the launcher inherits.
        die_with_agent: Tie the launcher to this process; a background job stays untied.

    Returns:
        The launcher process.

    Raises:
        JailBinaryError: The kernel refused to execute the binary.
    """
    try:
        with _sweep_lock:
            proc = subprocess.Popen(
                [str(binary), *args],
                stdin=stdin,
                stdout=stdout,
                stderr=stderr,
                pass_fds=pass_fds,
                start_new_session=True,
                preexec_fn=(  # noqa: PLW1509
                    die_with_parent(os.getpid(), sig=signal.SIGKILL) if die_with_agent else None
                ),
                env=_LAUNCHER_ENV,
            )
            _live_launchers.add(proc.pid)
    except OSError as exc:
        _raise_for_exec_failure(binary, exc)
    return proc


# Escapee reaping. `strict` has a PID namespace, so nothing outlives the child. `hardened` has
# none: a setsid() child survives the launcher's killpg and reparents to init, so the agent is a
# subreaper and kills what lands on it once the command returns. The launcher cannot do this:
# its Landlock ruleset denies /proc, and granting it would hand every jailed child the agent's
# environ. A process is the command's only if it appeared during the call and sits outside the
# agent's session; the launcher runs in its own, so a same-session child (git) is never swept.
_PR_SET_CHILD_SUBREAPER = 36
_SWEEP_DEADLINE_S = 5.0
_sweep_lock = threading.Lock()
_live_launchers: set[int] = set()
# Children started on purpose in their own session (a `/btw` ask, a lane, the claude_code
# provider's child): they look exactly like an escapee, so every such spawn registers here.
_own_detached: set[int] = set()


@functools.cache
def _become_subreaper() -> None:
    """Make this process a subreaper, once.

    Raises:
        JailUnavailableError: The prctl failed; a command could then outlive its call.
    """
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.restype = ctypes.c_int
    if libc.prctl(_PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
        err = ctypes.get_errno()
        raise JailUnavailableError(
            f"prctl(PR_SET_CHILD_SUBREAPER) failed: {os.strerror(err)}."
            " Without it a sandboxed command could leave a process running after it returns."
        )


def keep_out_of_the_sweep(pid: int) -> None:
    """Mark a pid as a child agent6 started deliberately, not an escapee.

    Called right after a detached spawn; never cleared, the set is bounded by the
    sessions one run opens.
    """
    _own_detached.add(pid)


def _own_children() -> dict[int, int]:
    """Return this process's children as `{pid: session id}`, right now.

    Read as bytes: comm is whatever a process named itself, so one hostile name
    must not break the scan. Empty without /proc: the sweep is a Linux mechanism,
    and an error here would leave a running background command untracked.
    """
    me = str(os.getpid()).encode()
    found: dict[int, int] = {}
    try:
        entries = list(pathlib.Path("/proc").iterdir())
    except OSError:
        return found
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_bytes()
        except OSError:
            continue
        # comm can hold spaces and parens; after its closing one: state, ppid, pgrp, session.
        fields = stat[stat.rfind(b")") + 1 :].split()
        if len(fields) > 3 and fields[1] == me:
            found[int(entry.name)] = int(fields[3])
    return found


def _forget_launcher(pid: int) -> None:
    """Drop a finished launcher from the set the sweep spares."""
    with _sweep_lock:
        _live_launchers.discard(pid)


def signal_group(pid: int, sig: int = signal.SIGKILL) -> None:
    """Signal a pid's process group, or the pid alone when it leads none or we are in it.

    A pgid is a leader's pid, reusable once that leader is reaped: one that is not
    the pid belongs to a leader we do not hold, and under sudo a recycled one would
    kill an unrelated group as root. The sweep holds its launchers unreaped, so
    their pgids cannot recycle; a stop's window is its identity check to the signal.

    Args:
        pid: The target.
        sig: The signal.
    """
    if pid == os.getpid():
        return
    with contextlib.suppress(OSError):
        pgid = os.getpgid(pid)
        if pgid == pid and pgid != os.getpgrp():
            os.killpg(pid, sig)
        else:
            os.kill(pid, sig)


def _kill_escapees(exclude: frozenset[int]) -> frozenset[int]:
    """Kill what the command left behind.

    Args:
        exclude: Pids that are not the command's.

    Returns:
        The pids still alive at the deadline.
    """
    our_session = os.getsid(0)
    deadline = time.monotonic() + _SWEEP_DEADLINE_S
    with _sweep_lock:
        while True:
            children = _own_children()
            # A spared pid that is no longer a child would spare the next process to get it.
            _live_launchers.intersection_update(children)
            _own_detached.intersection_update(children)
            escapees = {
                pid
                for pid, session in children.items()
                if session != our_session
                and pid not in exclude
                and pid not in _live_launchers
                and pid not in _own_detached
            }
            if not escapees:
                return frozenset()
            for pid in escapees:
                signal_group(pid)
                with contextlib.suppress(OSError):
                    # WNOHANG: a child in uninterruptible sleep must not hold the sweep lock.
                    os.waitpid(pid, os.WNOHANG)
            if time.monotonic() >= deadline:
                return frozenset(escapees)
            time.sleep(0.01)  # killing one layer reparents the next here


class JailedProcess:
    """A jailed child agent6 talks to for a whole session, such as an MCP server.

    `close` bounds the child's lifetime: it signals the launcher's group, reaps it,
    then sweeps the escapees that reparented onto the agent, since outside a PID
    namespace signalling the launcher pid alone leaves a setsid child running.

    Attributes:
        popen: The launcher process.
        stdin: The child's stdin, as the Popen holds it.
        stdout: The child's stdout.
        stderr: The child's stderr.
    """

    def __init__(self, proc: subprocess.Popen[bytes], before: frozenset[int] | None = None) -> None:
        self.popen = proc
        self.stdin = proc.stdin
        self.stdout = proc.stdout
        self.stderr = proc.stderr
        # The own-children snapshot from before the launcher; construction time for a direct caller.
        self._before = frozenset(_own_children()) if before is None else before

    def close(self) -> frozenset[int]:
        """End the child, its group and its escapees; best-effort and idempotent.

        Returns:
            The pids the sweep could not kill; the caller says so.
        """
        proc = self.popen
        if proc.stdin is not None:
            with contextlib.suppress(OSError):
                proc.stdin.close()
        if proc.poll() is None:
            # Its own group leader and held unreaped, so its pgid is its pid and cannot recycle.
            with contextlib.suppress(OSError):
                os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(OSError):
                    os.killpg(proc.pid, signal.SIGKILL)
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=1.0)
        _forget_launcher(proc.pid)
        return _kill_escapees(self._before | {proc.pid})


def spawn_in_jail(
    policy: kinds.JailPolicy,
    *,
    stdin: int | None = None,
    stdout: int | None = None,
    stderr: int | None = None,
    session_net: SessionNetwork | None = None,
) -> JailedProcess:
    """Start the policy's argv in the sandbox and return a handle to talk to it.

    The transport for a child agent6 talks to for the whole session (an MCP server
    on a JSON-RPC pipe), with the same policy, launcher and layers as `run_in_jail`.
    The child's stdio is the caller's, straight through every layer, so the policy
    travels on a separate descriptor the launcher reads and closes before the
    child exists. `isolation = "none"` spawns the command directly.

    Args:
        policy: The jail policy.
        stdin: The child's stdin.
        stdout: The child's stdout.
        stderr: The child's stderr.
        session_net: The run's session network, when one exists.

    Returns:
        The handle; its `close` bounds the child's lifetime.

    Raises:
        JailBinaryError: The launcher is missing or cannot be executed.
        JailUnavailableError: The policy needs a session network and none was wired.
    """
    argv = list(policy.argv)
    # Any later child of this process not in the snapshot escaped the server; close sweeps it.
    before = frozenset(_own_children())
    if policy.isolation == "none":
        with _sweep_lock:
            proc = subprocess.Popen(
                argv,
                stdin=stdin,
                stdout=stdout,
                stderr=stderr,
                env=dict(policy.env),
                cwd=policy.cwd,
                start_new_session=True,
                preexec_fn=die_with_parent(os.getpid(), sig=signal.SIGKILL),  # noqa: PLW1509
            )
            # In its own session, so unregistered a sibling handle's close would sweep it.
            _live_launchers.add(proc.pid)
        return JailedProcess(proc, before)
    binary = _require_jail_binary()
    spec = _policy_spec(policy)
    spec["mode"] = "exec"
    _become_subreaper()
    # pass_fds keeps the descriptor's number in the child, so the launcher is told the real one.
    join_args, join_fds = _join_args(policy, session_net)
    policy_r, policy_w = os.pipe()
    try:
        proc = _spawn_launcher(
            binary,
            ["--policy-fd", str(policy_r), *join_args],
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            pass_fds=(policy_r, *join_fds),
        )
    except BaseException:
        os.close(policy_w)
        raise
    finally:
        # Held here, the read end would keep the launcher from ever seeing EOF.
        os.close(policy_r)
    # After the spawn: a policy larger than the pipe buffer would block until the reader exists.
    with os.fdopen(policy_w, "wb") as handle:
        handle.write((json.dumps(spec) + "\n").encode())
    return JailedProcess(proc, before)


def run_in_jail(
    policy: kinds.JailPolicy, *, session_net: SessionNetwork | None = None
) -> kinds.CommandResult:
    """Run the policy's argv inside the sandbox and collect its result.

    The `none` isolation runs the command as a plain subprocess: the one place an
    LLM-influenced argv runs without the jail, so agent6 works where the kernel
    sandbox does not exist. `auto` reaches it only on a host with no confinement
    mechanism; an explicit `none`, `--dangerously-disable-sandbox` or
    `AGENT6_DANGEROUSLY_DISABLE_SANDBOX=1` selects it anywhere, and the CLI warns
    before any such run. Both real levels go through the Rust launcher.

    Args:
        policy: The jail policy.
        session_net: The run's session network, when one exists.

    Returns:
        The command's result; a timeout is rc 124, an unexecutable argv rc 127.

    Raises:
        JailUnavailableError: The launcher is missing, its setup failed, or a process
            the command left behind could not be killed.
    """
    if policy.isolation == "none":
        return _run_unsandboxed(policy)
    binary = _require_jail_binary()
    join_args, join_fds = _join_args(policy, session_net)
    spec = json.dumps(_policy_spec(policy))
    start = time.monotonic()
    _become_subreaper()
    # Any later child of this process not in the snapshot escaped the command.
    before = frozenset(_own_children())
    launcher = _spawn_launcher(
        binary,
        join_args,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        pass_fds=join_fds,
    )
    survivors: frozenset[int] = frozenset()
    try:
        result = _launcher_result(launcher, policy, spec, start, binary)
    finally:
        if launcher.poll() is None:
            # Abandoned mid-command: its own group leader and held unreaped, so its pgid is its pid.
            try:
                os.killpg(launcher.pid, signal.SIGKILL)
            except OSError:
                launcher.kill()
            with contextlib.suppress(subprocess.TimeoutExpired):
                launcher.communicate(timeout=5.0)
        _forget_launcher(launcher.pid)
        survivors = _kill_escapees(before | {launcher.pid})
    if survivors:
        raise JailUnavailableError(survivors_message(survivors))
    return result


def survivors_message(pids: frozenset[int]) -> str:
    """Return the error text for pids a sweep could not kill."""
    return (
        f"could not kill everything the command left running (pids {sorted(pids)});"
        " a process would have outlived this run."
    )


def _launcher_result(
    launcher: subprocess.Popen[bytes],
    policy: kinds.JailPolicy,
    spec: str,
    start: float,
    binary: pathlib.Path,
) -> kinds.CommandResult:
    """Feed the launcher its spec and read the command's result back.

    Args:
        launcher: The spawned launcher.
        policy: The policy it runs.
        spec: The encoded policy.
        start: When the command started, for the duration.
        binary: The launcher's path, for the error text.

    Returns:
        The command's result; a launcher timeout is rc 124, an unexecutable argv rc 127.

    Raises:
        JailUnavailableError: The launcher failed setup or wrote no result.
    """
    # The launcher enforces the command's deadline; this bounds only its teardown after it.
    wait_s = policy.timeout_s + 5.0 if policy.timeout_s > 0 else None
    try:
        raw_out, raw_err = launcher.communicate(input=spec.encode(), timeout=wait_s)
    except subprocess.TimeoutExpired as exc:
        try:
            os.killpg(launcher.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            launcher.kill()
        try:
            raw_out, raw_err = launcher.communicate(timeout=5.0)
        except subprocess.TimeoutExpired:
            raw_out, raw_err = b"", b""
        return kinds.CommandResult(
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
    # A non-zero launcher is a setup failure, except a child that could not be executed: that
    # is an ordinary 127, so the model fixes its argv instead of concluding the sandbox is broken.
    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        if "child execution failed" in stderr:
            return kinds.CommandResult(
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
    return _with_launcher_warnings(
        _result_from_json(result_json, tuple(policy.argv), duration), proc.stderr
    )


def _with_launcher_warnings(
    result: kinds.CommandResult, launcher_stderr: str
) -> kinds.CommandResult:
    """Return the result with the launcher's own diagnostics appended to its stderr.

    The child's stderr arrives in the result JSON, so the launcher's stderr is the
    jail reporting on itself: a mount it could not make, a grant it skipped. Read
    on success too; a degraded but working jail is the case they exist for.
    """
    warnings = launcher_stderr.strip()
    if not warnings:
        return result
    return dataclasses.replace(result, stderr=f"{result.stderr}\n{warnings}".strip())


def _result_from_json(
    result_json: dict[str, object], argv: tuple[str, ...], duration: float
) -> kinds.CommandResult:
    """Return the launcher's result object as a CommandResult.

    An `exec_failed` result gets the same wording as the one-shot path's 127, so the
    model reads one message however its run is jailed.
    """
    failed_exec = bool(result_json.get("exec_failed", False))
    stderr = str(result_json.get("stderr", ""))
    return kinds.CommandResult(
        argv=argv,
        returncode=int(str(result_json["returncode"])),
        stdout=str(result_json.get("stdout", "")),
        stderr=(
            f"{argv[0]}: command not found or not executable ({stderr})" if failed_exec else stderr
        ),
        duration_s=duration,
        exec_failed=failed_exec,
    )


# Detached commands. A background command's launcher stays registered live until `stop`, so
# the sweep spares it; only the launcher's result JSON is captured, so the exit code survives
# the turn that started the command.
_RESULT_NAME = "result.json"
_LAUNCHER_ERR_NAME = "launcher.err"


@dataclasses.dataclass(frozen=True, slots=True)
class BackgroundStatus:
    """What a detached command is doing, right now.

    Attributes:
        running: The process is alive; never an inference from output or age.
        returncode: The exit code once observed.
        error: Why the fate is unknown (a launcher that exited without a result);
            such a command is never reported as still running.
    """

    running: bool
    returncode: int | None
    error: str


@dataclasses.dataclass(frozen=True, slots=True)
class Stopped:
    """A stop request's answer.

    Attributes:
        returncode: The exit code, when this stop reaped the command.
        survivors: Under a PID namespace, the pids the launcher's sweep could not kill.
    """

    returncode: int | None
    survivors: frozenset[int]


def _write_outcome(outcome_dir: pathlib.Path, returncode: int) -> None:
    """Record a command's exit code where a surface in another process reads it."""
    with contextlib.suppress(OSError):
        (outcome_dir / _RESULT_NAME).write_text(
            json.dumps({"returncode": returncode}), encoding="utf-8"
        )


def _write_stopped(outcome_dir: pathlib.Path) -> None:
    """Record that the command was stopped, when its launcher died before reporting a code.

    No code is invented: nobody observed one. Written only over an empty result; a
    command that exited just before the kill keeps the code its launcher wrote, and
    a failed read is no grounds to overwrite.
    """
    result = outcome_dir / _RESULT_NAME
    try:
        existing = result.read_text(errors="replace").strip()
    except FileNotFoundError:
        existing = ""
    except OSError:
        return
    if existing:
        return
    with contextlib.suppress(OSError):
        result.write_text(json.dumps({"stopped": True}), encoding="utf-8")


def _stop_detached(proc: subprocess.Popen[bytes], descendants: frozenset[int], what: str) -> str:
    """Kill a detached process's group and sweep what it left behind.

    Unregistering before the sweep makes this the moment its escapees stop being
    spared; `run_in_jail`'s later sweeps would never catch them, since by then they
    are not new.

    Args:
        proc: The process.
        descendants: The children snapshot taken before it started.
        what: What the process is called in the answer.

    Returns:
        "" when the process and everything it started are gone, else why not.
    """
    if proc.poll() is None:
        with contextlib.suppress(OSError):
            os.killpg(proc.pid, signal.SIGKILL)
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=5.0)
    _forget_launcher(proc.pid)
    survivors = _kill_escapees(descendants)
    stuck = f"{what} {proc.pid} did not exit after SIGKILL" if proc.poll() is None else ""
    left = survivors_message(survivors) if survivors else ""
    return "; ".join(part for part in (stuck, left) if part)


class LocalJob:
    """A detached command running with no confinement (`none` isolation).

    With no launcher to write the exit code, this persists it on the first observed
    exit and on stop; otherwise `/shells` in another process reads the command as
    maybe still running for the run's life and after it.
    """

    def __init__(self, proc: subprocess.Popen[bytes], outcome_dir: pathlib.Path) -> None:
        self._proc = proc
        self._outcome_dir = outcome_dir
        self._descendants = frozenset(_own_children())
        self._final: BackgroundStatus | None = None

    @property
    def pid(self) -> int:
        """The command's pid."""
        return self._proc.pid

    def status(self) -> BackgroundStatus:
        """Return what the command is doing, persisting its exit code the first time."""
        if self._final is not None:
            return self._final
        if self._proc.poll() is None:
            return BackgroundStatus(running=True, returncode=None, error="")
        return self._settle(int(self._proc.returncode))

    def stop(self) -> str:
        """Kill the command and everything it started; idempotent.

        Returns:
            "" when the command is gone, else why it might not be.
        """
        answer = _stop_detached(self._proc, self._descendants, "the command")
        if self._proc.poll() is not None:
            self._settle(int(self._proc.returncode))
        return answer

    def _settle(self, returncode: int) -> BackgroundStatus:
        """Return the final status, recording the observed exit code."""
        self._final = BackgroundStatus(running=False, returncode=returncode, error="")
        _write_outcome(self._outcome_dir, returncode)
        return self._final


class BackgroundJob:
    """A jailed command detached from the call that started it.

    Its launcher writes the exit code to the outcome dir when the command ends.
    """

    def __init__(self, proc: subprocess.Popen[bytes], outcome_dir: pathlib.Path) -> None:
        self._proc = proc
        self._outcome_dir = outcome_dir
        self._descendants = frozenset(_own_children())

    @property
    def pid(self) -> int:
        """The launcher's pid."""
        return self._proc.pid

    def status(self) -> BackgroundStatus:
        """Return what the command is doing, from the live launcher or its written result."""
        if self._proc.poll() is None:
            return BackgroundStatus(running=True, returncode=None, error="")
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

    def stop(self) -> str:
        """Kill the command and everything it started; idempotent.

        The kill takes the launcher down before it can write the exit code, so the
        ending is recorded here.

        Returns:
            "" when the command is gone, else why it might not be.
        """
        answer = _stop_detached(self._proc, self._descendants, "the sandbox launcher")
        if self._proc.poll() is not None:
            _write_stopped(self._outcome_dir)
        return answer

    def _unregister(self) -> None:
        """Drop the finished launcher from the set the sweep spares."""
        _forget_launcher(self._proc.pid)


class SessionJob:
    """A command left running inside a `JailSession`.

    Its pid is namespace-local, so every question goes through the session. The
    terminal answer is kept: once the launcher has reaped the command, asking again
    gets ECHILD, which is not an unknown exit code.
    """

    def __init__(
        self,
        session: JailSession,
        pid: int,
        outcome_dir: pathlib.Path,
        *,
        before: kinds.ChildSnapshot,
    ) -> None:
        self._session = session
        self._pid = pid
        self._outcome_dir = outcome_dir
        # Taken before the command started: its own reparented daemon is not in it, a sibling's is.
        self._before = before
        self._final: BackgroundStatus | None = None
        session.open_job(pid, before)

    def status(self) -> BackgroundStatus:
        """Return what the command is doing, asking the session until it has ended."""
        if self._final is not None:
            return self._final
        try:
            status = self._session.status_background(self._pid)
        except JailUnavailableError as exc:
            self._final = BackgroundStatus(running=False, returncode=None, error=str(exc))
            return self._final
        if not status.running:
            self._settle(status)
        return status

    def stop(self) -> str:
        """Kill the command's group and sweep what it left outside it, on every stop.

        A command that exited on its own can have left a daemon too.

        Returns:
            "" when everything is gone, else the launcher's refusal or the pids no
            sweep could kill.
        """
        try:
            stopped = self._session.stop_background(self._pid)
        except JailUnavailableError as exc:
            if self._final is None:
                self._settle(BackgroundStatus(running=False, returncode=None, error=str(exc)))
            return str(exc)
        if self._final is None:
            self._settle(BackgroundStatus(running=False, returncode=stopped.returncode, error=""))
        survivors = stopped.survivors | self._session.sweep_for(self._pid, self._before)
        return survivors_message(survivors) if survivors else ""

    def _settle(self, status: BackgroundStatus) -> None:
        """Keep the terminal status and persist its exit code."""
        self._final = status
        if status.returncode is not None:
            _write_outcome(self._outcome_dir, status.returncode)


def start_in_jail(
    policy: kinds.JailPolicy, *, outcome_dir: pathlib.Path
) -> BackgroundJob | LocalJob:
    """Spawn the policy's argv in the sandbox and return without waiting.

    The same policy, launcher and confinement as `run_in_jail`. Nothing of the
    command's own output is captured, so its argv must redirect it; only the
    launcher's result JSON and stderr land in the outcome dir, which lets the exit
    code outlive the turn. The sweep spares the job while it lives and `stop`
    takes its whole group down.

    Args:
        policy: The jail policy.
        outcome_dir: Where the exit code is written.

    Returns:
        The job; a `LocalJob` under the `none` isolation.

    Raises:
        JailBinaryError: The launcher is missing or cannot be executed.
    """
    paths.mkdir_for_real_user(outcome_dir)
    if policy.isolation == "none":
        with _sweep_lock:
            proc = subprocess.Popen(
                list(policy.argv),
                cwd=str(policy.cwd),
                env=child_env.without_provider_keys({**os.environ, **dict(policy.env)}),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            # Unregistered, one stop sweeps every sibling, whose next status then reads as rc 0.
            _live_launchers.add(proc.pid)
        return LocalJob(proc, outcome_dir)
    binary = _require_jail_binary()
    spec = json.dumps(_policy_spec(policy))
    _become_subreaper()
    result = (outcome_dir / _RESULT_NAME).open("wb")
    errors = (outcome_dir / _LAUNCHER_ERR_NAME).open("wb")
    try:
        launcher = _spawn_launcher(
            binary, (), stdin=subprocess.PIPE, stdout=result, stderr=errors, die_with_agent=False
        )
    finally:
        result.close()
        errors.close()
    assert launcher.stdin is not None
    with contextlib.suppress(OSError):
        launcher.stdin.write(spec.encode())
    launcher.stdin.close()
    return BackgroundJob(launcher, outcome_dir)


def _abandon_launcher(proc: subprocess.Popen[bytes], interrupt_w: int) -> None:
    """Unregister, close and reap a launcher that failed setup.

    Closing stdin here swallows the EPIPE the dead peer forces on the buffered spec
    write; left to garbage collection it surfaces as unraisable noise, and the
    unreaped child is a zombie for the rest of the process.
    """
    _forget_launcher(proc.pid)
    with contextlib.suppress(OSError):
        os.close(interrupt_w)
    for pipe in (proc.stdin, proc.stdout, proc.stderr):
        if pipe is not None:
            with contextlib.suppress(OSError, ValueError):
                pipe.close()
    if proc.poll() is None:
        with contextlib.suppress(OSError):
            os.killpg(proc.pid, signal.SIGKILL)
    with contextlib.suppress(subprocess.TimeoutExpired, OSError):
        proc.wait(timeout=5.0)


@dataclasses.dataclass(slots=True)
class JailSession:
    """One long-lived launcher, serving every command of one run.

    The launcher sets up its namespaces, rootfs, Landlock and seccomp once, then
    reads one request per line, so the run's commands share a netns, a PID
    namespace and a /tmp: a server one command starts is reachable by the next.
    Every isolation level serves, `none` included, so the capture and background
    lifecycle are one implementation. Only `strict` has the PID namespace, so
    elsewhere the launcher sweeps what it backgrounded at EOF and this side sweeps
    each command's escapees. Not thread-safe: one loop drives it, one command at a
    time.

    Attributes:
        startup_stderr: What the launcher wrote to stderr during setup (a refused
            /proc mount, a skipped grant): the jail came up degraded but runs. The
            caller surfaces it once; "" when setup was clean.
    """

    _proc: subprocess.Popen[bytes]
    _binary: pathlib.Path
    # Without a PID namespace a `setsid` escapee reparents here, so each command's sweep runs here.
    _pid_namespaced: bool
    # One byte on the interrupt pipe asks the launcher to hand the running command back now; the
    # request pipe is in lockstep and this side is blocked reading that request's answer.
    _interrupt_w: int
    # Carried on every request: the launcher's own default would ignore the operator's cap.
    _memory_limit_mb: int
    startup_stderr: str = ""
    # What was already ours at open; without a PID namespace, anything beyond it at close is
    # this session's escapee.
    _opened_with: frozenset[int] = frozenset()
    # The start snapshot of every background command not yet stopped, by pid: what a stop sweeps
    # ends where the next of these began.
    _live_jobs: dict[int, kinds.ChildSnapshot] = dataclasses.field(default_factory=dict)
    _snapshots: int = 0

    @classmethod
    def open(
        cls, policy: kinds.JailPolicy, *, session_net: SessionNetwork | None = None
    ) -> JailSession:
        """Start a serving launcher confined by the policy; its argv is ignored.

        Args:
            policy: The confinement every request runs under.
            session_net: The run's session network, when one exists.

        Returns:
            The session, ready for its first request.

        Raises:
            JailUnavailableError: The launcher died during setup.
            JailBinaryError: The launcher is missing or cannot be executed.
        """
        binary = _require_jail_binary()
        join_args, join_fds = _join_args(policy, session_net)
        _become_subreaper()
        interrupt_r, interrupt_w = os.pipe()
        try:
            proc = _spawn_launcher(
                binary,
                ["--interrupt-fd", str(interrupt_r), *join_args],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                pass_fds=(interrupt_r, *join_fds),
            )
        except BaseException:
            os.close(interrupt_w)
            raise
        finally:
            os.close(interrupt_r)
        spec = _policy_spec(policy)
        spec["mode"] = "serve"
        assert proc.stdin is not None and proc.stdout is not None
        try:
            proc.stdin.write((json.dumps(spec) + "\n").encode())
            proc.stdin.flush()
            # The ready line keeps the lockstep and marks where a setup warning is on stderr.
            ready = proc.stdout.readline()
        except OSError as exc:
            # EPIPE: the launcher died before consuming the spec.
            err = _lossy_text(_read_available(proc.stderr)).strip()
            _abandon_launcher(proc, interrupt_w)
            raise JailUnavailableError(f"jail session died during setup: {err or exc}") from exc
        if not ready:
            err = _lossy_text(_read_available(proc.stderr)).strip()
            _abandon_launcher(proc, interrupt_w)
            raise JailUnavailableError(f"jail session died during setup: {err or 'no output'}")
        startup_stderr = _lossy_text(_read_available(proc.stderr, budget_s=0.1)).strip()
        pid_namespaced = policy.isolation == "strict"
        return cls(
            _proc=proc,
            _binary=binary,
            _pid_namespaced=pid_namespaced,
            _interrupt_w=interrupt_w,
            _memory_limit_mb=policy.memory_limit_mb,
            startup_stderr=startup_stderr,
            _opened_with=frozenset() if pid_namespaced else frozenset(_own_children()),
        )

    def run(
        self,
        argv: tuple[str, ...],
        *,
        env: tuple[tuple[str, str], ...] = (),
        timeout_s: float = 600.0,
        checkin_s: float = 0.0,
        log_dir: str = "",
        interrupted: Callable[[], bool] | None = None,
    ) -> kinds.CommandResult | kinds.BackgroundHandoff:
        """Run one command to completion in this session's namespaces.

        Without a PID namespace the command's escapees reparent to this process
        and are swept here; the sweep bounds a command's process lifetime.

        Args:
            argv: The command.
            env: Its environment entries.
            timeout_s: The command's deadline.
            checkin_s: When the launcher hands a still-running command back.
            log_dir: Where a handed-back command's output goes.
            interrupted: Polled while waiting; once true the launcher hands the
                command back at once, as the check-in would, and teardown stops it.

        Returns:
            The result, or the handoff of a command still running.

        Raises:
            JailUnavailableError: The launcher is gone, answered without a pid, or a
                process the command left behind could not be killed.
        """
        start = time.monotonic()
        before = self.child_snapshot()
        answer: dict[str, object] = {}
        survivors: frozenset[int] = frozenset()
        try:
            answer = self._request(
                {
                    "kind": "run",
                    "argv": list(argv),
                    "env": [list(p) for p in env],
                    "timeout_s": timeout_s,
                    "checkin_s": checkin_s,
                    "log_dir": log_dir,
                    "memory_limit_mb": self._memory_limit_mb,
                },
                interrupted=interrupted,
            )
        finally:
            # A handed-back command is still running; its children are not escapees yet.
            if not answer.get("backgrounded"):
                survivors = self._sweep(before.pids)
        if survivors:
            raise JailUnavailableError(survivors_message(survivors))
        elapsed = time.monotonic() - start
        if answer.get("backgrounded"):
            pid = answer.get("pid")
            if not isinstance(pid, int):
                raise JailUnavailableError(f"jail session handed back no pid: {answer}")
            return kinds.BackgroundHandoff(
                argv=argv,
                pid=pid,
                log=str(answer.get("log", "")),
                stdout=str(answer.get("stdout", "")),
                stderr=str(answer.get("stderr", "")),
                duration_s=elapsed,
                before=before,
            )
        return _result_from_json(answer, argv, elapsed)

    def start_background(
        self, argv: tuple[str, ...], *, env: tuple[tuple[str, str], ...] = ()
    ) -> int:
        """Start a command and leave it running for later commands to reach.

        Args:
            argv: The command.
            env: Its environment entries.

        Returns:
            Its pid, namespace-local: only this session can report on it or stop it.

        Raises:
            JailUnavailableError: The launcher started no command.
        """
        answer = self._request(
            {
                "kind": "background",
                "argv": list(argv),
                "env": [list(p) for p in env],
                "memory_limit_mb": self._memory_limit_mb,
            }
        )
        pid = answer.get("pid")
        if not isinstance(pid, int):
            raise JailUnavailableError(f"jail session started no command: {answer}")
        return pid

    def status_background(self, pid: int) -> BackgroundStatus:
        """Return what a backgrounded command is doing.

        Raises:
            JailUnavailableError: The launcher is gone.
        """
        answer = self._request({"kind": "status", "pid": pid})
        code = answer.get("returncode")
        return BackgroundStatus(
            running=bool(answer.get("running")),
            returncode=code if isinstance(code, int) else None,
            error=str(answer.get("error", "")),
        )

    def stop_background(self, pid: int) -> Stopped:
        """Kill a backgrounded command's group and, under a PID namespace, its escapees.

        Idempotent.

        Args:
            pid: The command's namespace-local pid.

        Returns:
            The stop's answer.

        Raises:
            JailUnavailableError: The launcher refused or is gone.
        """
        answer = self._request({"kind": "stop", "pid": pid})
        if not answer.get("stopped"):
            raise JailUnavailableError(f"jail session could not stop {pid}: {answer}")
        code = answer.get("returncode")
        listed = answer.get("survivors")
        survivors = cast(list[object], listed) if isinstance(listed, list) else []
        return Stopped(
            returncode=code if isinstance(code, int) else None,
            survivors=frozenset(p for p in survivors if isinstance(p, int)),
        )

    def _request(
        self, request: dict[str, object], *, interrupted: Callable[[], bool] | None = None
    ) -> dict[str, object]:
        """Send one request and read its one answer line.

        The channel is in lockstep: every request gets exactly one answer, or the
        next request reads this one's.

        Args:
            request: The request object.
            interrupted: Polled while waiting for the answer.

        Returns:
            The answer object.

        Raises:
            JailUnavailableError: The launcher is gone or answered badly; never the
                raw pipe error, which no handler in the run catches.
        """
        assert self._proc.stdin is not None and self._proc.stdout is not None
        try:
            self._proc.stdin.write((json.dumps(request) + "\n").encode())
            self._proc.stdin.flush()
            line = self._await_answer(interrupted)
        except (OSError, ValueError) as exc:  # ValueError: a closed pipe
            raise JailUnavailableError(f"jail session is gone: {exc}") from exc
        if not line:
            raise JailUnavailableError("jail session ended before answering")
        try:
            parsed = json.loads(line.decode(errors="replace"))
        except ValueError as exc:
            raise JailUnavailableError(
                f"jail session produced unparseable output: {line!r}"
            ) from exc
        if not isinstance(parsed, dict):
            raise JailUnavailableError(f"jail session answered with {type(parsed).__name__}")
        return parsed  # pyright: ignore[reportUnknownVariableType]

    def _await_answer(self, interrupted: Callable[[], bool] | None) -> bytes:
        """Return the launcher's answer line, waiting in a way an operator Stop can cut short.

        Selecting on the raw descriptor is safe because the channel is in lockstep:
        a complete answer can never sit in the reader's buffer while select says
        there is nothing to read. EOF selects ready and reads empty.
        """
        assert self._proc.stdout is not None
        if interrupted is None:
            return self._proc.stdout.readline()
        asked = False
        while not select.select([self._proc.stdout], [], [], _ANSWER_POLL_S)[0]:
            if not asked and interrupted():
                # Once: a second byte would hand back whatever command runs next.
                with contextlib.suppress(OSError):
                    os.write(self._interrupt_w, b"\x01")
                asked = True
        return self._proc.stdout.readline()

    def close(self) -> frozenset[int]:
        """Shut the request channel; the launcher exits on the EOF.

        `communicate()` closes stdin itself; a manual `stdin.close()` before it
        makes Python 3.12 and 3.13 raise `ValueError: flush of closed file`.

        Returns:
            The pids the sweep could not kill; empty under a PID namespace, which
            takes everything with it.
        """
        with contextlib.suppress(OSError):
            os.close(self._interrupt_w)
        with contextlib.suppress(subprocess.TimeoutExpired, ValueError, OSError):
            self._proc.communicate(timeout=10.0)
        if self._proc.poll() is None:
            with contextlib.suppress(OSError):
                os.killpg(self._proc.pid, signal.SIGKILL)
        _forget_launcher(self._proc.pid)
        return self._sweep(frozenset())

    def child_snapshot(self) -> kinds.ChildSnapshot:
        """Return the agent's children before a command starts.

        What reparents onto the agent after this is that command's own until the
        next command starts.

        Returns:
            The snapshot; empty under a PID namespace, which bounds them.
        """
        self._snapshots += 1
        pids = frozenset() if self._pid_namespaced else frozenset(_own_children())
        return kinds.ChildSnapshot(self._snapshots, pids)

    def open_job(self, pid: int, before: kinds.ChildSnapshot) -> None:
        """Record a background command's start snapshot until its stop."""
        self._live_jobs[pid] = before

    def sweep_for(self, pid: int, before: kinds.ChildSnapshot) -> frozenset[int]:
        """Kill what a stopped command left outside its process group.

        What appeared after it started and before the next still-running command
        did, whichever order the two stop in; the close sweeps whatever is left.

        Args:
            pid: The command's pid.
            before: The snapshot taken before it started.

        Returns:
            The pids the sweep could not kill.
        """
        self._live_jobs.pop(pid, None)
        younger = [b for b in self._live_jobs.values() if b.seq > before.seq]
        spare = before.pids
        if younger:
            spare |= frozenset(_own_children()) - min(younger, key=lambda b: b.seq).pids
        return self._sweep(spare)

    def _sweep(self, spare: frozenset[int]) -> frozenset[int]:
        """Kill what the session's commands left outside their process groups.

        Args:
            spare: Pids that are not escapees.

        Returns:
            The pids it could not kill; none under a PID namespace.
        """
        if self._pid_namespaced:
            return frozenset()
        return _kill_escapees(self._opened_with | spare | {self._proc.pid})
