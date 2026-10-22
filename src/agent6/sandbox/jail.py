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
import fcntl
import functools
import json
import os
import pathlib
import resource
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import termios
import threading
import time
from collections.abc import Callable, Sequence
from typing import IO, Any, NoReturn

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


def _tied_to_agent(controlling_terminal: bool, memory_limit_mib: int = 0) -> Callable[[], None]:
    """Return the `preexec_fn` of an unconfined child: the parent tie and the memory cap.

    Args:
        controlling_terminal: Make the terminal on stdin the child's new session's
            controlling terminal, so its signals and job control reach the foreground group.
        memory_limit_mib: The RLIMIT_DATA cap in MiB; 0 is no cap.

    Returns:
        The function to run in the child before exec.
    """
    tie = die_with_parent(os.getpid(), sig=signal.SIGKILL)
    cap = memory_cap(memory_limit_mib)

    def _setup() -> None:
        tie()
        cap()
        if controlling_terminal:
            fcntl.ioctl(0, termios.TIOCSCTTY, 0)

    return _setup


# The launcher hides its /proc entry from the command and runs with an empty environment, so its
# environ cannot leak the operator's provider key.
# The launcher reads nothing from its environment; the policy arrives on stdin.
_LAUNCHER_ENV: dict[str, str] = {}


class JailUnavailableError(Exception):
    """The launcher is missing, refused to set up, or left something running."""


class JailBinaryError(JailUnavailableError):
    """The launcher binary is missing or the kernel refuses to execute it.

    Says nothing about the host's namespaces.
    """


def lossy_text(v: object) -> str:
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


def read_available(pipe: IO[bytes] | None, budget_s: float = 0.5) -> bytes:
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
        binary = require_jail_binary()
        # Not `spawn_launcher`: nothing sweeps the holder, and `close` ends it through stdin.
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
                try:
                    code: int | None = proc.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    code = None
                    proc.kill()
                err = read_available(proc.stderr)
                if err.strip():
                    detail = err.decode(errors="replace")[-400:]
                elif code is not None:
                    detail = f"the launcher exited with code {code} before reporting ready"
                else:
                    detail = (
                        f"the launcher said nothing in {_HOLDER_READY_TIMEOUT_S:.0f}s"
                        " (a stale AGENT6_JAIL_BIN cannot hold one)"
                    )
                raise JailUnavailableError(f"the session network could not be created: {detail}")
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


def network_join_args(
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


def policy_spec(policy: kinds.JailPolicy) -> dict[str, Any]:
    """Return the launcher's policy spec, JSON-shaped; the caller encodes it.

    An absent `mode` is the launcher's "once"; the exec and serve callers add theirs.

    Args:
        policy: The jail policy.

    Returns:
        Each policy field under its launcher key; `hide_paths` also holds agent6's private dirs.
    """
    return {
        "isolation": policy.isolation,
        "cwd": str(policy.cwd),
        "argv": list(policy.argv),
        "env": [list(pair) for pair in policy.env],
        "network": policy.network,
        "extra_ro_paths": [str(p) for p in policy.extra_ro_paths],
        "extra_rw_paths": [str(p) for p in policy.extra_rw_paths],
        "grant_sources": [[str(path), str(real)] for path, real in policy.grant_sources],
        "extra_device_paths": [str(p) for p in policy.extra_device_paths],
        "extra_protect_paths": [str(p) for p in policy.extra_protect_paths],
        "tool_paths": [str(p) for p in policy.tool_paths],
        # The builtin private set is unioned at this one choke point, so no policy can omit it:
        # secrets and state never enter the jail even under a $HOME-wide grant.
        "hide_paths": sorted({str(p) for p in paths.hidden_paths(policy.hide_paths)}),
        "timeout_s": policy.timeout_s,
        # The launcher's wire key keeps the old suffix.
        "memory_limit_mb": policy.memory_limit_mib,
    }


def memory_cap(memory_limit_mib: int) -> Callable[[], None]:
    """Return a `preexec_fn` setting the child's RLIMIT_DATA, as the launcher does under `none`.

    Args:
        memory_limit_mib: The cap in MiB; 0 is no cap.

    Returns:
        The function to run in the child before exec.
    """

    def _setup() -> None:
        if memory_limit_mib > 0:
            limit = memory_limit_mib << 20
            resource.setrlimit(resource.RLIMIT_DATA, (limit, limit))

    return _setup


def _run_unsandboxed(policy: kinds.JailPolicy) -> kinds.CommandResult:
    """Run the policy's argv as a plain subprocess, with no confinement.

    The `none` isolation. The parent environment, overlaid with the policy's,
    minus agent6's provider keys: a jailed command never sees one. The
    sandbox-only knobs (network, paths) have no effect here; the memory cap and
    the timeout apply as they do in the launcher.

    Args:
        policy: The policy; only argv, cwd, env, memory cap and timeout apply.

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
            preexec_fn=_tied_to_agent(False, policy.memory_limit_mib),
            # <= 0 is no wall-clock kill (`agent6 exec`, a command whose check-in replaces it).
            timeout=policy.timeout_s if policy.timeout_s > 0 else None,
        )
    except subprocess.TimeoutExpired as exc:
        return kinds.CommandResult(
            argv=tuple(policy.argv),
            returncode=124,
            stdout=lossy_text(exc.stdout),
            stderr=lossy_text(exc.stderr),
            duration_s=time.monotonic() - start,
        )
    duration = time.monotonic() - start
    return kinds.CommandResult(
        argv=tuple(policy.argv),
        returncode=int(proc.returncode),
        stdout=lossy_text(proc.stdout),
        stderr=lossy_text(proc.stderr),
        duration_s=duration,
    )


@functools.lru_cache(maxsize=1)
def namespaces_work() -> bool:
    """Return whether the jail binary can run `full` `sandbox.isolation` on this host.

    The authoritative probe: `detect.probe_userns_supported` under-reports on an
    AppArmor-restricted host whose profile grants userns to the jail binary but not
    to `/usr/bin/unshare`. Cached for the process lifetime.

    Returns:
        Whether a trivial `full` policy ran.

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
                isolation="full",
                network="none",
                timeout_s=10.0,
            )
        )
    except JailBinaryError:
        raise
    except JailUnavailableError:
        return False
    return res.returncode == 0


def require_jail_binary() -> pathlib.Path:
    """Return the launcher binary.

    Returns:
        The path `locate_jail_binary` found.

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


def spawn_launcher(
    binary: pathlib.Path,
    args: Sequence[str],
    *,
    stdin: int | IO[bytes] | None,
    stdout: int | IO[bytes] | None,
    stderr: int | IO[bytes] | None,
    pass_fds: Sequence[int] = (),
    die_with_agent: bool = True,
    controlling_terminal: bool = False,
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
        controlling_terminal: Make the terminal on stdin the launcher's controlling terminal.

    Returns:
        The launcher process.

    Raises:
        JailBinaryError: The kernel refused to execute the binary.
    """
    try:
        with sweep_lock:
            proc = subprocess.Popen(
                [str(binary), *args],
                stdin=stdin,
                stdout=stdout,
                stderr=stderr,
                pass_fds=pass_fds,
                start_new_session=True,
                preexec_fn=(  # noqa: PLW1509
                    _tied_to_agent(controlling_terminal) if die_with_agent else None
                ),
                env=_LAUNCHER_ENV,
            )
            live_launchers.add(proc.pid)
    except OSError as exc:
        _raise_for_exec_failure(binary, exc)
    return proc


# Escapee reaping. `full` has a PID namespace, so nothing outlives the launcher. `partial` has
# none: a setsid() child survives the launcher's killpg and reparents to init, so the agent is a
# subreaper and kills what lands on it once the command returns. The launcher cannot do this:
# its Landlock ruleset denies /proc, and granting it would hand every jailed child the agent's
# environ. A process is the command's only if it appeared during the call and sits outside the
# agent's session; the launcher runs in its own, so a same-session child (git) is never swept.
_PR_SET_CHILD_SUBREAPER = 36
_SWEEP_DEADLINE_S = 5.0
sweep_lock = threading.Lock()
live_launchers: set[int] = set()
# Children started on purpose in their own session (a `/btw` ask, a lane, the claude_code
# provider's child): they look exactly like an escapee, so every such spawn registers here.
_own_detached: set[int] = set()


@functools.cache
def become_subreaper() -> None:
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

    Called right after a detached spawn; each sweep drops the pids that are no longer
    children, so a recycled pid is never spared.

    Args:
        pid: The child, started in a session of its own.
    """
    _own_detached.add(pid)


def own_children() -> dict[int, int]:
    """Return this process's children as `{pid: session id}`, right now.

    Read as bytes: comm is whatever a process named itself, so one hostile name
    must not break the scan.

    Returns:
        The children found. Empty without /proc: the sweep is a Linux mechanism, and an
        error here would leave a running background command untracked.
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


def forget_launcher(pid: int) -> None:
    """Drop a finished launcher from the set the sweep spares."""
    with sweep_lock:
        live_launchers.discard(pid)


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


def kill_escapees(exclude: frozenset[int]) -> frozenset[int]:
    """Kill what the command left behind.

    Args:
        exclude: Pids that are not the command's.

    Returns:
        The pids still alive at the deadline.
    """
    our_session = os.getsid(0)
    deadline = time.monotonic() + _SWEEP_DEADLINE_S
    with sweep_lock:
        while True:
            children = own_children()
            # A spared pid that is no longer a child would spare the next process to get it.
            live_launchers.intersection_update(children)
            _own_detached.intersection_update(children)
            escapees = {
                pid
                for pid, session in children.items()
                if session != our_session
                and pid not in exclude
                and pid not in live_launchers
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
        self._before = frozenset(own_children()) if before is None else before

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
        forget_launcher(proc.pid)
        return kill_escapees(self._before | {proc.pid})


def spawn_in_jail(
    policy: kinds.JailPolicy,
    *,
    stdin: int | None = None,
    stdout: int | None = None,
    stderr: int | None = None,
    session_net: SessionNetwork | None = None,
    controlling_terminal: bool = False,
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
        controlling_terminal: Make the terminal on stdin the controlling terminal of the
            child's session, with the command in its foreground group.

    Returns:
        The handle; its `close` bounds the child's lifetime.

    Raises:
        JailBinaryError: The launcher is missing or cannot be executed.
        JailUnavailableError: The policy needs a session network and none was wired.
    """
    argv = list(policy.argv)
    # Any later child of this process not in the snapshot escaped the server; close sweeps it.
    before = frozenset(own_children())
    if policy.isolation == "none":
        with sweep_lock:
            proc = subprocess.Popen(
                argv,
                stdin=stdin,
                stdout=stdout,
                stderr=stderr,
                env=dict(policy.env),
                cwd=policy.cwd,
                start_new_session=True,
                preexec_fn=_tied_to_agent(controlling_terminal, policy.memory_limit_mib),  # noqa: PLW1509
            )
            # In its own session, so unregistered a sibling handle's close would sweep it.
            live_launchers.add(proc.pid)
        return JailedProcess(proc, before)
    binary = require_jail_binary()
    spec = policy_spec(policy)
    spec["mode"] = "exec"
    become_subreaper()
    # pass_fds keeps the descriptor's number in the child, so the launcher is told the real one.
    join_args, join_fds = network_join_args(policy, session_net)
    policy_r, policy_w = os.pipe()
    try:
        proc = spawn_launcher(
            binary,
            ["--policy-fd", str(policy_r), *join_args],
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            pass_fds=(policy_r, *join_fds),
            controlling_terminal=controlling_terminal,
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
    mechanism; an explicit `none`, `--disable-sandbox` or
    `AGENT6_DISABLE_SANDBOX=1` selects it anywhere, and the CLI warns
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
    binary = require_jail_binary()
    join_args, join_fds = network_join_args(policy, session_net)
    spec = json.dumps(policy_spec(policy))
    start = time.monotonic()
    become_subreaper()
    # Any later child of this process not in the snapshot escaped the command.
    before = frozenset(own_children())
    launcher = spawn_launcher(
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
        forget_launcher(launcher.pid)
        survivors = kill_escapees(before | {launcher.pid})
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
            stdout=lossy_text(raw_out) or lossy_text(exc.stdout),
            stderr=lossy_text(raw_err) or lossy_text(exc.stderr),
            duration_s=time.monotonic() - start,
        )
    proc = subprocess.CompletedProcess(
        args=[str(binary)],
        returncode=launcher.returncode,
        stdout=lossy_text(raw_out),
        stderr=lossy_text(raw_err),
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
        result_from_json(result_json, tuple(policy.argv), duration), proc.stderr
    )


def _with_launcher_warnings(
    result: kinds.CommandResult, launcher_stderr: str
) -> kinds.CommandResult:
    """Return the result with the launcher's own diagnostics appended to its stderr.

    Read on success too; a degraded but working jail is the case they exist for.

    Args:
        result: The command's result.
        launcher_stderr: The jail's reports on itself, such as a skipped grant.

    Returns:
        The result with the diagnostics appended.
    """
    warnings = launcher_stderr.strip()
    if not warnings:
        return result
    return dataclasses.replace(result, stderr=f"{result.stderr}\n{warnings}".strip())


def result_from_json(
    result_json: dict[str, object], argv: tuple[str, ...], duration: float
) -> kinds.CommandResult:
    """Return the launcher's result object as a CommandResult.

    Args:
        result_json: The decoded result line; only `returncode` is required.
        argv: The command the launcher ran.
        duration: How long the command took in seconds, as the caller timed it.

    Returns:
        The result. An `exec_failed` result gets the same wording as the one-shot path's
        127, so the model reads one message however its run is jailed.
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
