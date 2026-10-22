# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Serve commands through a persistent jail launcher."""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import pathlib
import select
import signal
import subprocess
import time
from collections.abc import Callable
from typing import cast

from agent6 import kinds
from agent6.sandbox import jail

# The answer wait's poll interval: an operator Stop reads as immediate at this cost.
_ANSWER_POLL_S = 0.2


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


def _abandon_launcher(proc: subprocess.Popen[bytes], interrupt_w: int) -> None:
    """Unregister, close and reap a launcher that failed setup.

    Closing stdin here swallows the EPIPE the dead peer forces on the buffered spec
    write; left to garbage collection it surfaces as unraisable noise, and the
    unreaped child is a zombie for the rest of the process.

    Args:
        proc: The launcher; its group is killed if it still runs.
        interrupt_w: The write end of its interrupt pipe.
    """
    jail.forget_launcher(proc.pid)
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
    lifecycle are one implementation. Only `full` has the PID namespace, so
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
    _memory_limit_mib: int
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
        cls, policy: kinds.JailPolicy, *, session_net: jail.SessionNetwork | None = None
    ) -> JailSession:
        """Start a serving launcher confined by the policy; its argv is ignored.

        Args:
            policy: The confinement every request runs under.
            session_net: The run's session network, when one exists.

        Returns:
            The session, ready for its first request.

        Raises:
            jail.JailUnavailableError: The launcher died during setup.
            JailBinaryError: The launcher is missing or cannot be executed.
        """
        binary = jail.require_jail_binary()
        join_args, join_fds = jail.network_join_args(policy, session_net)
        jail.become_subreaper()
        interrupt_r, interrupt_w = os.pipe()
        try:
            proc = jail.spawn_launcher(
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
        spec = jail.policy_spec(policy)
        spec["mode"] = "serve"
        assert proc.stdin is not None and proc.stdout is not None
        try:
            proc.stdin.write((json.dumps(spec) + "\n").encode())
            proc.stdin.flush()
            # The ready line keeps the lockstep and marks where a setup warning is on stderr.
            ready = proc.stdout.readline()
        except OSError as exc:
            # EPIPE: the launcher died before consuming the spec.
            err = jail.lossy_text(jail.read_available(proc.stderr)).strip()
            _abandon_launcher(proc, interrupt_w)
            raise jail.JailUnavailableError(
                f"jail session died during setup: {err or exc}"
            ) from exc
        if not ready:
            err = jail.lossy_text(jail.read_available(proc.stderr)).strip()
            _abandon_launcher(proc, interrupt_w)
            raise jail.JailUnavailableError(f"jail session died during setup: {err or 'no output'}")
        startup_stderr = jail.lossy_text(jail.read_available(proc.stderr, budget_s=0.1)).strip()
        pid_namespaced = policy.isolation == "full"
        return cls(
            _proc=proc,
            _binary=binary,
            _pid_namespaced=pid_namespaced,
            _interrupt_w=interrupt_w,
            _memory_limit_mib=policy.memory_limit_mib,
            startup_stderr=startup_stderr,
            _opened_with=frozenset() if pid_namespaced else frozenset(jail.own_children()),
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
            jail.JailUnavailableError: The launcher is gone, answered without a pid, or a
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
                    # The launcher's wire key keeps the old suffix.
                    "memory_limit_mb": self._memory_limit_mib,
                },
                interrupted=interrupted,
            )
        finally:
            # A handed-back command is still running; its children are not escapees yet.
            if not answer.get("backgrounded"):
                survivors = self._sweep(before.pids)
        if survivors:
            raise jail.JailUnavailableError(jail.survivors_message(survivors))
        elapsed = time.monotonic() - start
        if answer.get("backgrounded"):
            pid = answer.get("pid")
            if not isinstance(pid, int):
                raise jail.JailUnavailableError(f"jail session handed back no pid: {answer}")
            return kinds.BackgroundHandoff(
                argv=argv,
                pid=pid,
                log=str(answer.get("log", "")),
                stdout=str(answer.get("stdout", "")),
                stderr=str(answer.get("stderr", "")),
                duration_s=elapsed,
                before=before,
            )
        return jail.result_from_json(answer, argv, elapsed)

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
            jail.JailUnavailableError: The launcher started no command.
        """
        answer = self._request(
            {
                "kind": "background",
                "argv": list(argv),
                "env": [list(p) for p in env],
                # The launcher's wire key keeps the old suffix.
                "memory_limit_mb": self._memory_limit_mib,
            }
        )
        pid = answer.get("pid")
        if not isinstance(pid, int):
            raise jail.JailUnavailableError(f"jail session started no command: {answer}")
        return pid

    def status_background(self, pid: int) -> BackgroundStatus:
        """Return what a backgrounded command is doing.

        Args:
            pid: The command's namespace-local pid.

        Returns:
            The launcher's answer; after one carrying the exit code, asking again gets an
            ECHILD `error`.

        Raises:
            jail.JailUnavailableError: The launcher is gone.
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
            jail.JailUnavailableError: The launcher refused or is gone.
        """
        answer = self._request({"kind": "stop", "pid": pid})
        if not answer.get("stopped"):
            raise jail.JailUnavailableError(f"jail session could not stop {pid}: {answer}")
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
            jail.JailUnavailableError: The launcher is gone or answered badly; never the
                raw pipe error, which no handler in the run catches.
        """
        assert self._proc.stdin is not None and self._proc.stdout is not None
        try:
            self._proc.stdin.write((json.dumps(request) + "\n").encode())
            self._proc.stdin.flush()
            line = self._await_answer(interrupted)
        except (OSError, ValueError) as exc:  # ValueError: a closed pipe
            raise jail.JailUnavailableError(f"jail session is gone: {exc}") from exc
        if not line:
            raise jail.JailUnavailableError("jail session ended before answering")
        try:
            parsed = json.loads(line.decode(errors="replace"))
        except ValueError as exc:
            raise jail.JailUnavailableError(
                f"jail session produced unparseable output: {line!r}"
            ) from exc
        if not isinstance(parsed, dict):
            raise jail.JailUnavailableError(f"jail session answered with {type(parsed).__name__}")
        return parsed  # pyright: ignore[reportUnknownVariableType]

    def _await_answer(self, interrupted: Callable[[], bool] | None) -> bytes:
        """Return the launcher's answer line, waiting in a way an operator Stop can cut short.

        Selecting on the raw descriptor is safe because the channel is in lockstep:
        a complete answer can never sit in the reader's buffer while select says
        there is nothing to read.

        Args:
            interrupted: Polled while waiting; true asks the launcher to hand the command
                back. None waits without polling.

        Returns:
            The answer line; empty at EOF.
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
        jail.forget_launcher(self._proc.pid)
        return self._sweep(frozenset())

    def child_snapshot(self) -> kinds.ChildSnapshot:
        """Return the agent's children before a command starts.

        What reparents onto the agent after this is that command's own until the
        next command starts.

        Returns:
            The snapshot; empty under a PID namespace, which bounds them.
        """
        self._snapshots += 1
        pids = frozenset() if self._pid_namespaced else frozenset(jail.own_children())
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
            spare |= frozenset(jail.own_children()) - min(younger, key=lambda b: b.seq).pids
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
        return jail.kill_escapees(self._opened_with | spare | {self._proc.pid})
