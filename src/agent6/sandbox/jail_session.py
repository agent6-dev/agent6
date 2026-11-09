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
import threading
import time
from collections.abc import Callable
from typing import IO, cast

import pydantic

from agent6 import kinds
from agent6.sandbox import jail

# The answer wait's poll interval: an operator Stop reads as immediate at this cost.
_ANSWER_POLL_S = 0.2
_PROTOCOL_VERSION = 1
_REQUEST_LIMIT = 1024 * 1024
_READY_LIMIT = 4096
_ANSWER_LIMIT = 128 * 1024 * 1024


class _Ready(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid", strict=True)
    ready: bool
    protocol: int


class _Answer(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow", strict=True)
    id: int = pydantic.Field(gt=0)


@dataclasses.dataclass(slots=True)
class _Stderr:
    """Drain transport diagnostics independently, retaining a bounded tail."""

    pipe: IO[bytes]
    _tail: bytearray = dataclasses.field(default_factory=bytearray)
    _lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)
    _done: threading.Event = dataclasses.field(default_factory=threading.Event)
    _thread: threading.Thread | None = None
    _truncated: bool = False

    def start(self) -> None:
        os.set_blocking(self.pipe.fileno(), False)
        self._thread = threading.Thread(target=self._drain, daemon=True)
        self._thread.start()

    def _read_available(self) -> bool:
        for _ in range(128):
            try:
                chunk = os.read(self.pipe.fileno(), 8192)
            except BlockingIOError:
                return True
            if not chunk:
                return False
            self._tail.extend(chunk)
            if len(self._tail) > 65536:
                del self._tail[:-65536]
                self._truncated = True
        return True

    def _drain(self) -> None:
        while not self._done.is_set():
            if select.select([self.pipe], [], [], 0.1)[0]:
                with self._lock:
                    if not self._read_available():
                        return

    def text(self) -> str:
        with self._lock:
            if not self.pipe.closed:
                self._read_available()
            prefix = "[jail stderr truncated]\n" if self._truncated else ""
            return prefix + jail.lossy_text(bytes(self._tail)).strip()

    def close(self) -> None:
        self._done.set()
        if self._thread is not None:
            self._thread.join(timeout=1)


def _frame(request: dict[str, object]) -> bytes:
    try:
        data = (json.dumps(request, allow_nan=False) + "\n").encode()
    except (TypeError, ValueError) as exc:
        raise jail.JailUnavailableError(f"jail session request is not valid JSON: {exc}") from exc
    if len(data) > _REQUEST_LIMIT:
        raise jail.JailUnavailableError(f"jail protocol frame exceeds {_REQUEST_LIMIT} bytes")
    return data


def _read_frame(
    proc: subprocess.Popen[bytes],
    limit: int,
    *,
    request_id: int = 0,
    interrupted: Callable[[], bool] | None = None,
    deadline: float | None = None,
) -> bytes:
    """Read one bounded frame without blocking cancellation on a partial line.

    Args:
        proc: The protocol peer.
        limit: Maximum frame bytes, including the newline.
        request_id: The active request, or zero during readiness.
        interrupted: Polled while reading; true sends one cancellation frame.
        deadline: The readiness deadline, or None for a command response.

    Returns:
        The complete frame, including its newline.

    Raises:
        EOFError: The peer closed stdout before a complete frame.
        jail.JailUnavailableError: The frame is oversized, malformed or late.
    """
    assert proc.stdout is not None and proc.stdin is not None
    line = bytearray()
    asked = False
    while True:
        if deadline is not None and time.monotonic() > deadline:
            raise jail.JailUnavailableError("jail session handshake timed out")
        if not asked and interrupted is not None and interrupted():
            proc.stdin.write(_frame({"kind": "cancel", "id": request_id}))
            proc.stdin.flush()
            asked = True
        if not select.select([proc.stdout], [], [], _ANSWER_POLL_S)[0]:
            continue
        chunk = os.read(proc.stdout.fileno(), min(65536, limit + 1 - len(line)))
        if not chunk:
            raise EOFError("jail session ended before a complete protocol frame")
        line.extend(chunk)
        if len(line) > limit:
            raise jail.JailUnavailableError(f"jail protocol frame exceeds {limit} bytes")
        if b"\n" in chunk:
            if line.find(b"\n") != len(line) - 1:
                raise jail.JailUnavailableError("jail session sent data after a protocol frame")
            return bytes(line)


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


def _abandon_launcher(proc: subprocess.Popen[bytes], stderr: _Stderr | None) -> None:
    """Kill, unregister and reap a launcher whose protocol cannot be trusted."""
    if proc.poll() is None:
        with contextlib.suppress(OSError):
            os.killpg(proc.pid, signal.SIGKILL)
    with contextlib.suppress(subprocess.TimeoutExpired, OSError):
        proc.wait(timeout=5.0)
    if stderr is not None:
        stderr.close()
    jail.forget_launcher(proc.pid)
    for pipe in (proc.stdin, proc.stdout, proc.stderr):
        if pipe is not None:
            with contextlib.suppress(OSError, ValueError):
                pipe.close()


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
    # Carried on every request: the launcher's own default would ignore the operator's cap.
    _memory_limit_mib: int
    startup_stderr: str = ""
    _stderr: _Stderr | None = None
    _request_id: int = 0
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
        spec = jail.policy_spec(policy)
        spec["mode"] = "serve"
        data = _frame(spec)
        proc = jail.spawn_launcher(
            binary,
            join_args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            pass_fds=join_fds,
        )
        assert proc.stdin is not None and proc.stdout is not None and proc.stderr is not None
        stderr = _Stderr(proc.stderr)
        stderr.start()
        try:
            proc.stdin.write(data)
            proc.stdin.flush()
            ready = _Ready.model_validate_json(
                _read_frame(proc, _READY_LIMIT, deadline=time.monotonic() + 10)
            )
            if not ready.ready or ready.protocol != _PROTOCOL_VERSION:
                raise jail.JailUnavailableError(
                    f"jail session handshake requires protocol {_PROTOCOL_VERSION}, got {ready}"
                )
            startup_stderr = stderr.text()
        except (EOFError, OSError, ValueError, jail.JailUnavailableError) as exc:
            err = stderr.text()
            _abandon_launcher(proc, stderr)
            if isinstance(exc, (EOFError, OSError)):
                reason = f"jail session died during setup: {err or exc}"
            else:
                reason = f"jail session handshake failed: {str(exc)[:500]}"
                if err:
                    reason += f": {err}"
            raise jail.JailUnavailableError(reason) from exc
        pid_namespaced = policy.isolation == "full"
        return cls(
            _proc=proc,
            _binary=binary,
            _pid_namespaced=pid_namespaced,
            _stderr=stderr,
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
            interrupted: Polled while waiting; true requests early hand-back when a
                check-in and log directory allow it, otherwise stops the command.

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

        Only cancellation may be sent before the response; it names this request
        and has no response of its own.

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
        self._request_id += 1
        data = _frame({**request, "id": self._request_id})
        try:
            self._proc.stdin.write(data)
            self._proc.stdin.flush()
            line = _read_frame(
                self._proc, _ANSWER_LIMIT, request_id=self._request_id, interrupted=interrupted
            )
            try:
                answer = _Answer.model_validate_json(line)
            except pydantic.ValidationError as exc:
                raise jail.JailUnavailableError(
                    "jail session answered with an invalid request ID or JSON: "
                    + str(exc.errors(include_input=False, include_url=False)[0]["msg"])
                ) from exc
            if answer.id != self._request_id:
                raise jail.JailUnavailableError(
                    f"jail session answered request ID {answer.id}, expected {self._request_id}"
                )
            return answer.model_dump()
        except (EOFError, OSError, ValueError, jail.JailUnavailableError) as exc:
            err = self._stderr.text() if self._stderr is not None else ""
            _abandon_launcher(self._proc, self._stderr)
            raise jail.JailUnavailableError(
                f"jail session protocol failed: {exc}" + (f": {err}" if err else "")
            ) from exc

    def close(self) -> frozenset[int]:
        """Close stdin, let EOF stop the launcher, then sweep command escapees.

        Returns:
            Pids the sweep could not kill; empty under a PID namespace.
        """
        deadline = time.monotonic() + 10.0
        if self._proc.stdin is not None:
            with contextlib.suppress(ValueError, OSError):
                self._proc.stdin.close()
        # An interrupted response can fill stdout and keep the launcher from reading stdin's EOF.
        with contextlib.suppress(ValueError, OSError):
            if self._proc.stdout is not None and not self._proc.stdout.closed:
                while self._proc.poll() is None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    readable, _, _ = select.select(
                        [self._proc.stdout], [], [], min(remaining, _ANSWER_POLL_S)
                    )
                    if readable and not os.read(self._proc.stdout.fileno(), 65536):
                        break
        with contextlib.suppress(subprocess.TimeoutExpired, OSError):
            self._proc.wait(timeout=max(0.0, deadline - time.monotonic()))
        _abandon_launcher(self._proc, self._stderr)
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
