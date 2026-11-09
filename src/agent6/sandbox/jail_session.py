# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Serve commands through a persistent jail launcher."""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import select
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
_PROTOCOL_VERSION = 2
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
        error: Why the outcome or descendant cleanup is unconfirmed.
    """

    running: bool
    returncode: int | None
    error: str


@dataclasses.dataclass(frozen=True, slots=True)
class Stopped:
    """A stop request's answer.

    Attributes:
        returncode: The observed exit code, retained across status and stop.
        survivors: Owned descendants the supervisor could not kill.
        error: Why the supervisor could not confirm cleanup.
    """

    returncode: int | None
    survivors: frozenset[int]
    error: str = ""


def _abandon_launcher(proc: subprocess.Popen[bytes], stderr: _Stderr | None) -> None:
    """Kill, unregister and reap a launcher whose protocol cannot be trusted."""
    if proc.poll() is None:
        with contextlib.suppress(OSError):
            # Killing the group would also kill the job subreapers that handle channel loss.
            proc.kill()
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

    The launcher shares its confinement policy across commands, applying the
    filesystem ruleset in each job owner. Under full isolation, commands share
    a netns, PID namespace and /tmp: a server is reachable by later commands.
    Every isolation level serves, `none` included, so the capture and background
    lifecycle are one implementation. The supervisor owns job handles, retained
    outcomes and descendant cleanup at every level. Not thread-safe: one loop
    drives it, one command at a time.

    Attributes:
        startup_stderr: What the launcher wrote to stderr during setup (a refused
            /proc mount, a skipped grant): the jail came up degraded but runs. The
            caller surfaces it once; "" when setup was clean.
    """

    _proc: subprocess.Popen[bytes]
    # Carried on every request: the launcher's own default would ignore the operator's cap.
    _memory_limit_mib: int
    startup_stderr: str = ""
    _stderr: _Stderr | None = None
    _request_id: int = 0
    _close_error: str | None = None

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
            die_with_agent=False,
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
        return cls(
            _proc=proc,
            _stderr=stderr,
            _memory_limit_mib=policy.memory_limit_mib,
            startup_stderr=startup_stderr,
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
            jail.JailUnavailableError: The launcher is gone, answered without a handle, or a
                process the command left behind could not be killed.
        """
        start = time.monotonic()
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
        listed = answer.get("survivors", [])
        if listed:
            raise jail.JailUnavailableError(f"command cleanup failed; surviving pids: {listed}")
        elapsed = time.monotonic() - start
        if answer.get("backgrounded"):
            handle = answer.get("handle")
            if not isinstance(handle, str) or not handle:
                raise jail.JailUnavailableError(f"jail session handed back no handle: {answer}")
            return kinds.BackgroundHandoff(
                argv=argv,
                handle=handle,
                log=str(answer.get("log", "")),
                stdout=str(answer.get("stdout", "")),
                stderr=str(answer.get("stderr", "")),
                duration_s=elapsed,
            )
        return jail.result_from_json(answer, argv, elapsed)

    def start_background(
        self, argv: tuple[str, ...], *, env: tuple[tuple[str, str], ...] = ()
    ) -> str:
        """Start a command and leave it running for later commands to reach.

        Args:
            argv: The command.
            env: Its environment entries.

        Returns:
            Its launcher-instance-scoped handle.

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
        handle = answer.get("handle")
        if not isinstance(handle, str) or not handle:
            raise jail.JailUnavailableError(f"jail session started no command: {answer}")
        return handle

    def status_background(self, handle: str) -> BackgroundStatus:
        """Return what a backgrounded command is doing.

        Args:
            handle: The command's launcher-instance-scoped handle.

        Returns:
            The launcher's retained answer, including any observed exit code.

        Raises:
            jail.JailUnavailableError: The launcher is gone.
        """
        answer = self._request({"kind": "status", "handle": handle})
        code = answer.get("returncode")
        return BackgroundStatus(
            running=bool(answer.get("running")),
            returncode=code if isinstance(code, int) else None,
            error=str(answer.get("error", "")),
        )

    def stop_background(self, handle: str) -> Stopped:
        """Stop a job and its descendants through its owning supervisor.

        Idempotent.

        Args:
            handle: The command's launcher-instance-scoped handle.

        Returns:
            The observed exit code and any cleanup failure, independently.

        Raises:
            jail.JailUnavailableError: The launcher did not answer.
        """
        answer = self._request({"kind": "stop", "handle": handle})
        code = answer.get("returncode")
        listed = answer.get("survivors")
        survivors = cast(list[object], listed) if isinstance(listed, list) else []
        return Stopped(
            returncode=code if isinstance(code, int) else None,
            survivors=frozenset(p for p in survivors if isinstance(p, int)),
            error=""
            if answer.get("stopped")
            else f"jail session could not stop {handle}: {answer}",
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

    def close(self) -> str:
        """Close stdin and let the supervisor settle its owned jobs.

        Returns:
            An empty string after clean supervisor exit, otherwise why cleanup is unknown.
        """
        if self._close_error is not None:
            return self._close_error
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
        code = self._proc.poll()
        detail = self._stderr.text() if self._stderr is not None else ""
        self._close_error = (
            "" if code == 0 else f"job cleanup unknown: jail supervisor exit {code}: {detail}"
        )
        _abandon_launcher(self._proc, self._stderr)
        return self._close_error
