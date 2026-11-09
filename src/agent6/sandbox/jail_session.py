# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Serve commands through a persistent jail launcher."""

from __future__ import annotations

import contextlib
import dataclasses
import os
import select
import subprocess
import time
from collections.abc import Callable
from typing import cast

import pydantic

from agent6 import kinds
from agent6.sandbox import jail, jail_protocol

_PROTOCOL_VERSION = 4
READ_LIMIT = 65536
_READY_LIMIT = 4096
_ANSWER_LIMIT = 128 * 1024 * 1024
# Direct argv needs a file, not a shell builtin; empty PATH entries denote the command cwd.
# Landlock leaves stat/access unconfined; opening checks grants without executing a candidate.
_EXECUTABLE_LOOKUP = """\
search=$PATH
unreadable=
while :; do
    directory=${search%%:*}
    candidate=${directory:-.}/$1
    if [ -f "$candidate" ] && [ -x "$candidate" ]; then
        if ( : < "$candidate" ) 2>/dev/null; then exit 0; fi
        if [ ! -r "$candidate" ]; then unreadable=$candidate; fi
    fi
    case "$search" in
        *:*) search=${search#*:} ;;
        *) break ;;
    esac
done
if [ -n "$unreadable" ]; then
    printf '%s\\n' "$unreadable: execute-only executable visibility is indeterminate" >&2
    printf '%s\\n' 'Use an explicit command path to skip this advisory lookup.' >&2
    exit 2
fi
exit 1
"""


class _Ready(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid", strict=True)
    ready: bool
    protocol: int
    instance: str = pydantic.Field(pattern=r"^[0-9a-f]{32}$")


class _Answer(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow", strict=True)
    id: int = pydantic.Field(gt=0)
    instance: str = pydantic.Field(pattern=r"^[0-9a-f]{32}$")


class _ReadAnswer(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid", strict=True)
    id: int
    offset: int = pydantic.Field(ge=0)
    next_offset: int = pydantic.Field(ge=0)
    size: int = pydantic.Field(ge=0)
    data: list[int] = pydantic.Field(max_length=READ_LIMIT)
    capture_error: str


class LauncherProbe(pydantic.BaseModel):
    """Observe target facts without copying environment secrets to disk."""

    model_config = pydantic.ConfigDict(extra="forbid", strict=True, frozen=True)
    uid: int = pydantic.Field(ge=0)
    gid: int = pydantic.Field(ge=0)
    architecture: str
    environment: dict[str, str]
    mounts: str


@dataclasses.dataclass(frozen=True, slots=True)
class BackgroundRead:
    """Raw captured bytes and the next byte offset in one size snapshot."""

    data: bytes
    next_offset: int
    size: int
    capture_error: str = ""


@dataclasses.dataclass(frozen=True, slots=True)
class BackgroundStatus:
    """What a detached command is doing, right now.

    Attributes:
        running: The process is alive; never an inference from output or age.
        returncode: The exit code once observed.
        error: Why the outcome or descendant cleanup is unconfirmed.
        capture_error: Why output is incomplete, independently of the command's exit.
    """

    running: bool
    returncode: int | None
    error: str
    capture_error: str = ""


@dataclasses.dataclass(frozen=True, slots=True)
class Stopped:
    """A stop request's answer.

    Attributes:
        returncode: The observed exit code, retained across status and stop.
        survivors: Owned descendants the supervisor could not kill.
        error: Why the supervisor could not confirm cleanup.
        capture_error: Why output is incomplete, independently of cleanup.
    """

    returncode: int | None
    survivors: frozenset[int]
    error: str = ""
    capture_error: str = ""


def _abandon_launcher(proc: subprocess.Popen[bytes], stderr: jail_protocol.Stderr | None) -> None:
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
    target: kinds.ExecutionTarget | None = None
    _stderr: jail_protocol.Stderr | None = None
    _request_id: int = 0
    _close_error: str | None = None

    @classmethod
    def open(
        cls,
        policy: kinds.JailPolicy,
        *,
        session_net: jail.SessionNetwork | None = None,
        controls: tuple[kinds.EffectiveControl, ...] = (),
    ) -> JailSession:
        """Start a serving launcher confined by the policy; its argv is ignored.

        Args:
            policy: The confinement every request runs under.
            session_net: The run's session network, when one exists.
            controls: The resolved local controls, without claiming per-control probes.

        Returns:
            The session, ready for its first request.

        Raises:
            jail.JailUnavailableError: The launcher died during setup.
            JailBinaryError: The launcher is missing or cannot be executed.
        """
        binary = jail.require_jail_binary()
        join_args, join_fds = jail.network_join_args(policy, session_net)
        proc = jail.spawn_launcher(
            binary,
            join_args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            pass_fds=join_fds,
            die_with_agent=False,
        )
        target = kinds.ExecutionTarget(
            transport="local",
            host_root=policy.cwd,
            command_root=policy.cwd,
            user=str(os.getuid()),
            home=dict(policy.env).get("HOME"),
            environment=kinds.ExecutionEnvironment(policy.env),
            controls=controls,
        )
        return cls.connect(proc, policy, target=target)

    @classmethod
    def connect(
        cls,
        proc: subprocess.Popen[bytes],
        policy: kinds.JailPolicy,
        *,
        target: kinds.ExecutionTarget,
        bootstrap: bytes = b"",
    ) -> JailSession:
        """Own a process transport and prepare its supervisor before returning.

        The transport must carry raw stdin/stdout, never a terminal. Failure closes it;
        it does not authorize selecting a different execution target.

        Args:
            proc: The process carrying the supervisor channel.
            policy: The local launcher confinement policy.
            target: Prepared transport and environment facts.
            bootstrap: Validated launcher bytes; nonempty requires installation acknowledgment.

        Returns:
            The ready session, owning the transport.

        Raises:
            jail.JailUnavailableError: The supervisor handshake failed.
        """
        assert proc.stdin is not None and proc.stdout is not None and proc.stderr is not None
        stderr = jail_protocol.Stderr(proc.stderr)
        stderr.start()
        try:
            spec = jail.policy_spec(policy)
            spec["mode"] = "serve"
            deadline = time.monotonic() + 10
            if bootstrap:
                jail_protocol.write_startup(proc, bootstrap, deadline)
                # Buffered installers must exit before any policy bytes enter their stdin.
                installed = jail_protocol.read_frame(proc, _READY_LIMIT, deadline=deadline)
                if installed != b'{"installed":true}\n':
                    raise jail.JailUnavailableError("invalid launcher installation acknowledgment")
            jail_protocol.write_startup(proc, jail_protocol.frame(spec), deadline)
            ready = _Ready.model_validate_json(
                jail_protocol.read_frame(proc, _READY_LIMIT, deadline=deadline)
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
            target=dataclasses.replace(
                target, protocol_version=ready.protocol, launcher_instance=ready.instance
            ),
        )

    def probe(self) -> LauncherProbe:
        """Read the launcher's identity, inherited environment and mount inventory.

        Returns:
            The observed target facts, including potentially secret environment values.

        Raises:
            jail.JailUnavailableError: The peer returned an invalid probe.
        """
        answer = self._request({"kind": "probe"})
        try:
            return LauncherProbe.model_validate(answer.get("probe"))
        except ValueError:
            raise jail.JailUnavailableError("invalid launcher probe") from None

    def prove_mapping(self, name: str, nonce: str) -> str:
        """Read the host nonce and write a target nonce through no-follow directory fds.

        Args:
            name: The fresh probe directory's basename.
            nonce: The independent host nonce.

        Returns:
            The independent target nonce written into the shared directory.

        Raises:
            jail.JailUnavailableError: The mapping proof failed.
        """
        answer = self._request({"kind": "mapping", "name": name, "nonce": nonce})
        value = answer.get("nonce")
        if not isinstance(value, str) or len(value) != 64 or answer.get("error"):
            raise jail.JailUnavailableError(
                f"workspace mapping proof failed: {answer.get('error')}"
            )
        return value

    def executable_available(self, executable: str) -> bool:
        """Look up a bare executable inside the prepared target's command environment.

        Paths remain advisory: setup may create a workspace-relative runner later.
        A fixed /bin/sh probe opens readable files without running them or a PATH-selected shell.

        Args:
            executable: The candidate executable name or advisory path.

        Returns:
            Whether the target can read and resolve the executable, or True for advisory paths.

        Raises:
            jail.JailUnavailableError: The probe failed, timed out or could not determine
                target visibility.
        """
        if "/" in executable:
            return True
        if self.target is None:
            raise jail.JailUnavailableError("command session has no prepared target")
        result = self.run(
            ("/bin/sh", "-c", _EXECUTABLE_LOOKUP, "agent6-lookup", executable),
            env=self.target.environment.variables,
            timeout_s=10,
        )
        if not isinstance(result, kinds.CommandResult) or result.exec_failed:
            raise jail.JailUnavailableError("target executable lookup could not run")
        if result.timed_out:
            raise jail.JailUnavailableError("target executable lookup timed out")
        if result.returncode not in (0, 1):
            detail = result.stderr.strip()[-400:]
            raise jail.JailUnavailableError(
                f"target executable lookup failed (exit {result.returncode})"
                + (f": {detail}" if detail else "")
            )
        return result.returncode == 0

    def run(
        self,
        argv: tuple[str, ...],
        *,
        env: tuple[tuple[str, str], ...] = (),
        timeout_s: float = 600.0,
        checkin_s: float = 0.0,
        interrupted: Callable[[], bool] | None = None,
    ) -> kinds.CommandResult | kinds.BackgroundHandoff:
        """Run one command to completion in this session's namespaces.

        Args:
            argv: The command.
            env: Its environment entries.
            timeout_s: The command's deadline.
            checkin_s: When the launcher hands a still-running command back.
            interrupted: Polled while waiting; true requests early hand-back when a
                check-in allows it, otherwise stops the command.

        Returns:
            The result, or the handoff of a command still running.

        Raises:
            jail.JailUnavailableError: The launcher is gone, answered without a handle, or a
                process the command left behind could not be killed.
        """
        start = time.monotonic()
        if self.target is not None and self.target.environment.inherit:
            env = tuple((dict(self.target.environment.variables) | dict(env)).items())
        answer = self._request(
            {
                "kind": "run",
                "argv": list(argv),
                "env": [list(p) for p in env],
                "timeout_s": timeout_s,
                "checkin_s": checkin_s,
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
        if self.target is not None and self.target.environment.inherit:
            env = tuple((dict(self.target.environment.variables) | dict(env)).items())
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
            capture_error=str(answer.get("capture_error", "")),
        )

    def read_background(
        self, handle: str, *, offset: int, limit: int = READ_LIMIT
    ) -> BackgroundRead:
        """Read at most 64 KiB of an owned job's capture, including after stop.

        Args:
            handle: The command's launcher-instance-scoped handle.
            offset: The first byte to copy.
            limit: Maximum bytes returned, from 1 to 65536.

        Returns:
            Raw bytes, the next offset and the size observed before reading.

        Raises:
            jail.JailUnavailableError: The handle, bounds or response is invalid.
        """
        if offset < 0 or not 1 <= limit <= READ_LIMIT:
            raise jail.JailUnavailableError("invalid background read bounds")
        answer = self._request(
            {"kind": "read", "handle": handle, "offset": offset, "limit": limit},
            answer_limit=READ_LIMIT * 6 + 4096,
        )
        if answer.get("error"):
            raise jail.JailUnavailableError(f"background read failed: {answer['error']}")
        try:
            read = _ReadAnswer.model_validate(answer)
            data = bytes(read.data)
        except ValueError as exc:
            _abandon_launcher(self._proc, self._stderr)
            raise jail.JailUnavailableError(f"invalid background read response: {exc}") from exc
        if (
            read.offset != offset
            or read.next_offset != offset + len(data)
            or read.next_offset > read.size
            or len(data) > limit
            or (not data and offset < read.size)
        ):
            _abandon_launcher(self._proc, self._stderr)
            raise jail.JailUnavailableError("invalid background read offsets or size")
        return BackgroundRead(
            data=data,
            next_offset=read.next_offset,
            size=read.size,
            capture_error=read.capture_error,
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
            error=str(answer.get("error", ""))
            if answer.get("stopped")
            else f"jail session could not stop {handle}: {answer}",
            capture_error=str(answer.get("capture_error", "")),
        )

    def _request(
        self,
        request: dict[str, object],
        *,
        interrupted: Callable[[], bool] | None = None,
        answer_limit: int = _ANSWER_LIMIT,
    ) -> dict[str, object]:
        """Send one request and read its one answer line.

        Only cancellation may be sent before the response; it names this request
        and has no response of its own.

        Args:
            request: The request object.
            interrupted: Polled while waiting for the answer.
            answer_limit: The maximum response frame size.

        Returns:
            The answer object.

        Raises:
            jail.JailUnavailableError: The launcher is gone or answered badly; never the
                raw pipe error, which no handler in the run catches.
        """
        assert self._proc.stdin is not None and self._proc.stdout is not None
        self._request_id += 1
        data = jail_protocol.frame({**request, "id": self._request_id})
        try:
            self._proc.stdin.write(data)
            self._proc.stdin.flush()
            line = jail_protocol.read_frame(
                self._proc, answer_limit, request_id=self._request_id, interrupted=interrupted
            )
            try:
                answer = _Answer.model_validate_json(line)
            except pydantic.ValidationError as exc:
                raise jail.JailUnavailableError(
                    "jail session answered with an invalid request ID or JSON: "
                    + str(exc.errors(include_input=False, include_url=False)[0]["msg"])
                ) from None
            if answer.id != self._request_id:
                raise jail.JailUnavailableError(
                    f"jail session answered request ID {answer.id}, expected {self._request_id}"
                )
            if self.target is None or answer.instance != self.target.launcher_instance:
                raise jail.JailUnavailableError("jail session launcher instance changed")
            return answer.model_dump(exclude={"instance"})
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
                        [self._proc.stdout], [], [], min(remaining, jail_protocol.ANSWER_POLL_S)
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
