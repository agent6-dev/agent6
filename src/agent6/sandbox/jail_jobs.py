# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Track detached commands and persist their outcomes."""

from __future__ import annotations

import contextlib
import json
import os
import pathlib
import signal
import subprocess

from agent6 import child_env, kinds, paths
from agent6.sandbox import jail, jail_session

# Detached commands. A background command's launcher stays registered live until `stop`, so
# the sweep spares it; only the launcher's result JSON is captured, so the exit code survives
# the turn that started the command.
_RESULT_NAME = "result.json"
_LAUNCHER_ERR_NAME = "launcher.err"


def _write_outcome(outcome_dir: pathlib.Path, returncode: int) -> None:
    """Record a command's exit code where a surface in another process reads it."""
    with contextlib.suppress(OSError):
        (outcome_dir / _RESULT_NAME).write_text(
            json.dumps({"returncode": returncode}), encoding="utf-8"
        )


def write_stopped(outcome_dir: pathlib.Path) -> None:
    """Record that the command was stopped, when its launcher died before reporting a code.

    No code is invented: nobody observed one. Written only over an empty result; a
    command that exited just before the kill keeps the code its launcher wrote, and
    a failed read is no grounds to overwrite.

    Args:
        outcome_dir: Where the launcher writes the command's exit code.
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
    jail.forget_launcher(proc.pid)
    survivors = jail.kill_escapees(descendants)
    stuck = f"{what} {proc.pid} did not exit after SIGKILL" if proc.poll() is None else ""
    left = jail.survivors_message(survivors) if survivors else ""
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
        self._descendants = frozenset(jail.own_children())
        self._final: jail_session.BackgroundStatus | None = None

    @property
    def pid(self) -> int:
        """The command's pid."""
        return self._proc.pid

    def status(self) -> jail_session.BackgroundStatus:
        """Return what the command is doing, persisting its exit code the first time."""
        if self._final is not None:
            return self._final
        if self._proc.poll() is None:
            return jail_session.BackgroundStatus(running=True, returncode=None, error="")
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

    def _settle(self, returncode: int) -> jail_session.BackgroundStatus:
        """Return the final status, recording the observed exit code."""
        self._final = jail_session.BackgroundStatus(running=False, returncode=returncode, error="")
        _write_outcome(self._outcome_dir, returncode)
        return self._final


class BackgroundJob:
    """A jailed command detached from the call that started it.

    Its launcher writes the exit code to the outcome dir when the command ends.
    """

    def __init__(self, proc: subprocess.Popen[bytes], outcome_dir: pathlib.Path) -> None:
        self._proc = proc
        self._outcome_dir = outcome_dir
        self._descendants = frozenset(jail.own_children())

    @property
    def pid(self) -> int:
        """The launcher's pid."""
        return self._proc.pid

    def status(self) -> jail_session.BackgroundStatus:
        """Return what the command is doing, from the live launcher or its written result."""
        if self._proc.poll() is None:
            return jail_session.BackgroundStatus(running=True, returncode=None, error="")
        self._unregister()
        raw = ""
        with contextlib.suppress(OSError):
            raw = (self._outcome_dir / _RESULT_NAME).read_text(errors="replace")
        with contextlib.suppress(ValueError, IndexError, KeyError):
            return jail_session.BackgroundStatus(
                running=False,
                returncode=int(json.loads(raw.strip().splitlines()[-1])["returncode"]),
                error="",
            )
        err = ""
        with contextlib.suppress(OSError):
            err = (self._outcome_dir / _LAUNCHER_ERR_NAME).read_text(errors="replace").strip()
        return jail_session.BackgroundStatus(
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
            write_stopped(self._outcome_dir)
        return answer

    def _unregister(self) -> None:
        """Drop the finished launcher from the set the sweep spares."""
        jail.forget_launcher(self._proc.pid)


class SessionJob:
    """A command left running inside a `jail_session.JailSession`.

    Its pid is namespace-local, so every question goes through the session. The
    terminal answer is kept: once the launcher has reaped the command, asking again
    gets ECHILD, which is not an unknown exit code.
    """

    def __init__(
        self,
        session: jail_session.JailSession,
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
        self._final: jail_session.BackgroundStatus | None = None
        session.open_job(pid, before)

    def status(self) -> jail_session.BackgroundStatus:
        """Return what the command is doing, asking the session until it has ended."""
        if self._final is not None:
            return self._final
        try:
            status = self._session.status_background(self._pid)
        except jail.JailUnavailableError as exc:
            self._final = jail_session.BackgroundStatus(
                running=False, returncode=None, error=str(exc)
            )
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
        except jail.JailUnavailableError as exc:
            if self._final is None:
                self._settle(
                    jail_session.BackgroundStatus(running=False, returncode=None, error=str(exc))
                )
            return str(exc)
        if self._final is None:
            self._settle(
                jail_session.BackgroundStatus(
                    running=False, returncode=stopped.returncode, error=""
                )
            )
        survivors = stopped.survivors | self._session.sweep_for(self._pid, self._before)
        return jail.survivors_message(survivors) if survivors else ""

    def _settle(self, status: jail_session.BackgroundStatus) -> None:
        """Keep the terminal status and persist its exit code."""
        self._final = status
        if status.returncode is not None:
            _write_outcome(self._outcome_dir, status.returncode)


def start_in_jail(
    policy: kinds.JailPolicy,
    *,
    outcome_dir: pathlib.Path,
    session_net: jail.SessionNetwork | None = None,
) -> BackgroundJob | LocalJob:
    """Spawn the policy's argv in the sandbox and return without waiting.

    The same policy, launcher, confinement and network as `run_in_jail`. Nothing of
    the command's own output is captured, so its argv must redirect it; only the
    launcher's result JSON and stderr land in the outcome dir, which lets the exit
    code outlive the turn. The sweep spares the job while it lives and `stop`
    takes its whole group down.

    Args:
        policy: The jail policy.
        outcome_dir: Where the exit code is written.
        session_net: The run's session network, when one exists.

    Returns:
        The job; a `LocalJob` under the `none` isolation.

    Raises:
        JailBinaryError: The launcher is missing or cannot be executed.
        JailUnavailableError: The policy needs a session network and none was wired.
    """
    paths.mkdir_for_real_user(outcome_dir)
    if policy.isolation == "none":
        with jail.sweep_lock:
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
            jail.live_launchers.add(proc.pid)
        return LocalJob(proc, outcome_dir)
    binary = jail.require_jail_binary()
    join_args, join_fds = jail.network_join_args(policy, session_net)
    spec = json.dumps(jail.policy_spec(policy))
    jail.become_subreaper()
    result = (outcome_dir / _RESULT_NAME).open("wb")
    errors = (outcome_dir / _LAUNCHER_ERR_NAME).open("wb")
    try:
        launcher = jail.spawn_launcher(
            binary,
            join_args,
            stdin=subprocess.PIPE,
            stdout=result,
            stderr=errors,
            pass_fds=join_fds,
            die_with_agent=False,
        )
    finally:
        result.close()
        errors.close()
    assert launcher.stdin is not None
    with contextlib.suppress(OSError):
        launcher.stdin.write(spec.encode())
    launcher.stdin.close()
    return BackgroundJob(launcher, outcome_dir)
