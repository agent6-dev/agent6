# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Detached commands: start one, read what it printed, stop it.

A background command outlives the tool call that started it but never the run:
`stop_all` at dispatcher close takes down whatever is still alive.

The launcher owns capture and completion. The host copies bytes into its own logs
and records observed outcomes for surfaces that replay the session.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import pathlib
import shlex
import time
from collections.abc import Callable

from agent6 import kinds, paths, terminal
from agent6.sandbox import jail, jail_jobs, jail_session
from agent6.sessions import process

# Host records are not granted to commands. Reads and writes retain the opened descriptor.
_LOG_ROOT = "logs"
_LOG_NAME = "out.log"
# Tail rendering reads at most 1 MiB from the host copy.
_TAIL_BYTES = 1 << 20
# Written at start so `/shells` and a dashboard read the roster off disk without the dispatcher.
SHELLS_DIR = "shells"  # under the session dir
_META_NAME = "meta.json"


# (argv, extra read-write paths) -> the sandbox policy; the dispatcher owns policy construction.
PolicyFor = Callable[[tuple[str, ...], tuple[pathlib.Path, ...]], kinds.JailPolicy]
# The network a policy's command joins: the run's for `session`, None for every other mode.
NetFor = Callable[[kinds.JailPolicy], jail.SessionNetwork | None]


class BackgroundError(Exception):
    """A background command could not be started, or its id is unknown."""


@dataclasses.dataclass(frozen=True, slots=True)
class ShellView:
    """One background command as a caller sees it.

    Attributes:
        id: The `bg<N>` id.
        command: The command as shell text.
        state: "running", "exited", "stopped", "died" or "stop failed".
        returncode: The exit code, when known.
        detail: Why the fate is unknown, cleanup failed or output is incomplete, else "".
    """

    id: str
    command: str
    state: str
    returncode: int | None
    detail: str

    def line(self) -> str:
        """Return the roster line."""
        code = "" if self.returncode is None else f" (exit {self.returncode})"
        detail = f" -- {self.detail}" if self.detail else ""
        return f"[{self.id}] {self.state}{code}: {self.command}{detail}"


@dataclasses.dataclass(slots=True)
class _Shell:
    id: str
    command: str
    dir: pathlib.Path
    job: jail_jobs.SessionJob
    # The host record's descriptor is never selected again by path.
    log_fd: int
    stopped: bool = False
    # Why a stop could not be confirmed, "" when the command is gone.
    stop_error: str = ""
    output_error: str = ""


def _seq_of(name: str) -> int:
    """Return the N in a `bg<N>` directory name, 0 for any other name."""
    return int(name[2:]) if name.startswith("bg") and name[2:].isdigit() else 0


def _highest_shell_seq(root: pathlib.Path) -> int:
    """Return the largest `bg<N>` already recorded under the root, or 0.

    Both `<root>/bg<N>` and `<root>/logs/bg<N>` are scanned: an execution that died between
    creating them leaves only one behind.

    Args:
        root: The shells dir under the session dir.

    Returns:
        The largest N, or 0.
    """
    highest = 0
    for directory in (root, root / _LOG_ROOT):
        with contextlib.suppress(OSError):
            for entry in directory.iterdir():
                highest = max(highest, _seq_of(entry.name))
    return highest


class BackgroundShells:
    """The run's background commands. Not thread-safe: one loop drives it.

    Args:
        root: The shells dir under the session dir.
    """

    def __init__(self, root: pathlib.Path) -> None:
        self._root = root
        self._shells: dict[str, _Shell] = {}
        # A resumed run reuses the session dir, and two commands never share a log.
        self._seq = _highest_shell_seq(root)
        # The ids an earlier execution issued: its end stopped every one of them.
        self._earlier = self._seq
        # Copied logs live on the host, outside the command's grants.
        self.log_root = root / _LOG_ROOT
        paths.mkdir_for_real_user(self.log_root)

    def start(
        self,
        argv: tuple[str, ...],
        policy_for: PolicyFor,
        *,
        session: jail_session.JailSession | None = None,
        net_for: NetFor | None = None,
    ) -> ShellView:
        """Start a command detached.

        Args:
            argv: The command.
            policy_for: Builds the sandbox policy for the original argv.
            session: The run's jail session; the command runs inside it, sharing its netns so
                a later command can reach it. None gives it a launcher of its own.
            net_for: The network a launcher of its own joins for the built policy; None joins
                none, which a `session` policy refuses.

        Returns:
            The command as registered.

        Raises:
            BackgroundError: The command could not be started or recorded.
        """
        self._seq += 1
        shell_id = f"bg{self._seq}"
        shell_dir = self._root / shell_id
        paths.mkdir_for_real_user(shell_dir)
        log_fd = self._open_log(shell_id)
        job: jail_jobs.SessionJob
        try:
            policy = policy_for(argv, ())
            job = jail_jobs.start_in_jail(
                policy,
                outcome_dir=shell_dir,
                session=session,
                session_net=None if net_for is None else net_for(policy),
            )
        except (jail.JailUnavailableError, OSError) as exc:
            os.close(log_fd)
            raise BackgroundError(f"could not start a background command: {exc}") from exc
        shell = _Shell(id=shell_id, command=shlex.join(argv), dir=shell_dir, job=job, log_fd=log_fd)
        return self._register(shell)

    def adopt(
        self, handoff: kinds.BackgroundHandoff, *, session: jail_session.JailSession
    ) -> ShellView:
        """Register a running command the launcher handed back, log and all.

        A run_command that outlived its check-in becomes an ordinary background job here.

        Args:
            handoff: The launcher's record of the command.
            session: The run's jail session the command runs in.

        Returns:
            The command as registered.

        Raises:
            BackgroundError: The log could not be opened or the record written; the command is
                stopped first.
        """
        self._seq += 1
        shell_id = f"bg{self._seq}"
        shell_dir = self._root / shell_id
        paths.mkdir_for_real_user(shell_dir)
        job = jail_jobs.SessionJob(session, handoff.handle, shell_dir)
        command = shlex.join(handoff.argv)
        # Only the host record is opened; the handle selects the launcher-owned capture.
        try:
            log_fd = self._open_log(shell_id)
        except (OSError, BackgroundError) as exc:
            rollback_error = self._rollback(
                _Shell(
                    id=shell_id,
                    command=command,
                    dir=shell_dir,
                    job=job,
                    log_fd=-1,
                    stopped=True,
                )
            )
            detail = f"; cleanup failed: {rollback_error}" if rollback_error else ""
            raise BackgroundError(
                f"could not open the handed-back command's log: {exc}{detail}"
            ) from exc
        shell = _Shell(id=shell_id, command=command, dir=shell_dir, job=job, log_fd=log_fd)
        return self._register(shell)

    def _register(self, shell: _Shell) -> ShellView:
        """Record and expose the shell, or stop it when the record cannot be written.

        Args:
            shell: The command just started or adopted.

        Returns:
            The shell's view.

        Raises:
            BackgroundError: The record could not be written.
        """
        meta = shell.dir / _META_NAME
        try:
            meta.write_text(
                json.dumps(
                    {
                        "id": shell.id,
                        "command": shell.command,
                    }
                ),
                encoding="utf-8",
            )
        except OSError as exc:
            rollback_error = self._rollback(shell)
            with contextlib.suppress(OSError):
                meta.unlink()
            detail = f"; cleanup failed: {rollback_error}" if rollback_error else ""
            raise BackgroundError(
                f"could not record background command {shell.id}: {exc}{detail}"
            ) from exc
        self._shells[shell.id] = shell
        return self._view(shell)

    def _rollback(self, shell: _Shell) -> str:
        """Retain ownership until failed registration has stopped, copied and closed.

        Args:
            shell: The command whose registration failed.

        Returns:
            An empty string after cleanup, otherwise the failure kept for teardown to retry.
        """
        self._shells[shell.id] = shell
        try:
            shell.stop_error = shell.job.stop()
            if shell.stop_error:
                return shell.stop_error
            if shell.log_fd >= 0:
                shell.output_error = shell.job.copy_output(shell.log_fd)
            shell.stop_error = shell.job.close()
            if shell.stop_error:
                return shell.stop_error
            log_fd, shell.log_fd = shell.log_fd, -1
            if log_fd >= 0:
                os.close(log_fd)
        except OSError as exc:
            shell.stop_error = f"rollback persistence or closure failed: {exc}"
            return shell.stop_error
        del self._shells[shell.id]
        return ""

    def _open_log(self, shell_id: str) -> int:
        """Create a private host log without following a replaced path.

        Args:
            shell_id: The command's `bg<N>` id.

        Returns:
            The read-write descriptor retained for all copies and tail reads.

        Raises:
            BackgroundError: The log directory already exists.
        """
        root_fd = os.open(
            self.log_root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        )
        try:
            try:
                os.mkdir(shell_id, 0o700, dir_fd=root_fd)
            except FileExistsError as exc:
                raise BackgroundError(
                    f"the background log directory for {shell_id} already exists;"
                    " a command may have created it"
                ) from exc
            dir_fd = os.open(
                shell_id,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=root_fd,
            )
            try:
                return os.open(
                    _LOG_NAME,
                    os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                    0o600,
                    dir_fd=dir_fd,
                )
            finally:
                os.close(dir_fd)
        finally:
            os.close(root_fd)

    def roster(self) -> list[ShellView]:
        """Return every background command this run started, live or not."""
        return [self._view(s) for s in self._shells.values()]

    def settle(self) -> None:
        """Observe every command, which is what writes an ending down.

        Only an observed exit reaches disk, so without this a surface in another process shows
        a command that ended in seconds as maybe-running for the rest of the run. Called at the
        turn boundary.
        """
        for shell in self._shells.values():
            self._view(shell)

    def read(
        self,
        shell_id: str,
        *,
        tail_lines: int,
        wait_s: float = 0.0,
        interrupted: Callable[[], bool] = lambda: False,
    ) -> tuple[ShellView, str]:
        """Return the tail of what the command has printed, optionally after waiting for it.

        Args:
            shell_id: The command's id.
            tail_lines: The most lines returned.
            wait_s: How long to wait for the command to end first; it returns as soon as the
                command ends, so one call replaces a dozen polls.
            interrupted: Cuts the wait short; the operator's Stop is polled at a turn boundary,
                which a tool call in flight never reaches.

        Returns:
            The command's view and its output, with a note when the tail cap or the line cap cut it.

        Raises:
            BackgroundError: The id is unknown.
        """
        shell = self._get(shell_id)
        if wait_s > 0:
            deadline = time.monotonic() + wait_s
            # Backs off to 2s: the status probe is a round trip to the launcher.
            pause = 0.1
            while shell.job.status().running and time.monotonic() < deadline:
                if interrupted():
                    break
                time.sleep(min(pause, max(0.0, deadline - time.monotonic())))
                pause = min(pause * 1.5, 2.0)
        view = self._view(shell)
        try:
            size = os.lseek(shell.log_fd, 0, os.SEEK_END)
            start = max(size - _TAIL_BYTES, 0)
            text = os.pread(shell.log_fd, size - start, start).decode(errors="replace")
            text = terminal.scrub_terminal_escapes(text)
            cut_mid_line = start > 0 and os.pread(shell.log_fd, 1, start - 1) != b"\n"
        except OSError as exc:
            return self._view(shell), f"(output unreadable: {exc})"
        lines = text.splitlines()
        if cut_mid_line:
            lines = lines[1:]  # the cap cut a line: its remainder is not a line
        notes: list[str] = [shell.output_error] if shell.output_error else []
        if start > 0:
            notes.append(
                f"earlier output cut at the {_TAIL_BYTES} byte tail cap ({size} bytes total)"
            )
        if len(lines) > tail_lines:
            notes.append(f"{len(lines) - tail_lines} earlier lines")
            lines = lines[-tail_lines:]
        if notes:
            lines = [f"... {'; '.join(notes)} ...", *lines]
        return view, "\n".join(lines)

    def stop(self, shell_id: str) -> ShellView:
        """Kill one command and sweep what it left behind.

        Args:
            shell_id: The command's id.

        Returns:
            The command's view after the stop.

        Raises:
            BackgroundError: The id is unknown.
        """
        shell = self._get(shell_id)
        self._stop(shell)
        return self._view(shell)

    def stop_all(self) -> list[ShellView]:
        """Kill everything this run started; idempotent and safe at teardown.

        Every shell is stopped, not just the live ones: a command that exited can have left a
        detached child behind.

        Returns:
            The commands that were still running.

        Raises:
            BackgroundError: A host record could not be persisted; teardown can retry it.
        """
        stopped: list[ShellView] = []
        failures: list[str] = []
        for shell in self._shells.values():
            try:
                if self._stop(shell):
                    stopped.append(self._view(shell))
            except OSError as exc:
                failures.append(f"{shell.id}: {exc}")
        if failures:
            raise BackgroundError("could not persist background output: " + "; ".join(failures))
        for shell in self._shells.values():
            error = shell.job.close()
            if error:
                shell.stop_error = error
            log_fd, shell.log_fd = shell.log_fd, -1
            if log_fd >= 0:
                os.close(log_fd)
        return stopped

    def _stop(self, shell: _Shell) -> bool:
        """Kill the shell and sweep what it left behind.

        Args:
            shell: The registered command.

        Returns:
            Whether it was still running; a command that had exited keeps its own ending.
        """
        try:
            was_running = shell.job.status().running
        finally:
            # Host persistence failure must not leave the job's descendants running.
            shell.stop_error = shell.job.stop()
        if shell.log_fd >= 0:
            shell.output_error = shell.job.copy_output(shell.log_fd)
        shell.stopped = shell.stopped or was_running
        return was_running

    def _get(self, shell_id: str) -> _Shell:
        shell = self._shells.get(shell_id)
        if shell is None:
            if 0 < _seq_of(shell_id) <= self._earlier:
                raise BackgroundError(
                    f"{shell_id} was started by an earlier execution of this run and ended with it"
                )
            known = ", ".join(self._shells) or "none"
            raise BackgroundError(f"no background command {shell_id!r} (started this run: {known})")
        return shell

    def _view(self, shell: _Shell) -> ShellView:
        if shell.log_fd >= 0:
            shell.output_error = shell.job.copy_output(shell.log_fd)
        status = shell.job.status()
        detail = "; ".join(
            dict.fromkeys(filter(None, (shell.stop_error, status.error, status.capture_error)))
        )
        # A stop that could not be confirmed outranks every other word: the command may be running.
        if shell.stop_error:
            return ShellView(shell.id, shell.command, "stop failed", status.returncode, detail)
        if status.running:
            return ShellView(shell.id, shell.command, "running", None, detail)
        if shell.stopped:
            return ShellView(shell.id, shell.command, "stopped", status.returncode, detail)
        # No exit code from the launcher means the fate is unknown, not a clean exit.
        if status.returncode is None:
            return ShellView(shell.id, shell.command, "died", None, detail)
        return ShellView(shell.id, shell.command, "exited", status.returncode, detail)


def shells_text(session_dir: pathlib.Path) -> str:
    """Return the roster as one block for a text view, or a line saying there is none."""
    return "\n".join(roster_from_dir(session_dir / SHELLS_DIR)) or "no background commands this run"


def shell_host_processes(root: pathlib.Path) -> list[process.ProcessIdentity]:
    """Return the live host processes the run's background commands recorded.

    Only historical host-process records carry these identities. Supervisor-owned
    jobs record no command pid. A stale record cannot target a recycled pid.

    Args:
        root: The run's shells dir.

    Returns:
        Each live process's pid and start time.
    """
    if not root.is_dir():
        return []
    processes: list[process.ProcessIdentity] = []
    try:
        directories = sorted(root.iterdir())
    except OSError:
        return []
    for d in directories:
        try:
            meta = json.loads((d / _META_NAME).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        pid = meta.get("pid") if isinstance(meta, dict) else None
        started = meta.get("pid_start") if isinstance(meta, dict) else None
        if type(pid) is int and pid > 0 and isinstance(started, str):
            identity = (pid, started)
            if process.process_is_alive(identity):
                processes.append(identity)
    return processes


def roster_from_dir(root: pathlib.Path) -> list[str]:
    """Return the run's background commands as lines, read off disk.

    For surfaces in another process: liveness needs the owning process, so this reports what
    each command was and how it ended, and says when it cannot tell.

    Args:
        root: The run's shells dir.

    Returns:
        One line per recorded command, in `bg<N>` order.
    """
    if not root.is_dir():
        return []
    lines: list[str] = []
    # By sequence: as text, bg10 sorts ahead of bg2.
    try:
        directories = sorted(root.iterdir(), key=lambda p: (_seq_of(p.name), p.name))
    except OSError:
        return []
    for d in directories:
        if not d.is_dir():
            continue
        try:
            meta = json.loads((d / _META_NAME).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(meta, dict):
            continue
        command = str(meta.get("command", ""))
        try:
            raw = (d / "result.json").read_text(errors="replace").strip()
        except OSError:
            raw = ""
        if not raw:
            lines.append(f"[{d.name}] still running (or the run that owns it ended): {command}")
            continue
        record: object = None
        with contextlib.suppress(ValueError, IndexError):
            record = json.loads(raw.splitlines()[-1])
        if not isinstance(record, dict):
            lines.append(f"[{d.name}] ended without a result: {command}")
            continue
        errors = "; ".join(
            dict.fromkeys(str(record[key]) for key in ("error", "capture_error") if record.get(key))
        )
        code = record.get("returncode")
        if isinstance(code, int):
            detail = f" -- {errors}" if errors else ""
            lines.append(f"[{d.name}] exited {code}: {command}{detail}")
        elif record.get("stopped"):
            # Historical records can carry a stop without an observed exit code.
            lines.append(f"[{d.name}] stopped: {command}")
        else:
            ending = f"outcome unknown ({errors})" if errors else "ended without a result"
            lines.append(f"[{d.name}] {ending}: {command}")
    return lines
