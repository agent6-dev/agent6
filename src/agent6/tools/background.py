# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Detached commands: start one, read what it printed, stop it.

A background command outlives the tool call that started it but never the run:
`stop_all` at dispatcher close takes down whatever is still alive.

Every state a caller can see is derived from the live process and the files on
disk, never from a cached guess, so a command that dies on its own reads as
dead the next time anyone looks. Nothing here blocks: there is no wait.
"""

from __future__ import annotations

import json
import os
import shlex
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from agent6.sandbox.jail import (
    BackgroundJob,
    JailSession,
    JailUnavailableError,
    SessionJob,
    start_in_jail,
)
from agent6.types import JailPolicy

# The command's own output goes to a file both sides can read: the jail gets
# the LOG directory read-write, and `exec` applies the redirect to the whole
# command. argv values ride as positional parameters, never as shell text.
#
# Every log lives under ONE root, `<root>/logs/<id>/`, and that root is the
# only thing granted. The run's jail session grants it when it opens, before
# any background command exists, which a per-shell grant cannot do. The cost is
# that a run's background commands share the root and can write each other's
# logs; the launcher's result and each command's identity stay OUTSIDE it, so
# what a command cannot do is rewrite its own exit code or its own name. (A
# command that exited 42 once reported "exited 0: npm test (all green)".)
#
# The grant includes MakeSym, so the agent NEVER resolves that path again: it
# reads through the descriptor it opened before the jail existed. Opening
# `out.log` by name, outside the jail and as the operator, let a command unlink
# it, symlink it at the operator's secrets, and have the next `read_background`
# hand them to the model -- or point it at a FIFO and hang the loop forever.
_LOG_ROOT = "logs"
_LOG_NAME = "out.log"
# How much of a log a read considers. A build can print gigabytes; only the
# tail is ever returned, so only the tail is read.
_TAIL_BYTES = 1 << 20
# What a surface needs that the run's own memory holds: the command, and when.
# Written at start so `/shells` and any dashboard widget read the roster off
# disk like every other run state, rather than needing the dispatcher.
_META_NAME = "meta.json"
_REDIRECT = f'exec >"$0/{_LOG_NAME}" 2>&1; exec "$@"'


# (argv, extra read-write paths) -> the sandbox policy to run it under. The
# dispatcher owns policy construction; this module only says what it needs.
PolicyFor = Callable[[tuple[str, ...], tuple[Path, ...]], JailPolicy]


class BackgroundError(Exception):
    """A background command could not be started, or its id is unknown."""


@dataclass(frozen=True, slots=True)
class ShellView:
    """One background command as a caller sees it."""

    id: str
    command: str
    state: str
    returncode: int | None
    detail: str

    def line(self) -> str:
        code = "" if self.returncode is None else f" (exit {self.returncode})"
        detail = f" -- {self.detail}" if self.detail else ""
        return f"[{self.id}] {self.state}{code}: {self.command}{detail}"


@dataclass(slots=True)
class _Shell:
    id: str
    command: str
    dir: Path
    job: BackgroundJob | SessionJob
    # Opened before the command could exist, held for the run: the one handle
    # to its output that no jailed process can redirect.
    log_fd: int
    stopped: bool = False
    # Why a stop could not be confirmed, "" when the command is gone.
    stop_error: str = ""


class BackgroundShells:
    """The run's background commands. Not thread-safe: one loop drives it."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._shells: dict[str, _Shell] = {}
        self._seq = 0
        # Eagerly: the run's jail session grants this path when it opens, and a
        # mount source has to exist by then.
        self.log_root = root / _LOG_ROOT
        self.log_root.mkdir(parents=True, exist_ok=True)

    def start(
        self, argv: tuple[str, ...], policy_for: PolicyFor, *, session: JailSession | None = None
    ) -> ShellView:
        """Start *argv* detached. With a *session*, it runs in the run's jail
        process, so it shares that netns and a later command can reach it;
        without one it gets a launcher of its own."""
        self._seq += 1
        shell_id = f"bg{self._seq}"
        shell_dir = self._root / shell_id
        log_dir = self.log_root / shell_id
        shell_dir.mkdir(parents=True, exist_ok=True)
        log_fd = self._open_log(shell_id)
        wrapped = ("/bin/sh", "-c", _REDIRECT, str(log_dir), *argv)
        policy = policy_for(wrapped, (log_dir,))
        job: BackgroundJob | SessionJob
        try:
            if session is None:
                job = start_in_jail(policy, outcome_dir=shell_dir)
            else:
                # The session is already confined; only the env comes from the
                # policy. Its grant of the log root is what makes the redirect
                # land.
                job = SessionJob(
                    session, session.start_background(wrapped, env=policy.env), shell_dir
                )
        except (JailUnavailableError, OSError) as exc:
            os.close(log_fd)
            raise BackgroundError(f"could not start a background command: {exc}") from exc
        # AFTER the start: this file is the whole roster for a surface in
        # another process, so writing it first listed a command that never
        # started as "still running" -- while this run's own roster did not
        # have it and read_background denied the id existed.
        (shell_dir / _META_NAME).write_text(
            json.dumps({"id": shell_id, "command": shlex.join(argv)}), encoding="utf-8"
        )
        shell = _Shell(id=shell_id, command=shlex.join(argv), dir=shell_dir, job=job, log_fd=log_fd)
        self._shells[shell_id] = shell
        return self._view(shell)

    def _open_log(self, shell_id: str) -> int:
        """Create this command's log directory and its log, and hand back the
        one descriptor every later read goes through.

        Every step is relative to a descriptor on the log root, never by path:
        that root is granted read-write to every command in the run, so one can
        plant `<log_root>/bg<N>` as a symlink, and `mkdir(exist_ok=True)` (like
        its `is_dir()` check) FOLLOWS it -- which had the agent, unconfined and
        outside the jail, create the log inside a directory a command named.
        O_NOFOLLOW on the leaf never covered the path above it. Creating the
        directory rather than accepting one also means a planted name fails
        here instead of quietly becoming this command's log.

        O_EXCL: we create the file, so it is a regular file we own. O_CLOEXEC:
        no child inherits the handle. The command's own `exec >` opens the same
        path from inside the jail and lands on this inode.
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
                    os.O_RDONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                    0o600,
                    dir_fd=dir_fd,
                )
            finally:
                os.close(dir_fd)
        finally:
            os.close(root_fd)

    def roster(self) -> list[ShellView]:
        """Every background command this run started, live or not."""
        return [self._view(s) for s in self._shells.values()]

    def read(self, shell_id: str, *, tail_lines: int) -> tuple[ShellView, str]:
        shell = self._get(shell_id)
        try:
            size = os.lseek(shell.log_fd, 0, os.SEEK_END)
            start = max(size - _TAIL_BYTES, 0)
            text = os.pread(shell.log_fd, size - start, start).decode(errors="replace")
        except OSError as exc:
            return self._view(shell), f"(output unreadable: {exc})"
        lines = text.splitlines()
        if len(lines) > tail_lines:
            lines = [f"... {len(lines) - tail_lines} earlier lines ...", *lines[-tail_lines:]]
        return self._view(shell), "\n".join(lines)

    def stop(self, shell_id: str) -> ShellView:
        shell = self._get(shell_id)
        shell.stop_error = shell.job.stop()
        shell.stopped = True
        return self._view(shell)

    def stop_all(self) -> list[ShellView]:
        """Kill everything this run started. Idempotent; safe at teardown.

        Every shell is stopped, not just the live ones: a command that already
        exited can still have left a detached child behind, and stop() is what
        sweeps those. Only the ones that WERE running are reported as stopped.
        """
        stopped: list[ShellView] = []
        for shell in self._shells.values():
            was_running = shell.job.status().running
            shell.stop_error = shell.job.stop()
            if was_running:
                shell.stopped = True
                stopped.append(self._view(shell))
        return stopped

    def _get(self, shell_id: str) -> _Shell:
        shell = self._shells.get(shell_id)
        if shell is None:
            known = ", ".join(self._shells) or "none"
            raise BackgroundError(f"no background command {shell_id!r} (started this run: {known})")
        return shell

    def _view(self, shell: _Shell) -> ShellView:
        status = shell.job.status()
        # A stop that could not be confirmed outranks every other word: the
        # command may well still be running, and "stopped" (or "running", with
        # the reason dropped) hides that the operator's stop did not take.
        if shell.stop_error:
            return ShellView(
                shell.id, shell.command, "stop failed", status.returncode, shell.stop_error
            )
        if status.running:
            return ShellView(shell.id, shell.command, "running", None, "")
        if shell.stopped:
            return ShellView(shell.id, shell.command, "stopped", status.returncode, "")
        # Exited on its own. A launcher that reported no exit code means the
        # command's fate is unknown -- say so rather than imply a clean exit.
        if status.returncode is None:
            return ShellView(shell.id, shell.command, "died", None, status.error)
        return ShellView(shell.id, shell.command, "exited", status.returncode, "")


def roster_from_dir(root: Path) -> list[str]:
    """The run's background commands, read off disk.

    For surfaces in another process (`/shells`, a dashboard widget): liveness
    needs the owning process, so this reports what each command WAS and how it
    ended, and says plainly when it cannot tell.
    """
    if not root.is_dir():
        return []
    lines: list[str] = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        try:
            meta = json.loads((d / _META_NAME).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        command = str(meta.get("command", ""))
        try:
            raw = (d / "result.json").read_text(errors="replace").strip()
        except OSError:
            raw = ""
        if not raw:
            lines.append(f"[{d.name}] still running (or the run that owns it ended): {command}")
            continue
        try:
            code = int(json.loads(raw.splitlines()[-1])["returncode"])
        except (ValueError, IndexError, KeyError):
            lines.append(f"[{d.name}] ended without a result: {command}")
            continue
        lines.append(f"[{d.name}] exited {code}: {command}")
    return lines
