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
import shlex
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from agent6.sandbox.jail import BackgroundJob, JailUnavailableError, start_in_jail
from agent6.types import JailPolicy

# The command's own output goes to a file both sides can read: the jail gets
# the LOG directory read-write, and `exec` applies the redirect to the whole
# command. argv values ride as positional parameters, never as shell text.
#
# The log dir is a CHILD of the shell dir, and the only thing granted. The
# launcher's result and the command's identity live in the shell dir itself,
# out of the command's reach: a command that can rewrite its own exit code and
# its own name is not an audit trail, it is a suggestion box. (Proven: a
# command that exited 42 reported "exited 0: npm test (all green)".)
_LOG_DIR = "log"
_LOG_NAME = "out.log"
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
    job: BackgroundJob
    stopped: bool = False


class BackgroundShells:
    """The run's background commands. Not thread-safe: one loop drives it."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._shells: dict[str, _Shell] = {}
        self._seq = 0

    def start(self, argv: tuple[str, ...], policy_for: PolicyFor) -> ShellView:
        self._seq += 1
        shell_id = f"bg{self._seq}"
        shell_dir = self._root / shell_id
        log_dir = shell_dir / _LOG_DIR
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / _LOG_NAME).touch()
        (shell_dir / _META_NAME).write_text(
            json.dumps({"id": shell_id, "command": shlex.join(argv)}), encoding="utf-8"
        )
        wrapped = ("/bin/sh", "-c", _REDIRECT, str(log_dir), *argv)
        try:
            job = start_in_jail(policy_for(wrapped, (log_dir,)), outcome_dir=shell_dir)
        except (JailUnavailableError, OSError) as exc:
            raise BackgroundError(f"could not start a background command: {exc}") from exc
        shell = _Shell(id=shell_id, command=shlex.join(argv), dir=shell_dir, job=job)
        self._shells[shell_id] = shell
        return self._view(shell)

    def roster(self) -> list[ShellView]:
        """Every background command this run started, live or not."""
        return [self._view(s) for s in self._shells.values()]

    def read(self, shell_id: str, *, tail_lines: int) -> tuple[ShellView, str]:
        shell = self._get(shell_id)
        text = ""
        try:
            text = (shell.dir / _LOG_DIR / _LOG_NAME).read_text(errors="replace")
        except OSError as exc:
            return self._view(shell), f"(output unreadable: {exc})"
        lines = text.splitlines()
        if len(lines) > tail_lines:
            lines = [f"... {len(lines) - tail_lines} earlier lines ...", *lines[-tail_lines:]]
        return self._view(shell), "\n".join(lines)

    def stop(self, shell_id: str) -> ShellView:
        shell = self._get(shell_id)
        shell.job.stop()
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
            shell.job.stop()
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
