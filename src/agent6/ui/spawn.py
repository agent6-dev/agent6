# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Find the agent6 executable and spawn it detached.

Shared by every front-end (TUI hub, machines page, web server) so a UI action
shells out to the same CLI a user would run, never doing the work in-process. A
leaf module (only stdlib) so any front-end depends on it without a cycle."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import IO

from agent6.sandbox.jail import keep_out_of_the_sweep
from agent6.sessions.layout import LOGS_NAME


def agent6_exe() -> str:
    """The agent6 executable that launched this TUI (so a spawned child uses the
    same install), falling back to the entry on PATH."""
    argv0 = Path(sys.argv[0])
    if argv0.name.startswith("agent6") and argv0.exists():
        return str(argv0.resolve())
    return shutil.which("agent6") or "agent6"


def spawn_detached_resume(
    cwd: Path,
    session_id: str,
    *,
    steer: str = "",
    preset: str = "",
    config_path: Path | None = None,
) -> str:
    """Fire-and-forget a detached `agent6 resume <session_id>` (new session, no
    stdio) so a run keeps going in the background after the operator detaches.

    A non-empty *steer* rides along as `--steer=TEXT` (the `=` form, so a
    follow-up starting with `-` cannot read as an option): the resume injects
    it as the first steering instruction. Operator-typed text, never LLM output.
    A non-empty *preset* is the `--preset` the leg continues under.

    The caller must have released the run's worker lock first, so the child
    acquires it cleanly. `AGENT6_STREAM_TO_LOG=1` keeps the headless child
    emitting delta events, so a later `agent6 attach` shows its full reasoning,
    not just tool calls. `AGENT6_DETACHED_AWAY=wait` makes the terminal-less
    child WAIT for a front-end at an ask/approval instead of fabricating an empty
    answer (every caller here is a front-end or a detach the operator re-attaches
    to). argv is the agent6 exe + the run id (never LLM output). Returns "" on
    success, else an error message."""
    argv = [agent6_exe(), "resume", session_id]
    if config_path is not None:
        # The overlay the parent ran under; a resume re-applies it only when
        # told, so the spawned continuation matches the run it continues.
        argv[1:1] = ["--config", str(config_path)]
    if preset:
        argv.append(f"--preset={preset}")
    if steer:
        argv.append(f"--steer={steer}")
    try:
        proc = subprocess.Popen(
            argv,
            cwd=str(cwd),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            env={**os.environ, "AGENT6_STREAM_TO_LOG": "1", "AGENT6_DETACHED_AWAY": "wait"},
        )
    except OSError as exc:
        return f"could not spawn background resume: {exc}"
    keep_out_of_the_sweep(proc.pid)
    return ""


# Subcommand groups whose verb is the SECOND argv word ("machine run",
# "sessions prune", "config set"); everything else is a one-word subcommand whose
# next arg is already a value.
_COMMAND_GROUPS = frozenset({"machine", "sessions", "config"})


def subcommand_label(argv: list[str]) -> str:
    """The agent6 subcommand named by *argv*, for diagnostics: "machine run",
    not a bare "machine" (or worse, "run" with the task word attached)."""
    if len(argv) < 2:
        return argv[0]
    label = argv[1]
    if label in _COMMAND_GROUPS and len(argv) > 2 and not argv[2].startswith("-"):
        return f"{label} {argv[2]}"
    return label


def _capture_message(stdout: str, stderr: str) -> str:
    """Captured CLI output as front-end message text. The CLI prefixes its own
    lines with "[agent6] " to stand apart from pass-through git output on a
    console; in a toast every line already comes from agent6, so the prefix is
    dropped."""
    lines = [ln.removeprefix("[agent6] ").strip() for ln in (stdout + "\n" + stderr).splitlines()]
    return "\n".join(ln for ln in lines if ln)


def _child_exit_message(label: str, rc: int | None, captured: str) -> str:
    """What a front-end shows when a spawned child ended before it began: the
    child's own words (its REFUSING / PARKED / ERROR message), or the exit
    code when it said nothing."""
    said = _capture_message("", captured)
    return said or f"agent6 {label} exited {rc} without a word"


def _not_started_message(label: str, timeout_s: float, captured: str) -> str:
    said = _capture_message("", captured)
    return f"agent6 {label} has not started after {timeout_s:.0f}s" + (f":\n{said}" if said else "")


def run_cli_capture(argv: list[str], cwd: Path, *, timeout_s: float = 120.0) -> tuple[bool, str]:
    """Run a quick agent6 subcommand synchronously, capturing its output, and
    return `(ok, message)`. For the fast, foreground CLI ops a front-end drives
    the same way a user would: `sessions merge`, `sessions prune`, `config set`. argv is
    fixed (the agent6 exe + operator-chosen args), never LLM output."""
    try:
        proc = subprocess.run(
            argv,
            cwd=str(cwd),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"failed to run agent6 {subcommand_label(argv)}: {exc}"
    message = _capture_message(proc.stdout, proc.stderr)
    return proc.returncode == 0, message or f"exit {proc.returncode}"


def spawn_and_confirm(
    argv: list[str],
    cwd: Path,
    *,
    started: Callable[[int], bool],
    timeout_s: float = 25.0,
) -> str:
    """Spawn *argv* detached (non-TTY stdio, new session, so the child never
    opens its own TUI) with an early-exit stderr capture: return "" once
    *started(child_pid)* reports the child took ownership of its work, or the
    stderr tail when the child exits nonzero first / nothing happens by the
    timeout. A child that exits 0 without the signal is a clean fast completion.

    The machine-run analogue of `spawn_and_locate`: `machine run` refusals (lock
    held, network refusal, bad bundle) print to stderr and exit nonzero without
    ever starting, which a fire-and-forget spawn (stderr to /dev/null) silently
    swallowed."""
    label = subcommand_label(argv)
    err = tempfile.NamedTemporaryFile(  # noqa: SIM115 - closed in finally
        mode="w+", suffix=".agent6-launch.err", delete=False
    )
    try:
        try:
            proc = subprocess.Popen(
                argv,
                cwd=str(cwd),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=err,
                start_new_session=True,
                # Driven from a hub (web/TUI) over the bridge: approvals and
                # questions WAIT for a front-end instead of being invented by
                # the headless default -- the same away-mode every detached
                # run spawn carries.
                env={**os.environ, "AGENT6_DETACHED_AWAY": "wait"},
            )
        except OSError as exc:
            return f"failed to start agent6 {label}: {exc}"
        keep_out_of_the_sweep(proc.pid)

        def err_tail() -> str:
            return _stderr_tail(err)

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if started(proc.pid):
                return ""
            rc = proc.poll()
            if rc is not None:
                # Recheck once: the signal may have landed in the same instant.
                if started(proc.pid) or rc == 0:
                    return ""
                return _child_exit_message(label, rc, err_tail())
            time.sleep(0.2)
        return _not_started_message(label, timeout_s, err_tail())
    finally:
        # Same lifetime note as spawn_and_locate: the detached child keeps the
        # unlinked-but-open inode as its stderr until it exits.
        err.close()
        Path(err.name).unlink(missing_ok=True)


def _stderr_tail(err: IO[str]) -> str:
    """The last 600 chars of a spawn's captured-stderr temp file."""
    err.flush()
    return Path(err.name).read_text(encoding="utf-8", errors="replace")[-600:]


def _located(list_dirs: Callable[[], list[Path]], before: set[Path]) -> Path | None:
    """The newest dir from *list_dirs* not in *before* whose logs.jsonl exists."""
    for d in list_dirs():
        if d not in before and (d / LOGS_NAME).exists():
            return d
    return None


def spawn_and_locate(
    argv: list[str],
    cwd: Path,
    *,
    before: set[Path],
    list_dirs: Callable[[], list[Path]],
    env: dict[str, str] | None = None,
    timeout_s: float = 25.0,
) -> tuple[Path | None, str]:
    """Spawn *argv* detached, then poll *list_dirs* for a NEW dir (not in *before*)
    whose `logs.jsonl` exists, and return `(dir, "")` so the caller can hand it
    to the dashboard. If the child exits before producing one (no git repo, bad
    config, ...), surface its stderr tail instead of waiting out the timeout;
    return `(None, message)` on any failure.

    The shared launch+watch path behind both "start a run" (hub) and "create a
    machine" (machines page): spawn the same CLI a user would, then watch the new
    log dir live."""
    label = subcommand_label(argv)
    err = tempfile.NamedTemporaryFile(  # noqa: SIM115 - closed in finally
        mode="w+", suffix=".agent6-launch.err", delete=False
    )
    try:
        try:
            proc = subprocess.Popen(
                argv,
                cwd=str(cwd),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=err,
                start_new_session=True,
                env=env,
            )
        except OSError as exc:
            return None, f"failed to start agent6 {label}: {exc}"
        keep_out_of_the_sweep(proc.pid)

        def err_tail() -> str:
            return _stderr_tail(err)

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            found = _located(list_dirs, before)
            if found is not None:
                return found, ""
            if proc.poll() is not None:
                # Child exited without a log dir; surface why (recheck once in case
                # the dir landed in the same instant the process exited).
                found = _located(list_dirs, before)
                if found is not None:
                    return found, ""
                return None, _child_exit_message(label, proc.returncode, err_tail())
            time.sleep(0.2)
        return None, _not_started_message(label, timeout_s, err_tail())
    finally:
        # On the success/timeout paths the child is a detached process still
        # holding this file as its stderr; closing + unlinking is intentional (its
        # real output is logs.jsonl, this capture only feeds the early-exit /
        # timeout diagnostic). On Linux the unlinked-but-open inode is freed when
        # the child exits.
        err.close()
        Path(err.name).unlink(missing_ok=True)
