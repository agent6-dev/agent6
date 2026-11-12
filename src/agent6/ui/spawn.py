# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Find the agent6 executable and spawn it detached.

Every front-end shells out to the same CLI an operator would run, never doing
the work in-process.
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from typing import IO

from agent6 import directive, kinds, paths
from agent6.models import validate
from agent6.sandbox import jail
from agent6.sessions import id, layout, lock, process
from agent6.viewmodel import listing


def agent6_exe() -> str:
    """Return the agent6 executable of this install, falling back to the one on PATH."""
    argv0 = pathlib.Path(sys.argv[0])
    if argv0.name.startswith("agent6") and argv0.exists():
        return str(argv0.resolve())
    # Under `python -m agent6.ui.tui` the install's binary sits beside its interpreter.
    beside = pathlib.Path(sys.executable).with_name("agent6")
    if beside.exists():
        return str(beside.resolve())
    return shutil.which("agent6") or "agent6"


def agent6_argv(config_paths: tuple[pathlib.Path, ...]) -> list[str]:
    """Return the argv head every spawn starts from: the exe, plus the front-end's `--config`."""
    argv = [agent6_exe()]
    for path in config_paths:
        argv += ["--config", str(path)]
    return argv


# Work a front-end drives over the bridge: approvals wait for it, and a run streams its deltas.
DETACHED_AWAY_ENV: dict[str, str] = {"AGENT6_DETACHED_AWAY": "wait"}
DETACHED_RUN_ENV: dict[str, str] = {"AGENT6_STREAM_TO_LOG": "1", **DETACHED_AWAY_ENV}


@dataclasses.dataclass(frozen=True, slots=True)
class RunOptions:
    """The run options a composer offers beside the mode, preset and model.

    Attributes:
        worktree: Work in a new git worktree (`run --worktree`); run mode only.
        seed_from: The session whose context seeds the run (`run --from`); with it a task
            may be empty, and a plan source becomes the task.
        max_usd: The metered spend ceiling (`--max-usd`), or None for the config's.
    """

    worktree: bool = False
    seed_from: str = ""
    max_usd: float | None = None

    def argv(self) -> list[str]:
        """Return the flags, in the order the help lists them."""
        flags: list[str] = []
        if self.worktree:
            flags.append("--worktree")
        if self.seed_from:
            flags += ["--from", self.seed_from]
        if self.max_usd is not None:
            flags += ["--max-usd", str(self.max_usd)]
        return flags


def spawn_new_work(  # noqa: PLR0911
    cwd: pathlib.Path,
    mode: str,
    task: str,
    *,
    preset: str = "",
    model: str = "",
    options: RunOptions | None = None,
    config_paths: tuple[pathlib.Path, ...] = (),
) -> tuple[pathlib.Path | None, str]:
    """Start a session detached from a hub.

    In run mode a `/parallel` message fans out one detached `run --parallel` per
    segment: a malformed directive refuses before any spawn, and any segment's
    failure fails the whole message while naming the lanes already running. A plain
    run into a checkout another run is driving refuses at once; a group's lanes
    clone the checkout and a worktree run has a checkout of its own, so neither
    takes that lock.

    Args:
        cwd: The checkout.
        mode: The session mode.
        task: The task text; empty only with a seed session.
        preset: The `--preset` value, when any.
        model: The `--model` value, when any.
        options: The run options beside the preset and model.
        config_paths: The front-end's explicit config, when any.

    Returns:
        The new session's dir (the first lane's for a group) and "", or None and why.
    """
    options = options or RunOptions()
    if (problem := _refusal(mode, task, options)) is not None:
        return None, problem
    segments = None
    if mode == "run":
        try:
            segments = directive.parse_directive(task)
        except directive.DirectiveError as exc:
            return None, str(exc)
    if segments is None:
        if (
            mode == "run"
            and not options.worktree
            and lock.repo_writer_held(state := paths.state_dir(cwd), cwd)
        ):
            holder = lock.repo_writer_holder(state, cwd) or "another run"
            return None, (
                f"run {holder} is already driving this checkout; steer it with this task"
                " (or /parallel it) from its run view, start this one in a worktree, or wait"
                " for it to finish"
            )
        return _spawn_run(
            cwd,
            mode,
            task,
            preset=preset,
            model=model,
            spec="",
            options=options,
            config_paths=config_paths,
        )
    if options.worktree or options.seed_from:
        return None, "/parallel lanes clone the checkout; drop the worktree and the seed session"
    refusal = validate.directive_model_refusal(
        cwd, segments, config_paths, preset=preset, model=model
    )
    if refusal is not None:
        return None, refusal
    first: pathlib.Path | None = None
    lines: list[str] = []
    failed = False
    for i, seg in enumerate(segments, 1):
        session_dir, err = _spawn_run(
            cwd,
            "run",
            seg.task,
            preset=preset,
            model=model,
            spec=seg.spec or "1",
            options=options,
            config_paths=config_paths,
        )
        if session_dir is None:
            lines.append(f"lane {i} ({seg.task}): {err}")
            failed = True
            continue
        lines.append(f"lane {i} ({seg.task}): running as {session_dir.name}")
        if first is None:
            first = session_dir
    if failed:
        # A partial failure must not vanish behind a surviving lane.
        return None, "\n".join(lines)
    assert first is not None
    return first, ""


def _refusal(mode: str, task: str, options: RunOptions) -> str | None:
    """Return why the request cannot start, before anything is spawned; None when it can."""
    if mode not in kinds.OPERATOR_MODES:
        return f"unknown mode {mode!r}"
    if not task.strip() and not options.seed_from:
        return "empty task"
    if (options.worktree or options.seed_from) and mode != "run":
        return f"a worktree and a seed session are run options; this is {mode}"
    # The child refuses this too; refusing here answers before anything is spawned.
    return directive.steer_problem(task)


def _spawn_run(
    cwd: pathlib.Path,
    mode: str,
    task: str,
    *,
    preset: str,
    model: str,
    spec: str,
    options: RunOptions,
    config_paths: tuple[pathlib.Path, ...],
) -> tuple[pathlib.Path | None, str]:
    """Spawn one detached session and locate it by its new session dir.

    Args:
        cwd: The checkout.
        mode: The session mode.
        task: The task text, passed after `--` so it never reads as a flag; "" passes none.
        preset: The `--preset` value, when any.
        model: The `--model` value, when any.
        spec: The `--parallel` spec of a lane; "" for a plain run.
        options: The run options beside the preset and model.
        config_paths: The front-end's explicit config, when any.

    Returns:
        The session dir and "", or None and why.
    """
    argv = [*agent6_argv(config_paths), mode]
    if preset:
        argv += ["--preset", preset]
    if model:
        argv += ["--model", model]
    if spec:
        argv += ["--parallel", spec]
    argv += options.argv()
    if task.strip():
        argv += ["--", task]
    state = paths.state_dir(cwd)
    return spawn_and_locate(
        argv,
        cwd,
        before=set(listing.session_dirs(state)),
        list_dirs=lambda: listing.session_dirs(state),
        env={**os.environ, **DETACHED_RUN_ENV},
    )


def spawn_detached_resume(
    cwd: pathlib.Path,
    session_id: str,
    *,
    steer: str = "",
    preset: str = "",
    model: str = "",
    config_paths: tuple[pathlib.Path, ...] = (),
    flags: Sequence[str] = (),
) -> str:
    """Start a detached `agent6 resume` so a run keeps going after the operator detaches.

    The child owns the run once its pid is the run's worker pid, which `resume`
    writes after its preflight; a child that exits before that hands back its own
    refusal. The caller must have released the run's worker lock first. Every argv
    word is the operator's, never model output.

    Args:
        cwd: The checkout whose state dir holds the session: a fork's origin, never
            its worktree.
        session_id: The session to resume.
        steer: The first steering instruction, passed as `--steer=TEXT`; a malformed
            directive refuses here with the message the child would print.
        preset: The `--preset` the execution continues under, when any.
        model: The `--model` the execution continues under, when any.
        config_paths: The front-end's explicit config, when any.
        flags: Further `resume` options.

    Returns:
        "" once the child owns the run, else why it did not.
    """
    if steer and (problem := directive.steer_problem(steer)) is not None:
        return problem
    try:
        session_dir = id.resolve_session(paths.state_dir(cwd), session_id).session_dir
    except id.SessionIdError as exc:
        return str(exc)
    argv = [*agent6_argv(config_paths), "resume", session_id]
    if preset:
        argv.append(f"--preset={preset}")
    if model:
        argv.append(f"--model={model}")
    if steer:
        argv.append(f"--steer={steer}")
    argv.extend(flags)
    return spawn_and_confirm(
        argv,
        cwd,
        started=lambda pid: process.read_worker_pid(session_dir) == pid,
        extra_env=DETACHED_RUN_ENV,
    )


# Subcommand groups whose verb is the second argv word; every other subcommand is one word.
_COMMAND_GROUPS = frozenset({"machine", "sessions", "config"})


def subcommand_label(argv: list[str]) -> str:
    """Return the subcommand the argv names, for diagnostics: "machine run", not "machine"."""
    if "-m" in argv and argv.index("-m") + 1 < len(argv):
        # A `python -m agent6.ui.cli.lane_agent` bootstrap reads as "lane agent".
        return argv[argv.index("-m") + 1].rsplit(".", 1)[-1].replace("_", " ")
    if len(argv) < 2:
        return argv[0]
    label = argv[1]
    if label in _COMMAND_GROUPS and len(argv) > 2 and not argv[2].startswith("-"):
        return f"{label} {argv[2]}"
    return label


def capture_message(*streams: str) -> str:
    """Return captured CLI output as message text, its `[agent6] ` and `ERROR: ` marks dropped."""
    lines = [
        ln.removeprefix("[agent6] ").removeprefix("ERROR: ").strip()
        for ln in "\n".join(streams).splitlines()
    ]
    return "\n".join(ln for ln in lines if ln)


def _child_exit_message(label: str, rc: int | None, captured: str) -> str:
    """Return what a front-end shows for a child that ended before it began."""
    said = capture_message(captured)
    return said or f"agent6 {label} exited {rc} without a word"


def _not_started_message(label: str, timeout_s: float, captured: str) -> str:
    """Return what a front-end shows when a child never reported starting."""
    said = capture_message(captured)
    return (
        f"agent6 {label} has not reported starting within {timeout_s:.0f}s"
        " (`agent6 ps` shows whether it is running)" + (f":\n{said}" if said else "")
    )


def run_cli_capture(
    argv: list[str], cwd: pathlib.Path, *, timeout_s: float = 120.0
) -> tuple[bool, str]:
    """Run a quick agent6 subcommand synchronously and capture its output.

    For the foreground CLI operations a front-end drives as an operator would; the
    argv is the operator's, never model output.

    Args:
        argv: The command.
        cwd: The checkout.
        timeout_s: How long the command may run.

    Returns:
        Whether it exited 0, and its output as message text or its exit code.
    """
    proc = _run_cli(argv, cwd, timeout_s=timeout_s)
    if isinstance(proc, str):
        return False, proc
    message = capture_message(proc.stdout, proc.stderr)
    return proc.returncode == 0, message or f"exit {proc.returncode}"


def run_cli_output(
    argv: list[str], cwd: pathlib.Path, *, timeout_s: float = 120.0
) -> tuple[bool, str]:
    """Run a subcommand whose stdout is the deliverable, such as a review's markdown.

    Args:
        argv: The command.
        cwd: The checkout.
        timeout_s: How long the command may run.

    Returns:
        Whether it exited 0, and its stdout alone on success or the captured message.
    """
    proc = _run_cli(argv, cwd, timeout_s=timeout_s)
    if isinstance(proc, str):
        return False, proc
    if proc.returncode == 0:
        return True, proc.stdout.strip()
    return False, capture_message(proc.stdout, proc.stderr) or f"exit {proc.returncode}"


def _run_cli(
    argv: list[str], cwd: pathlib.Path, *, timeout_s: float
) -> subprocess.CompletedProcess[str] | str:
    """Return the completed process, or the one-line reason it could not run."""
    try:
        return subprocess.run(
            argv,
            cwd=str(cwd),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"failed to run agent6 {subcommand_label(argv)}: {exc}"


def spawn_and_confirm(
    argv: list[str],
    cwd: pathlib.Path,
    *,
    started: Callable[[int], bool],
    extra_env: Mapping[str, str] | None = None,
    timeout_s: float = 25.0,
) -> str:
    """Spawn the argv detached and wait for the child to take ownership of its work.

    The pid-signalled analogue of `spawn_and_locate`: a refusal printed before the
    child starts is handed back. A child that exits 0 without the signal is a clean
    fast completion.

    Args:
        argv: The command.
        cwd: The checkout.
        started: Whether the child with this pid owns its work.
        extra_env: Environment entries beyond this process's and the away marker.
        timeout_s: How long to wait for ownership.

    Returns:
        "" once the child owns its work, else why it did not.
    """
    _, _pid, err = _spawn_and_wait(
        argv,
        cwd,
        ready=lambda pid: True if started(pid) else None,
        env={**os.environ, **DETACHED_AWAY_ENV, **(extra_env or {})},
        timeout_s=timeout_s,
        clean_exit=True,
    )
    return err


def _stderr_tail(err: IO[str], limit: int = 2000) -> str:
    """Return the end of a spawn's captured stderr, cut at a line start."""
    err.flush()
    text = pathlib.Path(err.name).read_text(encoding="utf-8", errors="replace")
    if len(text) <= limit:
        return text
    tail = text[-limit:]
    nl = tail.find("\n")
    return tail[nl + 1 :] if 0 <= nl < len(tail) - 1 else tail


def _located(
    list_dirs: Callable[[], list[pathlib.Path]], before: set[pathlib.Path], pid: int
) -> pathlib.Path | None:
    """Return the new dir the child with this pid owns, or None while none does.

    A session writes its worker pid before its journal's first event, so a dir recording
    another worker belongs to a session that appeared at the same moment, not to this
    child; a dir recording no worker (a machine draft) is the child's.

    Args:
        list_dirs: Lists the candidate dirs, newest first.
        before: The dirs that existed before the spawn.
        pid: The spawned child's pid.

    Returns:
        The child's dir, or None while it has not appeared.
    """
    for d in list_dirs():
        if d in before or not (d / layout.LOGS_NAME).exists():
            continue
        recorded = process.read_worker_pid(d)
        if recorded is None or recorded == pid:
            return d
    return None


def spawn_and_locate(
    argv: list[str],
    cwd: pathlib.Path,
    *,
    before: set[pathlib.Path],
    list_dirs: Callable[[], list[pathlib.Path]],
    env: dict[str, str] | None = None,
    timeout_s: float = 25.0,
) -> tuple[pathlib.Path | None, str]:
    """Spawn the argv detached and locate the session dir it creates.

    Args:
        argv: The command.
        cwd: The checkout.
        before: The session dirs that existed before the spawn.
        list_dirs: Lists the session dirs now.
        env: The child's environment; None inherits this process's.
        timeout_s: How long to wait for the dir.

    Returns:
        The new dir and "", or None and why.
    """
    found, _pid, err = spawn_lane(
        argv, cwd, before=before, list_dirs=list_dirs, env=env, timeout_s=timeout_s
    )
    return found, err


def spawn_lane(
    argv: list[str],
    cwd: pathlib.Path,
    *,
    before: set[pathlib.Path],
    list_dirs: Callable[[], list[pathlib.Path]],
    env: dict[str, str] | None = None,
    timeout_s: float = 25.0,
    tie_to_parent: bool = True,
) -> tuple[pathlib.Path | None, int, str]:
    """Spawn a lane detached, locate its session dir and keep the child's pid.

    The pid lets the parent stop a child that never reports a session dir; the parent
    tie (PDEATHSIG) ends the lane with a parent that dies, as a Ctrl-C stop would.

    Args:
        argv: The command.
        cwd: The clone.
        before: The session dirs that existed before the spawn.
        list_dirs: Lists the session dirs now.
        env: The child's environment; None inherits this process's.
        timeout_s: How long to wait for the dir.
        tie_to_parent: Deliver SIGTERM to the lane when this process dies.

    Returns:
        The new dir, the pid and "", or None, the pid (0 when nothing started) and why.
    """
    return _spawn_and_wait(
        argv,
        cwd,
        ready=lambda pid: _located(list_dirs, before, pid),
        env=env,
        timeout_s=timeout_s,
        preexec=jail.die_with_parent(os.getpid(), signal.SIGTERM) if tie_to_parent else None,
    )


def _spawn_and_wait[T](
    argv: list[str],
    cwd: pathlib.Path,
    *,
    ready: Callable[[int], T | None],
    env: dict[str, str] | None,
    timeout_s: float,
    clean_exit: T | None = None,
    preexec: Callable[[], None] | None = None,
) -> tuple[T | None, int, str]:
    """Spawn the argv detached with a stderr capture and poll until the child reports ready.

    Non-TTY stdio and a new session, so the child never opens its own TUI.

    Args:
        argv: The command.
        cwd: The checkout.
        ready: The answer for the child with this pid, or None while not ready.
        env: The child's environment; None inherits this process's.
        timeout_s: How long to poll.
        clean_exit: The answer for a child that exited 0 without one; None hands back
            its exit instead.
        preexec: Runs in the child before exec; a lane's parent tie.

    Returns:
        The answer, the child's pid and "", or None, the pid (0 when nothing started) and
        the child's stderr tail, its exit code when it said nothing, or what it said so far
        at the timeout.
    """
    label = subcommand_label(argv)
    err = tempfile.NamedTemporaryFile(  # noqa: SIM115  # closed in finally
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
                preexec_fn=preexec,  # noqa: PLW1509  # async-signal-minimal: one prctl
            )
        except OSError as exc:
            return None, 0, f"failed to start agent6 {label}: {exc}"
        jail.keep_out_of_the_sweep(proc.pid)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if (found := ready(proc.pid)) is not None:
                return found, proc.pid, ""
            rc = proc.poll()
            if rc is not None:
                # The answer may have landed in the same instant.
                if (found := ready(proc.pid)) is not None:
                    return found, proc.pid, ""
                if rc == 0 and clean_exit is not None:
                    return clean_exit, proc.pid, ""
                return None, proc.pid, _child_exit_message(label, rc, _stderr_tail(err))
            time.sleep(0.2)
        return None, proc.pid, _not_started_message(label, timeout_s, _stderr_tail(err))
    finally:
        # The child keeps the unlinked inode as its stderr; its real output is its own log.
        err.close()
        pathlib.Path(err.name).unlink(missing_ok=True)
