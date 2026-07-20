# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The web write side: drive a run/machine through the shared frontend bridge.

Every mutation the browser can make goes through here, and every one is either
the typed answer-file contract (`agent6.runs.ipc`) or spawning / running
the same `agent6` CLI a user would (`agent6.ui.spawn`). Nothing here
executes arbitrary input: new-work spawns fixed argv with the task as a single
argv element, answers are written to the run's own answer files, and the quick
ops (merge / prune / config set) shell the fixed agent6 subcommands. The browser
is trusted exactly as far as the operator behind the loopback/tailnet bind.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from agent6.config import ConfigError
from agent6.config.layer import load_effective, resolved_state_dir
from agent6.directive import DirectiveError, Segment, parse_directive, parse_spec
from agent6.machine import JournalError, MachineError, MachineJournal, load_machine
from agent6.models.validate import refusal_message, validate_spec_models
from agent6.runs.ipc import (
    read_worker_pid,
    request_compact,
    request_steer,
    request_stop,
    set_session_allow,
    worker_is_alive,
    write_answer,
    write_question_answers,
    write_steer_answer,
)
from agent6.runs.lock import repo_writer_held, repo_writer_holder
from agent6.ui.spawn import (
    agent6_exe,
    run_cli_capture,
    spawn_and_confirm,
    spawn_and_locate,
    spawn_detached_resume,
)
from agent6.ui.web import model
from agent6.viewmodel import newest_state_log

# Modes `agent6 web` can start as new work, mapped 1:1 to the CLI subcommand.
NEW_WORK_MODES = frozenset({"run", "plan", "ask"})


def spawn_new_work(  # noqa: PLR0911
    cwd: Path, mode: str, task: str, profile: str = ""
) -> tuple[str | None, str]:
    """Spawn `agent6 <mode> [--profile P] <task>` detached and return the new run
    id (its dir name) to open, or (None, diagnostic). Mirrors the TUI hub: the
    detached run is told to stream reasoning to its log so the dashboard is live.

    A `/parallel [spec] <task> ...` message (run mode only) fans out one detached
    `agent6 run --parallel <spec>` per segment (omitted spec = one isolated lane).
    A malformed directive is refused before any spawn (all-or-nothing); on success
    the first segment's run id is returned to open, the rest run in the hub."""
    if mode not in NEW_WORK_MODES:
        return None, f"unknown mode {mode!r}"
    if not task.strip():
        return None, "empty task"
    if mode == "run":
        # One live run-mode worker per checkout (acquire_repo_writer): refuse
        # up front so the composer shows why, instead of spawning a detached
        # run that parks itself and times out the locate. plan/ask are
        # read-only and spawn freely. Advisory probe; the lock is the boundary.
        state = resolved_state_dir(cwd)
        if repo_writer_held(state):
            holder = repo_writer_holder(state) or "another run"
            return None, (
                f"run {holder} is already driving this checkout; steer it with"
                " this task (or /parallel it) from its run view, or wait for it"
                " to finish"
            )
    segments = None
    if mode == "run":
        try:
            segments = parse_directive(task)
        except DirectiveError as exc:
            return None, str(exc)
    if segments is None:
        return _spawn_run(cwd, mode, task, profile, spec="")
    refusal = _model_refusal(cwd, segments)
    if refusal is not None:
        return None, refusal
    first: tuple[str | None, str] = (None, "")
    for seg in segments:
        result = _spawn_run(cwd, "run", seg.task, profile, spec=seg.spec or "1")
        if first[0] is None:
            first = result
    return first


def _model_refusal(cwd: Path, segments: list[Segment]) -> str | None:
    """Refuse a `/parallel` directive that names a model the configured providers'
    cache says doesn't exist, before any spawn (the composer's normal error path,
    nothing spawned). None = every model checks out, or there is no cache to check
    against (a fresh/offline machine proceeds; the detached lane's own preflight
    warns). A malformed spec surfaces its grammar error."""
    try:
        models = [m for seg in segments for m in parse_spec(seg.spec)]
    except DirectiveError as exc:
        return str(exc)
    try:
        cfg = load_effective(cwd).config
    except ConfigError:
        return None  # a broken config is its own separate error; don't mask it here
    verdict = validate_spec_models(models, cfg)
    return refusal_message(verdict, directive=True) if verdict.refused else None


def _spawn_run(
    cwd: Path, mode: str, task: str, profile: str, *, spec: str
) -> tuple[str | None, str]:
    """Spawn one detached `agent6 <mode>` (optionally `--parallel <spec>`) and
    return its located run dir name, or (None, diagnostic)."""
    argv = [agent6_exe(), mode]
    if profile:
        argv += ["--profile", profile]
    if spec:
        argv += ["--parallel", spec]
    # `--` ends option parsing: the body-derived task can start with `-` without
    # being read as a flag.
    argv += ["--", task]
    run_dir, err = spawn_and_locate(
        argv,
        cwd,
        before=set(model.run_dir_paths(cwd)),
        list_dirs=lambda: model.run_dir_paths(cwd),
        # AGENT6_DETACHED_AWAY=wait: this run is driven from the browser over the
        # bridge, so approvals and questions must WAIT for a viewer, not fabricate
        # empty answers when the tab is momentarily disconnected (see run.py).
        env={**os.environ, "AGENT6_STREAM_TO_LOG": "1", "AGENT6_DETACHED_AWAY": "wait"},
    )
    return (run_dir.name if run_dir is not None else None), err


def spawn_machine_create(cwd: Path, task: str) -> tuple[str | None, str]:
    """Spawn `agent6 machine create <task>` detached and return the draft dir name
    to watch (its logs.jsonl carries the authoring agent's reasoning), or None."""
    if not task.strip():
        return None, "empty task"
    draft, err = spawn_and_locate(
        [agent6_exe(), "machine", "create", "--", task],
        cwd,
        before=set(model.draft_dir_paths(cwd)),
        list_dirs=lambda: model.draft_dir_paths(cwd),
    )
    return (draft.name if draft is not None else None), err


def spawn_machine_run(cwd: Path, machine_file: str) -> tuple[bool, str]:
    """Spawn `agent6 machine run <file>` detached. `machine_file` must be one of
    the authored files the hub listed (validated against list_machine_files so the
    browser cannot point it at an arbitrary path).

    Started = the child wrote its own pid as the instance worker.pid (it does so
    right after taking the machine lock), so a refusal (lock held, network
    refusal, bad bundle: nonzero exit before that) surfaces its stderr in the
    toast instead of a false "started"."""
    allowed = {mf["path"] for mf in model.list_machine_files(cwd)}
    if machine_file not in allowed:
        return False, f"unknown machine file {machine_file!r}"
    try:
        spec = load_machine(Path(machine_file))
    except MachineError as exc:
        return False, f"invalid machine file: {exc}"
    instance = model.machines_root(cwd) / spec.machine
    err = spawn_and_confirm(
        [agent6_exe(), "machine", "run", machine_file],
        cwd,
        started=lambda pid: read_worker_pid(instance) == pid,
    )
    return (err == ""), (err or "started")


def approve(
    cwd: Path, run_id: str, prompt_id: str, approved: bool, *, session: bool = False
) -> tuple[bool, str]:
    """Answer a pending approval prompt (the run's `approval.prompt`). ``session``
    (the "allow session" button) also auto-approves every later run_command."""
    run_dir = model.run_dir_for(cwd, run_id)
    if run_dir is None:
        return False, f"no run {run_id!r}"
    if session:
        set_session_allow(run_dir)
    write_answer(run_dir, prompt_id, approved=approved)
    return True, "answered"


def answer_question(
    cwd: Path, run_id: str, question_id: str, answers: list[str]
) -> tuple[bool, str]:
    """Answer a pending `ask_user` prompt (one answer per question, by index)."""
    run_dir = model.run_dir_for(cwd, run_id)
    if run_dir is None:
        return False, f"no run {run_id!r}"
    write_question_answers(run_dir, question_id, answers)
    return True, "answered"


def steer(cwd: Path, run_id: str, text: str) -> tuple[bool, str]:
    """Steer a live run: pre-place the answer, then drop the request marker the
    run picks up at its next safe boundary. `text` is a free instruction; "" means
    continue, "abort" stops the run (the same contract the TUI steer modal uses)."""
    run_dir = model.run_dir_for(cwd, run_id)
    if run_dir is None:
        return False, f"no run {run_id!r}"
    write_steer_answer(run_dir, text)  # ready before the run reads it
    request_steer(run_dir)
    return True, "steer requested"


def resume_run(cwd: Path, run_id: str, text: str = "") -> tuple[bool, str]:
    """Resume a finished/stopped run detached, optionally seeding *text* as the
    first steering instruction (the composer's Enter on a finished run). Refused
    while the run's worker is alive: a live run is steered, not resumed."""
    run_dir = model.run_dir_for(cwd, run_id)
    if run_dir is None:
        return False, f"no run {run_id!r}"
    if worker_is_alive(run_dir):
        return False, "run is still live; steer it instead"
    err = spawn_detached_resume(cwd, run_dir.name, steer=text)
    return (err == ""), (err or "resuming")


def stop_after_step(cwd: Path, run_id: str) -> tuple[bool, str]:
    """Ask a live run to end cleanly at its next completed-iteration boundary
    (the finished step's tool results and auto-commit land first). The immediate
    stop stays the steer "abort" answer."""
    run_dir = model.run_dir_for(cwd, run_id)
    if run_dir is None:
        return False, f"no run {run_id!r}"
    if not worker_is_alive(run_dir):
        return False, "run is not live"
    request_stop(run_dir)
    return True, "stopping after the current step"


def compact_run(cwd: Path, run_id: str) -> tuple[bool, str]:
    """Ask a live run to compact its context at the next safe boundary."""
    run_dir = model.run_dir_for(cwd, run_id)
    if run_dir is None:
        return False, f"no run {run_id!r}"
    if not worker_is_alive(run_dir):
        return False, "run is not live"
    request_compact(run_dir)
    return True, "compaction requested"


def _machine_state_dir(cwd: Path, name: str, state: str = "") -> Path | None:
    """The per-state dir an answer belongs in (where its answer files live), or
    None when the machine name is unknown or no agent state is active.

    When *state* is given (the dir name the client rendered the prompt from,
    e.g. ``0001-work``) route to exactly that state, so an answer lands in the
    state it was shown for even if the machine has since advanced to another
    state that reuses the same prompt id. Falls back to the newest state when
    *state* is absent (a bare CLI/older client). *state* is validated as a
    single existing path component so a request body cannot traverse out.
    """
    machine_dir = model.machine_dir_for(cwd, name)
    if machine_dir is None:
        return None
    if state:
        if not model.is_safe_component(state):
            return None
        target = machine_dir / "states" / state
        return target if target.is_dir() else None
    log = newest_state_log(machine_dir)
    return log.parent if log is not None else None


def _machine_has_ended(cwd: Path, name: str) -> bool:
    """True when the instance's journal records a MachineEnd. Poking or steering
    an ended machine would report success while nothing will ever read the
    signal; an unreadable journal reads as not-ended so the real error surfaces
    from the operation itself."""
    machine_dir = model.machine_dir_for(cwd, name)
    if machine_dir is None:
        return False
    try:
        return model.machine_snapshot(machine_dir).get("ended") is not None
    except (MachineError, OSError):
        return False


def machine_poke(cwd: Path, name: str, *, data: Any = None, message: str = "") -> tuple[bool, str]:
    """Poke a waiting machine, optionally carrying a payload the next tool reads.
    `data` (any JSON) wins over `message` (a string); neither is a bare wake."""
    machine_dir = model.machine_dir_for(cwd, name)
    if machine_dir is None:
        return False, f"no machine {name!r}"
    if _machine_has_ended(cwd, name):
        return False, f"machine {name!r} has ended; nothing is waiting to wake"
    payload: Any = data if data is not None else (message or None)
    try:
        MachineJournal(machine_dir).poke(payload)
    except JournalError as exc:
        return False, str(exc)
    return True, "poked"


def machine_approve(
    cwd: Path, name: str, prompt_id: str, approved: bool, *, session: bool = False, state: str = ""
) -> tuple[bool, str]:
    """Answer a pending approval in the agent state the prompt was rendered from
    (``state``; newest when absent). ``session`` auto-approves every later
    run_command in that state."""
    state_dir = _machine_state_dir(cwd, name, state)
    if state_dir is None:
        return False, f"no active agent state for machine {name!r}"
    if session:
        set_session_allow(state_dir)
    write_answer(state_dir, prompt_id, approved=approved)
    return True, "answered"


def machine_answer(
    cwd: Path, name: str, question_id: str, answers: list[str], *, state: str = ""
) -> tuple[bool, str]:
    """Answer a pending `ask_user` prompt in the agent state the prompt was rendered
    from (``state``; newest when absent). One answer per question, by index."""
    state_dir = _machine_state_dir(cwd, name, state)
    if state_dir is None:
        return False, f"no active agent state for machine {name!r}"
    write_question_answers(state_dir, question_id, answers)
    return True, "answered"


def machine_steer(cwd: Path, name: str, text: str, *, state: str = "") -> tuple[bool, str]:
    """Steer the agent state the operator is viewing (``state``; newest when
    absent). Same contract as a run steer."""
    if _machine_has_ended(cwd, name):
        return False, f"machine {name!r} has ended; there is no state to steer"
    state_dir = _machine_state_dir(cwd, name, state)
    if state_dir is None:
        return False, f"no active agent state for machine {name!r}"
    write_steer_answer(state_dir, text)
    request_steer(state_dir)
    return True, "steer requested"


def merge_run(cwd: Path, run_id: str, strategy: str = "") -> tuple[bool, str]:
    """Merge a run's branch: `agent6 runs merge <id> [--strategy S]`. `--` before
    the client-supplied run id so a dashy value cannot be read as a flag."""
    argv = [agent6_exe(), "runs", "merge"]
    if strategy:
        argv += ["--strategy", strategy]
    argv += ["--", run_id]
    return run_cli_capture(argv, cwd)


def prune_runs(cwd: Path) -> tuple[bool, str]:
    """Prune merged/obsolete run branches: `agent6 runs prune`."""
    return run_cli_capture([agent6_exe(), "runs", "prune"], cwd)


def set_config(cwd: Path, key: str, value: str, *, repo: bool = False) -> tuple[bool, str]:
    """Set one config leaf: `agent6 config set <key> <value> [--repo]`. The CLI
    validates the key and value; the write lands in the global config by default.
    `--` before the body-derived key/value so a dashy value cannot be read as a
    flag."""
    argv = [agent6_exe(), "config", "set"]
    if repo:
        argv.append("--repo")
    argv += ["--", key, value]
    return run_cli_capture(argv, cwd)


def unset_config(cwd: Path, key: str, *, repo: bool = False) -> tuple[bool, str]:
    """Unset one config leaf: `agent6 config unset <key> [--repo]`, reverting it
    to the next-lower layer / built-in default. Same fixed-argv CLI bridge as
    set_config (`--` guards a dashy key)."""
    argv = [agent6_exe(), "config", "unset"]
    if repo:
        argv.append("--repo")
    argv += ["--", key]
    return run_cli_capture(argv, cwd)
