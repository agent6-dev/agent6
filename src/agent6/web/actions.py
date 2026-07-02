# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The web write side: drive a run/machine through the shared frontend bridge.

Every mutation the browser can make goes through here, and every one is either
the typed answer-file contract (`agent6.frontend.approval`) or spawning / running
the same `agent6` CLI a user would (`agent6.frontend.spawn`). Nothing here
executes arbitrary input: new-work spawns fixed argv with the task as a single
argv element, answers are written to the run's own answer files, and the quick
ops (merge / prune / config set) shell the fixed agent6 subcommands. The browser
is trusted exactly as far as the operator behind the loopback/tailnet bind.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from agent6.frontend.approval import (
    request_steer,
    write_answer,
    write_question_answer,
    write_steer_answer,
)
from agent6.frontend.spawn import (
    agent6_exe,
    run_cli_capture,
    spawn_and_locate,
    spawn_detached,
)
from agent6.machine import JournalError, MachineJournal
from agent6.viewmodel import newest_state_log
from agent6.web import model

# Modes `agent6 web` can start as new work, mapped 1:1 to the CLI subcommand.
NEW_WORK_MODES = frozenset({"run", "plan", "ask"})


def spawn_new_work(cwd: Path, mode: str, task: str, profile: str = "") -> tuple[str | None, str]:
    """Spawn `agent6 <mode> [--profile P] <task>` detached and return the new run
    id (its dir name) to open, or (None, diagnostic). Mirrors the TUI hub: the
    detached run is told to stream reasoning to its log so the dashboard is live."""
    if mode not in NEW_WORK_MODES:
        return None, f"unknown mode {mode!r}"
    if not task.strip():
        return None, "empty task"
    argv = [agent6_exe(), mode]
    if profile:
        argv += ["--profile", profile]
    argv.append(task)
    run_dir, err = spawn_and_locate(
        argv,
        cwd,
        before=set(model.run_dir_paths(cwd)),
        list_dirs=lambda: model.run_dir_paths(cwd),
        env={**os.environ, "AGENT6_STREAM_TO_LOG": "1"},
    )
    return (run_dir.name if run_dir is not None else None), err


def spawn_machine_create(cwd: Path, task: str) -> tuple[str | None, str]:
    """Spawn `agent6 machine create <task>` detached and return the draft dir name
    to watch (its logs.jsonl carries the authoring agent's reasoning), or None."""
    if not task.strip():
        return None, "empty task"
    draft, err = spawn_and_locate(
        [agent6_exe(), "machine", "create", task],
        cwd,
        before=set(model.draft_dir_paths(cwd)),
        list_dirs=lambda: model.draft_dir_paths(cwd),
    )
    return (draft.name if draft is not None else None), err


def spawn_machine_run(cwd: Path, machine_file: str) -> tuple[bool, str]:
    """Spawn `agent6 machine run <file>` detached. `machine_file` must be one of
    the authored files the hub listed (validated against list_machine_files so the
    browser cannot point it at an arbitrary path)."""
    allowed = {mf["path"] for mf in model.list_machine_files(cwd)}
    if machine_file not in allowed:
        return False, f"unknown machine file {machine_file!r}"
    err = spawn_detached([agent6_exe(), "machine", "run", machine_file], cwd)
    return (err == ""), (err or "started")


def approve(cwd: Path, run_id: str, prompt_id: str, approved: bool) -> tuple[bool, str]:
    """Answer a pending approval prompt (the run's `approval.prompt`)."""
    run_dir = model.run_dir_for(cwd, run_id)
    if run_dir is None:
        return False, f"no run {run_id!r}"
    write_answer(run_dir, prompt_id, approved=approved)
    return True, "answered"


def answer_question(cwd: Path, run_id: str, question_id: str, answer: str) -> tuple[bool, str]:
    """Answer a pending `ask_user` question."""
    run_dir = model.run_dir_for(cwd, run_id)
    if run_dir is None:
        return False, f"no run {run_id!r}"
    write_question_answer(run_dir, question_id, answer)
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


def machine_poke(cwd: Path, name: str, *, data: Any = None, message: str = "") -> tuple[bool, str]:
    """Poke a waiting machine, optionally carrying a payload the next tool reads.
    `data` (any JSON) wins over `message` (a string); neither is a bare wake."""
    machine_dir = model.machine_dir_for(cwd, name)
    if machine_dir is None:
        return False, f"no machine {name!r}"
    payload: Any = data if data is not None else (message or None)
    try:
        MachineJournal(machine_dir).poke(payload)
    except JournalError as exc:
        return False, str(exc)
    return True, "poked"


def machine_approve(
    cwd: Path, name: str, prompt_id: str, approved: bool, *, state: str = ""
) -> tuple[bool, str]:
    """Answer a pending approval in the agent state the prompt was rendered from
    (``state``; newest when absent)."""
    state_dir = _machine_state_dir(cwd, name, state)
    if state_dir is None:
        return False, f"no active agent state for machine {name!r}"
    write_answer(state_dir, prompt_id, approved=approved)
    return True, "answered"


def machine_answer(
    cwd: Path, name: str, question_id: str, answer: str, *, state: str = ""
) -> tuple[bool, str]:
    """Answer a pending `ask_user` question in the agent state the prompt was
    rendered from (``state``; newest when absent)."""
    state_dir = _machine_state_dir(cwd, name, state)
    if state_dir is None:
        return False, f"no active agent state for machine {name!r}"
    write_question_answer(state_dir, question_id, answer)
    return True, "answered"


def machine_steer(cwd: Path, name: str, text: str, *, state: str = "") -> tuple[bool, str]:
    """Steer the agent state the operator is viewing (``state``; newest when
    absent). Same contract as a run steer."""
    state_dir = _machine_state_dir(cwd, name, state)
    if state_dir is None:
        return False, f"no active agent state for machine {name!r}"
    write_steer_answer(state_dir, text)
    request_steer(state_dir)
    return True, "steer requested"


def merge_run(cwd: Path, run_id: str, strategy: str = "") -> tuple[bool, str]:
    """Merge a run's branch: `agent6 runs merge <id> [--strategy S]`."""
    argv = [agent6_exe(), "runs", "merge", run_id]
    if strategy:
        argv += ["--strategy", strategy]
    return run_cli_capture(argv, cwd)


def prune_runs(cwd: Path) -> tuple[bool, str]:
    """Prune merged/obsolete run branches: `agent6 runs prune`."""
    return run_cli_capture([agent6_exe(), "runs", "prune"], cwd)


def set_config(cwd: Path, key: str, value: str, *, repo: bool = False) -> tuple[bool, str]:
    """Set one config leaf: `agent6 config set <key> <value> [--repo]`. The CLI
    validates the key and value; the write lands in the global config by default."""
    argv = [agent6_exe(), "config", "set", key, value]
    if repo:
        argv.append("--repo")
    return run_cli_capture(argv, cwd)
