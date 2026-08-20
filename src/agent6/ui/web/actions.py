# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The web write side: drive a run/machine through the shared frontend bridge.

Every mutation the browser can make goes through here, and every one is either
the typed answer-file contract (`agent6.sessions.ipc`) or spawning / running
the same `agent6` CLI a user would (`agent6.ui.spawn`). Nothing here
executes arbitrary input: new-work spawns fixed argv with the task as a single
argv element, answers are written to the run's own answer files, and the quick
ops (merge / prune / config set) shell the fixed agent6 subcommands. The browser
is trusted exactly as far as the operator behind the loopback/tailnet bind.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent6.app.fork import undo_fork
from agent6.app.reporter import Reporter
from agent6.directive import parse_compact
from agent6.machine import (
    JournalError,
    MachineError,
    MachineJournal,
    load_machine,
    write_stop_request,
)
from agent6.sessions.ipc import (
    read_worker_pid,
    request_compact,
    request_steer,
    request_stop,
    worker_is_alive,
    write_answer,
    write_question_answers,
    write_steer_answer,
)
from agent6.ui.spawn import (
    agent6_argv,
    run_cli_capture,
    spawn_and_confirm,
    spawn_and_locate,
    spawn_detached_resume,
)
from agent6.ui.web import model
from agent6.viewmodel import newest_state_log, session_is_live
from agent6.viewmodel.listing import finished_needs_new_work, needs_new_work_refusal


# Modes `agent6 web` can start as new work, mapped 1:1 to the CLI subcommand.
def spawn_machine_create(
    cwd: Path, task: str, config_path: Path | None = None
) -> tuple[str | None, str]:
    """Spawn `agent6 machine create <task>` detached and return the draft dir name
    to watch (its logs.jsonl carries the authoring agent's reasoning), or None."""
    if not task.strip():
        return None, "empty task"
    draft, err = spawn_and_locate(
        [*agent6_argv(config_path), "machine", "create", "--", task],
        cwd,
        before=set(model.draft_dir_paths(cwd)),
        list_dirs=lambda: model.draft_dir_paths(cwd),
    )
    return (draft.name if draft is not None else None), err


def spawn_machine_run(
    cwd: Path, machine_file: str, config_path: Path | None = None
) -> tuple[bool, str]:
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
        [*agent6_argv(config_path), "machine", "run", machine_file],
        cwd,
        started=lambda pid: read_worker_pid(instance) == pid,
    )
    return (err == ""), (err or "started")


def approve(cwd: Path, session_id: str, prompt_id: str, answer: str) -> tuple[bool, str]:
    """Answer a pending approval prompt (the run's `approval.prompt`) with the
    operator's literal choice."""
    session_dir = model.session_dir_for(cwd, session_id)
    if session_dir is None:
        return False, f"no session {session_id!r}"
    if not session_is_live(session_dir):
        # The prompt box outlives the worker (it clears on the answer event,
        # which a dead run will never emit), so refuse like every sibling
        # action: nothing would consume the answer, and the next resume drops
        # it. A session grant would be just as stranded.
        return False, "the session is not live"
    write_answer(session_dir, prompt_id, answer)
    return True, "answered"


def answer_question(
    cwd: Path, session_id: str, question_id: str, answers: list[str]
) -> tuple[bool, str]:
    """Answer a pending `ask_user` prompt (one answer per question, by index)."""
    session_dir = model.session_dir_for(cwd, session_id)
    if session_dir is None:
        return False, f"no session {session_id!r}"
    if not session_is_live(session_dir):
        return False, "the session is not live"  # see approve()
    write_question_answers(session_dir, question_id, answers)
    return True, "answered"


def steer(cwd: Path, session_id: str, text: str) -> tuple[bool, str]:
    """Steer a live run: pre-place the answer, then drop the request marker the
    run picks up at its next safe boundary. `text` is a free instruction; "" means
    continue, "abort" stops the run (the same contract the TUI steer modal uses)."""
    session_dir = model.session_dir_for(cwd, session_id)
    if session_dir is None:
        return False, f"no session {session_id!r}"
    if not session_is_live(session_dir):
        # A crashed run folds as unfinished, so the composer still offers
        # steer; nothing would ever read the marker (and the next resume
        # deletes it), so refuse like the stop/compact siblings.
        return False, "the session is not live"
    focus = parse_compact(text)
    if focus is not None:
        # `/compact [focus]` is an out-of-band request, not steer text the
        # loop should read; /pin and /parallel stay steers the loop parses.
        if not request_compact(session_dir, focus=focus):
            return False, "could not write the compaction request"
        return True, "compaction requested"
    write_steer_answer(session_dir, text)  # ready before the run reads it
    request_steer(session_dir)
    return True, "steer requested"


def undo_session(cwd: Path, session_id: str) -> tuple[dict[str, str] | None, str]:
    """`/undo` on a finished run: fork it at the state before its last operator
    message, unstarted. Returns ({new_session_id, undone_text}, "") or
    (None, why). A live run is refused: stop it first, then undo."""
    session_dir = model.session_dir_for(cwd, session_id)
    if session_dir is None:
        return None, f"no session {session_id!r}"
    if session_is_live(session_dir):
        return None, "the session is live; /undo rides the steer channel from the composer"
    said: list[str] = []
    reporter = Reporter(out=said.append, err=said.append)
    result = undo_fork(None, session_dir.name, cwd=cwd, reporter=reporter)
    if result is None:
        return None, (said[-1].strip() if said else "undo failed")
    child, text = result
    return {"new_session_id": child, "undone_text": text}, ""


def resume_run(
    cwd: Path, session_id: str, text: str = "", config_path: Path | None = None
) -> tuple[bool, str]:
    """Resume a finished/stopped run detached, optionally seeding *text* as the
    first steering instruction (the composer's Enter on a finished run). Refused
    while the run's worker is alive: a live run is steered, not resumed."""
    session_dir = model.session_dir_for(cwd, session_id)
    if session_dir is None:
        return False, f"no session {session_id!r}"
    if session_is_live(session_dir):
        return False, "the session is still live; steer it instead"
    if not text.strip() and finished_needs_new_work(session_dir):
        # The spawn is DETACHED, so the same refusal from `agent6 resume` would
        # land on a process nobody is reading and the composer would report
        # "resuming" for a run that never started.
        return False, needs_new_work_refusal(session_id)
    err = spawn_detached_resume(cwd, session_dir.name, steer=text, config_path=config_path)
    return (err == ""), (err or "resuming")


def stop_after_step(cwd: Path, session_id: str) -> tuple[bool, str]:
    """Ask a live run to end cleanly at its next completed-iteration boundary
    (the finished step's tool results and auto-commit land first). The immediate
    stop stays the steer "abort" answer."""
    session_dir = model.session_dir_for(cwd, session_id)
    if session_dir is None:
        return False, f"no session {session_id!r}"
    if not session_is_live(session_dir):
        return False, "the session is not live"
    request_stop(session_dir)
    return True, "stopping after the current step"


def compact_run(cwd: Path, session_id: str) -> tuple[bool, str]:
    """Ask a live run to compact its context at the next safe boundary."""
    session_dir = model.session_dir_for(cwd, session_id)
    if session_dir is None:
        return False, f"no session {session_id!r}"
    if not session_is_live(session_dir):
        return False, "the session is not live"
    if not request_compact(session_dir):
        return False, "could not write the compaction request"
    return True, "compaction requested"


def _machine_state_dir(cwd: Path, name: str, state: str = "") -> Path | None:
    """The per-state dir an answer belongs in (where its answer files live), or
    None when the machine name is unknown or no agent state is active.

    When *state* is given (the dir name the client rendered the prompt from,
    e.g. `0001-work`) route to exactly that state, so an answer lands in the
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


def machine_stop(cwd: Path, name: str) -> tuple[bool, str]:
    """Write the durable stop marker for a running machine (parks at its next
    transition boundary; resumable). Not-running is a refusal, not a marker."""
    machine_dir = model.machine_dir_for(cwd, name)
    if machine_dir is None:
        return False, f"no machine {name!r}"
    if _machine_has_ended(cwd, name):
        return False, f"machine {name!r} has ended; nothing to stop"
    if not worker_is_alive(machine_dir):
        return False, f"machine {name!r} is not running; nothing to stop"
    write_stop_request(machine_dir)
    return True, "stop requested; the machine parks at its next transition boundary"


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


def _machine_unavailable(cwd: Path, name: str, *, ended: str, stopped: str) -> str:
    """Why *name* cannot receive an answer, or "" when it can.

    Ordered so the message names the real state: an unknown machine is not a
    stopped one. Reading a missing instance dir as LIVE sent the caller past
    this gate to fail on "no active agent state", pointing the operator at a
    state belonging to a machine that never existed.

    The newest state dir of a parked or stopped machine is a FINISHED agent
    state whose loop has exited, so a marker written there is polled by nobody.
    """
    machine_dir = model.machine_dir_for(cwd, name)
    if machine_dir is None:
        return f"no machine {name!r}"
    if _machine_has_ended(cwd, name):
        return ended
    if not worker_is_alive(machine_dir):
        return stopped
    return ""


def machine_approve(
    cwd: Path, name: str, prompt_id: str, answer: str, *, state: str = ""
) -> tuple[bool, str]:
    """Answer a pending approval in the agent state the prompt was rendered from
    (`state`; newest when absent)."""
    refusal = _machine_unavailable(
        cwd,
        name,
        ended=f"machine {name!r} has ended; the prompt is closed",
        stopped=f"machine {name!r} is not running; poke it to wake a waiting machine",
    )
    if refusal:
        return False, refusal
    state_dir = _machine_state_dir(cwd, name, state)
    if state_dir is None:
        return False, f"no active agent state for machine {name!r}"
    write_answer(state_dir, prompt_id, answer)
    return True, "answered"


def machine_answer(
    cwd: Path, name: str, question_id: str, answers: list[str], *, state: str = ""
) -> tuple[bool, str]:
    """Answer a pending `ask_user` prompt in the agent state the prompt was rendered
    from (`state`; newest when absent). One answer per question, by index."""
    refusal = _machine_unavailable(
        cwd,
        name,
        ended=f"machine {name!r} has ended; the prompt is closed",
        stopped=f"machine {name!r} is not running; poke it to wake a waiting machine",
    )
    if refusal:
        return False, refusal
    state_dir = _machine_state_dir(cwd, name, state)
    if state_dir is None:
        return False, f"no active agent state for machine {name!r}"
    write_question_answers(state_dir, question_id, answers)
    return True, "answered"


def machine_steer(cwd: Path, name: str, text: str, *, state: str = "") -> tuple[bool, str]:
    """Steer the agent state the operator is viewing (`state`; newest when
    absent). Same contract as a run steer."""
    refusal = _machine_unavailable(
        cwd,
        name,
        ended=f"machine {name!r} has ended; there is no state to steer",
        stopped=(
            f"machine {name!r} is not running, so no agent state would read a steer"
            " (poke it to wake a waiting machine)"
        ),
    )
    if refusal:
        return False, refusal
    state_dir = _machine_state_dir(cwd, name, state)
    if state_dir is None:
        return False, f"no active agent state for machine {name!r}"
    write_steer_answer(state_dir, text)
    request_steer(state_dir)
    return True, "steer requested"


def merge_run(
    cwd: Path, session_id: str, strategy: str = "", config_path: Path | None = None
) -> tuple[bool, str]:
    """Merge a run's branch: `agent6 sessions merge <id> [--strategy S]`. `--` before
    the client-supplied run id so a dashy value cannot be read as a flag."""
    argv = [*agent6_argv(config_path), "sessions", "merge"]
    if strategy:
        argv += ["--strategy", strategy]
    argv += ["--", session_id]
    return run_cli_capture(argv, cwd)


def prune_sessions(cwd: Path, config_path: Path | None = None) -> tuple[bool, str]:
    """Prune merged/obsolete run branches: `agent6 sessions prune`."""
    return run_cli_capture([*agent6_argv(config_path), "sessions", "prune"], cwd)


def remove_session(cwd: Path, session_id: str, config_path: Path | None = None) -> tuple[bool, str]:
    """Delete one run's history: `agent6 sessions rm <id>`. History only -- the run
    branch is git's, and `sessions prune` is the branch verb. The CLI refuses a live
    run, so this surface inherits that."""
    return run_cli_capture([*agent6_argv(config_path), "sessions", "rm", "--", session_id], cwd)


def remove_asks(cwd: Path, config_path: Path | None = None) -> tuple[bool, str]:
    """Clear every saved ask: `agent6 sessions rm --asks`. The bucket that
    accumulates, since an ask runs in any directory."""
    return run_cli_capture([*agent6_argv(config_path), "sessions", "rm", "--asks"], cwd)


def set_config(
    cwd: Path, key: str, value: str, *, repo: bool = False, config_path: Path | None = None
) -> tuple[bool, str]:
    """Set one config leaf: `agent6 config set <key> <value> [--repo]`. The CLI
    validates the key and value; the write lands in the global config by default.
    `--` before the body-derived key/value so a dashy value cannot be read as a
    flag."""
    argv = [*agent6_argv(config_path), "config", "set"]
    if repo:
        argv.append("--repo")
    argv += ["--", key, value]
    return run_cli_capture(argv, cwd)


def unset_config(
    cwd: Path, key: str, *, repo: bool = False, config_path: Path | None = None
) -> tuple[bool, str]:
    """Unset one config leaf: `agent6 config unset <key> [--repo]`, reverting it
    to the next-lower layer / built-in default. Same fixed-argv CLI bridge as
    set_config (`--` guards a dashy key)."""
    argv = [*agent6_argv(config_path), "config", "unset"]
    if repo:
        argv.append("--repo")
    argv += ["--", key]
    return run_cli_capture(argv, cwd)
