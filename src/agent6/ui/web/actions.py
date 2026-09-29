# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Drive a run or a machine from the browser.

Every mutation is either the answer-file contract (`agent6.sessions.ipc`) or the
same `agent6` CLI a user would run (`agent6.ui.spawn`) with fixed argv: the task
is one argv element, the quick ops shell fixed subcommands, nothing executes
arbitrary input. The browser is trusted as far as the operator behind the bind.
"""

from __future__ import annotations

import contextlib
import os
import pathlib
from typing import Any

from agent6 import errors
from agent6 import paths as agent6_paths
from agent6.app import fork, stop, undo
from agent6.app import reporter as app_reporter
from agent6.config import io, write
from agent6.machine import (
    JournalError,
    MachineError,
    MachineJournal,
    load_machine,
    write_stop_request,
)
from agent6.sessions import ipc, layout, manifest
from agent6.ui import directives, spawn
from agent6.ui.web import model
from agent6.viewmodel import (
    listing,
    machine_state,
    newest_state_log,
    open_approval,
    open_question,
    session_is_live,
)


def spawn_machine_create(
    cwd: pathlib.Path, task: str, config_path: pathlib.Path | None = None
) -> tuple[str | None, str]:
    """Spawn `agent6 machine create <task>` detached.

    Args:
        cwd: The repository.
        task: What the machine is for.
        config_path: An explicit config file, or None.

    Returns:
        The draft dir name to watch and "", or None and the reason.
    """
    if not task.strip():
        return None, "empty task"
    draft, err = spawn.spawn_and_locate(
        [*spawn.agent6_argv(config_path), "machine", "create", "--", task],
        cwd,
        before=set(model.draft_dir_paths(cwd)),
        list_dirs=lambda: model.draft_dir_paths(cwd),
    )
    return (draft.name if draft is not None else None), err


def spawn_machine_run(
    cwd: pathlib.Path, machine_file: str, config_path: pathlib.Path | None = None
) -> tuple[bool, str]:
    """Spawn `agent6 machine run <file>` detached.

    Started means the child wrote its pid as the instance's worker.pid, so a refusal
    before that surfaces its stderr instead of a false "started".

    Args:
        cwd: The repository.
        machine_file: A listed machine file, by path or listed name (never an arbitrary path).
        config_path: An explicit config file, or None.

    Returns:
        Whether it started, and the note or the refusal.
    """
    listed = model.list_machine_files(cwd)
    by_name = {mf["name"]: mf["path"] for mf in listed}
    paths = {mf["path"] for mf in listed}
    if machine_file in by_name and machine_file not in paths:
        machine_file = by_name[machine_file]
    if machine_file not in paths:
        return False, (
            f"unknown machine file {machine_file!r}: give a listed path or name"
            f" ({', '.join(sorted(by_name)) or 'none listed'})"
        )
    try:
        spec = load_machine(pathlib.Path(machine_file))
    except MachineError as exc:
        return False, f"invalid machine file: {exc}"
    instance = layout.machines_root(agent6_paths.state_dir(cwd)) / spec.machine
    err = spawn.spawn_and_confirm(
        [*spawn.agent6_argv(config_path), "machine", "run", machine_file],
        cwd,
        started=lambda pid: ipc.read_worker_pid(instance) == pid,
    )
    return (err == ""), (err or "started")


def _live_session_dir(cwd: pathlib.Path, session_id: str) -> pathlib.Path | tuple[bool, str]:
    """Return the session dir a verb acts on, or its refusal.

    A dead run is refused whatever the page still offers: nothing would consume the
    answer or the marker, and the next resume drops them.
    """
    session_dir = model.session_dir_for(cwd, session_id)
    if session_dir is None:
        return False, f"no session {session_id!r}"
    if not session_is_live(session_dir):
        return False, "the session is not live; resume it instead"
    return session_dir


def approve(cwd: pathlib.Path, session_id: str, prompt_id: str, answer: str) -> tuple[bool, str]:
    """Answer a pending approval prompt with the operator's literal choice.

    Returns:
        Whether the answer landed, and the note or the refusal.
    """
    session_dir = _live_session_dir(cwd, session_id)
    if isinstance(session_dir, tuple):
        return session_dir
    prompt = open_approval(session_dir)
    if prompt is None or prompt.id != prompt_id:
        return False, "that approval is no longer open"
    if not ipc.write_answer(session_dir, prompt_id, answer):
        return False, ipc.ANSWERED_ELSEWHERE
    return True, "answered"


def answer_question(
    cwd: pathlib.Path, session_id: str, question_id: str, answers: list[str]
) -> tuple[bool, str]:
    """Answer a pending `ask_user` prompt, one answer per question by index.

    Returns:
        Whether the answers landed, and the note or the refusal.
    """
    session_dir = _live_session_dir(cwd, session_id)
    if isinstance(session_dir, tuple):
        return session_dir
    prompt = open_question(session_dir)
    if prompt is None or prompt.id != question_id:
        return False, "that question is no longer open"
    if len(answers) != len(prompt.questions):
        # The asking side raises on a mismatch after consuming the file, losing the text.
        return False, f"that prompt has {len(prompt.questions)} question(s)"
    if not ipc.write_question_answers(session_dir, question_id, answers):
        return False, ipc.ANSWERED_ELSEWHERE
    return True, "answered"


def steer(cwd: pathlib.Path, session_id: str, text: str) -> tuple[bool, str]:
    """Steer a live run at its next safe boundary.

    Args:
        cwd: The repository.
        session_id: The run.
        text: A free instruction; "" continues, "abort" stops the run.

    Returns:
        Whether the request landed, and the note or the refusal.
    """
    session_dir = _live_session_dir(cwd, session_id)
    if isinstance(session_dir, tuple):
        return session_dir
    return directives.submit_composer_line(session_dir, text)


def fork_run(
    cwd: pathlib.Path, session_id: str, config_path: pathlib.Path | None = None
) -> tuple[dict[str, str] | None, str]:
    """Fork a run at its latest checkpoint into a new, unstarted run.

    Returns:
        The new session id in a dict and "", or None and the reason.
    """
    session_dir = model.session_dir_for(cwd, session_id)
    if session_dir is None:
        return None, f"no session {session_id!r}"
    said: list[str] = []
    reporter = app_reporter.Reporter(out=said.append, err=said.append)
    child, rc = fork.create_fork(config_path, session_dir.name, cwd=cwd, reporter=reporter)
    if rc != 0:
        return None, (spawn.capture_message(said[-1]) if said else "fork failed")
    return {"new_session_id": child}, ""


def undo_session(cwd: pathlib.Path, session_id: str) -> tuple[dict[str, str] | None, str]:
    """Fork a finished run at the state before its last operator message, unstarted.

    A live run is refused: its `/undo` rides the steer channel.

    Returns:
        The new session id and the undone text in a dict and "", or None and the reason.
    """
    session_dir = model.session_dir_for(cwd, session_id)
    if session_dir is None:
        return None, f"no session {session_id!r}"
    if session_is_live(session_dir):
        return None, "the session is live; /undo rides the steer channel from the composer"
    said: list[str] = []
    reporter = app_reporter.Reporter(out=said.append, err=said.append)
    result = undo.undo_fork(None, session_dir.name, cwd=cwd, reporter=reporter)
    if result is None:
        return None, (spawn.capture_message(said[-1]) if said else "undo failed")
    child, text = result
    return {"new_session_id": child, "undone_text": text}, ""


def resume_run(
    cwd: pathlib.Path,
    session_id: str,
    text: str = "",
    *,
    preset: str = "",
    route: str = "",
    config_path: pathlib.Path | None = None,
) -> tuple[bool, str]:
    """Resume a finished or stopped run detached.

    Args:
        cwd: The repository.
        session_id: The run.
        text: The first steering instruction, or "".
        preset: The preset to continue under; "" as recorded.
        route: The `[provider/]model` to continue on; "" as recorded.
        config_path: An explicit config file, or None.

    Returns:
        Whether it started, and the note or the refusal; a live run is refused.
    """
    session_dir = model.session_dir_for(cwd, session_id)
    if session_dir is None:
        return False, f"no session {session_id!r}"
    if session_is_live(session_dir):
        return False, "the session is still live; steer it instead"
    if not text.strip() and listing.finished_needs_new_work(session_dir):
        # The CLI's own refusal would land on a detached process nobody reads.
        return False, (
            f"run {session_id!r} already finished (the agent called finish_session);"
            " type what to do next (Enter resumes it with the instruction)"
        )
    err = spawn.spawn_detached_resume(
        cwd, session_dir.name, steer=text, preset=preset, model=route, config_path=config_path
    )
    return (err == ""), (err or "resuming")


def run_plan(
    cwd: pathlib.Path, session_id: str, config_path: pathlib.Path | None = None
) -> tuple[dict[str, str] | None, str]:
    """Execute a finished plan by spawning `agent6 run --from <id>` detached.

    The plan session is untouched, so revising it keeps working.

    Returns:
        The new run id in a dict and "", or None and the reason.
    """
    session_dir = model.session_dir_for(cwd, session_id)
    if session_dir is None:
        return None, f"no session {session_id!r}"
    manifest_mode = ""
    with contextlib.suppress(manifest.ManifestError):
        manifest_mode = manifest.read_manifest(session_dir).mode
    if manifest_mode != "plan":
        return None, f"{session_id!r} is not a plan"
    plan_path = session_dir / "plan.md"
    try:
        plan_md = plan_path.read_text(encoding="utf-8")
    except OSError:
        return None, f"plan {session_id!r} has no plan.md that can be read yet"
    if not plan_md.strip():
        return None, f"plan {session_id!r} has an empty plan.md"
    runs = layout.bucket_dir(agent6_paths.state_dir(cwd), "runs")
    agent6_paths.mkdir_for_real_user(runs)
    new_dir, err = spawn.spawn_and_locate(
        [*spawn.agent6_argv(config_path), "run", "--from", session_id],
        cwd,
        before={p for p in runs.iterdir() if p.is_dir()},
        list_dirs=lambda: [p for p in runs.iterdir() if p.is_dir()],
        env={**os.environ, **spawn.DETACHED_RUN_ENV},
    )
    if new_dir is None:
        return None, err or "could not start the run"
    return {"run_id": new_dir.name}, ""


def stop_run(cwd: pathlib.Path, session_id: str, *, after_step: bool) -> tuple[bool, str]:
    """Stop a run now, or after the current step's tool results and auto-commit.

    Returns:
        Whether the stop landed, and the message.
    """
    session_dir = model.session_dir_for(cwd, session_id)
    if session_dir is None:
        return False, f"no session {session_id!r}"
    out = stop.stop_session(session_dir, after_step=after_step)
    return out.ok, out.message


def compact_run(cwd: pathlib.Path, session_id: str) -> tuple[bool, str]:
    """Ask a live run to compact its context at the next safe boundary.

    Returns:
        Whether the request landed, and the note or the refusal.
    """
    session_dir = _live_session_dir(cwd, session_id)
    if isinstance(session_dir, tuple):
        return session_dir
    if not ipc.request_compact(session_dir):
        return False, "could not write the compaction request"
    return True, "compaction requested"


def _machine_state_dir(cwd: pathlib.Path, name: str, state: str = "") -> pathlib.Path | None:
    """Return the per-state dir an answer belongs in.

    Args:
        cwd: The repository.
        name: The machine instance.
        state: The state dir the client rendered the prompt from (`0001-work`), or ""
            for the newest state; validated as one existing path component.

    Returns:
        The state dir, or None for an unknown machine or no active agent state.
    """
    machine_dir = model.machine_dir_for(cwd, name)
    if machine_dir is None:
        return None
    if state:
        if not layout.is_safe_session_id(state):
            return None
        target = machine_dir / "states" / state
        return target if target.is_dir() else None
    log = newest_state_log(machine_dir)
    return log.parent if log is not None else None


def _machine_dir_or_missing(cwd: pathlib.Path, name: str) -> pathlib.Path:
    """Return the instance dir the verb refusal reads; a missing one names the machine unknown."""
    return (
        model.machine_dir_for(cwd, name) or layout.machines_root(agent6_paths.state_dir(cwd)) / name
    )


def machine_stop(cwd: pathlib.Path, name: str) -> tuple[bool, str]:
    """Write the stop marker for a running machine; it parks at its next transition.

    Returns:
        Whether it succeeded, and the note or the refusal; nothing to stop is a note.
    """
    machine_dir = _machine_dir_or_missing(cwd, name)
    ok, answer = machine_state.verb_answer(machine_dir, name, "stop")
    if not ok or answer:
        return ok, answer
    write_stop_request(machine_dir)
    return True, "stop requested; the machine parks at its next transition boundary"


def machine_poke(
    cwd: pathlib.Path, name: str, *, data: Any = None, message: str = ""
) -> tuple[bool, str]:
    """Poke a waiting machine, optionally with a payload the next tool reads.

    Args:
        cwd: The repository.
        name: The machine instance.
        data: Any JSON payload; wins over the message.
        message: A string payload.

    Returns:
        Whether the poke landed, and the note or the refusal.
    """
    machine_dir = _machine_dir_or_missing(cwd, name)
    ok, refusal = machine_state.verb_answer(machine_dir, name, "poke")
    if not ok or refusal:
        return False, refusal
    payload: Any = data if data is not None else (message or None)
    try:
        MachineJournal(machine_dir).poke(payload)
    except JournalError as exc:
        return False, str(exc)
    return True, "poked"


def _state_dir_for_verb(
    cwd: pathlib.Path, name: str, verb: machine_state.MachineVerb, state: str
) -> pathlib.Path | tuple[bool, str]:
    """Return the agent-state dir a prompt answer or a steer lands in, or the refusal.

    A state the machine has left reads nothing and its prompt ids repeat in the next,
    so it is refused rather than rerouted.

    Args:
        cwd: The repository.
        name: The machine instance.
        verb: The verb, for the refusal.
        state: The state dir the client rendered, or "" for the newest.
    """
    machine_dir = _machine_dir_or_missing(cwd, name)
    ok, refusal = machine_state.verb_answer(machine_dir, name, verb)
    if not ok or refusal:
        return False, refusal
    agent_state = _machine_state_dir(cwd, name, state)
    if agent_state is None:
        return False, f"no active agent state for machine {name!r}"
    newest = newest_state_log(machine_dir)
    if newest is not None and agent_state != newest.parent:
        return False, (
            f"machine {name!r} has moved on from {agent_state.name} to {newest.parent.name}:"
            " the prompt shown is closed"
        )
    return agent_state


def machine_approve(
    cwd: pathlib.Path, name: str, prompt_id: str, answer: str, *, state: str = ""
) -> tuple[bool, str]:
    """Answer a pending approval in the agent state the prompt was rendered from.

    Returns:
        Whether the answer landed, and the note or the refusal.
    """
    target = _state_dir_for_verb(cwd, name, "answer", state)
    if not isinstance(target, pathlib.Path):
        return target
    prompt = open_approval(target)
    if prompt is None or prompt.id != prompt_id:
        return False, "that approval is no longer open"
    if not ipc.write_answer(target, prompt_id, answer):
        return False, ipc.ANSWERED_ELSEWHERE
    return True, "answered"


def machine_answer(
    cwd: pathlib.Path, name: str, question_id: str, answers: list[str], *, state: str = ""
) -> tuple[bool, str]:
    """Answer a pending `ask_user` prompt in the agent state it was rendered from.

    Returns:
        Whether the answers landed, and the note or the refusal.
    """
    target = _state_dir_for_verb(cwd, name, "answer", state)
    if not isinstance(target, pathlib.Path):
        return target
    prompt = open_question(target)
    if prompt is None or prompt.id != question_id:
        return False, "that question is no longer open"
    if len(answers) != len(prompt.questions):
        return False, f"that prompt has {len(prompt.questions)} question(s)"
    if not ipc.write_question_answers(target, question_id, answers):
        return False, ipc.ANSWERED_ELSEWHERE
    return True, "answered"


def machine_steer(cwd: pathlib.Path, name: str, text: str, *, state: str = "") -> tuple[bool, str]:
    """Steer the agent state the operator is viewing, under the run steer's contract.

    Returns:
        Whether the request landed, and the note or the refusal.
    """
    target = _state_dir_for_verb(cwd, name, "steer", state)
    if not isinstance(target, pathlib.Path):
        return target
    if not ipc.submit_steer(target, text):
        return False, "could not write the steer request"
    return True, "steer requested"


def merge_run(
    cwd: pathlib.Path, session_id: str, strategy: str = "", config_path: pathlib.Path | None = None
) -> tuple[bool, str]:
    """Merge a run's branch through `agent6 sessions merge`.

    Returns:
        Whether the CLI succeeded, and its message.
    """
    argv = [*spawn.agent6_argv(config_path), "sessions", "merge"]
    if strategy:
        argv += ["--strategy", strategy]
    argv += ["--", session_id]
    return spawn.run_cli_capture(argv, cwd)


def review_run(
    cwd: pathlib.Path, session_id: str, config_path: pathlib.Path | None = None
) -> tuple[dict[str, str] | None, str]:
    """Review a finished run's record through `agent6 sessions review`, a call of minutes.

    Returns:
        The review markdown in a dict and "", or None and the CLI's refusal.
    """
    ok, text = spawn.run_cli_output(
        [*spawn.agent6_argv(config_path), "sessions", "review", "--", session_id],
        cwd,
        timeout_s=900.0,
    )
    return ({"review": text}, "") if ok else (None, text)


def prune_sessions(
    cwd: pathlib.Path, *, delete_squashed: bool = False, config_path: pathlib.Path | None = None
) -> tuple[bool, str]:
    """Prune merged run branches through `agent6 sessions prune`.

    Args:
        cwd: The repository.
        delete_squashed: Pass `--delete-squashed`, without which a squash-merged branch stays.
        config_path: An explicit config file, or None.

    Returns:
        Whether the CLI succeeded, and its message.
    """
    argv = [*spawn.agent6_argv(config_path), "sessions", "prune"]
    if delete_squashed:
        argv.append("--delete-squashed")
    return spawn.run_cli_capture(argv, cwd)


def remove_session(
    cwd: pathlib.Path, session_id: str, config_path: pathlib.Path | None = None
) -> tuple[bool, str]:
    """Delete one run's history through `agent6 sessions rm`; the branch is prune's.

    Returns:
        Whether the CLI succeeded, and its message; a live run is refused.
    """
    return spawn.run_cli_capture(
        [*spawn.agent6_argv(config_path), "sessions", "rm", "--", session_id], cwd
    )


def remove_asks(cwd: pathlib.Path, config_path: pathlib.Path | None = None) -> tuple[bool, str]:
    """Clear every saved ask through `agent6 sessions rm --asks`.

    Returns:
        Whether the CLI succeeded, and its message.
    """
    return spawn.run_cli_capture([*spawn.agent6_argv(config_path), "sessions", "rm", "--asks"], cwd)


def set_config(
    cwd: pathlib.Path,
    key: str,
    value: str,
    *,
    repo: bool = False,
    config_path: pathlib.Path | None = None,
) -> tuple[bool, str]:
    """Set one config leaf through `agent6 config set`, which validates key and value.

    Returns:
        Whether the CLI succeeded, and its message.
    """
    argv = [*spawn.agent6_argv(config_path), "config", "set"]
    if repo:
        argv.append("--repo")
    argv += ["--", key, value]
    return spawn.run_cli_capture(argv, cwd)


def add_provider(
    cwd: pathlib.Path,
    name: str,
    *,
    api_format: str,
    deployment: str = "",
    base_url: str = "",
    api_key_env: str = "",
    repo: bool = False,
) -> tuple[bool, str]:
    """Add or update `[providers.<name>]` through the shared config writer.

    An existing block keeps its other keys; a blank optional field is omitted and
    defaults from the format and deployment.

    Returns:
        Whether the write landed, and the note or the refusal.
    """
    name = name.strip()
    if not name:
        return False, "a provider name is required"
    fields: dict[str, io.ConfigLeafValue] = {"api_format": api_format}
    if deployment and deployment != "direct":
        fields["deployment"] = deployment
    if base_url.strip():
        fields["base_url"] = base_url.strip()
    if api_key_env.strip():
        fields["api_key_env"] = api_key_env.strip()
    try:
        err = write.set_config_leaves(cwd, f"providers.{name}", fields, to_repo=repo)
    except errors.OperatorError as exc:
        return False, str(exc)
    if err:
        return False, err
    return True, f"set [providers.{name}]"


def unset_config(
    cwd: pathlib.Path, key: str, *, repo: bool = False, config_path: pathlib.Path | None = None
) -> tuple[bool, str]:
    """Unset one config leaf through `agent6 config unset`.

    Returns:
        Whether the CLI succeeded, and its message.
    """
    argv = [*spawn.agent6_argv(config_path), "config", "unset"]
    if repo:
        argv.append("--repo")
    argv += ["--", key]
    return spawn.run_cli_capture(argv, cwd)
