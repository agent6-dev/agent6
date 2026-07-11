# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Pure JSON payload builders for the web UI.

The web server is a thin renderer: every payload it serves is built here from the
shared read-side (viewmodel folds, config_layer, transcript_render, the machine
spec/journal). Pure functions, no HTTP or threads, so the run/machine snapshots
are exactly `run_state_as_dict` / `machine_state_as_dict` (identical to
`agent6 watch --json`).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from agent6.config.layer import load_effective, resolved_state_dir
from agent6.machine import JournalError, MachineError, MachineJournal, load_machine
from agent6.transcript_render import fold_conversation, load_transcripts
from agent6.ui.viewmodel import (
    fold_machine,
    fold_run,
    machine_state_as_dict,
    newest_state_log,
    run_state_as_dict,
    summarize_run_dir,
    tail_events,
    task_snippet,
)
from agent6.ui.viewmodel.config_view import render_show

RUN_SUBDIRS = ("runs", "asks")


# --- directory layout --------------------------------------------------------


def state_dir_for(cwd: Path) -> Path:
    """The per-repo agent6 state dir (runs/asks/machines live under it)."""
    return resolved_state_dir(cwd)


def runs_root(cwd: Path) -> Path:
    return state_dir_for(cwd) / "runs"


def asks_root(cwd: Path) -> Path:
    return state_dir_for(cwd) / "asks"


def machines_root(cwd: Path) -> Path:
    return state_dir_for(cwd) / "machines"


def is_safe_component(name: str) -> bool:
    """True iff *name* is a single path component (no separator, not `.`/`..`),
    so a browser-supplied run id or machine name cannot traverse out of its dir."""
    return bool(name) and "/" not in name and "\\" not in name and name not in {".", ".."}


_safe_component = is_safe_component


def run_dir_for(cwd: Path, run_id: str) -> Path | None:
    """Locate a run dir by exact id across runs/ and asks/ (no prefix match: the
    web client always sends the full id from the hub payload). Rejects a run_id
    that is not a single safe path component."""
    if not _safe_component(run_id):
        return None
    for sub in RUN_SUBDIRS:
        d = state_dir_for(cwd) / sub / run_id
        if d.is_dir():
            return d
    return None


def machine_dir_for(cwd: Path, name: str) -> Path | None:
    if not _safe_component(name):
        return None
    d = machines_root(cwd) / name
    return d if d.is_dir() else None


def draft_dir_for(cwd: Path, name: str) -> Path | None:
    """A `machine create` draft dir by name. Its logs.jsonl is a run-style log of
    the authoring agent, so it is watched through the run endpoints."""
    if not _safe_component(name):
        return None
    d = state_dir_for(cwd) / "machine-drafts" / name
    return d if d.is_dir() else None


def run_dir_paths(cwd: Path) -> list[Path]:
    """Every run/ask directory (unordered): the before/after set for spawn-and-locate."""
    out: list[Path] = []
    for sub in RUN_SUBDIRS:
        d = state_dir_for(cwd) / sub
        if d.is_dir():
            out.extend(p for p in d.iterdir() if p.is_dir())
    return out


def draft_dir_paths(cwd: Path) -> list[Path]:
    """Every machine-create draft directory (where `machine create` writes)."""
    d = state_dir_for(cwd) / "machine-drafts"
    return [p for p in d.iterdir() if p.is_dir()] if d.is_dir() else []


# --- hub listing -------------------------------------------------------------


def _run_summary(run_dir: Path) -> dict[str, Any]:
    """The hub's one-line run summary, from the shared scanner: id, mode, task,
    status (+ reason detail), when, usd. The status words come from
    ``viewmodel.status_word``, so a provider_error death reads "failed" here
    exactly as in the TUI hub and `agent6 runs list`."""
    s = summarize_run_dir(run_dir)
    return {
        "id": s.run_id,
        "mode": s.mode,
        "task": task_snippet(s.task)[:100],
        "status": s.status,
        "reason": s.reason,
        "mtime": s.mtime,
        "usd": s.cost_usd,
    }


def _list_runs(cwd: Path) -> list[dict[str, Any]]:
    """All runs (runs/ + asks/) summarized, newest first by last-activity time."""
    dirs: list[Path] = []
    for sub in RUN_SUBDIRS:
        d = state_dir_for(cwd) / sub
        if d.is_dir():
            dirs.extend(p for p in d.iterdir() if p.is_dir())
    summaries = [_run_summary(p) for p in dirs]
    summaries.sort(key=lambda s: s["mtime"], reverse=True)
    return summaries


def _list_machines(cwd: Path) -> list[dict[str, Any]]:
    """Machine instances under the state machines/ dir, newest first. Each is a
    watchable run of an authored machine (holds machine.asm.toml + journal)."""
    root = machines_root(cwd)
    if not root.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for d in root.iterdir():
        if not d.is_dir() or not (d / "machine.asm.toml").is_file():
            continue
        entry: dict[str, Any] = {"name": d.name, "mtime": _machine_mtime(d), "status": "—"}
        try:
            spec = load_machine(d / "machine.asm.toml")
            ms = fold_machine(spec, MachineJournal(d).read())
        except (MachineError, OSError):
            # A corrupt source or journal (JournalError is a MachineError) must
            # not drop the instance from the hub or 500 the listing; show it as
            # unreadable so the operator sees something is wrong.
            entry["status"] = "unreadable"
        else:
            entry["machine"] = ms.machine
            entry["current"] = ms.current
            entry["status"] = ms.ended.status if ms.ended is not None else "running"
        out.append(entry)
    out.sort(key=lambda e: e["mtime"], reverse=True)
    return out


def _machine_mtime(machine_dir: Path) -> float:
    for candidate in (machine_dir / "journal.jsonl", machine_dir):
        try:
            return candidate.stat().st_mtime
        except OSError:
            continue
    return 0.0


def list_machine_files(cwd: Path) -> list[dict[str, str]]:
    """Authored .asm.toml machine source files (cwd top level + machines/ subdir):
    the ones a user can run or use as a create starting point."""
    found: set[Path] = set(cwd.glob("*.asm.toml"))
    sub = cwd / "machines"
    if sub.is_dir():
        found.update(sub.glob("*.asm.toml"))
    return [{"path": str(p), "name": p.name} for p in sorted(found)]


def hub_payload(cwd: Path) -> dict[str, Any]:
    """The hub: every run and machine instance, plus the authored machine files
    (to run or create from), summarized for the listing."""
    return {
        "runs": _list_runs(cwd),
        "machines": _list_machines(cwd),
        "machine_files": list_machine_files(cwd),
    }


# --- run snapshot + transcript ----------------------------------------------


def run_snapshot(run_dir: Path) -> dict[str, Any]:
    """A run's folded RunState as the wire dict. Identical to
    `agent6 watch <id> --json` so `curl` and the CLI agree."""
    logs = run_dir / "logs.jsonl"
    return run_state_as_dict(fold_run(tail_events(logs, follow=False)))


def transcript_payload(run_dir: Path) -> dict[str, Any]:
    """The full conversation as ordered turns (provider-agnostic)."""
    turns = fold_conversation(load_transcripts(run_dir / "transcripts"))
    return {"run_id": run_dir.name, "turns": [asdict(t) for t in turns]}


# --- machine snapshot (structure + watch + reasoning) -----------------------


def machine_snapshot(machine_dir: Path) -> dict[str, Any]:
    """A machine instance's folded MachineState as the wire dict. Identical to
    `agent6 watch <name> --json`."""
    spec = load_machine(machine_dir / "machine.asm.toml")
    ms = fold_machine(spec, MachineJournal(machine_dir).read())
    return machine_state_as_dict(ms)


def machine_is_parked(machine_dir: Path) -> bool:
    """True when the instance is parked in an armed wait (a PendingWait is
    persisted). Under --exit-on-wait scheduling a parked machine legitimately
    has no live process, so liveness probes must not read "dead pid" as
    "crashed" while this holds. A corrupt wait file counts as parked: better
    to keep streaming than to close on a guess."""
    try:
        return MachineJournal(machine_dir).read_pending_wait() is not None
    except JournalError:
        return True


def machine_reasoning_snapshot(machine_dir: Path) -> dict[str, Any]:
    """The RunState of the machine's most recent agent-state execution: the live
    reasoning + tool calls inside the state the machine is running. Empty when no
    agent state has produced a log yet.

    Carries ``state_dir`` (the per-state dir name, e.g. ``0001-work``) so a
    client echoes it back when answering a prompt: prompt ids reset per state
    (``approval-1`` in every state), so routing an answer to whichever state is
    newest AT POST TIME would misdeliver it if the machine advanced meanwhile.
    """
    log = newest_state_log(machine_dir)
    if log is None:
        return {}
    snap = run_state_as_dict(fold_run(tail_events(log, follow=False)))
    snap["state_dir"] = log.parent.name
    return snap


# --- config ------------------------------------------------------------------


def config_payload(cwd: Path, config_path: Path | None = None) -> dict[str, Any]:
    """The effective config as a per-leaf view (value/effective/default/source/
    modified/adaptive/type/choices), keyed by dotted key. The same structure
    `agent6 config show --json` prints; never includes secrets."""
    eff = load_effective(cwd, config_path)
    return json.loads(render_show(eff, as_json=True))
