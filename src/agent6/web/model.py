# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Pure JSON payload builders for the web UI.

The web server is a thin renderer: every payload it serves is built here from the
shared read-side (viewmodel folds, config_layer, transcript_render, the machine
spec/journal), with no HTTP, no threads, no UI toolkit. Keeping the payloads
pure makes them unit-testable and keeps the wire form identical to
`agent6 watch --json` (the run/machine snapshots ARE `run_state_as_dict` /
`machine_state_as_dict`).
"""

from __future__ import annotations

import contextlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from agent6.config_layer import load_effective, render_show, resolved_state_dir
from agent6.machine import MachineError, MachineJournal, load_machine
from agent6.transcript_render import fold_conversation, load_transcripts
from agent6.viewmodel import (
    fold_machine,
    fold_run,
    machine_state_as_dict,
    newest_state_log,
    run_state_as_dict,
    tail_events,
)

RUN_SUBDIRS = ("runs", "asks")
_STALE_AFTER_S = 600.0


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


def run_dir_for(cwd: Path, run_id: str) -> Path | None:
    """Locate a run dir by exact id across runs/ and asks/ (no prefix match: the
    web client always sends the full id from the hub payload)."""
    for sub in RUN_SUBDIRS:
        d = state_dir_for(cwd) / sub / run_id
        if (d / "logs.jsonl").is_file() or d.is_dir():
            return d
    return None


def machine_dir_for(cwd: Path, name: str) -> Path | None:
    d = machines_root(cwd) / name
    return d if d.is_dir() else None


# --- hub listing -------------------------------------------------------------


def _run_mtime(run_dir: Path) -> float:
    """Last-activity time of a run: the mtime of its logs.jsonl (not the dir,
    which a viewer bumps merely by opening it). Falls back to the dir mtime."""
    for candidate in (run_dir / "logs.jsonl", run_dir):
        try:
            return candidate.stat().st_mtime
        except OSError:
            continue
    return 0.0


def _run_summary(run_dir: Path) -> dict[str, Any]:
    """A cheap one-line summary for the hub: id, mode, task, status, when, usd.

    Single pass over logs.jsonl reading run.start (mode/task), the last run.end
    (status), and the last budget.update (usd cost), so it stays fast over a
    directory of many runs."""
    logs = run_dir / "logs.jsonl"
    mode, task, status, usd = "?", "", "running", 0.0
    mtime = _run_mtime(run_dir)
    if not logs.is_file():
        return {
            "id": run_dir.name,
            "mode": mode,
            "task": "(no logs)",
            "status": "—",
            "mtime": mtime,
            "usd": 0.0,
        }
    try:
        with logs.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                etype = ev.get("type")
                if etype == "run.start":
                    mode = str(ev.get("mode", mode))
                    task = str(ev.get("user_task", ""))
                elif etype == "run.end":
                    status = "ok" if ev.get("all_passed") else "done"
                elif etype == "budget.update":
                    usd = float(ev.get("usd_total", usd) or 0.0)
        if status == "running" and (time.time() - logs.stat().st_mtime) > _STALE_AFTER_S:
            status = "stale"
    except OSError:
        pass
    if mode == "ask":
        transcript = run_dir / "transcript.md"
        with contextlib.suppress(OSError):
            task = transcript.read_text(encoding="utf-8")
    return {
        "id": run_dir.name,
        "mode": mode,
        "task": _first_task_line(task)[:100],
        "status": status,
        "mtime": mtime,
        "usd": usd,
    }


def _first_task_line(task: str) -> str:
    """First non-blank line of the task, skipping the ask context preamble."""
    for line in task.splitlines():
        s = line.strip()
        if s in {"# agent6 ask", "## Question"} or s.startswith("<"):
            continue
        if s == "## Answer":
            break
        if s:
            return s
    return task.strip()


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
        with contextlib.suppress(MachineError, OSError):
            spec = load_machine(d / "machine.asm.toml")
            ms = fold_machine(spec, MachineJournal(d).read())
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


def hub_payload(cwd: Path) -> dict[str, Any]:
    """The hub: every run and machine instance, summarized for the listing."""
    return {"runs": _list_runs(cwd), "machines": _list_machines(cwd)}


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


def machine_reasoning_snapshot(machine_dir: Path) -> dict[str, Any]:
    """The RunState of the machine's most recent agent-state execution: the live
    reasoning + tool calls inside the state the machine is running. Empty when no
    agent state has produced a log yet."""
    log = newest_state_log(machine_dir)
    if log is None:
        return {}
    return run_state_as_dict(fold_run(tail_events(log, follow=False)))


# --- config ------------------------------------------------------------------


def config_payload(cwd: Path, config_path: Path | None = None) -> dict[str, Any]:
    """The effective config as a per-leaf view (value/effective/default/source/
    modified/adaptive/type/choices), keyed by dotted key. The same structure
    `agent6 config show --json` prints; never includes secrets."""
    eff = load_effective(cwd, config_path)
    return json.loads(render_show(eff, as_json=True))
