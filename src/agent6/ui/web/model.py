# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Pure JSON payload builders for the web UI.

The web server is a thin renderer: every payload it serves is built here from the
shared read-side (viewmodel folds, config_layer, transcript_render, the machine
spec/journal). Pure functions, no HTTP or threads, so the run/machine snapshots
are exactly `run_state_as_dict` / `machine_state_as_dict` (identical to
`agent6 attach --json`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent6.config import Config, ConfigError
from agent6.config.layer import load_effective, resolved_state_dir
from agent6.machine import JournalError, MachineError, MachineJournal, load_machine
from agent6.models.cache import cached_models, list_models
from agent6.models.validate import known_models
from agent6.runs.manifest import ManifestError, read_manifest
from agent6.secrets import resolve_api_key
from agent6.viewmodel import (
    fold_machine,
    fold_run,
    fold_transcript,
    is_run_husk,
    is_winner,
    machine_state_as_dict,
    newest_state_log,
    run_compare,
    run_state_as_dict,
    summarize_run_dir,
    tail_events,
    task_snippet,
)
from agent6.viewmodel.config_view import render_show
from agent6.viewmodel.transcript_style import item_lines

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
    that is not a single safe path component. Husks are skipped so an orphaned
    dir in runs/ cannot shadow a real ask of the same id."""
    if not _safe_component(run_id):
        return None
    for sub in RUN_SUBDIRS:
        d = state_dir_for(cwd) / sub / run_id
        if d.is_dir() and not is_run_husk(d):
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
        "winner": is_winner(run_dir),  # fan-out compare winner: a ★ on the hub row
    }


def _list_runs(cwd: Path) -> list[dict[str, Any]]:
    """All runs (runs/ + asks/) summarized, newest first by last-activity time.
    Husks (never-started dirs) are skipped, the same rule as `agent6 runs`."""
    dirs: list[Path] = []
    for sub in RUN_SUBDIRS:
        d = state_dir_for(cwd) / sub
        if d.is_dir():
            dirs.extend(p for p in d.iterdir() if p.is_dir() and not is_run_husk(p))
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


def _list_drafts(cwd: Path) -> list[dict[str, Any]]:
    """`machine create` drafts summarized like runs (their logs.jsonl is a
    run-style authoring log), newest first, so the machines page can link to
    the #/draft/<name> view."""
    summaries = [_run_summary(p) for p in draft_dir_paths(cwd)]
    summaries.sort(key=lambda s: s["mtime"], reverse=True)
    return summaries


def list_machine_files(cwd: Path) -> list[dict[str, str]]:
    """Authored .asm.toml machine source files (cwd top level + machines/ subdir):
    the ones a user can run or use as a create starting point."""
    found: set[Path] = set(cwd.glob("*.asm.toml"))
    sub = cwd / "machines"
    if sub.is_dir():
        found.update(sub.glob("*.asm.toml"))
    return [{"path": str(p), "name": p.name} for p in sorted(found)]


def hub_payload(cwd: Path) -> dict[str, Any]:
    """The hub: every run, machine instance, and machine-create draft, plus the
    authored machine files (to run or create from), summarized for the listing."""
    return {
        "runs": _list_runs(cwd),
        "machines": _list_machines(cwd),
        "machine_files": list_machine_files(cwd),
        "drafts": _list_drafts(cwd),
    }


# --- run snapshot + conversation ----------------------------------------------


def manifest_branches(run_dir: Path) -> dict[str, str]:
    """Branch facts from the run's manifest (run_branch / base_branch /
    merged_into) for the run header. The event fold doesn't carry them, but a
    web user needs to SEE where a run's work lives and where Merge lands --
    consecutive spawns chain branches invisibly otherwise. Empty for a run with
    no manifest (or branch_per_run off)."""
    try:
        manifest = read_manifest(run_dir)
    except ManifestError:
        return {}
    out: dict[str, str] = {}
    if manifest.run_branch:
        out["run_branch"] = manifest.run_branch
    if manifest.base_branch:
        out["base_branch"] = manifest.base_branch
    if manifest.merged and manifest.merged.into:
        out["merged_into"] = manifest.merged.into
    return out


def manifest_header(run_dir: Path) -> dict[str, Any]:
    """Manifest-derived run-header fields the event fold doesn't carry: branch
    facts (run/base/merged) and the fan-out compare outcome (rank/winner/
    rationale). Merged into the RunState snapshot by BOTH the one-shot
    `/api/run/<id>` and the SSE stream, so the header the page paints from can
    never drift between them. Empty for a run with no (readable) manifest."""
    header: dict[str, Any] = dict(manifest_branches(run_dir))
    compare = run_compare(run_dir)
    if compare is not None:
        header["compare"] = compare.model_dump(mode="json")
    return header


def run_snapshot(run_dir: Path) -> dict[str, Any]:
    """A run's folded RunState as the wire dict (the same fold as
    `agent6 attach <id> --json`), plus dir-derived metadata: the authoritative
    run id and the manifest's branch/compare facts."""
    logs = run_dir / "logs.jsonl"
    snap = run_state_as_dict(fold_run(tail_events(logs, follow=False)))
    # The dir we looked up under is the authoritative run id: stamp it so the
    # payload never carries an empty run_id (older logs predate run.start
    # carrying one) and matches sibling endpoints like /conversation.
    snap["run_id"] = snap.get("run_id") or run_dir.name
    snap.update(manifest_header(run_dir))
    return snap


def conversation_items(log_path: Path) -> list[dict[str, Any]]:
    """The log folded into rendered conversation items, one entry per
    ``TranscriptItem``: its ``kind``, the collapsed ``lines`` (lists of
    ``[text, style]`` spans from the shared ``item_lines`` renderer, the same
    fold the CLI stream and the TUI conversation view draw), and ``full`` (the
    expanded rendering) only when it differs, so the page can offer per-item
    expansion without re-implementing any clipping client-side."""
    out: list[dict[str, Any]] = []
    for item in fold_transcript(list(tail_events(log_path, follow=False))):
        collapsed = item_lines(item, detail="collapsed")
        expanded = item_lines(item, detail="expanded")
        entry: dict[str, Any] = {"kind": item.kind, "lines": collapsed}
        if expanded != collapsed:
            entry["full"] = expanded
        out.append(entry)
    return out


def conversation_payload(run_dir: Path) -> dict[str, Any]:
    """A run's conversation, folded from its event log."""
    return {"run_id": run_dir.name, "items": conversation_items(run_dir / "logs.jsonl")}


def machine_conversation_payload(machine_dir: Path) -> dict[str, Any]:
    """The conversation of the machine's most recent agent-state execution
    (empty when no agent state has produced a log yet), plus the per-state dir
    it came from so a client can tell when the machine advanced."""
    log = newest_state_log(machine_dir)
    if log is None:
        return {"state_dir": "", "items": []}
    return {"state_dir": log.parent.name, "items": conversation_items(log)}


# --- machine snapshot (structure + watch + reasoning) -----------------------


def machine_snapshot(machine_dir: Path) -> dict[str, Any]:
    """A machine instance's folded MachineState as the wire dict. Identical to
    `agent6 attach <name> --json`."""
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


def config_suggestions(cwd: Path, key: str) -> list[str]:
    """Value suggestions for one open-text config leaf, from the same sources
    the TUI config page and CLI TAB completion use: ``models.<role>.provider``
    offers the configured provider names, ``models.<role>.model`` the role's
    provider's model ids (cache-first, refreshed from the live listing; the
    fetch dials only that operator-configured provider's base_url). Enum leaves
    already carry their choices in the config payload; everything else (and any
    error) suggests nothing -- suggestions are best-effort, never a failure.

    The pseudo-key ``parallel.models`` (the new-work composer's ``/parallel``
    spec autocomplete) returns the worker's model plus the worker provider's
    cached listing (lanes inherit the worker provider) -- exactly the set
    `run --parallel` validation accepts -- cache-only so a keystroke never
    blocks the server on the network."""
    if key == "parallel.models":
        return _parallel_model_suggestions(cwd)
    parts = key.split(".")
    if len(parts) != 3 or parts[0] != "models" or parts[2] not in ("provider", "model"):
        return []
    try:
        cfg = load_effective(cwd).config
    except ConfigError:
        return []
    return _role_value_suggestions(cfg, parts[1], parts[2])


def _role_value_suggestions(cfg: Config, role: str, kind: str) -> list[str]:
    """`models.<role>.provider` -> configured provider names; `.model` -> that
    role's provider's model ids (cache-first, refreshed from the live listing)."""
    if kind == "provider":
        return sorted(cfg.providers)
    provider = getattr(getattr(cfg.models, role, None), "provider", None)
    if not provider:
        return []
    entry = cfg.providers.get(provider)
    if entry is None:
        return cached_models(provider)
    api_key = resolve_api_key(provider, getattr(entry, "api_key_env", None))
    return list_models(provider, entry, api_key)


def _parallel_model_suggestions(cwd: Path) -> list[str]:
    """Model ids a `/parallel` lane can run (worker-scoped, via `known_models`),
    for the spec autocomplete in the new-work composer. Cache-only; a broken
    config suggests nothing."""
    try:
        cfg = load_effective(cwd).config
    except ConfigError:
        return []
    return sorted(known_models(cfg))
