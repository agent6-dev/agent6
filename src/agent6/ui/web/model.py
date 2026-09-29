# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Build the web UI's JSON payloads.

Every payload comes from the shared read side (the viewmodel folds, the config
layer, the machine spec and journal), with no HTTP or threads, so the run and
machine snapshots are the same dicts `agent6 attach --json` prints.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from agent6.app.confine import resolved_config_values
from agent6.app.parallel import subordinate_workdir_root
from agent6.config import ConfigError
from agent6.config.io import format_toml_value
from agent6.config.layer import available_preset_names, load_effective
from agent6.git_ops import EMPTY_TREE, commit_diff, diff_range, run_ref_tips
from agent6.models.choices import (
    available_routes,
    config_value_choices,
    default_label,
    default_preset,
    default_route,
    resume_defaults,
)
from agent6.paths import state_dir
from agent6.sessions.ipc import worker_is_alive
from agent6.sessions.layout import (
    HUB_BUCKETS,
    LOGS_NAME,
    bucket_dir,
    is_safe_session_id,
    machines_root,
)
from agent6.sessions.manifest import ManifestError, read_manifest
from agent6.viewmodel import (
    MachineSummary,
    NewestExecutionFold,
    fold_session,
    fold_transcript,
    is_session_husk,
    is_winner,
    machine_files,
    machine_instance_dirs,
    newest_state_log,
    operator_inputs,
    restate,
    session_dirs,
    session_state_as_dict,
    summarize_machine_dir,
    summarize_session_dir,
    summary_row,
    tail_events,
)
from agent6.viewmodel.config_view import render_show
from agent6.viewmodel.format import format_when, status_label, status_level
from agent6.viewmodel.listing import nested_rows, row_json
from agent6.viewmodel.transcript_style import item_lines


def session_dir_for(cwd: Path, session_id: str) -> Path | None:
    """Locate a session dir by its exact id across the hub buckets.

    Husks are skipped so an orphaned dir cannot shadow a real session of the same id.

    Args:
        cwd: The repository.
        session_id: The full id from the hub payload.

    Returns:
        The session dir, or None for an unsafe id, a missing session or an id in two buckets.
    """
    if not is_safe_session_id(session_id):
        return None
    found: Path | None = None
    for sub in HUB_BUCKETS:
        d = bucket_dir(state_dir(cwd), sub) / session_id
        if d.is_dir() and not is_session_husk(d):
            if found is not None:
                return None
            found = d
    return found


def machine_dir_for(cwd: Path, name: str) -> Path | None:
    """Return a machine instance's dir by name, or None for an unsafe or unknown name."""
    if not is_safe_session_id(name):
        return None
    d = machines_root(state_dir(cwd)) / name
    return d if d.is_dir() else None


def draft_dir_for(cwd: Path, name: str) -> Path | None:
    """Return a `machine create` draft's dir by name, or None when there is none.

    The draft's logs.jsonl is the authoring agent's run-style log, watched through
    the run endpoints.
    """
    if not is_safe_session_id(name):
        return None
    d = bucket_dir(state_dir(cwd), "machines") / name
    return d if d.is_dir() and not is_session_husk(d) else None


def draft_workspace(cwd: Path, name: str, config_path: Path | None) -> Path | None:
    """Return the workspace a `machine create` draft commits in, or None once it is gone.

    The workspace is a repository of its own beside the other subordinate working
    trees; publishing the draft removes it.
    """
    try:
        cfg = load_effective(cwd, config_path).config
    except ConfigError:
        return None
    workspace = subordinate_workdir_root(cfg, cwd, name)
    return workspace if (workspace / ".git").exists() else None


def draft_step_diff_payload(
    workspace: Path, sha: str, *, cumulative: bool
) -> tuple[dict[str, Any] | None, str]:
    """Return the patch one draft step introduced, or the whole bundle as of that step.

    A draft starts from the empty tree, so the cumulative patch is the bundle so far.

    Args:
        workspace: The draft's workspace.
        sha: The step's commit.
        cumulative: Whole bundle instead of the one step.

    Returns:
        The payload and "", or None and the reason.
    """
    return _diff_payload(
        workspace, sha, base=EMPTY_TREE, cumulative=cumulative, miss="not a commit of this draft"
    )


def _diff_payload(
    repo: Path, sha: str, *, base: str, cumulative: bool, miss: str
) -> tuple[dict[str, Any] | None, str]:
    """Return one step's patch, or the chain `base..sha` when cumulative and a base is known.

    Args:
        repo: The repository.
        sha: The step's commit.
        base: The chain's base; "" when unknown.
        cumulative: The chain instead of the one step.
        miss: The reason to give when the sha has no diff.

    Returns:
        The payload (saying which it holds) and "", or None and the reason.
    """
    if not re.fullmatch(r"[0-9a-f]{7,40}", sha):
        return None, f"not a commit sha: {sha!r}"
    whole = cumulative and bool(base)
    patch = diff_range(repo, base, sha) if whole else commit_diff(repo, sha)
    if not patch:
        return None, f"no diff for {sha[:12]} ({miss})"
    return {"cumulative": whole, "patch": patch}, ""


def draft_dir_paths(cwd: Path) -> list[Path]:
    """Return every `machine create` draft directory."""
    d = bucket_dir(state_dir(cwd), "machines")
    return [p for p in d.iterdir() if p.is_dir()] if d.is_dir() else []


# --- hub listing -------------------------------------------------------------


def _list_sessions(cwd: Path) -> list[dict[str, Any]]:
    """Return every session the hub lists, newest first, a fan-out's lanes under its row."""
    tips = run_ref_tips(cwd)
    dirs = session_dirs(state_dir(cwd))
    winners = {p.name for p in dirs if is_winner(p)}
    rows = nested_rows(summarize_session_dir(p, branch_tips=tips) for p in dirs)
    return [row_json(r, winners=winners) for r in rows]


def _machine_row(s: MachineSummary) -> dict[str, Any]:
    """Return one machine-instance row for the hub."""
    entry: dict[str, Any] = {
        "name": s.name,
        "mtime": s.mtime,
        "when": format_when(s.mtime) if s.mtime else "",
        "status": s.status,
        "level": status_level(s.status),
    }
    if s.status != "unreadable":
        entry["machine"] = s.machine
        entry["current"] = s.current
    if s.reason:
        # A live machine blocked on an operator prompt carries a reason too.
        entry["label"] = status_label(s.status, s.reason)
    return entry


def _list_machines(cwd: Path) -> list[dict[str, Any]]:
    """Return the machine instances, newest first, summarized by the shared fold."""
    dirs = machine_instance_dirs(state_dir(cwd))
    return [_machine_row(summarize_machine_dir(d)) for d in dirs]


def _list_drafts(cwd: Path) -> list[dict[str, Any]]:
    """Return the `machine create` drafts summarized like runs, newest first."""
    summaries: list[dict[str, Any]] = [
        summary_row(summarize_session_dir(p, branch_tips={}), winner=is_winner(p))
        for p in draft_dir_paths(cwd)
        if not is_session_husk(p)
    ]
    summaries.sort(key=lambda s: s["mtime"], reverse=True)
    return summaries


def list_machine_files(cwd: Path) -> list[dict[str, str]]:
    """Return the hub's machine-file rows."""
    return [{"path": str(p), "name": p.name} for p in machine_files(cwd)]


def routes_payload(
    cwd: Path, config_path: Path | None, *, mode: str, preset: str
) -> dict[str, Any]:
    """Return the new-work composer's model picker.

    Args:
        cwd: The repository.
        config_path: An explicit config file, or None.
        mode: The session mode the default route is for.
        preset: The preset the default route is under.

    Returns:
        Every `provider/model` the config can run, and the label of the no-flag entry.
    """
    return {
        "routes": available_routes(cwd, config_path),
        "default_label": default_label(default_route(cwd, config_path, mode, preset)),
    }


def resume_defaults_payload(
    cwd: Path, config_path: Path | None, session_dir: Path, *, preset: str
) -> dict[str, str]:
    """Return the resume row's no-flag preset and model labels under a picked preset."""
    preset_label, model_label = resume_defaults(cwd, config_path, session_dir, preset=preset)
    return {"preset_label": preset_label, "model_label": model_label}


def hub_payload(cwd: Path, config_path: Path | None = None) -> dict[str, Any]:
    """Return the hub: sessions, machines, drafts, machine files and the preset choices."""
    return {
        "sessions": _list_sessions(cwd),
        "machines": _list_machines(cwd),
        "machine_files": list_machine_files(cwd),
        "drafts": _list_drafts(cwd),
        "presets": available_preset_names(cwd, config_path),
        "preset_default_label": default_label(default_preset(cwd, config_path)),
    }


# --- run snapshot + conversation ----------------------------------------------


def conversation_items(
    events: list[dict[str, Any]], *, worker_dead: bool = False
) -> list[dict[str, Any]]:
    """Fold events into rendered conversation items.

    Each item carries its `kind`, the collapsed `lines` as `[text, style]` spans from
    the shared renderer, and `full` only when the expanded rendering differs.

    Args:
        events: The session's events.
        worker_dead: Settle the calls still open as never returned.

    Returns:
        One entry per transcript item.
    """
    out: list[dict[str, Any]] = []
    for item in fold_transcript(events, worker_dead=worker_dead):
        collapsed = item_lines(item, detail="collapsed")
        expanded = item_lines(item, detail="expanded")
        entry: dict[str, Any] = {"kind": item.kind, "lines": collapsed}
        if expanded != collapsed:
            entry["full"] = expanded
        out.append(entry)
    return out


def conversation_payload(session_dir: Path) -> dict[str, Any]:
    """Return a run's conversation and the operator's past inputs, from one read of the log."""
    events = list(tail_events(session_dir / LOGS_NAME, follow=False))
    return {
        "items": conversation_items(events, worker_dead=not worker_is_alive(session_dir)),
        "operator_inputs": operator_inputs(events),
    }


def restate_payload(session_dir: Path) -> dict[str, Any]:
    """Return `/restate` for the web composer, over the session's whole journal."""
    events = list(tail_events(session_dir / LOGS_NAME, follow=False))
    return {"text": restate(events, worker_dead=not worker_is_alive(session_dir))}


def machine_conversation_payload(machine_dir: Path) -> dict[str, Any]:
    """Return the conversation of the machine's newest agent-state execution, or no items."""
    log = newest_state_log(machine_dir)
    if log is None:
        return {"items": []}
    events = list(tail_events(log, follow=False))
    # The machine's worker (one pid for every state) is the one to probe.
    items = conversation_items(events, worker_dead=not worker_is_alive(machine_dir))
    return {"items": items}


# --- machine snapshot (structure + watch + reasoning) -----------------------


def machine_reasoning_snapshot(
    machine_dir: Path, *, fold: NewestExecutionFold | None = None
) -> dict[str, Any]:
    """Return the session state of the machine's newest agent-state execution.

    The snapshot carries `state_dir` so a client echoes it when answering a prompt
    (prompt ids reset per state), and `last_event_ep` rather than an age so the
    payload changes only when something happened.

    Args:
        machine_dir: The machine instance's directory.
        fold: A fold to refresh, or None to fold the newest log afresh.

    Returns:
        The wire form plus the two keys, or empty before any agent state has a log.
    """
    if fold is not None:
        log, state = fold.refresh(machine_dir), fold.state
    else:
        log = newest_state_log(machine_dir)
        state = fold_session(tail_events(log, follow=False)) if log is not None else None
    if log is None or state is None:
        return {}
    snap = session_state_as_dict(state)
    snap["state_dir"] = log.parent.name
    if state.last_event_ep is not None:
        snap["last_event_ep"] = state.last_event_ep
    return snap


# --- config ------------------------------------------------------------------


def config_payload(cwd: Path, config_path: Path | None = None) -> dict[str, Any]:
    """Return the effective config per leaf, keyed by dotted key.

    The shared fields are what `agent6 config show --json` prints; `input` is each
    value's round-trippable editor text. No secret is included.
    """
    eff = load_effective(cwd, config_path)
    resolved = resolved_config_values(eff.config)
    payload: dict[str, Any] = json.loads(render_show(eff, as_json=True, resolved=resolved))
    for setting in payload.values():
        value = setting["value"]
        setting["input"] = (
            "" if value is None else value if isinstance(value, str) else format_toml_value(value)
        )
    return payload


def config_suggestions(cwd: Path, key: str, config_path: Path | None = None) -> list[str]:
    """Return value suggestions for one open-text config leaf.

    `preset` offers the preset names; every other key what `config_value_choices`
    offers. A config error suggests nothing.
    """
    if key == "preset":
        return available_preset_names(cwd, config_path)
    try:
        eff = load_effective(cwd, config_path)
    except ConfigError:
        return []
    return config_value_choices(eff, key)


def step_diff_payload(
    repo: Path, session_dir: Path, sha: str, *, cumulative: bool
) -> tuple[dict[str, Any] | None, str]:
    """Return the patch one run step introduced, or the whole chain up to it.

    Args:
        repo: The repository the run worked in.
        session_dir: The run's directory.
        sha: The step's commit.
        cumulative: The chain `base..sha` instead of the one step.

    Returns:
        The payload and "", or None and the reason; a model-controlled run has no chain.
    """
    try:
        m = read_manifest(session_dir)
    except ManifestError as exc:
        return None, f"unreadable manifest: {exc}"
    if m.git_control == "model":
        return None, "the model owns git in this run: no step chain to select from"
    return _diff_payload(
        repo,
        sha,
        base=m.base_sha,
        cumulative=cumulative,
        miss="pruned, or not a commit of this run",
    )
