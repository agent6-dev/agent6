# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Build the web UI's JSON payloads.

Every payload comes from the shared read side (the viewmodel folds, the config
layer, the machine spec and journal), with no HTTP or threads, so the run and
machine snapshots are the same dicts `agent6 attach --json` prints.
"""

from __future__ import annotations

import json
import pathlib
import re
from typing import Any

from agent6 import paths
from agent6.app import confine, workdirs
from agent6.config import ConfigError, io, layer, layer_data, layer_presets, write
from agent6.models import choices
from agent6.sessions import layout, manifest, process
from agent6.ui.web import diff_highlight
from agent6.vcs import git_history, git_read
from agent6.viewmodel import (
    MachineSummary,
    config_view,
    fold_session,
    fold_transcript,
    format,
    is_session_husk,
    listing,
    machine_files,
    machine_instance_dirs,
    operator_inputs,
    restate,
    session_state_as_dict,
    summarize_machine_dir,
    tail_events,
    transcript_style,
)
from agent6.viewmodel.machine import logs as machine_logs


def session_dir_for(cwd: pathlib.Path, session_id: str) -> pathlib.Path | None:
    """Locate a session dir by its exact id across the hub buckets.

    Husks are skipped so an orphaned dir cannot shadow a real session of the same id.

    Args:
        cwd: The repository.
        session_id: The full id from the hub payload.

    Returns:
        The session dir, or None for an unsafe id, a missing session or an id in two buckets.
    """
    if not layout.is_safe_session_id(session_id):
        return None
    found: pathlib.Path | None = None
    for sub in layout.HUB_BUCKETS:
        d = layout.bucket_dir(paths.state_dir(cwd), sub) / session_id
        if d.is_dir() and not is_session_husk(d):
            if found is not None:
                return None
            found = d
    return found


def machine_dir_for(cwd: pathlib.Path, name: str) -> pathlib.Path | None:
    """Return a machine instance's dir by name, or None for an unsafe or unknown name."""
    if not layout.is_safe_session_id(name):
        return None
    d = layout.machines_root(paths.state_dir(cwd)) / name
    return d if d.is_dir() else None


def draft_dir_for(cwd: pathlib.Path, name: str) -> pathlib.Path | None:
    """Return a `machine create` draft's dir by name, or None when there is none.

    The draft's logs.jsonl is the authoring agent's run-style log, watched through
    the run endpoints.

    Args:
        cwd: The repository.
        name: The draft's dir name.

    Returns:
        The draft's dir, or None for an unsafe name, a missing draft or one that never
        started.
    """
    if not layout.is_safe_session_id(name):
        return None
    d = layout.bucket_dir(paths.state_dir(cwd), "machines") / name
    return d if d.is_dir() and not is_session_husk(d) else None


def draft_workspace(
    cwd: pathlib.Path, name: str, config_paths: tuple[pathlib.Path, ...]
) -> pathlib.Path | None:
    """Return the workspace a `machine create` draft commits in, or None once it is gone.

    The workspace is a repository of its own beside the other subordinate working
    trees; publishing the draft removes it.

    Args:
        cwd: The repository.
        name: The draft's dir name.
        config_paths: The `--config` files in command-line order.

    Returns:
        The workspace, or None once it is gone or when the config does not load.
    """
    try:
        cfg = layer.load_effective(cwd, config_paths).config
    except ConfigError:
        return None
    workspace = workdirs.subordinate_workdir_root(cfg, cwd, name)
    return workspace if (workspace / ".git").exists() else None


def draft_commit_diff_payload(
    workspace: pathlib.Path, session_dir: pathlib.Path, sha: str, *, cumulative: bool
) -> tuple[dict[str, Any] | None, str]:
    """Return the patch one draft commit introduced, or the whole bundle as of that commit.

    A draft starts from the empty tree, so the cumulative patch is the bundle so far.

    Args:
        workspace: The draft's workspace.
        session_dir: The draft's journal directory.
        sha: The commit sha; "" selects the latest journaled commit.
        cumulative: Whole bundle instead of the one commit.

    Returns:
        The payload and "", or None and the reason.
    """
    if not sha:
        commits = fold_session(tail_events(session_dir / layout.LOGS_NAME, follow=False)).commits
        if not commits:
            return None, "no commit yet"
        sha = commits[-1].sha
    return _diff_payload(
        workspace,
        sha,
        base=git_read.EMPTY_TREE,
        cumulative=cumulative,
        miss="not a commit of this draft",
    )


def _diff_payload(
    repo: pathlib.Path, sha: str, *, base: str, cumulative: bool, miss: str
) -> tuple[dict[str, Any] | None, str]:
    """Return one commit's patch, or the chain `base..sha` when cumulative and a base is known.

    Attributes:
        patch: The unified patch for the selected commit or chain.
        old_spans: Base paths to per-line [start, end, class] spans in UTF-16 units.
            Line 1 is index 0; undisplayed lines and unhighlighted sides are empty.
        new_spans: Head paths to per-line [start, end, class] spans in UTF-16 units.
            Line 1 is index 0; undisplayed lines and unhighlighted sides are empty.

        cumulative: Whether the patch covers the chain rather than one commit.

    Args:
        repo: The repository.
        sha: The commit sha.
        base: The chain's base; "" when unknown.
        cumulative: The chain instead of the one commit.
        miss: The reason to give when the sha has no diff.

    Returns:
        The payload (saying which it holds) and "", or None and the reason.
    """
    if not re.fullmatch(r"[0-9a-f]{7,40}", sha):
        return None, f"not a commit sha: {sha!r}"
    whole = cumulative and bool(base)
    patch = git_read.diff_range(repo, base, sha) if whole else git_read.commit_diff(repo, sha)
    if not patch:
        return None, f"no diff for {sha[:12]} ({miss})"
    if not whole:
        parents = git_history.commit_info(repo, sha).parents
        base = parents[0] if parents else git_read.EMPTY_TREE
    old_spans, new_spans = diff_highlight.revision_spans(repo, base, sha, patch)
    return {
        "cumulative": whole,
        "patch": patch,
        "old_spans": old_spans,
        "new_spans": new_spans,
    }, ""


def draft_dir_paths(cwd: pathlib.Path) -> list[pathlib.Path]:
    """Return every `machine create` draft directory."""
    d = layout.bucket_dir(paths.state_dir(cwd), "machines")
    return [p for p in d.iterdir() if p.is_dir()] if d.is_dir() else []


# --- hub listing -------------------------------------------------------------


def _list_sessions(cwd: pathlib.Path) -> list[listing.SummaryRow]:
    """Return every session the hub lists, newest first, a fan-out's lanes under its row."""
    rows, winners = listing.rows_for(cwd)
    return [listing.row_json(r, winners=winners) for r in rows]


def _machine_row(s: MachineSummary) -> dict[str, Any]:
    """Return one machine-instance row for the hub.

    Attributes:
        name: The machine instance name.
        mtime: The latest activity time in epoch seconds.
        when: The rendered latest activity time.
        status: The machine's status word.
        status_level: The status presentation level.
        machine: The declared machine name, absent for unreadable instances.
        current: The current state, absent for unreadable instances.
        status_label: The rendered status, with its reason when it has one.

    Args:
        s: The machine summary.

    Returns:
        The machine listing row.
    """
    entry: dict[str, Any] = {
        "name": s.name,
        "mtime": s.mtime,
        "when": format.format_when(s.mtime) if s.mtime else "",
        "status": s.status,
        "status_label": format.status_label(s.status, s.reason),
        "status_level": format.status_level(s.status),
    }
    if s.status != "unreadable":
        entry["machine"] = s.machine
        entry["current"] = s.current
    return entry


def _list_machines(cwd: pathlib.Path) -> list[dict[str, Any]]:
    """Return the machine instances, newest first, summarized by the shared fold."""
    dirs = machine_instance_dirs(paths.state_dir(cwd))
    return [_machine_row(summarize_machine_dir(d)) for d in dirs]


def _list_drafts(cwd: pathlib.Path) -> list[listing.SummaryRow]:
    """Return the `machine create` drafts as listing rows, newest first."""
    rows, winners = listing.rows_for(cwd, buckets=("machines",))
    return [listing.row_json(r, winners=winners) for r in rows]


def list_machine_files(cwd: pathlib.Path) -> list[dict[str, str]]:
    """Return the hub's machine-file rows.

    Attributes:
        path: The path accepted by machine run.
        name: The machine file's basename.
        [].path: The path accepted by machine run.
        [].name: The machine file's basename.

    Args:
        cwd: The repository.

    Returns:
        The authored machine files.
    """
    return [{"path": str(p), "name": p.name} for p in machine_files(cwd)]


def routes_payload(
    cwd: pathlib.Path, config_paths: tuple[pathlib.Path, ...], *, mode: str, preset: str
) -> dict[str, Any]:
    """Return the new-work composer's model picker.

    Attributes:
        routes: The provider/model routes the config can run.
        default_label: The composer's no-flag model label.

    Args:
        cwd: The repository.
        config_paths: The `--config` files in command-line order.
        mode: The session mode the default route is for.
        preset: The preset the default route is under.

    Returns:
        Every `provider/model` the config can run, and the label of the no-flag entry.
    """
    return {
        "routes": choices.available_routes(cwd, config_paths),
        "default_label": choices.default_label(
            choices.default_route(cwd, config_paths, mode, preset)
        ),
    }


def resume_defaults_payload(
    cwd: pathlib.Path,
    config_paths: tuple[pathlib.Path, ...],
    session_dir: pathlib.Path,
    *,
    preset: str,
) -> dict[str, str]:
    """Return the resume row's no-flag preset and model labels under a picked preset.

    Attributes:
        preset_label: The resume's no-flag preset label.
        model_label: The resume's no-flag model label under the selected preset.

    Args:
        cwd: The repository.
        config_paths: Config overlays in command-line order.
        session_dir: The session directory.
        preset: The selected preset override.

    Returns:
        The no-flag preset and model labels.
    """
    preset_label, model_label = choices.resume_defaults(
        cwd, config_paths, session_dir, preset=preset
    )
    return {"preset_label": preset_label, "model_label": model_label}


def hub_payload(cwd: pathlib.Path, config_paths: tuple[pathlib.Path, ...] = ()) -> dict[str, Any]:
    """Return the hub: sessions, machines, drafts, machine files, presets.

    Attributes:
        sessions: Session listing rows, newest activity first.
        machines: Machine-instance listing rows.
        machine_files: Authored machine files available to run.
        drafts: Machine-authoring session listing rows.
        presets: Available preset names.
        preset_default_label: The composer's no-flag preset label.

    Args:
        cwd: The repository.
        config_paths: Config overlays in command-line order.

    Returns:
        The hub listings and preset choices.
    """
    return {
        "sessions": _list_sessions(cwd),
        "machines": _list_machines(cwd),
        "machine_files": list_machine_files(cwd),
        "drafts": _list_drafts(cwd),
        "presets": layer_presets.available_preset_names(cwd, config_paths),
        "preset_default_label": choices.default_label(choices.default_preset(cwd, config_paths)),
    }


# --- run snapshot + conversation ----------------------------------------------


def conversation_items(
    events: list[dict[str, Any]], *, worker_dead: bool = False
) -> list[dict[str, Any]]:
    """Fold events into rendered conversation items.

    Each item carries its `kind`, the collapsed `lines` as `[text, style]` spans from
    the shared renderer, and `full` for every tool, task and steer or when expansion differs.

    Attributes:
        kind: The transcript item kind.
        lines: Collapsed lines of text/style span pairs.
        full: Expanded lines of text/style span pairs, only when different from lines.

    Args:
        events: The session's events.
        worker_dead: Settle the calls still open as never returned.

    Returns:
        One entry per transcript item.
    """
    out: list[dict[str, Any]] = []
    for item in fold_transcript(events, worker_dead=worker_dead):
        collapsed = transcript_style.item_lines(item, detail="collapsed")
        expanded = transcript_style.item_lines(item, detail="expanded")
        entry: dict[str, Any] = {"kind": item.kind, "lines": collapsed}
        if item.kind in ("tool", "task", "operator") or expanded != collapsed:
            entry["full"] = expanded
        out.append(entry)
    return out


def conversation_payload(session_dir: pathlib.Path) -> dict[str, Any]:
    """Return a run's conversation and the operator's past inputs, from one read of the log.

    Attributes:
        items: The folded conversation items.
        operator_inputs: The operator's past inputs for composer history.

    Args:
        session_dir: The session directory.

    Returns:
        The conversation and operator input history.
    """
    events = list(tail_events(session_dir / layout.LOGS_NAME, follow=False))
    return {
        "items": conversation_items(events, worker_dead=not process.worker_is_alive(session_dir)),
        "operator_inputs": operator_inputs(events),
    }


def restate_payload(session_dir: pathlib.Path) -> dict[str, Any]:
    """Return `/restate` for the web composer, over the session's whole journal.

    Attributes:
        text: The rendered restatement of the session.

    Args:
        session_dir: The session directory.

    Returns:
        The restatement text.
    """
    events = list(tail_events(session_dir / layout.LOGS_NAME, follow=False))
    return {"text": restate(events, worker_dead=not process.worker_is_alive(session_dir))}


def machine_conversation_payload(machine_dir: pathlib.Path) -> dict[str, Any]:
    """Return the conversation of the machine's newest agent-state execution, or no items.

    Attributes:
        items: The newest agent-state conversation, empty before any execution.

    Args:
        machine_dir: The machine instance directory.

    Returns:
        The newest agent-state conversation.
    """
    log = machine_logs.newest_state_log(machine_dir)
    if log is None:
        return {"items": []}
    events = list(tail_events(log, follow=False))
    # The machine's worker (one pid for every state) is the one to probe.
    items = conversation_items(events, worker_dead=not process.worker_is_alive(machine_dir))
    return {"items": items}


# --- machine snapshot (structure + watch + reasoning) -----------------------


def machine_reasoning_snapshot(
    machine_dir: pathlib.Path, *, fold: machine_logs.NewestExecutionFold | None = None
) -> dict[str, Any]:
    """Return the session state of the machine's newest agent-state execution.

    The snapshot carries `state_dir` so a client echoes it when answering a prompt
    (prompt ids reset per state). Its `last_event_at` is a time, not an age, so the
    payload changes only when something happened.

    Attributes:
        state_dir: The agent-state execution directory to echo when answering a prompt.

    Args:
        machine_dir: The machine instance's directory.
        fold: A fold to refresh, or None to fold the newest log afresh.

    Returns:
        The wire form plus `state_dir`, or empty before any agent state has a log.
    """
    if fold is not None:
        log, state = fold.refresh(machine_dir), fold.state
    else:
        log = machine_logs.newest_state_log(machine_dir)
        state = fold_session(tail_events(log, follow=False)) if log is not None else None
    if log is None or state is None:
        return {}
    snap = session_state_as_dict(state)
    snap["state_dir"] = log.parent.name
    return snap


# --- config ------------------------------------------------------------------


def selected_preset(cwd: pathlib.Path, config_paths: tuple[pathlib.Path, ...] = ()) -> str:
    """Return the preset the config page edits: the effective `agent6.preset`, "" for `none`.

    Args:
        cwd: The repository.
        config_paths: The `--config` files in command-line order.

    Returns:
        The preset whose `[presets.<name>]` table a page write lands in, or "".
    """
    return write.selected_preset(layer.load_effective(cwd, config_paths))


def config_payload(
    cwd: pathlib.Path, config_paths: tuple[pathlib.Path, ...] = ()
) -> dict[str, Any]:
    """Return the effective config per leaf, keyed by dotted key.

    Attributes:
        input: The value formatted for round-trip editing.
        entry: The named config entry, only for entry-table settings.
        entry.table: The entry table's dotted path.
        entry.name: The entry's name.
        entry.source: The highest config layer holding the entry.

    Args:
        cwd: The repository.
        config_paths: The `--config` files in command-line order.

    Returns:
        Each leaf's display fields, round-trippable `input`, and named `entry` when present.
        Scalar entry tables get individual rows. No secret is included.
    """
    eff = layer.load_effective(cwd, config_paths)
    resolved = confine.resolved_config_values(eff.config)
    payload: dict[str, Any] = json.loads(
        config_view.render_show(eff, as_json=True, resolved=resolved)
    )
    entry_sources: dict[tuple[str, str], str] = {}
    for source in eff.layers:
        for key in layer_data.flatten_leaves(source.data):
            if entry := write.entry_of(key):
                table, name = entry
                entry_sources[table.path, name] = source.source
    for key, setting in payload.items():
        if entry := write.entry_of(key):
            table, name = entry
            setting["entry"] = {
                "table": table.path,
                "name": name,
                "source": entry_sources.get((table.path, name), "default"),
            }
        value = setting["value"]
        setting["input"] = (
            ""
            if value is None
            else value
            if isinstance(value, str)
            else io.format_toml_value(value)
        )
    return payload


def config_suggestions(
    cwd: pathlib.Path,
    key: str,
    config_paths: tuple[pathlib.Path, ...] = (),
    *,
    provider: str | None = None,
) -> list[str]:
    """Return value suggestions for one open-text config leaf.

    Args:
        cwd: The repository.
        key: The leaf's dotted key.
        config_paths: The `--config` files in command-line order.
        provider: The model editor's unsaved provider choice, restricted to configured names.

    Returns:
        The preset names for `agent6.preset`; for any other key, what `config_value_choices`
        offers, or [] on a config error.
    """
    if key == "agent6.preset":
        return layer_presets.available_preset_names(cwd, config_paths)
    try:
        eff = layer.load_effective(cwd, config_paths)
    except ConfigError:
        return []
    if provider is not None and re.fullmatch(r"models\.(planner|worker|reviewer)\.model", key):
        return (
            [
                f"{provider}/{model}"
                for model in choices.provider_model_choices(eff.config, provider)
            ]
            if provider in eff.config.providers
            else []
        )
    return choices.config_value_choices(eff, key)


def commit_diff_payload(
    repo: pathlib.Path, session_dir: pathlib.Path, sha: str, *, cumulative: bool
) -> tuple[dict[str, Any] | None, str]:
    """Return the patch one run commit introduced, or the whole chain up to it.

    Args:
        repo: The repository the run worked in.
        session_dir: The run's directory.
        sha: The commit sha; "" selects the latest journaled commit.
        cumulative: The chain `base..sha` instead of the one commit.

    Returns:
        The payload and "", or None and the reason; a model-controlled run has no chain.
    """
    try:
        m = manifest.read_manifest(session_dir)
    except manifest.ManifestError as exc:
        return None, f"unreadable manifest: {exc}"
    if m.git_control == "model":
        return None, "the model owns git in this run: no commit chain to select from"
    if not sha:
        commits = fold_session(tail_events(session_dir / layout.LOGS_NAME, follow=False)).commits
        if not commits:
            return None, "no commit yet"
        sha = commits[-1].sha
    return _diff_payload(
        repo,
        sha,
        base=m.base_sha,
        cumulative=cumulative,
        miss="pruned, or not a commit of this run",
    )
