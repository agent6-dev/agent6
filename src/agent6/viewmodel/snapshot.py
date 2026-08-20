# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The one-object wire snapshots: a session's folded state and a machine
instance's, as `agent6 attach --json` prints them and the web serves them.
One fold each, so the two never disagree."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent6.machine import MachineJournal, load_machine
from agent6.sessions.layout import LOGS_NAME
from agent6.sessions.manifest import ManifestError, read_manifest
from agent6.viewmodel.listing import session_compare
from agent6.viewmodel.machine_state import fold_machine, machine_state_as_dict
from agent6.viewmodel.state import fold_session, session_state_as_dict
from agent6.viewmodel.tail import tail_events


def manifest_branches(session_dir: Path) -> dict[str, str]:
    """Branch facts from the run's manifest (run_branch / base_branch /
    merged_into) for the run header. The event fold does not carry them, and
    an operator needs to SEE where a run's work lives and where Merge lands
    (consecutive spawns chain branches invisibly otherwise). Empty for a run
    with no manifest (or branch_per_run off)."""
    try:
        manifest = read_manifest(session_dir)
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


def manifest_header(session_dir: Path) -> dict[str, Any]:
    """Manifest-derived session-header fields the event fold does not carry:
    the branch facts and the fan-out compare outcome (rank/winner/rationale).
    Merged into every session snapshot (one-shot and streamed) so the header
    a page paints from cannot drift. Empty for a run with no (readable)
    manifest."""
    header: dict[str, Any] = dict(manifest_branches(session_dir))
    compare = session_compare(session_dir)
    if compare is not None:
        header["compare"] = compare.model_dump(mode="json")
    return header


def session_snapshot(session_dir: Path) -> dict[str, Any]:
    """A session's folded state as the wire dict, with the dir-aware status
    (parked / stale / waiting, not the fold's blanket "running"), the
    dir-backed identity fill, and the manifest header. A session with no log
    yet (a parked submission, a `fork --no-run`) folds nothing and lets the
    dir supply the word."""
    state = fold_session(tail_events(session_dir / LOGS_NAME, follow=False))
    snap = session_state_as_dict(state, session_dir)
    snap.update(manifest_header(session_dir))
    return snap


def machine_snapshot(machine_dir: Path) -> dict[str, Any]:
    """A machine instance's folded MachineState as the wire dict. Raises
    MachineError for an unloadable source and JournalError for a corrupt
    journal; the callers word those."""
    spec = load_machine(machine_dir / "machine.asm.toml")
    ms = fold_machine(spec, MachineJournal(machine_dir).read())
    return machine_state_as_dict(ms, machine_dir)
