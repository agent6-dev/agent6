# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The machines listing: one row per machine, an instance (its status and
current state) joined with the authored `.asm.toml` that declares it (its
spec validity), then the authored files no instance has run. `agent6
machine` and the TUI machines page render these rows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agent6.app.machine._bundle import summarize_machine_file
from agent6.viewmodel import machine_files, machine_instance_dirs, summarize_machine_dir


@dataclass(frozen=True, slots=True)
class MachineRow:
    name: str  # the machine's name ("-" for an unparsable file with no instance)
    file: Path | None  # the authored file, when one declares it
    states: str  # the file's state count, or "-"
    spec: str  # the file's validity ("valid", "N issue(s)", "invalid"), or "-" without a file
    status: str  # the instance's status word; "" for a file no instance ran
    reason: str  # a failed instance's reason, else ""
    current: str  # the instance's current state; "" without an instance
    mtime: float  # the instance's last activity; 0.0 without one


def machine_rows(cwd: Path, state_dir: Path) -> list[MachineRow]:
    """Instances newest first, each joined with the first authored file
    declaring its name (a second file with the same name, or an unparsable
    one named "-", keeps its own row), then the files no instance ran."""
    files = [(p, summarize_machine_file(p)) for p in machine_files(cwd)]
    rows: list[MachineRow] = []
    joined: set[Path] = set()
    for inst in (summarize_machine_dir(d) for d in machine_instance_dirs(state_dir)):
        own = next(((p, f) for p, f in files if f.name == inst.name and p not in joined), None)
        if own is not None:
            joined.add(own[0])
        rows.append(
            MachineRow(
                name=inst.name,
                file=own[0] if own else None,
                states=own[1].states if own else "-",
                spec=own[1].spec if own else "-",
                status=inst.status,
                reason=inst.reason,
                current=inst.current,
                mtime=inst.mtime,
            )
        )
    for path, f in files:
        if path not in joined:
            rows.append(MachineRow(f.name, path, f.states, f.spec, "", "", "", 0.0))
    return rows
