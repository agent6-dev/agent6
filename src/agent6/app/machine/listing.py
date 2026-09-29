# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Build the machines listing `agent6 machine` and the TUI machines page render.

One row per machine: an instance joined with the authored `.asm.toml` that declares it, then
the authored files no instance has run.
"""

from __future__ import annotations

import dataclasses
import pathlib

from agent6.app.machine import _bundle
from agent6.viewmodel import machine_files, machine_instance_dirs, summarize_machine_dir


@dataclasses.dataclass(frozen=True, slots=True)
class MachineRow:
    """One listing row: an instance, an authored file, or both.

    Attributes:
        name: The machine's name ("-" for an unparsable file with no instance).
        file: The authored file, when one declares it.
        states: The file's state count, or "-".
        spec: The file's validity ("valid", "N issue(s)", "invalid"), or "-" without a file.
        status: The instance's status word; "" for a file no instance ran.
        reason: A failed instance's reason, else "".
        current: The instance's current state; "" without an instance.
        mtime: The instance's last activity; 0.0 without one.
    """

    name: str
    file: pathlib.Path | None
    states: str
    spec: str
    status: str
    reason: str
    current: str
    mtime: float


def machine_rows(cwd: pathlib.Path, state_dir: pathlib.Path) -> list[MachineRow]:
    """List instances newest first, then the authored files no instance ran.

    An instance joins the first authored file declaring its name; a second file with the same
    name, or an unparsable one named "-", keeps its own row.

    Args:
        cwd: The repository the authored files are found under.
        state_dir: The repo's state directory holding the instances.

    Returns:
        The rows in listing order.
    """
    files = [(p, _bundle.summarize_machine_file(p)) for p in machine_files(cwd)]
    rows: list[MachineRow] = []
    joined: set[pathlib.Path] = set()
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
