# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Build the machines listing `agent6 machine` and the TUI machines page render.

One row per machine: an instance joined with the authored `.asm.toml` that declares it, then
the authored files no instance has run.
"""

from __future__ import annotations

import dataclasses
import pathlib

from agent6.app.machine import bundle
from agent6.viewmodel.machine import state as machine_state


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
        start_actions: The starts available for the authored file.
    """

    name: str
    file: pathlib.Path | None
    states: str
    spec: str
    status: str
    reason: str
    current: str
    mtime: float
    start_actions: tuple[str, ...]


def machine_summaries(state_dir: pathlib.Path) -> list[machine_state.MachineSummary]:
    """Return the summaries of a project's instances, newest first."""
    return [
        machine_state.summarize_machine_dir(d)
        for d in machine_state.machine_instance_dirs(state_dir)
    ]


def machine_rows(cwd: pathlib.Path, state_dir: pathlib.Path) -> list[MachineRow]:
    """List a repository's instances and authored machine files.

    Args:
        cwd: The repository containing the authored files.
        state_dir: The repository's state directory.

    Returns:
        The instances newest first, then the remaining files.
    """
    return join_machine_rows(cwd, machine_summaries(state_dir))


def join_machine_rows(
    cwd: pathlib.Path, instances: list[machine_state.MachineSummary]
) -> list[MachineRow]:
    """List instances newest first, then the authored files no instance ran.

    An instance joins the first authored file declaring its name; a second file with the same
    name, or an unparsable one named "-", keeps its own row.

    Args:
        cwd: The repository the authored files are found under.
        instances: The repo's instance summaries, newest first.

    Returns:
        The rows in listing order.
    """
    files = [(p, bundle.summarize_machine_file(p)) for p in machine_state.machine_files(cwd)]
    rows: list[MachineRow] = []
    joined: set[pathlib.Path] = set()
    starts = {inst.name: inst.start_actions for inst in instances}
    for inst in instances:
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
                start_actions=inst.start_actions if own else (),
            )
        )
    for path, f in files:
        if path not in joined:
            rows.append(
                MachineRow(
                    f.name, path, f.states, f.spec, "", "", "", 0.0, starts.get(f.name, ("Run",))
                )
            )
    return rows
