# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The unified `agent6 watch <target>`: follow a run or a machine, live.

Resolves <target> to a run (id or unique prefix) or a machine (by name) and
dispatches to the right viewer. A run defaults to the textual dashboard
(`--plain` for a no-deps line tail); a machine streams its state overview +
reasoning. `--json` prints a one-shot snapshot of the folded state, the same
wire form a web client reads. An empty target watches the most recent run. A
target that is both a run prefix and a machine name resolves as the run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from agent6.cli._common import _machines_dir, _runs_dir
from agent6.cli.machine_cmds import _cmd_machine_watch
from agent6.cli.plan_watch import _cmd_watch as _watch_run
from agent6.cli.plan_watch import _resolve_run_dir
from agent6.machine import MachineError, MachineJournal, load_machine
from agent6.run_id import RunIdError, resolve_run_id
from agent6.viewmodel import (
    fold_machine,
    fold_run,
    machine_state_as_dict,
    run_state_as_dict,
    tail_events,
)


def _is_run_target(runs_dir: Path, target: str) -> bool:
    """True if *target* resolves to a run id (exact or unique prefix)."""
    try:
        resolve_run_id(runs_dir, target)
        return True
    except RunIdError:
        return False


def _run_json_snapshot(run_dir: Path) -> int:
    """Print a run's folded RunState as one JSON object (the web wire form)."""
    logs = run_dir / "logs.jsonl"
    if not logs.is_file():
        print(f"ERROR: no logs.jsonl in {run_dir}", file=sys.stderr)
        return 2
    print(json.dumps(run_state_as_dict(fold_run(tail_events(logs, follow=False)))))
    return 0


def _machine_json_snapshot(machine_dir: Path) -> int:
    """Print a machine's folded MachineState as one JSON object (the web wire form)."""
    source = machine_dir / "machine.asm.toml"
    try:
        spec = load_machine(source)
    except MachineError as exc:
        print(f"FAIL: {source}: {'; '.join(exc.problems)}", file=sys.stderr)
        return 1
    ms = fold_machine(spec, MachineJournal(machine_dir).read())
    print(json.dumps(machine_state_as_dict(ms)))
    return 0


def _cmd_watch_target(target: str, *, plain: bool, json_out: bool, since: int) -> int:
    """Resolve *target* to a run or machine and follow it (or snapshot it)."""
    cwd = Path.cwd()
    runs_dir = _runs_dir(cwd)
    machines_dir = _machines_dir(cwd)

    # Empty target, or one that resolves to a run id: watch the run.
    if not target or _is_run_target(runs_dir, target):
        if not json_out:
            return _watch_run(target, plain=plain, since=since)
        run_dir = _resolve_run_dir(runs_dir, target)
        if run_dir is None or not run_dir.is_dir():
            print(f"ERROR: no run found ({target or 'latest'}) under {runs_dir}", file=sys.stderr)
            return 2
        return _run_json_snapshot(run_dir)

    # Else a machine by name.
    machine_dir = machines_dir / target
    if machine_dir.is_dir():
        return _machine_json_snapshot(machine_dir) if json_out else _cmd_machine_watch(target)

    print(
        f"ERROR: no run or machine matches {target!r} (looked under {runs_dir} and {machines_dir})",
        file=sys.stderr,
    )
    return 2
