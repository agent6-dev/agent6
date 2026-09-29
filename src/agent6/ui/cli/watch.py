# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 attach <target>`: follow a run or a machine, live.

A run renders its conversation, or with `--raw` the line tail of logs.jsonl; a machine
streams its state overview and reasoning; `--tui` opens the full-screen dashboard;
`--json` prints one snapshot of the folded state, the same wire form a web client reads.
An empty target watches the most recent run. A target that is both a run prefix and a
machine name resolves as the run.
"""

from __future__ import annotations

import json
import pathlib
import sys

from agent6 import paths
from agent6.machine import JournalError, MachineError, load_machine
from agent6.sessions import id
from agent6.ui.cli import _common, machine_cmds, plan_watch
from agent6.viewmodel import (
    machine_snapshot,
    session_snapshot,
)


def _run_intent(repo_root: pathlib.Path, target: str) -> tuple[bool, str | None]:
    """Resolve a target against the session buckets.

    Args:
        repo_root: The repo whose state dir holds the sessions.
        target: The id or prefix.

    Returns:
        `(True, None)` when it resolves; `(False, None)` on no match, so the caller may try
        a machine; `(False, message)` for an ambiguous prefix or a husk, which the caller
        surfaces instead of falling through to machine lookup.
    """
    try:
        _common.resolve_session_layout(repo_root, target)
    except id.SessionIdError as exc:
        return (False, None) if exc.no_match else (False, str(exc))
    return (True, None)


def _machine_json_snapshot(machine_dir: pathlib.Path) -> int:
    """Print a machine's snapshot as one JSON object.

    Returns:
        The exit code; 1 when the machine or its journal cannot be read.
    """
    try:
        snap = machine_snapshot(machine_dir)
    except JournalError as exc:  # a MachineError too: the corrupt-journal wording first
        _common.error(f"{exc}")
        return 1
    except MachineError as exc:
        source = machine_dir / "machine.asm.toml"
        print(f"FAIL: {source}: {'; '.join(exc.problems)}", file=sys.stderr)
        return 1
    print(json.dumps(snap))
    return 0


def _machine_watch_tui(machine_dir: pathlib.Path) -> int:
    """Open the full-screen machine watch.

    Returns:
        The exit code; 3 when the TUI cannot be imported.
    """
    source = machine_dir / "machine.asm.toml"
    try:
        spec = load_machine(source)
    except MachineError as exc:
        print(f"FAIL: {source}: {'; '.join(exc.problems)}", file=sys.stderr)
        return 1
    try:
        from agent6.ui.tui import machines  # noqa: PLC0415  # noqa: PLC0415
    except ImportError as e:
        _common.error(f"{e}")
        print("HINT: drop --tui for the plain text follow.", file=sys.stderr)
        return 3
    return machines.run_machine_watch_tui(machine_dir, spec)


def _cmd_watch_target(  # noqa: PLR0911
    target: str,
    *,
    tui: bool,
    json_out: bool,
    since: int | None,
    raw: bool,
    config_path: pathlib.Path | None = None,
) -> int:
    """Resolve a target to a run or a machine and follow or snapshot it.

    Args:
        target: A run id or prefix, a machine name, or "" for the newest run.
        tui: Open the full-screen dashboard.
        json_out: Print one snapshot and exit.
        since: Replay from this event line; `--raw` only.
        raw: Tail the log lines instead of rendering the conversation.
        config_path: The `--config` file, if any.

    Returns:
        The exit code; 2 when the flags conflict or nothing matches.
    """
    if since is not None and since < 0:
        _common.error("--since must be non-negative.")
        return 2
    if since is not None and not raw:
        # --since replays event lines, which only the --raw tail renders.
        _common.error("--since applies to --raw only.")
        return 2
    since = since or 0
    cwd = pathlib.Path.cwd()

    # An ambiguous prefix or a husk is surfaced, not fallen through to machine lookup.
    is_run, run_error = (True, None) if not target else _run_intent(cwd, target)
    if run_error is not None:
        _common.error(f"{run_error}")
        return 2

    # Empty target, or one that resolves to a run id: watch the run.
    if is_run:
        if not json_out:
            return plan_watch._cmd_watch(
                target, tui=tui, since=since, raw=raw, config_path=config_path
            )
        layout = _common.resolve_target(target)
        if layout is None:
            return 2
        session_dir = layout.session_dir
        # The wire form the web serves, the merged claim checked against the repo.
        print(json.dumps(session_snapshot(session_dir, repo=cwd)))
        return 0

    # Else a machine by name.
    machine_dir = machine_cmds.machine_instance_root(target, cwd)
    if machine_dir is not None and machine_dir.is_dir():
        if raw:
            _common.error("--raw applies to run sessions, not machines.")
            return 2
        if json_out:
            return _machine_json_snapshot(machine_dir)
        return _machine_watch_tui(machine_dir) if tui else machine_cmds._cmd_machine_watch(target)

    _common.error(f"no run or machine matches {target!r} (looked under {paths.state_dir(cwd)})")
    return 2
