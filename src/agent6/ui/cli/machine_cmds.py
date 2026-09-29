# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 machine` lifecycle subcommands: argv adaptation and console rendering.

The read-only commands (list, status, replay, poke, stop, the watch follower) load and
render directly; run and create hand the lifecycle to `app.machine` behind the
`MachineFrontend` seam. The interactive network-refusal resolver stays here, since it
needs a TTY. The offline authoring gate is `machine_check`.
"""

from __future__ import annotations

import contextlib
import difflib
import json
import pathlib
import sys
import time
from typing import Any

from agent6 import budget, kinds, paths
from agent6.app import _setup, reporter
from agent6.app.machine import (
    MachineFrontend,
    NetworkRefusal,
    listing,
    machine_network_refusal,
    machine_spend,
)
from agent6.config import (
    Config,
    ConfigError,
    io,
    layer,
)
from agent6.machine import (
    EngineError,
    JournalError,
    MachineError,
    MachineJournal,
    PendingWait,
    StepEvent,
    ToolState,
    drive,
    load_machine,
    write_stop_request,
)
from agent6.sandbox import detect
from agent6.sessions import ipc, layout
from agent6.ui import notify
from agent6.ui.cli import _common, machine_check, plan_watch
from agent6.viewmodel import (
    MachineState,
    MachineWatchCursor,
    armed_wait,
    event_epoch,
    fold_machine,
    format,
    machine_state,
    machine_word_for_dir,
    newest_agent_execution,
)


def _cmd_machine_list() -> int:
    """List this repo's machines: every instance newest first, then the files no instance ran.

    The rows are `app.machine.listing.machine_rows`, the ones the TUI machines page shows.

    Returns:
        The exit code, 0.
    """
    cwd = pathlib.Path.cwd()
    machines = listing.machine_rows(cwd, paths.state_dir(cwd))
    if not machines:
        print('no machines yet. Draft one with `agent6 machine create "<task>"`.')
        return 0
    color = sys.stdout.isatty()
    rows: list[tuple[str, str, str, str, str, str, str]] = []
    for m in machines:
        styled, plain = (
            _common.styled_status(m.status, m.reason, color=color) if m.status else ("-", "-")
        )
        rows.append(
            (
                format.format_when(m.mtime) if m.mtime else "-",
                styled,
                plain,
                m.current or "-",
                m.name,
                m.spec,
                str(m.file.relative_to(cwd)) if m.file is not None else "-",
            )
        )
    status_w = max(6, *(len(plain) for _, _, plain, *_ in rows))
    state_w = max(5, *(len(r[3]) for r in rows))
    name_w = max(7, *(len(r[4]) for r in rows))
    spec_w = max(4, *(len(r[5]) for r in rows))
    print(
        f"{'updated':<11}  {'status':<{status_w}}  {'state':<{state_w}}  {'machine':<{name_w}}"
        f"  {'spec':<{spec_w}}  file"
    )
    for when, styled, plain, state, name, spec, file in rows:
        pad = " " * (status_w - len(plain))
        print(
            f"{when:<11}  {styled}{pad}  {state:<{state_w}}  {name:<{name_w}}"
            f"  {spec:<{spec_w}}  {file}"
        )
    return 0


def _resolve_network_refusal(  # noqa: PLR0911
    path: pathlib.Path,
    refusal: NetworkRefusal,
    cfg: Config,
    isolation: kinds.IsolationLevel,
    tool_states: list[ToolState],
    cwd: pathlib.Path,
    overlay: dict[str, Any],
) -> int | tuple[Config, kinds.IsolationLevel]:
    """Turn a hard network refusal into a choice.

    Interactively: explain it, then offer to apply the minimal config fix and continue,
    simulate the machine offline, or stop. Headless: print the exact fix and the simulate
    command and exit non-zero; a sandbox setting is never relaxed unattended.

    Args:
        path: The machine file.
        refusal: The refusal.
        cfg: The effective config.
        isolation: The resolved isolation.
        tool_states: The machine's tool states.
        cwd: The repo.
        overlay: The machine's `[config]` overlay.

    Returns:
        The new `(cfg, isolation)` when the fix applied and re-validates clear, else the exit
        code.
    """
    _common.refuse(refusal.message)
    fix = refusal.fix
    if not fix:
        print(
            f"  No sandbox-config change fixes this on the '{isolation}' isolation.",
            file=sys.stderr,
        )
        print(f"  Simulate it offline instead:  agent6 machine test {path}", file=sys.stderr)
        return 2
    if not sys.stdin.isatty():
        print("  To allow it, apply this to the per-repo config and re-run:", file=sys.stderr)
        for key, value in fix:
            print(f"    agent6 config set {key} {value} --repo", file=sys.stderr)
        print(f"  Or simulate it offline now:    agent6 machine test {path}", file=sys.stderr)
        return 2
    print("  agent6 can apply the minimal fix now (writes the per-repo config):", file=sys.stderr)
    for key, value in fix:
        print(f"    {key} = {value}", file=sys.stderr)
    choice = (_common.safe_input("  [a]pply & run, [s]imulate offline, or [Q]uit? ") or "").lower()
    if choice == "s":
        return machine_check._cmd_machine_test(path, blackboard=None)
    if choice != "a":
        print("Stopped; nothing changed.", file=sys.stderr)
        return 2
    target = paths.repo_config_path(cwd)
    paths.mkdir_for_real_user(target.parent)
    for key, value in fix:
        io.upsert_toml_leaf(target, key, value)
    paths.chown_to_real_user(target.parent)
    paths.chown_to_real_user(target)
    try:
        new_cfg = layer.load_effective_with_overlay(cwd, overlay).config
        new_profile = detect.resolve_isolation(new_cfg.sandbox.isolation, _setup.detect_env())
    except (ConfigError, detect.IsolationUnavailableError) as exc:
        print(f"  Applied, but the config no longer validates: {exc}", file=sys.stderr)
        return 2
    if machine_network_refusal(new_cfg, new_profile, tool_states) is not None:
        print("  Applied, but a conflict remains; review the per-repo config.", file=sys.stderr)
        return 2
    print(f"  Applied to {target}. Continuing the run.", file=sys.stderr)
    return new_cfg, new_profile


def _no_instance_hint(machine_id: str, cwd: pathlib.Path) -> str:
    """Return a "Did you mean" suffix for a missing-instance error, or "".

    `machine run` takes a file; status, replay, poke, stop and `agent6 attach` take an
    instance id. Given an `.asm.toml` file, its `machine` name suggests the instance id;
    else the closest existing instance name is offered.

    Args:
        machine_id: The argument as given.
        cwd: The repo.
    """
    machines = layout.machines_root(paths.state_dir(cwd))
    existing = sorted(p.name for p in machines.iterdir() if p.is_dir()) if machines.is_dir() else []
    candidate = pathlib.Path(machine_id)
    if machine_id.endswith(".asm.toml") or candidate.is_file():
        name = ""
        with contextlib.suppress(MachineError, OSError):
            name = load_machine(candidate).machine
        if name in existing:
            return (
                f" Did you mean the instance id {name!r}?"
                " (`machine run` takes the FILE; status/replay/poke/stop and"
                " `agent6 attach` take the ID.)"
            )
        if name:
            return f" That is a machine file; run it first with `agent6 machine run {machine_id}`."
        return (
            f" That is a machine file that does not load; see `agent6 machine check {machine_id}`."
        )
    close = difflib.get_close_matches(candidate.name, existing, n=1)
    return f" Did you mean {close[0]!r}?" if close else ""


def machine_instance_root(machine_id: str, cwd: pathlib.Path) -> pathlib.Path | None:
    """Return the instance dir for one leaf id, or None for a path or symlink escape."""
    candidate = pathlib.Path(machine_id)
    if candidate.name != machine_id:
        return None
    machines = layout.machines_root(paths.state_dir(cwd))
    root = machines / machine_id
    try:
        if root.resolve().parent != machines.resolve():
            return None
    except (OSError, RuntimeError):
        return None
    return root


def _existing_machine_root(machine_id: str, cwd: pathlib.Path) -> pathlib.Path | None:
    """Resolve an instance id, printing the shared missing-instance error.

    Returns:
        The instance dir, or None after the error.
    """
    root = machine_instance_root(machine_id, cwd)
    if root is not None and root.is_dir():
        return root
    machines = layout.machines_root(paths.state_dir(cwd))
    location = f"at {root}" if root is not None else f"for id {machine_id!r} under {machines}"
    _common.error(f"no machine instance {location}.{_no_instance_hint(machine_id, cwd)}")
    return None


def _cmd_machine_replay(machine_id: str) -> int:
    """Re-run a machine's journal offline and print the result.

    Returns:
        The exit code; 1 when the replay did not end ok.
    """
    cwd = pathlib.Path.cwd()
    root = _existing_machine_root(machine_id, cwd)
    if root is None:
        return 2
    source_path = root / "machine.asm.toml"
    try:
        spec = load_machine(source_path)
    except MachineError as exc:
        return machine_check._fail(source_path, exc.problems)
    journal = MachineJournal(root)
    try:
        result = drive(spec, journal, None, live=False)
    except (JournalError, EngineError) as exc:
        _common.error(f"{exc}")
        return 1
    print(
        f"{result.status.upper()}: {spec.machine} replayed to {result.state!r}"
        f" after {_common.plural(result.transitions, 'transition')} ({result.reason})"
    )
    return 0 if result.status in ("ok", "incomplete") else 1


def _armed_wait_tolerant(root: pathlib.Path, ms: MachineState) -> tuple[PendingWait | None, str]:
    """Return the armed wait and a note; a corrupt wait.json yields the note instead.

    The readout goes on, as the shared dir word tolerates it (parked, keep streaming),
    instead of the `JournalError` aborting the command.

    Args:
        root: The instance dir.
        ms: The machine state.
    """
    try:
        return armed_wait(root, ms), ""
    except JournalError as exc:
        return None, str(exc)


def _cmd_machine_status(machine_id: str) -> int:
    """Print an instance's state, current wait and recent history.

    Returns:
        The exit code; 1 when the journal cannot be read.
    """
    cwd = pathlib.Path.cwd()
    root = _existing_machine_root(machine_id, cwd)
    if root is None:
        return 2
    source_path = root / "machine.asm.toml"
    try:
        spec = load_machine(source_path)
    except MachineError as exc:
        return machine_check._fail(source_path, exc.problems)
    journal = MachineJournal(root)
    try:
        result = drive(spec, journal, None, live=False)
        events = journal.read()
        snapshot = journal.latest_snapshot()
    except (JournalError, EngineError) as exc:
        _common.error(f"{exc}")
        return 1
    ms = fold_machine(spec, events)
    pending, pending_note = _armed_wait_tolerant(root, ms)

    alive = ipc.worker_is_alive(root)
    spend, inflight_state = machine_spend(events, root, alive=alive)
    # machine_word_for_dir owns running/waiting/stopped for every surface; parked beats alive.
    word = machine_word_for_dir(ms, root)

    print(f"machine: {spec.machine} (v{spec.version})")
    if alive and word == "running":
        running_in = f", running {inflight_state!r}" if inflight_state else ""
        print(f"  status: running (worker pid {ipc.read_worker_pid(root)} alive){running_in}")
    else:
        # A live worker blocked on an operator prompt: "waiting", naming the state to answer in.
        print(
            f"  status: {word}"
            + (
                f" (an approval open in {blocked_in}: answer it in the TUI machine view"
                " or the web page)"
                if alive and (blocked_in := newest_agent_execution(root).blocked_in)
                else ""
            )
        )
    print(f"  state: {result.state!r}")
    print(f"  transitions: {result.transitions}")
    cached = (
        f", cache_r={spend.cache_read_tokens} tok, cache_c={spend.cache_creation_tokens} tok"
        if spend.cache_read_tokens or spend.cache_creation_tokens
        else ""
    )
    print(
        f"  spend: {budget.format_usd(spend.usd, partial=spend.partial)}"
        f" (in={spend.input_tokens} tok, out={spend.output_tokens} tok{cached})"
    )
    if pending is not None:
        # The armed record is the wait a poke wakes; a timed one wakes on its own too.
        print("  " + machine_state.wait_line(machine_id, pending.state, pending.wake_at))
    if pending_note:
        print(f"  pending wait: unreadable ({pending_note})")
    poked, poke_payload = journal.read_pending_poke()
    if poked:
        print("  poke pending: " + ("bare" if poke_payload is None else repr(poke_payload)))
    if snapshot is not None and snapshot.blackboard:
        print("  blackboard:")
        for key, value in snapshot.blackboard.items():
            print(f"    {key} = {value!r}")
    step_events = [e for e in events if isinstance(e, StepEvent)]
    if step_events:
        print("  recent steps:")
        for event in step_events[-5:]:
            print(
                f"    {format.format_transition(event.seq, event.state, event.label, event.goto)}"
            )
    return 0


def _cmd_machine_poke(
    machine_id: str, *, data: str | None = None, message: str | None = None
) -> int:
    """Send a signal to a parked machine's armed wait.

    Args:
        machine_id: The instance.
        data: The payload as text, if any.
        message: A message for the wake, if any.

    Returns:
        The exit code; 2 when the machine has ended or has no armed wait.
    """
    cwd = pathlib.Path.cwd()
    root = _existing_machine_root(machine_id, cwd)
    if root is None:
        return 2
    # An ended machine consumes no signals, so the wake reply would be a lie: refuse.
    ok, refusal = machine_state.verb_answer(root, machine_id, "poke")
    if not ok or refusal:
        _common.refuse(f"{refusal}")
        return 2
    journal = MachineJournal(root)
    if message is not None:
        payload: Any = message
    elif data is not None:
        try:
            payload = json.loads(data)
        except json.JSONDecodeError as exc:
            _common.error(f"--data is not valid JSON: {exc}")
            return 2
    else:
        payload = None
    try:
        journal.poke(payload)
    except JournalError as exc:
        _common.error(f"{exc}")
        return 1
    carried = "" if payload is None else " (with payload)"
    print(f"poked {machine_id}: it will wake on its next signal check{carried}")
    return 0


def _cmd_machine_stop(machine_id: str) -> int:
    """Write the durable stop marker for a running machine.

    The engine parks at its next transition boundary (or wakes out of a sleep) without
    journaling an end, so the instance stays resumable. A machine that is not running gets
    the note and exit 0, as `agent6 stop` answers, and no marker to ambush the next run.

    Returns:
        The exit code.
    """
    cwd = pathlib.Path.cwd()
    root = _existing_machine_root(machine_id, cwd)
    if root is None:
        return 2
    ok, answer = machine_state.verb_answer(root, machine_id, "stop")
    if not ok:
        _common.refuse(f"{answer}")
        return 2
    if answer:
        print(f"[agent6] {answer}", file=sys.stderr)
        return 0
    write_stop_request(root)
    print(f"stop requested: {machine_id} parks at its next transition boundary")
    return 0


def _render_overview(ms: MachineState) -> str:
    """Return the state list with the current and visited states marked, from the shared fold."""
    lines = [f"machine: {ms.machine} (v{ms.version})  initial={ms.initial}", "states:"]
    for s in ms.states:
        mark = s.mark
        lines.append(f"  {mark} {s.name:<22} [{s.kind}]")
    return "\n".join(lines)


def _watch_liveness_exit(root: pathlib.Path, machine_id: str, ms: MachineState) -> int | None:
    """Return the watch's exit code when nothing will ever append to the journal.

    Parked (an armed `--exit-on-wait` wait, no worker) or stopped (no live worker, no end,
    no wait: an operator stop and a crash leave the same dir, since the worker clears its
    pid on every unwound exit). Routed through `machine_word_for_dir`, the one owner of the
    running, waiting and stopped words, so watch agrees with status, the TUI and the web.

    Args:
        root: The instance dir.
        machine_id: The instance.
        ms: The machine state.

    Returns:
        The exit code, or None while a live worker may still write, one blocked in a
        foreground wait included.
    """
    word = machine_word_for_dir(ms, root)
    current = next((st.name for st in ms.states if st.is_current), "?")
    if word == "waiting" and not ipc.worker_is_alive(root):
        print(
            f"\nWAITING in {current!r} (poke to resume):"
            f" agent6 machine poke {machine_id} [--message TEXT]"
        )
        return 0
    if word == "stopped":
        print(
            f"\nSTOPPED in {current!r}: no worker is running and the machine has not ended;"
            " resume with `agent6 machine run`",
            file=sys.stderr,
        )
        return 1
    return None


def _cmd_machine_watch(machine_id: str) -> int:  # noqa: PLR0911, PLR0912
    """Follow a running machine: the overview, each transition, the live reasoning.

    Read-only. Exits when the worker is dead (parked or crashed), when the instance ended,
    or on Ctrl-C.

    Returns:
        The exit code.
    """
    cwd = pathlib.Path.cwd()
    root = _existing_machine_root(machine_id, cwd)
    if root is None:
        return 2
    source = root / "machine.asm.toml"
    try:
        spec = load_machine(source)
    except MachineError as exc:
        return machine_check._fail(source, exc.problems)
    journal = MachineJournal(root)
    try:
        ms = fold_machine(spec, journal.read())
    except JournalError as exc:
        _common.error(f"{exc}")
        return 1
    print(_render_overview(ms), flush=True)
    if ms.ended is not None:
        print(f"\n{ms.ended.status.upper()}: ended in {ms.ended.state!r} ({ms.ended.reason})")
        return 0 if ms.ended.status == "ok" else 1
    code = _watch_liveness_exit(root, machine_id, ms)
    if code is not None:
        return code

    print("\n[agent6] watching (Ctrl-C to stop)...", file=sys.stderr)
    print(
        "[agent6] poke a waiting machine from another shell: "
        f"agent6 machine poke {machine_id} [--message TEXT]",
        file=sys.stderr,
    )
    cursor = MachineWatchCursor(seen_steps=len(ms.transitions))
    cursor.seed_notifications(ms)  # history already rendered by the overview
    anchor: float | None = None
    try:
        while True:
            try:
                ms = fold_machine(spec, journal.read())
            except JournalError as exc:
                # The same degradation `machine status` gives a corrupt journal, never a traceback.
                _common.error(f"{exc}")
                return 1
            for t in cursor.new_transitions(ms):
                print(f"  {t.line}", flush=True)
            for n in cursor.new_notifications(ms):
                # The bell and a desktop notification, so an operator watching over ssh is alerted.
                print(f"\a  🔔 [{n.level}] {n.state}: {n.message}", flush=True)
                notify.desktop_notify(f"agent6: {ms.machine}", n.message)
            newest, switched = cursor.advance_log(root)
            if switched:
                # Each state log re-derives its elapsed-time base, else states 2..N read inflated.
                anchor = None
                if newest is not None:
                    print(f"  -- agent state: {newest.parent.name} --", file=sys.stderr)
            for line in cursor.read_log_lines():
                if anchor is None:
                    with contextlib.suppress(json.JSONDecodeError):
                        anchor = event_epoch(json.loads(line).get("ts"))
                print(
                    "    " + plan_watch.format_plain_event(line, session_start_ts=anchor),
                    flush=True,
                )
            if ms.ended is not None:
                print(
                    f"\n{ms.ended.status.upper()}: ended in {ms.ended.state!r} after"
                    f" {ms.ended.transitions} transitions ({ms.ended.reason})"
                )
                return 0 if ms.ended.status == "ok" else 1
            code = _watch_liveness_exit(root, machine_id, ms)
            if code is not None:
                return code
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[agent6] watch: stopped.", file=sys.stderr)
        return 0


def _machine_frontend() -> MachineFrontend:
    """Return the presentation seam `machine run` and `machine create` drive.

    Stdio output plus the interactive network-refusal resolver, which needs a TTY, so it
    stays on the CLI side; `create_machine` uses only the reporter.
    """
    return MachineFrontend(
        reporter=reporter.STDIO_REPORTER, resolve_network_fix=_resolve_network_refusal
    )
