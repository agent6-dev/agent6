# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 machine` subcommands: argv adaptation + console rendering.

The read-only commands (check/test/graph/status/replay/poke/watch) load + render
directly; run/create adapt argv and hand the lifecycle to `app.machine` behind
the `MachineFrontend` seam. The interactive network-refusal resolver stays here
(it needs a TTY)."""

from __future__ import annotations

import contextlib
import datetime as _dt
import json
import sys
import time
import tomllib
from pathlib import Path
from typing import Any, Literal

from agent6.app._setup import detect_env
from agent6.app.machine import (
    MachineFrontend,
    lint_and_typecheck,
    machine_network_refusal,
    machine_spend,
    run_offline_tests,
    validate_bundle,
)
from agent6.app.machine.create import create_machine
from agent6.app.machine.run import run_machine
from agent6.app.reporter import STDIO_REPORTER
from agent6.config import (
    Config,
    ConfigError,
)
from agent6.config.io import upsert_toml_leaf
from agent6.config.layer import (
    load_effective_with_overlay,
    repo_config_path_for,
)
from agent6.machine import (
    DryRunReport,
    EngineError,
    JournalError,
    MachineEnd,
    MachineError,
    MachineJournal,
    MachineSpec,
    StepEvent,
    ToolState,
    drive,
    dry_run,
    load_machine,
    render,
)
from agent6.paths import chown_to_real_user
from agent6.runs.ipc import read_worker_pid, worker_is_alive
from agent6.sandbox.detect import ProfileUnavailableError, select_profile
from agent6.types import SandboxProfile
from agent6.ui.cli._common import _machines_dir
from agent6.ui.cli.plan_watch import event_epoch, format_plain_event
from agent6.ui.notify import desktop_notify
from agent6.viewmodel import (
    MachineState,
    MachineWatchCursor,
    fold_machine,
)


def _fail(path: Path, problems: list[str], label: str = "") -> int:
    """Print a FAIL header + problem bullets to stderr; always returns 1."""
    suffix = f" ({label})" if label else ""
    print(f"FAIL: {path}{suffix}", file=sys.stderr)
    for problem in problems:
        print(f"  - {problem}", file=sys.stderr)
    return 1


def _load_validated(path: Path) -> tuple[MachineSpec | None, list[str], str]:
    """Shared `check`/`test` front half: load + structural bundle validation.

    Returns (spec, problems, label). spec is None when validation failed;
    label names the failing stage for the FAIL header.
    """
    try:
        spec = load_machine(path)
    except MachineError as exc:
        return None, list(exc.problems), ""
    bundle_problems = validate_bundle(spec, path)
    if bundle_problems:
        return None, bundle_problems, "bundle"
    return spec, [], ""


def _cmd_machine_check(path: Path) -> int:
    spec, problems, label = _load_validated(path)
    if spec is None:
        return _fail(path, problems, label)
    script_problems = lint_and_typecheck(path.parent / "scripts")
    if script_problems:
        return _fail(path, script_problems, "scripts")
    print(f"OK: {path} ({spec.machine}, {len(spec.states)} states)")
    return 0


def _cmd_machine_test(path: Path, *, blackboard: Path | None) -> int:
    # `machine test` is the offline simulation: `machine check`'s structural +
    # bundle validation, plus running the bundle's `*_test.py` mocks in a jail
    # (no network), plus a pure dry-run. Reuse the same load + bundle validation
    # so a malformed machine fails the same way.
    spec, problems, label = _load_validated(path)
    if spec is None:
        return _fail(path, problems, label)
    # Static (lint + types) then the offline mock tests in a no-network jail.
    script_problems = lint_and_typecheck(path.parent / "scripts")
    script_problems.extend(run_offline_tests(path.parent, detect_env().detected_profile))
    if script_problems:
        return _fail(path, script_problems, "scripts")
    fixture: dict[str, Any] | None = None
    if blackboard is not None:
        if not blackboard.is_file():
            print(f"ERROR: blackboard fixture not found: {blackboard}", file=sys.stderr)
            return 2
        try:
            fixture = tomllib.loads(blackboard.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as exc:
            print(f"ERROR: blackboard fixture is not valid TOML: {exc}", file=sys.stderr)
            return 2
    report = dry_run(spec, fixture)
    _print_dry_run_report(spec, report)
    if report.ok:
        print(
            f"\nOK: {path} dry-run passed ({_plural(len(report.states), 'state')}, "
            f"{_plural(len(report.branches), 'branch', 'branches')})"
        )
        return 0
    print(f"\nFAIL: {path} dry-run found problems", file=sys.stderr)
    return 1


def _print_dry_run_report(spec: MachineSpec, report: DryRunReport) -> None:
    """Render the per-state and per-branch dry-run tables."""
    mark = {True: "ok", False: "FAIL"}
    print(f"machine {spec.machine!r}: per-state dry-run")
    print(f"  {'STATE':<16} {'KIND':<9} {'->LABEL':<9} {'GOTO':<14} STATUS  DETAIL")
    for s in report.states:
        print(
            f"  {s.name:<16} {s.kind:<9} {(s.label or '-'):<9} {(s.goto or '-'):<14}"
            f" {mark[s.ok]:<6}  {s.detail}"
        )
    if report.branches:
        print("\nper-branch routing (fixture overlaid on defaults)")
        print(f"  {'STATE':<16} {'CLAUSE':<7} {'GOTO':<14} STATUS  PREDICATE")
        for b in report.branches:
            clause = "-" if b.clause_index is None else f"[{b.clause_index}]"
            pred = b.detail if not b.ok else (b.predicate or "")
            print(f"  {b.name:<16} {clause:<7} {(b.goto or '-'):<14} {mark[b.ok]:<6}  {pred}")


def _cmd_machine_graph(path: Path, *, fmt: str) -> int:
    try:
        spec = load_machine(path)
    except MachineError as exc:
        print(f"FAIL: {path}", file=sys.stderr)
        for problem in exc.problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    render_fmt: Literal["mermaid", "dot"] = "dot" if fmt == "dot" else "mermaid"
    print(render(spec, render_fmt), end="")
    return 0


def _safe_input(prompt: str) -> str | None:
    """``input`` that returns None on EOF / non-interactive stdin instead of raising."""
    try:
        return input(prompt)
    except (EOFError, KeyboardInterrupt):
        return None


def _suggested_network_fix(
    cfg: Config, profile: SandboxProfile, tool_states: list[ToolState]
) -> dict[str, str] | None:
    """The minimal sandbox-config change that lets this machine's tool states run
    ON THIS PROFILE, or None if no config change can (a tool that REQUIRES network
    isolation needs `strict`, which config can't conjure).

    Two refusals this resolves: a tool that opted in (`allow_network = "allow"`)
    under a config that blocks egress, and -- on `hardened`, which can't give any
    tool its own netns -- a plain tool refused under `tool_network = "block"`. The
    returned dict is applied in order, with `agent_network` before `tool_network`
    so `config set`-style sequential writes never trip the combo validator."""
    if not tool_states:
        return None
    has_allow = any(s.allow_network == "allow" for s in tool_states)
    has_block = any(s.allow_network == "block" for s in tool_states)
    if has_block:
        # A tool REQUIRES no network; only strict's per-tool netns isolates it.
        return None
    if profile == "strict":
        # Plain no-network tools already run on strict; only a tool that opted
        # into the network needs the explicit-per-tool egress mode.
        return {"sandbox.tool_network": "only_explicit_states"} if has_allow else None
    if profile == "hardened":
        # hardened can't isolate one tool's netns, so EVERY tool (networked or
        # not) shares the host network; the combo validator then requires
        # agent_network = "open". Same fix whether or not a tool opted in.
        return {"sandbox.agent_network": "open", "sandbox.tool_network": "allow"}
    return None


def _resolve_network_refusal(  # noqa: PLR0911
    path: Path,
    refusal: str,
    cfg: Config,
    profile: SandboxProfile,
    tool_states: list[ToolState],
    cwd: Path,
    overlay: dict[str, Any],
) -> int | tuple[Config, SandboxProfile]:
    """A hard network refusal becomes a choice, not a dead end: explain it, then
    (interactively) offer to apply the minimal config fix and continue, simulate
    the machine offline, or stop. Headless prints the exact fix + simulate
    command and exits non-zero, it never relaxes a sandbox setting unattended.
    Returns the new ``(cfg, profile)`` when the fix applied and re-validates
    clear, else an exit code."""
    print(f"REFUSING: {refusal}", file=sys.stderr)
    fix = _suggested_network_fix(cfg, profile, tool_states)
    if fix is None:
        print(
            f"  No sandbox-config change fixes this on the '{profile}' profile"
            " (a tool needs isolation only 'strict' provides).",
            file=sys.stderr,
        )
        print(f"  Simulate it offline instead:  agent6 machine test {path}", file=sys.stderr)
        return 2
    if not sys.stdin.isatty():
        print("  To allow it, apply this to the per-repo config and re-run:", file=sys.stderr)
        for key, value in fix.items():
            print(f"    agent6 config set {key} {value} --repo", file=sys.stderr)
        print(f"  Or simulate it offline now:    agent6 machine test {path}", file=sys.stderr)
        return 2
    print("  agent6 can apply the minimal fix now (writes the per-repo config):", file=sys.stderr)
    for key, value in fix.items():
        print(f"    {key} = {value}", file=sys.stderr)
    choice = (_safe_input("  [a]pply & run, [s]imulate offline, or [Q]uit? ") or "").strip().lower()
    if choice == "s":
        return _cmd_machine_test(path, blackboard=None)
    if choice != "a":
        print("Stopped; nothing changed.", file=sys.stderr)
        return 2
    target = repo_config_path_for(cwd)
    target.parent.mkdir(parents=True, exist_ok=True)
    for key, value in fix.items():
        upsert_toml_leaf(target, key, value)
    chown_to_real_user(target.parent)
    chown_to_real_user(target)
    try:
        new_cfg = load_effective_with_overlay(cwd, overlay).config
        new_profile = select_profile(new_cfg.sandbox.profile, detect_env())
    except (ConfigError, ProfileUnavailableError) as exc:
        print(f"  Applied, but the config no longer validates: {exc}", file=sys.stderr)
        return 2
    if machine_network_refusal(new_cfg, new_profile, tool_states) is not None:
        print("  Applied, but a conflict remains; review the per-repo config.", file=sys.stderr)
        return 2
    print(f"  Applied to {target}. Continuing the run.", file=sys.stderr)
    return new_cfg, new_profile


def _cmd_machine_run(
    path: Path, *, exit_on_wait: bool = False, disable_sandbox: bool = False
) -> int:
    return run_machine(
        path, _machine_frontend(), exit_on_wait=exit_on_wait, disable_sandbox=disable_sandbox
    )


def _cmd_machine_replay(machine_id: str) -> int:
    root = _machines_dir(Path.cwd()) / machine_id
    if not root.is_dir():
        print(f"ERROR: no machine instance at {root}", file=sys.stderr)
        return 1
    source_path = root / "machine.asm.toml"
    try:
        spec = load_machine(source_path)
    except MachineError as exc:
        print(f"FAIL: {source_path}", file=sys.stderr)
        for problem in exc.problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    journal = MachineJournal(root)
    try:
        result = drive(spec, journal, None, live=False)
    except (JournalError, EngineError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(
        f"{result.status.upper()}: {spec.machine} replayed to {result.state!r}"
        f" after {result.transitions} transitions ({result.reason})"
    )
    return 0 if result.status in ("ok", "incomplete") else 1


def _plural(n: int, singular: str, plural: str | None = None) -> str:
    """'1 transition' / '3 transitions' -- no '1 branches' in user-facing counts."""
    word = singular if n == 1 else (plural or singular + "s")
    return f"{n} {word}"


def _cmd_machine_status(machine_id: str) -> int:  # noqa: PLR0912
    root = _machines_dir(Path.cwd()) / machine_id
    if not root.is_dir():
        print(f"ERROR: no machine instance at {root}", file=sys.stderr)
        return 1
    source_path = root / "machine.asm.toml"
    try:
        spec = load_machine(source_path)
    except MachineError as exc:
        print(f"FAIL: {source_path}", file=sys.stderr)
        for problem in exc.problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    journal = MachineJournal(root)
    try:
        result = drive(spec, journal, None, live=False)
        events = journal.read()
        snapshot = journal.latest_snapshot()
        pending = journal.read_pending_wait()
    except (JournalError, EngineError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    alive = worker_is_alive(root)
    spend, inflight_state = machine_spend(events, root, alive=alive)

    print(f"machine: {spec.machine} (v{spec.version})")
    if alive:
        pid = read_worker_pid(root)
        running_in = f" -- running {inflight_state!r}" if inflight_state else ""
        print(f"  status: running (worker pid {pid} alive){running_in}")
    else:
        print(f"  status: {result.status}")
    print(f"  state: {result.state!r}")
    print(f"  transitions: {result.transitions}")
    print(f"  spend: ${spend.usd:.4f} (in={spend.input_tokens} tok, out={spend.output_tokens} tok)")
    if pending is not None:
        if pending.wake_epoch is not None:
            wake = _dt.datetime.fromtimestamp(pending.wake_epoch, tz=_dt.UTC).isoformat()
            print(f"  next wake: {wake} (waiting in {pending.state!r})")
        else:
            print(f"  waiting for a signal poke (in {pending.state!r})")
    if snapshot is not None and snapshot.blackboard:
        print("  blackboard:")
        for key, value in snapshot.blackboard.items():
            print(f"    {key} = {value!r}")
    step_events = [e for e in events if isinstance(e, StepEvent)]
    if step_events:
        print("  recent steps:")
        for event in step_events[-5:]:
            print(f"    [{event.seq}] {event.state!r} --{event.label}--> {event.goto!r}")
    return 0


def _cmd_machine_poke(
    machine_id: str, *, data: str | None = None, message: str | None = None
) -> int:
    root = _machines_dir(Path.cwd()) / machine_id
    if not root.is_dir():
        print(f"ERROR: no machine instance at {root}", file=sys.stderr)
        return 1
    journal = MachineJournal(root)
    try:
        events = journal.read()
    except JournalError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    # An ended machine consumes no signals: a poke would sit unread, so the
    # "it will wake on its next signal check" reply would be a lie. Refuse.
    if events and isinstance(events[-1], MachineEnd):
        end = events[-1]
        print(
            f"ERROR: {machine_id} already ended in {end.state!r} ({end.status}: {end.reason});"
            " a poke would never be consumed.",
            file=sys.stderr,
        )
        return 1
    if message is not None:
        payload: Any = message
    elif data is not None:
        try:
            payload = json.loads(data)
        except json.JSONDecodeError as exc:
            print(f"ERROR: --data is not valid JSON: {exc}", file=sys.stderr)
            return 2
    else:
        payload = None
    try:
        journal.poke(payload)
    except JournalError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    carried = "" if payload is None else " (with payload)"
    print(f"poked {machine_id}: it will wake on its next signal check{carried}")
    return 0


def _render_overview(ms: MachineState) -> str:
    """The state list with the current state marked (`>`) and visited ones (`.`)
    -- the at-a-glance overview, rendered from the shared fold."""
    lines = [f"machine: {ms.machine} (v{ms.version})  initial={ms.initial}", "states:"]
    for s in ms.states:
        mark = ">" if s.is_current else ("." if s.is_visited else " ")
        lines.append(f"  {mark} {s.name:<22} [{s.kind}]")
    return "\n".join(lines)


def _cmd_machine_watch(machine_id: str) -> int:  # noqa: PLR0912
    """Follow a running machine: the state overview, each transition as it lands,
    and the current agent state's live reasoning (its per-state logs.jsonl). Exits
    when the machine ends/waits, or on Ctrl-C. Read-only."""
    root = _machines_dir(Path.cwd()) / machine_id
    if not root.is_dir():
        print(f"ERROR: no machine instance at {root}", file=sys.stderr)
        return 1
    source = root / "machine.asm.toml"
    try:
        spec = load_machine(source)
    except MachineError as exc:
        print(f"FAIL: {source}", file=sys.stderr)
        for problem in exc.problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    journal = MachineJournal(root)
    ms = fold_machine(spec, journal.read())
    print(_render_overview(ms), flush=True)
    if ms.ended is not None:
        print(f"\n{ms.ended.status.upper()}: ended in {ms.ended.state!r} ({ms.ended.reason})")
        return 0 if ms.ended.status == "ok" else 1

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
            ms = fold_machine(spec, journal.read())
            for t in cursor.new_transitions(ms):
                print(f"  [{t.seq:>3}] {t.state} --{t.label}--> {t.goto}", flush=True)
            for n in cursor.new_notifications(ms):
                # Ring the bell + fire a desktop notification (if notify-send is
                # present) so an operator watching over ssh is alerted.
                print(f"\a  🔔 [{n.level}] {n.state}: {n.message}", flush=True)
                desktop_notify(f"agent6: {ms.machine}", n.message)
            newest, switched = cursor.advance_log(root)
            if switched:
                # Reset the elapsed-time anchor too: each state log re-derives its
                # own base from its first event, else states 2..N read inflated.
                anchor = None
                if newest is not None:
                    print(f"  -- agent state: {newest.parent.name} --", file=sys.stderr)
            for line in cursor.read_log_lines():
                if anchor is None:
                    with contextlib.suppress(json.JSONDecodeError):
                        anchor = event_epoch(json.loads(line).get("ts"))
                print("    " + format_plain_event(line, run_start_ts=anchor), flush=True)
            if ms.ended is not None:
                print(
                    f"\n{ms.ended.status.upper()}: ended in {ms.ended.state!r} after"
                    f" {ms.ended.transitions} transitions ({ms.ended.reason})"
                )
                return 0 if ms.ended.status == "ok" else 1
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[agent6] watch: stopped.", file=sys.stderr)
        return 0


def _machine_frontend() -> MachineFrontend:
    """The presentation seam `app.machine` run/create drive: stdio output plus
    the interactive network-refusal resolver (needs a TTY, so it stays cli-side;
    `create_machine` uses only the reporter)."""
    return MachineFrontend(reporter=STDIO_REPORTER, resolve_network_fix=_resolve_network_refusal)


def _cmd_machine_create(task: str, *, output: Path | None, max_attempts: int) -> int:
    return create_machine(task, _machine_frontend(), output=output, max_attempts=max_attempts)
