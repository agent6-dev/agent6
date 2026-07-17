# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 machine` subcommands + bundle validation + agent runner."""

from __future__ import annotations

import contextlib
import datetime as _dt
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from agent6.app._setup import check_provider_keys as _check_provider_keys
from agent6.app._setup import detect_env
from agent6.app.egress import (
    check_network_profile,
    resolve_strict_egress_viability,
    warn_if_unsandboxed,
)
from agent6.app.machine import (
    Spend,
    build_machine_notify_hook,
    hard_usd_preflight_error,
    machine_network_refusal,
    machine_protect_paths,
    machine_spend,
    read_budget_totals,
    validate_bundle,
)
from agent6.config import (
    Config,
    ConfigError,
)
from agent6.config.io import upsert_toml_leaf
from agent6.config.layer import (
    load_effective,
    load_effective_with_overlay,
    repo_config_path_for,
)
from agent6.events import EventSink
from agent6.git_ops import CommitIdentity, GitError, verify_git_identity
from agent6.machine import (
    SCRIPTS_PAYLOAD_KEY,
    TOML_PAYLOAD_KEY,
    AgentExecResult,
    AgentRequest,
    AgentState,
    DryRunReport,
    EngineError,
    JournalError,
    LiveWorld,
    MachineError,
    MachineJournal,
    MachineSpec,
    StepEvent,
    ToolState,
    build_authoring_prompt,
    drive,
    dry_run,
    extract_scripts,
    extract_toml,
    load_machine,
    machine_lock,
    render,
    write_source,
)
from agent6.models.pricing import lookup_price
from agent6.paths import chown_to_real_user
from agent6.runs.id import new_friendly_id
from agent6.runs.ipc import read_worker_pid, worker_is_alive, write_worker_pid
from agent6.sandbox.detect import ProfileUnavailableError, select_profile
from agent6.types import SandboxProfile
from agent6.ui.cli._common import _machines_dir, _state_dir
from agent6.ui.cli.plan_watch import event_epoch, format_plain_event
from agent6.ui.cli.scriptcheck import lint_and_typecheck, run_offline_tests
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


def _build_machine_agent_runner(
    overlay: dict[str, Any],
    cwd: Path,
    profile: SandboxProfile,
    transcript_dir: Path,
    protect_paths: tuple[Path, ...] = (),
    commit_identity: CommitIdentity | None = None,
) -> Callable[[AgentRequest, Path | None], AgentExecResult]:
    """Build the runner an `agent` state uses to drive a confined agent6 loop.

    The machine engine is a host-netns supervisor; each `agent` state runs in
    its OWN subprocess (`agent6.ui.cli.machine_agent`) which confines its egress
    per `sandbox.agent_network` before running the loop, independently of the
    engine and of sibling `tool` states. The subprocess is spawned with a fixed
    argv (no LLM-derived content) and handed the request via a temp file; the
    operator-authored prompt travels in that file, never on the command line.
    ``timeout_secs`` is enforced by killing the subprocess's whole process group
    (true mid-call cancellation, and the per-agent broker is torn down with it).

    ``events_log`` is per CALL: the live World passes each agent-state execution
    its own ``<instance>/states/<seq>-<state>/logs.jsonl`` and `machine create`
    passes the draft log, so the subprocess writes a watchable event stream there.
    """

    def run_agent(request: AgentRequest, events_log: Path | None = None) -> AgentExecResult:
        def salvaged(reason: str) -> AgentExecResult:
            # No result.json (killed/timed-out/crashed): recover the loop's
            # running budget.update totals from the state's own event log, else a
            # timed-out state books $0 and the budget guard never trips.
            spend = read_budget_totals(events_log) if events_log is not None else Spend()
            return AgentExecResult(
                reason=reason,
                payload=None,
                usd=spend.usd,
                input_tokens=spend.input_tokens,
                output_tokens=spend.output_tokens,
            )

        payload = {
            "cwd": str(cwd),
            "root": str(cwd),
            "overlay": overlay,
            "profile": profile,
            "transcript_dir": str(transcript_dir),
            # When set, the agent subprocess writes a watchable logs.jsonl here
            # (role.*_delta + tool.* events), so `machine create` is followable in
            # the TUI dashboard exactly like a run.
            "events_log": str(events_log) if events_log is not None else None,
            "protect_paths": [str(p) for p in protect_paths],
            # Resolved on the host (pre-Landlock, so it sees global git config);
            # the confined agent subprocess can't read ~/.gitconfig, so its
            # mode="run" commits would otherwise fail with "Author identity
            # unknown". None for read-only (mode="agent"/"machine") states.
            "commit_identity": (
                {"name": commit_identity.name, "email": commit_identity.email}
                if commit_identity is not None
                else None
            ),
            "request": {
                "model": request.model,
                "prompt": request.prompt,
                "timeout_s": request.timeout_s,
                "provider": request.provider,
                "thinking": request.thinking,
                "temperature": request.temperature,
                "max_usd": request.max_usd,
                "max_input_tokens": request.max_input_tokens,
                "max_output_tokens": request.max_output_tokens,
                "mode": request.mode,
            },
        }
        with tempfile.TemporaryDirectory(prefix="agent6-machine-agent-") as td:
            req_file = Path(td) / "request.json"
            out_file = Path(td) / "result.json"
            req_file.write_text(json.dumps(payload), encoding="utf-8")
            argv = [
                sys.executable,
                "-m",
                "agent6.ui.cli.machine_agent",
                str(req_file),
                str(out_file),
            ]
            # Own session/process group so the timeout kill takes the agent
            # subprocess AND its broker/jail children with it.
            proc = subprocess.Popen(argv, start_new_session=True)
            try:
                proc.wait(timeout=request.timeout_s)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.wait()
                return salvaged("timeout")
            if proc.returncode != 0 or not out_file.is_file():
                return salvaged("error")
            try:
                out = json.loads(out_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return salvaged("error")
            result_payload = out.get("payload")
            return AgentExecResult(
                reason=str(out.get("reason", "error")),
                payload=result_payload if isinstance(result_payload, dict) else None,
                usd=float(out.get("usd", 0.0)),
                input_tokens=int(out.get("input_tokens", 0)),
                output_tokens=int(out.get("output_tokens", 0)),
            )

    return run_agent


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


def _cmd_machine_run(  # noqa: PLR0911, PLR0912, PLR0915
    path: Path, *, exit_on_wait: bool = False, disable_sandbox: bool = False
) -> int:
    if disable_sandbox:
        # Set the env setter so this supervisor's select_profile resolves to
        # none; it then passes that profile to each agent subprocess in its
        # request (the subprocess trusts req["profile"], it does not re-resolve).
        # Using the env (vs mutating cfg) is the simplest single knob; the env
        # is operator-controlled and the LLM cannot reach it.
        os.environ["AGENT6_DANGEROUSLY_DISABLE_SANDBOX"] = "1"
    try:
        spec = load_machine(path)
    except MachineError as exc:
        print(f"FAIL: {path}", file=sys.stderr)
        for problem in exc.problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    # Re-validate the script bundle before executing anything: `load_machine`
    # does not, and on a profile that can't RO-bind the bundle a `scripts/`
    # symlink escaping it (which `machine check` rejects) would otherwise be read
    # by a tool. Security boundary, so run enforces it too, not just check.
    bundle_problems = validate_bundle(spec, path)
    if bundle_problems:
        return _fail(path, bundle_problems, "bundle")
    cwd = Path.cwd()
    states = list(spec.states.values())
    has_agent_state = any(getattr(s, "kind", None) == "agent" for s in states)
    # mode="run" agent states edit + commit; they need a resolved git identity.
    has_run_agent = any(isinstance(s, AgentState) and s.mode == "run" for s in states)
    tool_states = [s for s in states if isinstance(s, ToolState)]
    agent_runner: Callable[[AgentRequest, Path | None], AgentExecResult] | None = None
    # Default profile for confinement-free machines: resolve from the host.
    profile: SandboxProfile = detect_env().detected_profile
    # The running machine's own file + scripts bundle are read-only in every
    # run jail, so a tool/agent can't rewrite its own logic or bundled scripts.
    protect_paths = machine_protect_paths(path, cwd)
    # Load the effective config (machine [config] overlay included) for EVERY
    # machine: a pure wait/branch machine still reads [machine] snapshot_keep from
    # it, and validating the overlay up front means a bad overlay or an ignored
    # snapshot_keep never slips through to a pure machine. The agent/tool block
    # below adds the provider/sandbox checks only those state kinds need.
    try:
        cfg = load_effective_with_overlay(cwd, spec.config).config
    except ConfigError as exc:
        print(f"CONFIG ERROR:\n{exc}", file=sys.stderr)
        return 2
    snapshot_keep = cfg.machine.snapshot_keep
    if has_agent_state or tool_states:
        try:
            if has_agent_state:
                cfg.require_runnable("worker")
        except ConfigError as exc:
            print(f"CONFIG ERROR:\n{exc}", file=sys.stderr)
            return 2
        try:
            profile = select_profile(cfg.sandbox.profile, detect_env())
        except ProfileUnavailableError as exc:
            print(f"REFUSING: {exc}", file=sys.stderr)
            return 2
        agent_profile = profile
        if has_agent_state:
            # Same as run/resume: a strict that can't run the egress broker on
            # this process (surgical AppArmor profile) downgrades to hardened or
            # refuses, so the per-state agent subprocess gets a profile it can
            # actually use. Tool states keep `profile`: the jail launcher itself
            # can still run strict and give each tool its own namespace.
            agent_profile, egress_err = resolve_strict_egress_viability(cfg, profile)
            if egress_err is not None:
                print(egress_err, file=sys.stderr)
                return 2
        snapshot_keep = cfg.machine.snapshot_keep
        refusal = machine_network_refusal(cfg, profile, tool_states)
        if refusal is not None:
            outcome = _resolve_network_refusal(
                path, refusal, cfg, profile, tool_states, cwd, spec.config
            )
            if isinstance(outcome, int):
                return outcome
            cfg, profile = outcome  # fix applied + re-validated clear; continue
        if has_agent_state:
            missing = _check_provider_keys(cfg)
            if missing is not None:
                print(missing, file=sys.stderr)
                return 2
            # After _check_provider_keys so the price cache has been refreshed.
            usd_err = hard_usd_preflight_error(spec, cfg)
            if usd_err is not None:
                print(f"REFUSING: {usd_err}", file=sys.stderr)
                return 2
            # Resolve the commit identity HERE on the host, where global git
            # config is visible, so a mode="run" state's confined agent (which
            # can't read ~/.gitconfig under Landlock) still commits cleanly. A
            # missing identity fails loudly up front, not as mid-loop noise.
            commit_identity: CommitIdentity | None = None
            if has_run_agent:
                base = CommitIdentity(
                    name=cfg.git.commit.name,
                    email=cfg.git.commit.email,
                    coauthor=cfg.git.commit.coauthor,
                )
                try:
                    name, email = verify_git_identity(cwd, base)
                except GitError as exc:
                    print(f"ERROR: {exc}", file=sys.stderr)
                    return 2
                commit_identity = CommitIdentity(name=name, email=email)
            root = _machines_dir(cwd) / spec.machine
            # The engine is a host-netns supervisor; each agent state confines
            # itself in its own subprocess per sandbox.agent_network.
            agent_runner = _build_machine_agent_runner(
                spec.config,
                cwd,
                agent_profile,
                root / "agent_transcripts",
                protect_paths,
                commit_identity,
            )
    warn_if_unsandboxed(profile)
    root = _machines_dir(cwd) / spec.machine
    journal = MachineJournal(root, snapshot_keep=snapshot_keep)
    # Persistent, writable scratch for tool scripts (see LiveWorld.data_dir).
    data_dir = root / "data"
    try:
        with machine_lock(root):
            journal.ensure_dirs()
            data_dir.mkdir(parents=True, exist_ok=True)
            # Liveness marker for watchers (the web SSE stream probes it to
            # tell a crashed machine from a parked one), mirroring cli/run.py.
            write_worker_pid(root, os.getpid())
            if not journal.exists():
                write_source(root, path.read_text(encoding="utf-8"))
            world = LiveWorld(
                cwd=cwd,
                journal=journal,
                agent_runner=agent_runner,
                profile=profile,
                protect_paths=protect_paths,
                data_dir=data_dir,
                # Each agent state writes its own watchable logs.jsonl here, so a
                # running machine is followable like a run (pruned to keep recent).
                state_log_root=root / "states",
                # Operator argv fired on machine.notify/machine.end, on the host
                # outside the jail (None when [machine.notify].on_event is unset).
                notify_hook=build_machine_notify_hook(cfg, spec.machine, root),
                memory_limit_mb=cfg.sandbox.memory_limit_mb,
            )
            result = drive(spec, journal, world, live=True, exit_on_wait=exit_on_wait)
    except (JournalError, EngineError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if result.status == "waiting":
        print(
            f"WAITING: {spec.machine} paused in {result.state!r}"
            f" after {_plural(result.transitions, 'transition')} ({result.reason})"
        )
        return 0
    spend, _ = machine_spend(journal.read(), root, alive=False)
    print(
        f"{result.status.upper()}: {spec.machine} ended in {result.state!r}"
        f" after {_plural(result.transitions, 'transition')} ({result.reason})"
        f" -- spent ${spend.usd:.4f}"
    )
    return 0 if result.status == "ok" else 1


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
        MachineJournal(root).poke(payload)
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


_CREATE_TIMEOUT_S = 900.0


_CREATE_STOP_REASONS = frozenset(
    {"budget_exhausted", "timeout", "provider_error", "prompt_revision_failed"}
)


def _write_scripts(base_dir: Path, scripts: dict[str, str]) -> None:
    """Write the bundle's helper scripts (keys are bundle-relative, already
    validated by extract_scripts to live under scripts/ with no `..`).

    Defense-in-depth: unlink a pre-existing symlink at the target before writing
    so a planted `scripts/<name>` -> elsewhere link can't redirect the write out
    of the bundle. `validate_bundle` (run by check/run before any execution) is
    the comprehensive backstop for symlinks anywhere in the tree."""
    for rel, content in scripts.items():
        p = base_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.is_symlink():
            p.unlink()
        p.write_text(content if content.endswith("\n") else content + "\n", encoding="utf-8")


def _check_machine_text(
    text: str, scripts: dict[str, str], scratch: Path
) -> tuple[MachineSpec | None, list[str]]:
    """Validate a candidate `.asm.toml` + its scripts via `load_machine`.

    The scripts are written into the scratch bundle first so the missing-script
    check resolves against this attempt's files only (stale scripts from a prior
    attempt are cleared). Returns the parsed spec + empty problems on success, or
    `(None, problems)` when the source or its script bundle is invalid.
    """
    candidate_path = scratch / "candidate.asm.toml"
    candidate_path.write_text(text, encoding="utf-8")
    shutil.rmtree(scratch / "scripts", ignore_errors=True)
    _write_scripts(scratch, scripts)
    try:
        spec = load_machine(candidate_path)
    except MachineError as exc:
        return None, list(exc.problems)
    bundle_problems = validate_bundle(spec, candidate_path)
    if bundle_problems:
        return None, bundle_problems
    return spec, []


def _cmd_machine_create(  # noqa: PLR0911, PLR0912, PLR0915
    task: str, *, output: Path | None, max_attempts: int
) -> int:
    if max_attempts < 1:
        print("ERROR: --max-attempts must be >= 1.", file=sys.stderr)
        return 2
    cwd = Path.cwd()
    try:
        cfg = load_effective(cwd, None).config
        cfg.require_runnable("worker")
    except ConfigError as exc:
        print(f"CONFIG ERROR:\n{exc}", file=sys.stderr)
        return 2
    missing = _check_provider_keys(cfg)
    if missing is not None:
        print(missing, file=sys.stderr)
        return 2
    try:
        profile = select_profile(cfg.sandbox.profile, detect_env())
    except ProfileUnavailableError as exc:
        print(f"REFUSING: {exc}", file=sys.stderr)
        return 2
    profile, egress_err = resolve_strict_egress_viability(cfg, profile)
    if egress_err is not None:
        print(egress_err, file=sys.stderr)
        return 2
    net_err = check_network_profile(cfg, profile)
    if net_err is not None:
        print(f"REFUSING: {net_err}", file=sys.stderr)
        return 2
    warn_if_unsandboxed(profile)

    scratch = _state_dir(cwd) / "machine-drafts" / new_friendly_id()
    scratch.mkdir(parents=True, exist_ok=True)
    # Persist the natural-language task that drove this draft, so the draft dir is
    # self-describing (the agent_transcripts/ embed it inside the authoring prompt,
    # but a plain prompt.txt is what a human looks for).
    (scratch / "prompt.txt").write_text(task, encoding="utf-8")
    # A watchable event log for the draft: the TUI opens the dashboard on this dir
    # and follows the authoring agent live. The parent owns the run.start header
    # (the NL task) + the per-attempt markers + the final run.end; each attempt's
    # subprocess appends its own role.*_delta / tool.* events to the same file.
    events_log = scratch / "logs.jsonl"
    events = EventSink(events_log)
    events.emit("run.start", user_task=task, mode="machine")
    # Authoring can take minutes with nothing on this terminal; say where the
    # live reasoning streams so the operator can follow instead of wondering.
    print(
        f"machine create: drafting as {scratch.name} (follow live: agent6 attach {scratch.name})",
        file=sys.stderr,
    )
    # Authoring drafts a machine; it has no machine [config] overlay of its own.
    runner = _build_machine_agent_runner({}, cwd, profile, scratch / "agent_transcripts")

    # The drafted machine's agent states inherit this worker model. If it is
    # unpriced (anthropic-direct, local), steer the draft to best_effort_usd_limit
    # so the freshly-created machine actually runs -- a hard max_usd would refuse.
    # Checked after _check_provider_keys refreshed the price cache.
    worker = cfg.models.resolve("worker")
    worker_unpriced = worker is None or lookup_price(worker.model) is None

    prior_toml: str | None = None
    prior_scripts: dict[str, str] = {}
    diagnostics: list[str] | None = None
    spec: MachineSpec | None = None
    valid_toml: str | None = None
    valid_scripts: dict[str, str] = {}
    total_usd = 0.0
    for attempt in range(1, max_attempts + 1):
        prompt = build_authoring_prompt(
            task,
            attempt=attempt,
            prior_toml=prior_toml,
            diagnostics=diagnostics,
            prior_scripts=prior_scripts,
            worker_unpriced=worker_unpriced,
        )
        print(f"machine create: attempt {attempt}/{max_attempts}...", file=sys.stderr)
        events.emit("loop.note", text=f"attempt {attempt}/{max_attempts}")
        # model omitted (=None): inherit the operator's effective worker model.
        # mode="machine": authoring system prompt + read-only tools (see loop.py).
        # thinking="off": authoring is transcription of a described design, not
        # deep derivation. "low" is already the provider default and did not
        # help: kimi-k2.6 spiralled into 30-minute length-capped thinks and
        # timed out on every attempt (0/3 drafts across two spec sizes). With
        # the reasoning channel off it drafted in ~2.5 minutes for $0.02.
        result = runner(
            AgentRequest(
                prompt=prompt, timeout_s=_CREATE_TIMEOUT_S, mode="machine", thinking="off"
            ),
            events_log,
        )
        total_usd += result.usd
        candidate = extract_toml(result.payload)
        if candidate is None:
            diagnostics = [
                f"You did not return a draft: call finish_run with result.{TOML_PAYLOAD_KEY}"
                " set to the complete .asm.toml source as a single string."
                f" (agent loop reason: {result.reason})"
            ]
            prior_toml = None
            prior_scripts = {}
            if result.reason in _CREATE_STOP_REASONS:
                break
            continue
        candidate_scripts = extract_scripts(result.payload)
        candidate_spec, problems = _check_machine_text(candidate, candidate_scripts, scratch)
        if candidate_spec is None:
            # Structural / bundle failure. A missing-script problem (only produced
            # here, never by the lint/test pass below) gets an extra hint pointing
            # the agent at result.scripts.
            if any("not found in bundle" in p for p in problems):
                hint = (
                    f"Return each missing scripts/... file in finish_run"
                    f" result.{SCRIPTS_PAYLOAD_KEY} (a map of the path to its complete source)."
                )
                problems = [*problems, hint]
        else:
            # Structurally valid. Now make it production-ready: lint + type-check
            # the scripts, run their offline `*_test.py` mocks in a jail, and
            # dry-run the routing (synthesized facts through the real reducer;
            # catches e.g. a branch reading a field the schema doesn't declare).
            # Any failure becomes a retry diagnostic so the agent fixes it itself.
            print("machine create: linting + offline-testing scripts...", file=sys.stderr)
            events.emit("loop.note", text="linting + offline-testing the draft")
            problems = lint_and_typecheck(scratch / "scripts")
            problems.extend(run_offline_tests(scratch, profile))
            report = dry_run(candidate_spec, None)
            problems.extend(
                f"dry-run state {c.name!r}: {c.detail}"
                for c in (*report.states, *report.branches)
                if not c.ok
            )
            if not problems:
                spec = candidate_spec
                valid_toml = candidate
                valid_scripts = candidate_scripts
                break
        prior_toml = candidate
        prior_scripts = candidate_scripts
        diagnostics = problems
        if result.reason in _CREATE_STOP_REASONS:
            break

    print(f"machine create: spent ~${total_usd:.4f}", file=sys.stderr)
    # End the watchable session (the file-write below is fast and event-less);
    # all_passed marks whether a valid machine was authored, for the TUI status.
    events.emit(
        "run.end",
        all_passed=spec is not None and valid_toml is not None,
        reason="machine create finished",
    )

    if spec is None or valid_toml is None:
        print(f"FAILED: no valid machine after {max_attempts} attempt(s).", file=sys.stderr)
        if diagnostics:
            print("Last diagnostics:", file=sys.stderr)
            for problem in diagnostics:
                print(f"  - {problem}", file=sys.stderr)
        if prior_toml is not None:
            print("The last (invalid) draft is on stdout for reference.", file=sys.stderr)
            print(prior_toml if prior_toml.endswith("\n") else prior_toml + "\n", end="")
        return 1

    payload = valid_toml if valid_toml.endswith("\n") else valid_toml + "\n"
    target = output if output is not None else cwd / f"{spec.machine}.asm.toml"
    if output is None and target.exists():
        print(f"REFUSING to overwrite existing {target}.", file=sys.stderr)
        print(
            "The validated draft is on stdout; redirect it or re-run with -o <file>.",
            file=sys.stderr,
        )
        print(payload, end="")
        return 1
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(payload, encoding="utf-8")
    _write_scripts(target.parent, valid_scripts)
    scripts_note = f" + {len(valid_scripts)} script(s)" if valid_scripts else ""
    print(
        f"OK: wrote draft to {target} ({spec.machine}, {len(spec.states)} states){scripts_note}.",
        file=sys.stderr,
    )
    # The scratch validation ran against a clean copy; re-run the STRUCTURAL
    # bundle check on the output dir, which can differ from scratch (e.g. a
    # pre-existing symlink under scripts/). Lint/types are NOT re-run: the
    # written files are byte-identical to the scratch copy that just passed.
    out_problems = validate_bundle(spec, target)
    if out_problems:
        print("WARNING: the written bundle has problems and won't run yet:", file=sys.stderr)
        for problem in out_problems:
            print(f"  - {problem}", file=sys.stderr)
    print(
        "Review and commit it; `machine run` only accepts committed machines.",
        file=sys.stderr,
    )
    return 0
