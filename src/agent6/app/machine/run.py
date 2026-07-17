# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`agent6 machine run`: compose the engine and drive a machine to completion.

The engine (`agent6.machine`) is a host-netns supervisor; this module resolves
the sandbox profile, egress viability, provider keys, budget-price and git
identity preflight, builds the per-`agent`-state runner and the `LiveWorld`, and
calls `drive`. Output routes through the injected `MachineFrontend.reporter`; a
hard tool-network refusal is handed to `frontend.resolve_network_fix` (the one
interactive step, held cli-side). The machine ENGINE is unchanged.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from agent6.app._setup import check_provider_keys, detect_env
from agent6.app.egress import resolve_strict_egress_viability, warn_if_unsandboxed
from agent6.app.machine._bundle import validate_bundle
from agent6.app.machine._frontend import MachineFrontend
from agent6.app.machine._preflight import (
    build_machine_notify_hook,
    hard_usd_preflight_error,
    machine_network_refusal,
    machine_protect_paths,
)
from agent6.app.machine._spend import machine_spend
from agent6.app.machine_agent import build_machine_agent_runner
from agent6.app.reporter import Reporter
from agent6.config import ConfigError
from agent6.config.layer import load_effective_with_overlay, resolved_state_dir
from agent6.git_ops import CommitIdentity, GitError, is_git_repo, paths_dirty, verify_git_identity
from agent6.machine import (
    AgentExecResult,
    AgentRequest,
    AgentState,
    EngineError,
    JournalError,
    LiveWorld,
    MachineError,
    MachineJournal,
    ToolState,
    drive,
    load_machine,
    machine_lock,
    write_source,
)
from agent6.runs.ipc import write_worker_pid
from agent6.sandbox.detect import ProfileUnavailableError, select_profile
from agent6.types import SandboxProfile


def _fail(reporter: Reporter, path: Path, problems: list[str], label: str = "") -> int:
    """Print a FAIL header + problem bullets to stderr; always returns 1."""
    suffix = f" ({label})" if label else ""
    reporter.err(f"FAIL: {path}{suffix}")
    for problem in problems:
        reporter.err(f"  - {problem}")
    return 1


def _transitions(n: int) -> str:
    return f"{n} transition{'' if n == 1 else 's'}"


def _uncommitted_refusal(path: Path, cwd: Path) -> str | None:
    """A refusal message if the machine file has uncommitted changes, else None.

    `machine run` only accepts a committed machine (docs state-machines.md
    §7.1/§9; the `machine create` hint promises it): a tool/agent reads the file
    as trusted logic, so an untracked or dirty `.asm.toml` is unreviewed. Skipped
    outside a git repo (nothing to commit against) and for a file that resolves
    outside the repo tree."""
    if not is_git_repo(cwd):
        return None
    try:
        rel = path.resolve().relative_to(cwd.resolve()).as_posix()
    except ValueError:
        return None
    try:
        if not paths_dirty(cwd, (rel,)):
            return None
    except GitError:
        return None
    return (
        f"{path} has uncommitted changes; `machine run` only accepts a committed"
        " machine. Review and commit the .asm.toml first."
    )


def run_machine(  # noqa: PLR0911, PLR0912, PLR0915
    path: Path,
    frontend: MachineFrontend,
    *,
    exit_on_wait: bool = False,
    disable_sandbox: bool = False,
) -> int:
    reporter = frontend.reporter
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
        return _fail(reporter, path, list(exc.problems))
    # Re-validate the script bundle before executing anything: `load_machine`
    # does not, and on a profile that can't RO-bind the bundle a `scripts/`
    # symlink escaping it (which `machine check` rejects) would otherwise be read
    # by a tool. Security boundary, so run enforces it too, not just check.
    bundle_problems = validate_bundle(spec, path)
    if bundle_problems:
        return _fail(reporter, path, bundle_problems, "bundle")
    cwd = Path.cwd()
    # Machines are operator artifacts: refuse an uncommitted file before running
    # anything (docs §7.1/§9), so a tool/agent never executes unreviewed logic.
    uncommitted = _uncommitted_refusal(path, cwd)
    if uncommitted is not None:
        reporter.err(f"REFUSING: {uncommitted}")
        return 1
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
        reporter.err(f"CONFIG ERROR:\n{exc}")
        return 2
    snapshot_keep = cfg.machine.snapshot_keep
    if has_agent_state or tool_states:
        try:
            if has_agent_state:
                cfg.require_runnable("worker")
        except ConfigError as exc:
            reporter.err(f"CONFIG ERROR:\n{exc}")
            return 2
        try:
            profile = select_profile(cfg.sandbox.profile, detect_env())
        except ProfileUnavailableError as exc:
            reporter.err(f"REFUSING: {exc}")
            return 2
        agent_profile = profile
        if has_agent_state:
            # Same as run/resume: a strict that can't run the egress broker on
            # this process (surgical AppArmor profile) downgrades to hardened or
            # refuses, so the per-state agent subprocess gets a profile it can
            # actually use. Tool states keep `profile`: the jail launcher itself
            # can still run strict and give each tool its own namespace.
            agent_profile, egress_err = resolve_strict_egress_viability(
                cfg, profile, reporter=reporter
            )
            if egress_err is not None:
                reporter.err(egress_err)
                return 2
        snapshot_keep = cfg.machine.snapshot_keep
        refusal = machine_network_refusal(cfg, profile, tool_states)
        if refusal is not None:
            outcome = frontend.resolve_network_fix(
                path, refusal, cfg, profile, tool_states, cwd, spec.config
            )
            if isinstance(outcome, int):
                return outcome
            cfg, profile = outcome  # fix applied + re-validated clear; continue
        if has_agent_state:
            missing = check_provider_keys(cfg)
            if missing is not None:
                reporter.err(missing)
                return 2
            # After check_provider_keys so the price cache has been refreshed.
            usd_err = hard_usd_preflight_error(spec, cfg)
            if usd_err is not None:
                reporter.err(f"REFUSING: {usd_err}")
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
                    reporter.err(f"ERROR: {exc}")
                    return 2
                commit_identity = CommitIdentity(name=name, email=email)
            root = resolved_state_dir(cwd) / "machines" / spec.machine
            # The engine is a host-netns supervisor; each agent state confines
            # itself in its own subprocess per sandbox.agent_network.
            agent_runner = build_machine_agent_runner(
                spec.config,
                cwd,
                agent_profile,
                root / "agent_transcripts",
                protect_paths,
                commit_identity,
            )
    warn_if_unsandboxed(profile, reporter=reporter)
    root = resolved_state_dir(cwd) / "machines" / spec.machine
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
        reporter.err(f"ERROR: {exc}")
        return 1
    if result.status == "waiting":
        reporter.out(
            f"WAITING: {spec.machine} paused in {result.state!r}"
            f" after {_transitions(result.transitions)} ({result.reason})"
        )
        return 0
    spend, _ = machine_spend(journal.read(), root, alive=False)
    reporter.out(
        f"{result.status.upper()}: {spec.machine} ended in {result.state!r}"
        f" after {_transitions(result.transitions)} ({result.reason})"
        f" -- spent ${spend.usd:.4f}"
    )
    return 0 if result.status == "ok" else 1
