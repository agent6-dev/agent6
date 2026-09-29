# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Compose the engine and drive a machine to completion, for `agent6 machine run`.

Resolves the isolation, egress, provider-key, budget and git-identity preflight, builds the
per-`agent`-state runner and the `LiveWorld`, and calls `drive`. A tool-network refusal is
handed to `frontend.resolve_network_fix`, the one interactive step.
"""

from __future__ import annotations

import contextlib
import dataclasses
import os
import shutil
import sys
import uuid
from collections.abc import Callable
from pathlib import Path

from pydantic import ValidationError

from agent6.app._session import resolve_isolation_or_refuse
from agent6.app._setup import check_provider_keys, detect_env
from agent6.app.confine import (
    check_hide_paths_support,
    config_refusal,
    warn_cleartext_credential_endpoints,
    warn_sandbox_gaps,
)
from agent6.app.frontend import apply_spawned_away_default, approval_scopes
from agent6.app.machine._bundle import validate_bundle
from agent6.app.machine._frontend import MachineFrontend
from agent6.app.machine._preflight import (
    NetworkRefusal,
    build_machine_notify_hook,
    machine_network_refusal,
    machine_pass_env_refusal,
    machine_protect_paths,
)
from agent6.app.machine._spend import book_crashed_attempt
from agent6.app.machine_agent import build_machine_agent_runner, clone_at_machine_chain
from agent6.app.parallel import subordinate_workdir_root
from agent6.app.preflight import SessionRefusedError, budget_preflight
from agent6.app.reporter import Reporter
from agent6.config import Config, ConfigError
from agent6.config.layer import load_effective_with_overlay
from agent6.git_ops import (
    CommitIdentity,
    GitError,
    branch_exists,
    chain_tip,
    delete_ref,
    is_ancestor,
    is_git_repo,
    machine_branch_for,
    machine_chain_ref_for,
    paths_dirty,
    verify_git_identity,
)
from agent6.harness.subrun import SubrunError
from agent6.kinds import CommandResult, IsolationLevel, JailPolicy, NetworkMode
from agent6.machine import (
    AgentExecResult,
    AgentRequest,
    AgentState,
    EngineError,
    JournalError,
    LiveWorld,
    MachineEnd,
    MachineError,
    MachineJournal,
    ToolPolicyFactory,
    ToolState,
    bundle_drift,
    clear_stop_request,
    drive,
    load_machine,
    machine_lock,
    write_bundle,
)
from agent6.paths import mkdir_for_real_user, state_dir
from agent6.sandbox.jail import JailUnavailableError, run_in_jail
from agent6.sessions.ipc import clear_worker_pid, write_worker_pid
from agent6.sessions.layout import machines_root
from agent6.tools.policy import jail_policy, passthrough_env
from agent6.viewmodel.format import format_usd
from agent6.viewmodel.machine_state import machine_spend, wait_line


def _fail(reporter: Reporter, path: Path, problems: list[str], label: str = "") -> int:
    """Return the exit code 1 after printing a FAIL header and the problems."""
    suffix = f" ({label})" if label else ""
    reporter.err(f"FAIL: {path}{suffix}")
    for problem in problems:
        reporter.err(f"  - {problem}")
    return 1


def _transitions(n: int) -> str:
    """Return the transition count with its noun."""
    return f"{n} transition{'' if n == 1 else 's'}"


def uncommitted_refusal(path: Path, cwd: Path) -> str | None:
    """Return the refusal when the machine's bundle has uncommitted changes, else None.

    A tool or agent executes the bundle as trusted logic, so an uncommitted piece is
    unreviewed; `machine test` is the ungated iteration loop. Skipped outside a git repo and
    for pieces that resolve outside the repo tree.
    """
    if not is_git_repo(cwd):
        return None
    scripts = path.parent / "scripts"
    pieces = [(path, "machine")] + ([(scripts, "scripts bundle")] if scripts.exists() else [])
    for piece, label in pieces:
        try:
            # Resolving the entry itself would turn a retargeted symlink into its clean target.
            rel = (piece.parent.resolve() / piece.name).relative_to(cwd.resolve()).as_posix()
        except ValueError:
            continue
        try:
            dirty = paths_dirty(cwd, (rel,))
        except GitError as exc:
            # A review-discipline gate, not a security boundary: fail open, never silently.
            print(
                f"[agent6] WARNING: could not check {rel} for uncommitted changes: {exc}",
                file=sys.stderr,
            )
            continue
        if dirty:
            return (
                f"{piece} has uncommitted changes; `machine run` only accepts a"
                f" committed {label}. Review and commit the bundle first."
            )
    return None


def machine_tool_runner(
    cwd: Path, machine_id: str, clone_root: Path
) -> Callable[[JailPolicy], CommandResult]:
    """Return a jail runner that executes each tool policy in a fresh clone at the chain tip.

    Run states commit to the machine chain and never touch the checkout, so a tool state
    jailed there could not see their work. Tree writes are scratch, discarded with the clone;
    the durable channels are the blackboard and `$AGENT6_MACHINE_DATA_DIR`.

    Args:
        cwd: The repository.
        machine_id: The machine whose chain the clones start from.
        clone_root: Where the per-call clones are made.

    Returns:
        The runner.
    """

    def run(policy: JailPolicy) -> CommandResult:
        """Run the policy in a fresh clone, with the protect paths remapped into it.

        Returns:
            The command's result.

        Raises:
            JailUnavailableError: The clone could not be made.
        """
        dest = clone_root / f"tool-{uuid.uuid4().hex[:12]}"
        try:
            clone_at_machine_chain(cwd, dest, machine_chain_ref_for(machine_id))
        except (SubrunError, GitError) as exc:
            shutil.rmtree(dest, ignore_errors=True)
            raise JailUnavailableError(f"machine tree clone failed: {exc}") from exc
        try:
            return run_in_jail(
                dataclasses.replace(
                    policy,
                    cwd=dest,
                    extra_protect_paths=tuple(
                        dest / p.relative_to(cwd) if p.is_relative_to(cwd) else p
                        for p in policy.extra_protect_paths
                    ),
                )
            )
        finally:
            shutil.rmtree(dest, ignore_errors=True)

    return run


def machine_tool_policy_factory(
    cfg: Config,
    cwd: Path,
    isolation: IsolationLevel,
    *,
    protect_paths: tuple[Path, ...],
    data_dir: Path | None,
) -> ToolPolicyFactory:
    """Return the per-call tool-jail policy builder for a machine.

    The one shared `jail_policy` plus the machine deltas (the bundle's protect paths, the data
    dir's grant and `$AGENT6_MACHINE_DATA_DIR`), so operator grants, `protect_git`, hidden
    paths, env and tool mounts hold in machine tool jails as in run commands.

    Args:
        cfg: The resolved config.
        cwd: The repository.
        isolation: The isolation level.
        protect_paths: The bundle's read-only paths.
        data_dir: The machine's writable scratch, when it has one.

    Returns:
        The builder.
    """
    env_base = passthrough_env()
    extra_rw: tuple[Path, ...] = ()
    if data_dir is not None:
        # extra_rw_paths mount at their real locations on every isolation level.
        env_base["AGENT6_MACHINE_DATA_DIR"] = str(data_dir)
        extra_rw = (data_dir,)

    def build(
        argv: tuple[str, ...], timeout_s: float, network: NetworkMode, pass_env: tuple[str, ...]
    ) -> JailPolicy:
        """Return one call's policy; the pass_env names were allowed at startup."""
        env = {**env_base, **{k: os.environ[k] for k in pass_env if k in os.environ}}
        return jail_policy(
            cwd,
            cfg,
            isolation,
            argv,
            timeout_s=timeout_s,
            network=network,
            extra_rw_paths=extra_rw,
            extra_protect_paths=protect_paths,
            env_base=env,
        )

    return build


def run_machine(  # noqa: C901, PLR0911, PLR0912, PLR0915  # the refusals, one per surface, in order
    path: Path,
    frontend: MachineFrontend,
    *,
    config_path: Path | None = None,
    exit_on_wait: bool = False,
    disable_sandbox: bool = False,
    auto_approve: bool = False,
    no_commands: bool = False,
) -> int:
    """Run a machine to its end, wait or stop.

    Args:
        path: The `.asm.toml`.
        frontend: The reporter and the interactive network fix.
        config_path: An explicit config file, else the effective one.
        exit_on_wait: Return at the first wait state instead of blocking through it.
        disable_sandbox: Run unconfined.
        auto_approve: Approve every run_command in the agent states.
        no_commands: Withhold the command tools from the agent states.

    Returns:
        The exit code: 0 when the machine ended ok, waits or stopped, 1 on a failure, 2 on a
        refusal.
    """
    reporter = frontend.reporter
    # The three flags reach each agent subprocess through the env, which the LLM cannot reach.
    if disable_sandbox:
        os.environ["AGENT6_DANGEROUSLY_DISABLE_SANDBOX"] = "1"
    if auto_approve:
        os.environ["AGENT6_AUTO_APPROVE"] = "1"
    if no_commands:
        os.environ["AGENT6_NO_COMMANDS"] = "1"
    try:
        spec = load_machine(path)
    except MachineError as exc:
        return _fail(reporter, path, list(exc.problems))
    # A security boundary: `load_machine` does not validate the bundle, so run does, like check.
    bundle_problems = validate_bundle(spec, path)
    if bundle_problems:
        return _fail(reporter, path, bundle_problems, "bundle")
    cwd = Path.cwd()
    uncommitted = uncommitted_refusal(path, cwd)
    if uncommitted is not None:
        reporter.refuse(uncommitted)
        return 2
    states = list(spec.states.values())
    has_agent_state = any(getattr(s, "kind", None) == "agent" for s in states)
    # mode="run" agent states edit and commit; they need a resolved git identity.
    has_run_agent = any(isinstance(s, AgentState) and s.mode == "run" for s in states)
    tool_states = [s for s in states if isinstance(s, ToolState)]
    agent_runner: Callable[[AgentRequest, Path | None], AgentExecResult] | None = None
    env = detect_env()
    isolation: IsolationLevel = env.detected_isolation
    # The machine's own file and scripts are read-only in every jail: no state rewrites them.
    protect_paths = machine_protect_paths(path, cwd)
    # Every machine loads the overlay: a pure wait/branch machine still reads snapshot_keep.
    try:
        eff = load_effective_with_overlay(cwd, spec.config, explicit_path=config_path)
        cfg = eff.config
    except ConfigError as exc:
        reporter.error(str(exc))
        return 2
    cfg = cfg.with_sandbox_overrides(auto_approve=auto_approve, no_commands=no_commands)
    if has_run_agent and cfg.sandbox.run_commands == "ask":
        # An unattended machine auto-denies every run_command under 'ask'.
        reporter.note(
            "this machine has mode='run' agent state(s) and"
            " sandbox.run_commands='ask'; an unattended machine auto-denies"
            " run_command. Approve for this invocation with --auto-approve, or"
            " set `agent6 config set --repo sandbox.run_commands yes` to always"
            " allow. Edits and the auto-commit need no approval; verify shares"
            " run_command's gate, so an unattended execution ends unverified."
        )
    snapshot_keep = cfg.machine.snapshot_keep
    # One clone base for the agent states' per-state clones and the tool states' per-call trees.
    clone_root = subordinate_workdir_root(cfg, cwd, f"machine-{spec.machine}")
    agent_states = [s for s in spec.states.values() if isinstance(s, AgentState)]
    if has_agent_state or tool_states:
        try:
            for state in agent_states:
                state_cfg = cfg.with_machine_agent_overrides(
                    provider=state.provider,
                    model=None if state.model == "inherit" else state.model,
                    effort=state.effort,
                    temperature=state.temperature,
                    max_usd=state.max_usd,
                    max_tokens_fallback=state.max_tokens_fallback,
                )
                state_cfg.require_runnable("worker")
        except (ConfigError, ValidationError) as exc:
            reporter.error(str(exc))
            return 2
        try:
            isolation = resolve_isolation_or_refuse(cfg, env, reporter=reporter)
        except SessionRefusedError as refusal:
            return refusal.rc
        # Its fix is an allowlist entry, never a network change: ahead of the network-fix flow.
        if (denied := machine_pass_env_refusal(cfg, spec.states)) is not None:
            reporter.refuse(denied)
            return 2
        refusal = machine_network_refusal(cfg, isolation, tool_states)
        if refusal is None and (hidden := check_hide_paths_support(cfg, isolation, cwd)):
            refusal = NetworkRefusal(hidden)
        if refusal is not None:
            outcome = frontend.resolve_network_fix(
                path, refusal, cfg, isolation, tool_states, cwd, spec.config
            )
            if isinstance(outcome, int):
                return outcome
            cfg, isolation = outcome  # the fix applied and re-validated clear
        cfg_err = config_refusal(cfg, isolation, cwd, explicit_leaves=eff.explicit_leaves)
        if cfg_err is not None:
            reporter.refuse(cfg_err)
            return 2
        if has_agent_state:
            # Every agent state's pin is checked now: a dead route found later wastes the run.
            routes = []
            for state in agent_states:
                state_cfg = cfg.with_machine_agent_overrides(
                    provider=state.provider,
                    model=None if state.model == "inherit" else state.model,
                )
                route = state_cfg.models.resolve("worker")
                if route is not None:
                    routes.append((route.provider, route.model))
            pinned_providers = [provider for provider, _model in routes]
            missing = check_provider_keys(cfg, extra_providers=pinned_providers)
            if missing is not None:
                reporter.err(missing)
                return 2
            # After check_provider_keys, which refreshed the price cache.
            budget_err = budget_preflight(cfg, extra_routes=routes, reporter=reporter)
            if budget_err is not None:
                reporter.refuse(budget_err)
                return 2
            # Resolved on the host: a confined agent cannot read ~/.gitconfig.
            commit_identity: CommitIdentity | None = None
            if has_run_agent:
                base = CommitIdentity(name=cfg.git.commit.name, email=cfg.git.commit.email)
                try:
                    name, email = verify_git_identity(cwd, base)
                except GitError as exc:
                    reporter.error(str(exc))
                    return 2
                commit_identity = CommitIdentity(name=name, email=email)
            root = machines_root(state_dir(cwd)) / spec.machine
            # The complete effective config: the child cannot rediscover the --config layer, and
            # omitting values equal to defaults loses an explicit reset over a non-default global.
            agent_overlay = cfg.model_dump(mode="json")
            agent_overlay.pop("preset", None)  # an overlay cannot select a preset
            agent_runner = build_machine_agent_runner(
                agent_overlay,
                cwd,
                isolation,
                root / "agent_transcripts",
                protect_paths,
                commit_identity,
                # A machine that writes never touches the checkout: each state works a fresh clone.
                machine_id=spec.machine if has_run_agent else None,
                clone_root=clone_root if has_run_agent else None,
            )
    try:
        warn_sandbox_gaps(isolation, env, cfg, root=cwd, reporter=reporter)
    except JailUnavailableError as exc:
        # The hardened exposure scan builds the run's policy, which creates the jail's HOME.
        reporter.refuse(str(exc))
        return 2
    warn_cleartext_credential_endpoints(cfg, reporter=reporter)
    root = machines_root(state_dir(cwd)) / spec.machine
    journal = MachineJournal(root, snapshot_keep=snapshot_keep)
    data_dir = root / "data"
    try:
        with machine_lock(root):
            journal.ensure_dirs()
            # Refused before any worker.pid stamp: a stamped pid would read "started" and then
            # "running" forever in `machine status`.
            events = journal.read()
            if events and isinstance(events[-1], MachineEnd):
                end = events[-1]
                reporter.refuse(
                    f"{spec.machine} already ended in {end.state!r}"
                    f" ({end.status}: {end.reason})."
                    " Replay it with `agent6 machine replay`, or archive the"
                    f" instance directory to start fresh: {journal.root}"
                )
                return 2
            if journal.exists():
                # A live instance runs the bundle it recorded.
                drift = bundle_drift(root, path)
                if drift is not None:
                    reporter.refuse(
                        f"{drift}. A live instance runs the bundle it"
                        " recorded; archive the instance directory to start"
                        f" fresh with the edited machine: {journal.root}"
                    )
                    return 2
            mkdir_for_real_user(data_dir)
            # A leftover stop marker would park this invocation at its first boundary.
            clear_stop_request(root)
            # Before the drive re-runs the state, which starts a fresh log over the crashed one.
            book_crashed_attempt(journal, root)
            apply_spawned_away_default(root, approval_scopes(cfg))
            if not journal.exists():
                # The chain ref outlives an archived instance dir; its tip is that instance's work.
                ref = machine_chain_ref_for(spec.machine)
                branch = machine_branch_for(spec.machine)
                tip = chain_tip(cwd, ref) if has_run_agent else None
                if tip is not None and is_ancestor(cwd, tip, "HEAD"):
                    delete_ref(cwd, ref)
                    reporter.note(
                        f"the previous instance's work on {branch!r} is in HEAD;"
                        " this instance starts from HEAD"
                    )
                elif tip is not None:
                    reporter.refuse(
                        f"no journal for {spec.machine!r}, but its chain branch"
                        f" {branch!r} exists with a previous instance's work."
                        f" Keep it (`git merge {branch}`) or discard it"
                        f" (`git branch -D {branch}` and `git update-ref -d {ref}`),"
                        " then rerun."
                    )
                    return 2
                write_bundle(root, path)
            operator_hook = build_machine_notify_hook(cfg, spec.machine, root)

            def surface_notify(kind: str, state: str, message: str, level: str) -> None:
                """Show a notify here, where the foreground run is its own watcher, then hook."""
                # Presentation never affects control flow: a dead stderr is swallowed.
                if kind == "notify":
                    with contextlib.suppress(OSError):
                        reporter.note(f"notify [{level}] {state!r}: {message}")
                if operator_hook is not None:
                    operator_hook(kind, state, message, level)

            def say_where_it_parked() -> None:
                """Print the wait line; a silent in-process wait reads as a hang."""
                pending = journal.read_pending_wait()
                if pending is not None:
                    reporter.note(wait_line(spec.machine, pending.state, pending.wake_at))

            world = LiveWorld(
                cwd=cwd,
                journal=journal,
                agent_runner=agent_runner,
                tool_policy=machine_tool_policy_factory(
                    cfg, cwd, isolation, protect_paths=protect_paths, data_dir=data_dir
                ),
                # A machine that writes runs its tool states in its own tree,
                # so an edit-then-check loop sees the run states' commits.
                jail_runner=(
                    machine_tool_runner(cwd, spec.machine, clone_root) if has_run_agent else None
                ),
                data_dir=data_dir,
                state_log_root=root / "states",
                state_log_keep=cfg.machine.state_log_keep,
                notify_hook=surface_notify,
                on_wait=say_where_it_parked,
            )
            # Stamped after the last refusal; the finally always clears it.
            write_worker_pid(root, os.getpid())
            try:
                result = drive(spec, journal, world, live=True, exit_on_wait=exit_on_wait)
            finally:
                clear_worker_pid(root)
    except (JournalError, EngineError) as exc:
        reporter.error(str(exc))
        return 1
    if result.status == "waiting":
        reporter.out(
            f"WAITING: {spec.machine} paused in {result.state!r}"
            f" after {_transitions(result.transitions)} ({result.reason})"
        )
    elif result.status == "stopped":
        reporter.out(
            f"STOPPED: {spec.machine} parked in {result.state!r}"
            f" after {_transitions(result.transitions)} ({result.reason});"
            " resume with `agent6 machine run`."
        )
    else:
        spend, _ = machine_spend(journal.read(), root, alive=False)
        reporter.out(
            f"{result.status.upper()}: {spec.machine} ended in {result.state!r}"
            f" after {_transitions(result.transitions)} ({result.reason});"
            f" spent {format_usd(spend.usd, partial=spend.partial)}"
        )
    # A machine with run states commits to its own branch: the ending names it, as a run's does.
    branch = machine_branch_for(spec.machine)
    if has_run_agent and branch_exists(cwd, branch) and not is_ancestor(cwd, branch, "HEAD"):
        reporter.out(f"\nchanges are on {branch}")
        reporter.out(f"  merge with:  git merge {branch}")
        reporter.out(f"  inspect:     git diff HEAD...{branch}")
    return 0 if result.status in ("ok", "waiting", "stopped") else 1
