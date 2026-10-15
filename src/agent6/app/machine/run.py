# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Compose the engine and drive a machine to completion, for `agent6 machine run`.

Resolves one config per state, the isolation, egress, provider-key, budget and git-identity
preflight, builds the per-`agent`-state runner and the `LiveWorld`, and calls `drive`.
"""

from __future__ import annotations

import contextlib
import dataclasses
import os
import pathlib
import shutil
import sys
import uuid
from collections.abc import Callable, Iterator, Mapping

import pydantic

from agent6 import budget, git_ops, kinds, paths
from agent6.app import _setup, _workdirs, confine, machine_agent, preflight, workspace_setup
from agent6.app import frontend as app_frontend
from agent6.app import reporter as app_reporter
from agent6.app.machine import _bundle, _preflight, _spend, _states, _workspace
from agent6.config import Config, ConfigError
from agent6.harness import subrun
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
    ToolExecResult,
    ToolPolicyFactory,
    ToolState,
    bundle_drift,
    clear_stop_request,
    drive,
    load_machine,
    machine_lock,
    write_bundle,
)
from agent6.machine import journal as machine_journal
from agent6.sandbox import detect, jail
from agent6.sessions import ipc, layout
from agent6.tools import _result_format
from agent6.tools import policy as tools_policy
from agent6.viewmodel import machine_state


def _fail(
    reporter: app_reporter.Reporter, path: pathlib.Path, problems: list[str], label: str = ""
) -> int:
    """Return the exit code 1 after printing a FAIL header and the problems."""
    suffix = f" ({label})" if label else ""
    reporter.err(f"FAIL: {path}{suffix}")
    for problem in problems:
        reporter.err(f"  - {problem}")
    return 1


def _transitions(n: int) -> str:
    """Return the transition count with its noun."""
    return f"{n} transition{'' if n == 1 else 's'}"


def uncommitted_refusal(path: pathlib.Path, cwd: pathlib.Path) -> str | None:
    """Return the refusal when the machine's bundle has uncommitted changes, else None.

    A tool or agent executes the bundle as trusted logic, so an uncommitted piece is
    unreviewed; `machine test` is the ungated iteration loop. Skipped outside a git repo and
    for pieces that resolve outside the repo tree.
    """
    if not git_ops.is_git_repo(cwd):
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
            dirty = git_ops.paths_dirty(cwd, (rel,))
        except git_ops.GitError as exc:
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
    cwd: pathlib.Path,
    machine_id: str,
    clone_root: pathlib.Path,
    states: Mapping[str, _states.StateConfig],
    reporter: app_reporter.Reporter,
) -> Callable[[str, kinds.JailPolicy, pathlib.Path | None], kinds.CommandResult]:
    """Return a jail runner that executes each tool policy in a fresh clone at the chain tip.

    Run states commit to the machine chain and never touch the checkout, so a tool state
    jailed there could not see their work. Tree writes are scratch, discarded with the clone;
    the durable channels are the blackboard and `$AGENT6_MACHINE_DATA_DIR`.

    Args:
        cwd: The repository.
        machine_id: The machine whose chain the clones start from.
        clone_root: Where the per-call clones are made.
        states: The resolved config for each state.
        reporter: Receives setup output.

    Returns:
        The runner.
    """

    def run(
        state: str, policy: kinds.JailPolicy, events_log: pathlib.Path | None
    ) -> kinds.CommandResult:
        """Run the policy in a fresh clone, with the protect paths remapped into it.

        Returns:
            The command's result.

        Raises:
            JailUnavailableError: The clone or jail is unavailable.
            EngineError: The caller named an unknown state.
        """
        resolved = states.get(state)
        if resolved is None:
            raise EngineError(f"no config was resolved for state {state!r}")
        dest = clone_root / f"tool-{uuid.uuid4().hex[:12]}"
        try:
            machine_agent.clone_at_machine_chain(
                cwd, dest, git_ops.machine_chain_ref_for(machine_id)
            )
        except (subrun.SubrunError, git_ops.GitError) as exc:
            shutil.rmtree(dest, ignore_errors=True)
            raise jail.JailUnavailableError(f"machine tree clone failed: {exc}") from exc
        try:
            protect = tuple(
                dest / p.relative_to(cwd) if p.is_relative_to(cwd) else p
                for p in policy.extra_protect_paths
            )
            cfg = resolved.config
            policy = dataclasses.replace(policy, cwd=dest, extra_protect_paths=protect)
            # Setup and the tool each get the state's timeout_secs.
            failed = _workspace.tool_setup_failure(cfg, policy, events_log, reporter)
            if failed is not None:
                return failed
            return jail.run_in_jail(policy)
        finally:
            shutil.rmtree(dest, ignore_errors=True)

    return run


def machine_tool_policy_factory(
    states: Mapping[str, _states.StateConfig],
    cwd: pathlib.Path,
    *,
    protect_paths: tuple[pathlib.Path, ...],
    data_dir: pathlib.Path | None,
    bundle_dir: pathlib.Path | None = None,
) -> ToolPolicyFactory:
    """Return the per-call tool-jail policy builder for a machine.

    The one shared `jail_policy` over the state's own config, plus the machine deltas (the
    bundle's protect paths, the data dir's grant and `$AGENT6_MACHINE_DATA_DIR`), so operator
    grants, `protect_git`, hidden paths, env and tool mounts hold in a tool jail as in a run
    command.

    Args:
        states: The resolved config of each state, bounded at startup.
        cwd: The repository.
        protect_paths: The bundle's read-only paths.
        data_dir: The machine's writable scratch, when it has one.
        bundle_dir: The directory holding the machine file, which a command's `scripts/`
            path names; None when it is the repository.

    Returns:
        The builder.
    """
    env_base = _result_format.passthrough_env()
    extra_rw: tuple[pathlib.Path, ...] = ()
    if data_dir is not None:
        # extra_rw_paths mount at their real locations on every isolation level.
        env_base["AGENT6_MACHINE_DATA_DIR"] = str(data_dir)
        extra_rw = (data_dir,)
    # Resolved once per state: every tool jail of the run binds the same directories.
    grants = {name: tools_policy.resolve_grants(sc.config, cwd) for name, sc in states.items()}

    def build(
        state: str, argv: tuple[str, ...], timeout_s: float, pass_env: tuple[str, ...]
    ) -> kinds.JailPolicy:
        """Return one call's policy; the pass_env names were allowed at startup.

        Raises:
            JailUnavailableError: No config was resolved for the state.
        """
        resolved = states.get(state)
        if resolved is None:
            raise jail.JailUnavailableError(f"no config was resolved for state {state!r}")
        cfg, isolation = resolved.config, resolved.isolation
        if bundle_dir is not None:
            argv = _bundle.anchored_argv(argv, bundle_dir, cwd)
        env = {**env_base, **{k: os.environ[k] for k in pass_env if k in os.environ}}
        # A tool's processes die with its state, so its network is its own, never a shared one.
        network = tools_policy.resolve_network(cfg, isolation)
        return tools_policy.jail_policy(
            cwd,
            cfg,
            isolation,
            argv,
            timeout_s=timeout_s,
            network="host" if network == "host" else "none",
            extra_rw_paths=extra_rw,
            extra_protect_paths=protect_paths,
            env_base=env,
            grants=grants[state],
        )

    return build


def _labelled(reporter: app_reporter.Reporter, where: str) -> app_reporter.Reporter:
    """Return a reporter whose stderr lines name the overlay they are about."""
    if not where:
        return reporter

    def err(msg: str) -> None:
        reporter.err(f"{msg} ({where})")

    return dataclasses.replace(reporter, err=err)


def _checked_configs(
    configs: _states.MachineConfigs,
) -> Iterator[tuple[str, _states.StateConfig]]:
    """Yield each distinct config the run uses with the label a refusal names it by."""
    yield "", configs.operator
    yield "", configs.machine
    for name, resolved in configs.states.items():
        if resolved is not configs.machine:
            yield f"[states.{name}.config]: ", resolved


def run_machine(  # noqa: C901, PLR0911, PLR0912, PLR0915  # the refusals, one per surface, in order
    path: pathlib.Path,
    reporter: app_reporter.Reporter,
    *,
    config_path: pathlib.Path | None = None,
    exit_on_wait: bool = False,
    disable_sandbox: bool = False,
    auto_approve: bool = False,
    no_commands: bool = False,
) -> int:
    """Run a machine to its end, wait or stop.

    Args:
        path: The `.asm.toml`.
        reporter: The two output channels.
        config_path: An explicit config file, part of the operator's ceiling.
        exit_on_wait: Return at the first wait state instead of blocking through it.
        disable_sandbox: Run unconfined.
        auto_approve: Approve every run_command in the agent states.
        no_commands: Withhold the command tools from the agent states.

    Returns:
        The exit code: 0 when the machine ended ok, waits or stopped, 1 on a failure, 2 on a
        refusal.
    """
    try:
        spec = load_machine(path)
    except MachineError as exc:
        return _fail(reporter, path, list(exc.problems))
    # A security boundary: `load_machine` does not validate the bundle, so run does, like check.
    bundle_problems = _bundle.validate_bundle(spec, path)
    if bundle_problems:
        return _fail(reporter, path, bundle_problems, "bundle")
    cwd = pathlib.Path.cwd()
    uncommitted = uncommitted_refusal(path, cwd)
    if uncommitted is not None:
        reporter.refuse(uncommitted)
        return 2
    agent_states = {n: s for n, s in spec.states.items() if isinstance(s, AgentState)}
    # mode="run" agent states edit and commit; they need a resolved git identity.
    has_run_agent = any(s.mode == "run" for s in agent_states.values())
    has_tool_state = any(isinstance(s, ToolState) for s in spec.states.values())
    agent_runner: Callable[[AgentRequest, pathlib.Path | None], AgentExecResult] | None = None
    env = _setup.detect_env()
    # The machine's own file and scripts are read-only in every jail: no state rewrites them.
    protect_paths = _preflight.machine_protect_paths(path, cwd, config_path)
    # Every machine resolves its configs: a pure wait/branch machine still reads snapshot_keep.
    # The three flags travel in them, to each agent state as its overlay and resolved isolation.
    try:
        configs = _states.resolve_configs(
            spec,
            cwd,
            env,
            config_path=config_path,
            disable_sandbox=disable_sandbox,
            auto_approve=auto_approve,
            no_commands=no_commands,
        )
    except _states.SandboxWidenedError as exc:
        for problem in exc.problems:
            reporter.refuse(problem)
        return 2
    except detect.IsolationUnavailableError as exc:
        reporter.refuse(str(exc))
        return 2
    except ConfigError as exc:
        reporter.error(str(exc))
        return 2
    cfg = configs.machine.config
    if any(
        state.mode == "run" and configs.states[name].config.sandbox.run_commands == "ask"
        for name, state in agent_states.items()
    ):
        # An unattended machine auto-denies every run_command under 'ask'.
        always = (
            f'set sandbox.run_commands = "yes" in {config_path}'
            if config_path is not None
            else "set `agent6 config set --repo sandbox.run_commands yes`"
        )
        reporter.note(
            "this machine has mode='run' agent state(s) and"
            " sandbox.run_commands='ask'; an unattended machine auto-denies"
            f" run_command. Approve for this invocation with --auto-approve, or {always}"
            " to always allow. Edits and the auto-commit need no approval; verify shares"
            " run_command's gate, so an unattended execution ends unverified."
        )
    root = layout.machines_root(paths.state_dir(cwd)) / spec.machine
    base_commit = None
    # has_run_agent supplies both clone arguments: every tool and agent state then runs in a clone.
    if has_run_agent:
        try:
            events = MachineJournal(root).read()
            if events:
                begin = events[0]
                if not isinstance(begin, machine_journal.MachineBegin) or not begin.base_commit:
                    reporter.refuse(
                        "this machine's journal has no starting commit.\n"
                        + machine_journal.fresh_start_remedy(root)
                    )
                    return 2
                base_commit = begin.base_commit
            else:
                base_commit = git_ops.status(cwd).head_sha
        except (git_ops.GitError, JournalError) as exc:
            reporter.refuse(f"cannot resolve the machine's starting commit: {exc}")
            return 2
        if not base_commit:
            reporter.refuse("machine clones need a committed starting point")
            return 2
        resolved_states = dict(configs.states)
        inferred_setup: tuple[str, ...] = ()
        if any(
            not configs.states[name].config.parallel.setup_command
            for name, state in spec.states.items()
            if isinstance(state, (AgentState, ToolState))
        ):
            inferred_setup = workspace_setup.infer_if_unset(
                cfg.with_setup_command(()), cwd, reporter=reporter, at=base_commit
            ).parallel.setup_command
        for name, state in spec.states.items():
            if not isinstance(state, (AgentState, ToolState)):
                continue
            resolved = resolved_states[name]
            state_cfg = resolved.config
            if not state_cfg.parallel.setup_command:
                state_cfg = state_cfg.with_setup_command(inferred_setup)
            if state_cfg.parallel.setup_command and state_cfg.sandbox.run_commands == "ask":
                reporter.refuse(
                    f"state {name!r} needs workspace setup, but sandbox.run_commands='ask'; "
                    "use --auto-approve or set sandbox.run_commands='yes' to allow setup"
                )
                return 2
            resolved_states[name] = dataclasses.replace(resolved, config=state_cfg)
        configs = dataclasses.replace(configs, states=resolved_states)
    snapshot_keep = cfg.machine.snapshot_keep
    # One clone base for the agent states' per-state clones and the tool states' per-call trees.
    clone_root = _workdirs.subordinate_workdir_root(cfg, cwd, f"machine-{spec.machine}")

    def agent_config(name: str, state: AgentState) -> Config:
        """Return an agent state's config with its spend cap applied."""
        return configs.states[name].config.with_machine_agent_overrides(max_usd=state.max_usd)

    if agent_states or has_tool_state:
        try:
            for name, state in agent_states.items():
                agent_config(name, state).require_runnable("worker")
        except (ConfigError, pydantic.ValidationError) as exc:
            reporter.error(str(exc))
            return 2
        # The allowlist is the operator's own: no overlay is merged into it.
        denied = _preflight.machine_pass_env_refusal(configs.operator.config, spec.states)
        if denied is not None:
            reporter.refuse(denied)
            return 2
        # The operator's own config refuses here as it would for a plain run; then each state's.
        for label, resolved in _checked_configs(configs):
            refusal = confine.check_network_support(
                resolved.config, resolved.isolation
            ) or confine.config_refusal(
                resolved.config, resolved.isolation, cwd, explicit_leaves=resolved.explicit_leaves
            )
            if refusal is not None:
                reporter.refuse(f"{label}{refusal}")
                return 2
        if agent_states:
            # A state can inherit roles besides its worker; the ceiling itself makes no calls.
            for name, state in agent_states.items():
                state_cfg = agent_config(name, state)
                missing = _setup.check_provider_keys(state_cfg)
                if missing is not None:
                    reporter.err(missing)
                    return 2
                routes = {
                    (route.provider, route.model)
                    for route in state_cfg.models.configured().values()
                }
                # The state's own budget: its config may set a cap the machine's does not. After
                # check_provider_keys, which refreshed the price cache.
                budget_err = preflight.budget_preflight(state_cfg, routes=routes, reporter=reporter)
                if budget_err is not None:
                    reporter.refuse(f"{name}: {budget_err}")
                    return 2
            # Resolved on the host: a confined agent cannot read ~/.gitconfig.
            commit_identity: git_ops.CommitIdentity | None = None
            if has_run_agent:
                base = git_ops.CommitIdentity(name=cfg.git.commit.name, email=cfg.git.commit.email)
                try:
                    name, email = git_ops.verify_git_identity(cwd, base)
                except git_ops.GitError as exc:
                    reporter.error(str(exc))
                    return 2
                commit_identity = git_ops.CommitIdentity(name=name, email=email)
            agent_runner = machine_agent.build_machine_agent_runner(
                {
                    state_name: machine_agent.state_overlay(
                        configs.states[state_name].config, configs.states[state_name].isolation
                    )
                    for state_name in agent_states
                },
                cwd,
                root / "agent_transcripts",
                protect_paths,
                commit_identity,
                # A machine that writes never touches the checkout: each state works a fresh clone.
                machine_id=spec.machine if has_run_agent else None,
                clone_root=clone_root if has_run_agent else None,
            )
    try:
        for label, resolved in _checked_configs(configs):
            if resolved is configs.operator:
                continue  # the ceiling runs nothing itself
            confine.warn_sandbox_gaps(
                resolved.isolation,
                env,
                resolved.config,
                root=cwd,
                reporter=_labelled(reporter, label.removesuffix(": ")),
            )
    except jail.JailUnavailableError as exc:
        # The hardened exposure scan builds the run's policy, which creates the jail's HOME.
        reporter.refuse(str(exc))
        return 2
    confine.warn_cleartext_credential_endpoints(cfg, reporter=reporter)
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
                    f" Replay it with `agent6 machine replay {spec.machine}`."
                    f" {machine_journal.fresh_start_remedy(journal.root)}"
                )
                return 2
            if journal.exists():
                # A live instance runs the bundle it recorded.
                drift = bundle_drift(root, path)
                if drift is not None:
                    reporter.refuse(
                        f"{drift}. A live instance runs the bundle it recorded."
                        f" {machine_journal.fresh_start_remedy(journal.root)}"
                    )
                    return 2
            paths.mkdir_for_real_user(data_dir)
            # A leftover stop marker would park this invocation at its first boundary.
            clear_stop_request(root)
            # Before the drive re-runs the state, which starts a fresh log over the crashed one.
            _spend.book_crashed_attempt(journal, root)
            app_frontend.apply_spawned_away_default(root, app_frontend.approval_scopes(cfg))
            if not journal.exists():
                # The chain ref outlives an archived instance dir; its tip is that instance's work.
                ref = git_ops.machine_chain_ref_for(spec.machine)
                branch = git_ops.machine_branch_for(spec.machine)
                tip = git_ops.chain_tip(cwd, ref) if has_run_agent else None
                if tip is not None and git_ops.is_ancestor(cwd, tip, "HEAD"):
                    git_ops.delete_ref(cwd, ref)
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
            operator_hook = _preflight.build_machine_notify_hook(cfg, spec.machine, root)

            def surface_notify(kind: str, state: str, message: str, level: str) -> None:
                """Show a notify here, where the foreground run is its own watcher, then hook."""
                # Presentation never affects control flow: a dead stderr is swallowed.
                if kind == "notify":
                    with contextlib.suppress(OSError):
                        reporter.note(f"notify [{level}] {state!r}: {message}")
                if operator_hook is not None:
                    operator_hook(kind, state, message, level)

            poke_hint_shown = False

            def say_where_it_parked() -> None:
                """Print the wait line; a silent in-process wait reads as a hang."""
                nonlocal poke_hint_shown
                pending = journal.read_pending_wait()
                if pending is not None:
                    # This process is the worker, so the wait is live.
                    reporter.note(
                        machine_state.wait_line(
                            spec.machine,
                            pending.state,
                            pending.wake_at,
                            alive=True,
                            include_poke=not poke_hint_shown,
                        )
                    )
                    poke_hint_shown = True

            def say_tool_failure(state: str, outcome: ToolExecResult) -> None:
                """Name a failed tool step here: a bare `nonzero` edge reads as progress."""
                detail = machine_state.tool_failure(
                    outcome.exit_code, outcome.timed_out, outcome.stderr or outcome.stdout
                )
                reporter.note(f"state {state!r}: {detail}")

            world = LiveWorld(
                cwd=cwd,
                journal=journal,
                agent_runner=agent_runner,
                tool_policy=machine_tool_policy_factory(
                    configs.states,
                    cwd,
                    protect_paths=protect_paths,
                    data_dir=data_dir,
                    bundle_dir=path.parent,
                ),
                # A machine that writes runs its tool states in its own tree,
                # so an edit-then-check loop sees the run states' commits.
                jail_runner=(
                    machine_tool_runner(cwd, spec.machine, clone_root, configs.states, reporter)
                    if has_run_agent
                    else None
                ),
                data_dir=data_dir,
                state_log_root=root / "states",
                state_log_keep=cfg.machine.state_log_keep,
                notify_hook=surface_notify,
                on_wait=say_where_it_parked,
                on_tool_failure=say_tool_failure,
            )
            # Stamped after the last refusal; the finally always clears it.
            ipc.write_worker_pid(root, os.getpid())
            try:
                result = drive(
                    spec,
                    journal,
                    world,
                    live=True,
                    exit_on_wait=exit_on_wait,
                    base_commit=base_commit,
                )
            finally:
                ipc.clear_worker_pid(root)
    except (JournalError, EngineError) as exc:
        reporter.error(str(exc))
        return 1
    spend, _ = machine_state.machine_spend(journal.read(), root, alive=False)
    spend_line = (
        f"spent {budget.format_usd(spend.usd, partial=spend.partial)}"
        f" (in={spend.input_tokens} tok, out={spend.output_tokens} tok)"
    )
    if result.status == "waiting":
        reporter.out(
            f"WAITING: {spec.machine} paused in {result.state!r}"
            f" after {_transitions(result.transitions)} ({result.reason}); {spend_line}"
        )
    elif result.status == "stopped":
        reporter.out(
            f"STOPPED: {spec.machine} parked in {result.state!r}"
            f" after {_transitions(result.transitions)} ({result.reason});"
            f" resume with `agent6 machine run {spec.machine}`. {spend_line}"
        )
    else:
        reporter.out(
            f"{result.status.upper()}: {spec.machine} ended in {result.state!r}"
            f" after {_transitions(result.transitions)} ({result.reason});"
            f" {spend_line}"
        )
    # A machine with run states commits to its own branch: the ending names it, as a run's does.
    branch = git_ops.machine_branch_for(spec.machine)
    if (
        has_run_agent
        and git_ops.branch_exists(cwd, branch)
        and not git_ops.is_ancestor(cwd, branch, "HEAD")
    ):
        reporter.out(f"\nchanges are on {branch}")
        reporter.out(f"  merge with:  git merge {branch}")
        reporter.out(f"  inspect:     git diff HEAD...{branch}")
    return 0 if result.status in ("ok", "waiting", "stopped") else 1
