# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Build the isolation, providers, dispatcher and tools a run and a resume share.

The lifecycles keep their own workspace steps and Harness wiring.
"""

from __future__ import annotations

import dataclasses
import functools
import pathlib
import shutil
from collections.abc import Callable

import agent6
from agent6 import budget as agent6_budget
from agent6 import kinds
from agent6.app import (
    confine,
    conversation_state,
    environment_lease,
    frontend,
    preflight,
    review_actions,
    role_providers,
    session_setup,
    stamps,
)
from agent6.app import reporter as app_reporter
from agent6.config import ClaudeCodeProviderEntry, Config, RoleModel, RoleName
from agent6.graph import curator as graph_curator
from agent6.harness import compaction_elision, review_gate
from agent6.providers import Provider, TranscriptSink
from agent6.sandbox import container, detect, jail, jail_session
from agent6.sessions import event_log
from agent6.sessions import layout as sessions_layout
from agent6.tools import command_tools, dispatch, mcp_client, operator_prompts, target_policy
from agent6.tools import policy as tools_policy


def resolve_isolation_or_refuse(
    cfg: Config, env: detect.Environment, *, reporter: app_reporter.Reporter
) -> kinds.IsolationLevel:
    """Return the isolation level the config resolves to on this host.

    An `auto` degrades inside `resolve_isolation`.

    Args:
        cfg: The resolved config.
        env: The detected host environment.
        reporter: Where the refusal goes.

    Returns:
        The isolation level.

    Raises:
        SessionRefusedError: An explicit level is unavailable on this host.
    """
    try:
        return detect.resolve_isolation(cfg.sandbox.isolation, env)
    except detect.IsolationUnavailableError as exc:
        reporter.refuse(str(exc))
        raise preflight.SessionRefusedError(2) from exc


def tool_result_cap_bytes(cfg: Config, role: RoleName) -> int:
    """Return the byte bound on one tool result entering the conversation of the role's provider.

    Args:
        cfg: The resolved config.
        role: The role that drives the session.

    Returns:
        The loop's default; for a role on Claude Code, a bound under its persist threshold.
    """
    rm = cfg.models.resolve(role)
    entry = cfg.providers.get(rm.provider) if rm is not None else None
    if isinstance(entry, ClaudeCodeProviderEntry):
        return compaction_elision.CLAUDE_CODE_RESULT_CAP_BYTES
    return compaction_elision.TOOL_RESULT_CAP_BYTES


def select_isolation(
    cfg: Config,
    *,
    cwd: pathlib.Path,
    confirm_unconfined: Callable[[kinds.IsolationLevel, Config], bool],
    reporter: app_reporter.Reporter,
    worktree_git_dir: pathlib.Path | None = None,
) -> kinds.IsolationLevel:
    """Pick the sandbox isolation and refuse what it cannot honor.

    Confirms an unconfined autorun; refuses a network mode, egress or budget the isolation
    cannot honor, or a workspace no tool could read.

    Args:
        cfg: The resolved config.
        cwd: The workspace.
        confirm_unconfined: Confirms an unconfined autorun with the operator.
        reporter: Where the refusals go.
        worktree_git_dir: The linked worktree's git dir, when the run is in one.

    Returns:
        The isolation level.

    Raises:
        SessionRefusedError: The host, the config or the operator refused the run.
    """
    try:
        env = detect.probed_environment()
    except jail.JailUnavailableError as exc:
        # The jail binary cannot run, so no jailed command can either.
        reporter.refuse(str(exc))
        raise preflight.SessionRefusedError(2) from exc
    selected = resolve_isolation_or_refuse(cfg, env, reporter=reporter)
    cfg_err = confine.startup_refusal(cfg, selected, cwd, worktree_git_dir=worktree_git_dir)
    if cfg_err is not None:
        reporter.refuse(cfg_err)
        raise preflight.SessionRefusedError(2)
    try:
        confine.warn_sandbox_gaps(
            selected, env, cfg, root=cwd, worktree_git_dir=worktree_git_dir, reporter=reporter
        )
    except jail.JailUnavailableError as exc:
        # The exposure scan builds the jail policy, which creates the jail's HOME.
        reporter.refuse(str(exc))
        raise preflight.SessionRefusedError(2) from exc
    confine.warn_cleartext_credential_endpoints(cfg, reporter=reporter)
    # A container is the boundary the operator chose; the unconfined confirm is for `none`.
    if cfg.sandbox.isolation != "container" and not confirm_unconfined(selected, cfg):
        reporter.note("aborted.")
        raise preflight.SessionRefusedError(2)
    budget_err = preflight.budget_preflight(cfg, reporter=reporter)
    if budget_err is not None:
        reporter.refuse(budget_err)
        raise preflight.SessionRefusedError(2)
    return selected


def note_state_disk_space(state_dir: pathlib.Path, *, reporter: app_reporter.Reporter) -> None:
    """Report low free space on the state filesystem without refusing the execution.

    Args:
        state_dir: The existing state directory whose filesystem holds the session.
        reporter: Receives the note below ten percent free, or an inspection error.
    """
    try:
        usage = shutil.disk_usage(state_dir)
    except OSError as exc:
        reporter.note(f"could not inspect free space at {state_dir}: {exc}")
        return
    if usage.free * 10 < usage.total:
        percent = 100 * usage.free / usage.total
        free_space = (
            f"{usage.free / 1024**3:.1f} GiB"
            if usage.free >= 1024**3
            else f"{usage.free / 1024**2:.0f} MiB"
        )
        reporter.note(f"low disk space: {free_space} free ({percent:.1f}%) at {state_dir}")


def install_inside_workspace(cwd: pathlib.Path) -> pathlib.Path | None:
    """Return agent6's install root when it sits inside the workspace, else None.

    An in-tree install is inside the jail's writable workspace, so a jailed command can
    rewrite the running agent.

    Args:
        cwd: The workspace.

    Returns:
        The resolved package directory, or None when it is outside the workspace.
    """
    root = pathlib.Path(agent6.__file__).resolve().parent
    return root if root.is_relative_to(cwd.resolve()) else None


def warn_install_inside_workspace(cwd: pathlib.Path, *, reporter: app_reporter.Reporter) -> None:
    """Warn when agent6 is installed inside the workspace; agent6 developing agent6 is that."""
    if (root := install_inside_workspace(cwd)) is not None:
        reporter.warn(
            f"agent6 is installed inside this workspace ({root});"
            " a jailed command can rewrite the running agent. Install it outside"
            " the project (pipx / uv tool)."
        )


@dataclasses.dataclass(frozen=True, slots=True)
class SessionProviders:
    """Hold the run's providers, all metering into one tracker.

    Attributes:
        budget: The run's tracker.
        rm_role: The driving role's resolved model.
        provider: The driving role's instrumented provider.
        summariser_provider: The summariser seat, when one is configured.
        review_seats: The in-loop review panel.
    """

    budget: agent6_budget.BudgetTracker
    rm_role: RoleModel
    provider: Provider
    summariser_provider: Provider | None
    review_seats: list[review_gate.ReviewSeat]

    def close(self) -> None:
        """Release every provider's held process (a `claude_code` session)."""
        role_providers.close_provider(self.provider)
        if self.summariser_provider is not None:
            role_providers.close_provider(self.summariser_provider)
        for seat in self.review_seats:
            role_providers.close_provider(seat.provider)


def build_session_providers(
    cfg: Config,
    *,
    role: RoleName,
    events: event_log.EventSink,
    transcript_sink: TranscriptSink,
    stream_text: bool,
    reporter: app_reporter.Reporter = app_reporter.STDIO_REPORTER,
    conversations: conversation_state.Conversations | None = None,
) -> SessionProviders:
    """Build the driving role's provider and the summariser and review seats.

    Args:
        cfg: The resolved config.
        role: The driving role.
        events: The run's event sink.
        transcript_sink: Where every provider call is transcribed.
        stream_text: Whether the provider streams text into the sink.
        reporter: Where the prompt-override warning goes.
        conversations: The session's conversation state; None holds it for this process.

    Returns:
        The providers, metering into one tracker.
    """
    conversations = conversations or conversation_state.Conversations()
    budget = session_setup.budget_tracker(cfg)
    inner = role_providers.build_role_provider(
        cfg, role, transcript_sink=transcript_sink, budget=budget, events=events
    )
    rm_role = cfg.models.resolve(role)
    assert rm_role is not None  # require_runnable validated this
    preflight.warn_if_prompt_override_incomplete(cfg, reporter=reporter)
    provider: Provider = role_providers.InstrumentedProvider(
        inner=inner,
        role=role,
        model=rm_role.model,
        provider_name=rm_role.provider,
        events=events,
        budget=budget,
        stream_text=stream_text,
        conversations=conversations,
    )
    summariser_provider = role_providers.helper_seat_provider(
        cfg,
        "summariser",
        transcript_sink=transcript_sink,
        budget=budget,
        events=events,
        conversations=conversations,
    )
    # The panel is the in-loop review: a trigger with no seats builds one seat on the reviewer.
    review_seats = (
        role_providers.build_review_seats(
            cfg,
            transcript_sink=transcript_sink,
            budget=budget,
            events=events,
            conversations=conversations,
        )
        if cfg.review.trigger != "off"
        else []
    )
    return SessionProviders(
        budget=budget,
        rm_role=rm_role,
        provider=provider,
        summariser_provider=summariser_provider,
        review_seats=review_seats,
    )


@dataclasses.dataclass(frozen=True, slots=True)
class SessionTools:
    """Hold the curator and dispatcher pair and the model-derived loop knobs.

    Attributes:
        curator: The run's DAG curator.
        dispatcher: The run's tool dispatcher.
        compact_drop_at_chars: The compactor's drop threshold.
        compact_summarise_at_chars: The compactor's summarise threshold.
        keep_recent_chars: How much recent context the compactor keeps.
        cfg: The decompose-resolved config the Harness is built with.
    """

    curator: graph_curator.GraphCurator
    dispatcher: dispatch.ToolDispatcher
    compact_drop_at_chars: int
    compact_summarise_at_chars: int
    keep_recent_chars: int
    cfg: Config


def prepare_local_execution(
    cfg: Config,
    root: pathlib.Path,
    isolation: kinds.IsolationLevel,
    *,
    name: str = "run",
    session_net: jail.SessionNetwork | None = None,
    worktree_git_dir: pathlib.Path | None = None,
    extra_protect_paths: tuple[pathlib.Path, ...] = (),
    shared_writes: tuple[pathlib.Path, ...] = (),
) -> command_tools.JailAccess:
    """Resolve local permissions and prepare a session before any workspace command.

    The returned access owns its temporary directory and any network it creates.
    Preparation failure releases both and propagates, never choosing another transport.

    Args:
        cfg: The resolved config.
        root: The host workspace.
        isolation: The resolved local isolation level.
        name: The session's temporary-directory label.
        session_net: An existing network shared with MCP servers; None creates one if needed.
        worktree_git_dir: The recorded repository git directory.
        extra_protect_paths: Additional read-only command paths.
        shared_writes: Other machine states' writable regions for grant resolution.

    Returns:
        The policy, prepared session and owned resources.
    """
    grants = tools_policy.resolve_grants(cfg, root, shared_writes=shared_writes)
    temp = tools_policy.command_temp(isolation, name)
    own_net = False
    try:
        policy = tools_policy.jail_policy(
            root.resolve(),
            cfg,
            isolation,
            (),
            grants=grants,
            temp_dir=pathlib.Path(temp.name) if temp is not None else None,
            worktree_git_dir=worktree_git_dir,
            extra_protect_paths=extra_protect_paths,
        )
        if policy.network == "session" and session_net is None:
            session_net = jail.SessionNetwork.open()
            own_net = True
        session = jail_session.JailSession.open(
            policy,
            session_net=session_net,
            controls=target_policy.effective_controls(cfg, isolation),
        )
    except BaseException:
        if temp is not None:
            temp.cleanup()
        if own_net and session_net is not None:
            session_net.close()
        raise
    return command_tools.JailAccess(
        policy=policy,
        session=session,
        grants=grants,
        temp=temp,
        network=session_net,
        owns_network=own_net,
    )


def container_options(cfg: Config) -> container.ContainerOptions:
    """Return the `[container]` table as the sandbox's typed selection.

    Args:
        cfg: The resolved config.

    Returns:
        The options; an empty `workspace` means none.
    """
    return container.ContainerOptions(
        exec=cfg.container.exec,
        runtime=cfg.container.runtime,
        workspace=pathlib.Path(cfg.container.workspace) if cfg.container.workspace else None,
        start=cfg.container.start,
        stop=cfg.container.stop,
    )


def mounted_root(cwd: pathlib.Path, worktree_git_dir: pathlib.Path | None) -> pathlib.Path:
    """Return the workspace a dev container mounts: the repository of a linked worktree, else cwd.

    Args:
        cwd: The checkout commands run in.
        worktree_git_dir: The repository's `.git` recorded for a linked worktree, or None.

    Returns:
        The host root discovery looks up by label.
    """
    if worktree_git_dir is not None and worktree_git_dir.name == ".git":
        return worktree_git_dir.parent
    return cwd


@dataclasses.dataclass(frozen=True, slots=True)
class ContainerReport:
    """What `check sandbox` says about the container a run would select, inspected only.

    Attributes:
        detail: The container, user and workspace path, or the opaque prefix.
        exposures: Host paths and sockets the container mounts from outside the workspace.
    """

    detail: str
    exposures: tuple[str, ...]


def describe_container(cfg: Config, root: pathlib.Path) -> ContainerReport:
    """Inspect the container the config selects, starting and stopping nothing.

    Args:
        cfg: The resolved config.
        root: The host workspace.

    Returns:
        The report.

    Raises:
        jail.JailUnavailableError: Discovery refused; the message names the fix.
    """
    found = container.discover(root, container_options(cfg))
    target = found.target
    detail = (
        f"container {target.container_id or 'behind the exec prefix'}; user"
        f" {target.user or 'unknown'}; workspace {target.command_root}"
    )
    return ContainerReport(detail, container.exposures(found))


def prepare_container_execution(
    cfg: Config,
    root: pathlib.Path,
    options: container.ContainerOptions,
    *,
    state_dir: pathlib.Path,
    session_id: str,
    execution_id: str,
    checkout: pathlib.Path | None = None,
    worktree_git_dir: pathlib.Path | None = None,
    controls: tuple[kinds.EffectiveControl, ...] = (),
    reporter: app_reporter.Reporter = app_reporter.STDIO_REPORTER,
) -> command_tools.JailAccess:
    """Acquire the container, check the checkout and the hidden paths, and prepare the transport.

    `sandbox.isolation = "container"` selects this path. A failure after the acquisition
    releases the lease, which stops a container this run started.

    Args:
        cfg: The resolved config.
        root: The host workspace the container mounts; a lane's origin, not its clone.
        options: The operator's container selection, start and stop commands included.
        state_dir: The repository's state directory, where the lease lives; a lane's origin's.
        session_id: The acquiring session.
        execution_id: Its current execution.
        checkout: The checkout commands run in; None means the workspace itself.
        worktree_git_dir: The checkout's repository git dir, when it is a linked worktree.
        controls: Effective controls the caller resolved.
        reporter: Receives the ownership line and what the container exposes.

    Returns:
        The prepared access, whose lifecycle hook releases the lease at close.

    Raises:
        container.ContainerError: Discovery, start, the mapping check, a hidden path a mount
            exposes, or preparation refused.
        environment_lease.LeaseError: The lease record is malformed.
        OSError: The host side of the mapping proof failed.
    """
    acquired = environment_lease.acquire(
        state_dir, root, options, session_id=session_id, execution_id=execution_id
    )
    reporter.note(f"container: {acquired.reason}")
    for line in container.exposures(acquired.discovery):
        reporter.warn(f"the container exposes a {line}")
    settle = functools.partial(
        environment_lease.release,
        state_dir,
        acquired.ref,
        session_id=session_id,
        host_root=root,
        options=options,
    )
    try:
        container.check_checkout(acquired.discovery, checkout or root, worktree_git_dir)
        container.check_hidden_paths(acquired.discovery, cfg.sandbox.hide_paths)
        discovery = dataclasses.replace(
            acquired.discovery,
            target=dataclasses.replace(acquired.discovery.target, lifecycle_ref=acquired.ref),
        )
        session = container.prepare(
            discovery, memory_limit_mib=cfg.sandbox.memory_limit_mib, controls=controls
        )
    except (container.ContainerError, OSError):
        settle()
        raise
    return command_tools.JailAccess(
        policy=kinds.JailPolicy(
            cwd=discovery.target.command_root,
            argv=(),
            isolation="none",
            network="host",
            memory_limit_mib=cfg.sandbox.memory_limit_mib,
        ),
        session=session,
        grants=tools_policy.resolve_grants(cfg, root),
        lifecycle=settle,
    )


def build_session_tools(
    cfg: Config,
    *,
    cwd: pathlib.Path,
    state_dir: pathlib.Path,
    layout: sessions_layout.SessionLayout,
    isolation: kinds.IsolationLevel,
    mode: kinds.ResumableMode,
    events: event_log.EventSink,
    prompts: operator_prompts.OperatorPrompts,
    loop_log: Callable[[str], None],
    mcp_manager: mcp_client.MCPManager | None,
    rm_role: RoleModel,
    command_access: command_tools.JailAccess | None = None,
    execution_id: str | None = None,
    session_net: jail.SessionNetwork | None = None,
    worktree_git_dir: pathlib.Path | None = None,
) -> SessionTools:
    """Build the curator, the dispatcher and the model-derived loop knobs.

    Args:
        cfg: The resolved config.
        cwd: The workspace.
        state_dir: The repo's state directory.
        layout: The session's directory layout.
        isolation: The isolation level.
        command_access: An already-prepared session; None prepares the local target here.
        execution_id: The lifecycle's execution identity, or None to allocate one.
        mode: The run mode.
        events: The run's event sink.
        prompts: The operator prompt gate.
        loop_log: The loop's logger.
        mcp_manager: The MCP servers, when enabled.
        rm_role: The driving role's resolved model.
        session_net: The session's network namespace, when one exists.
        worktree_git_dir: The linked worktree's git dir, when the run is in one.

    Returns:
        The tools and the decompose-resolved config.
    """
    # The curator runs in-process: the run's worker.lock already makes this the sole writer.
    curator = graph_curator.GraphCurator(layout)
    compact_drop, compact_summarise, keep_recent = role_providers.resolve_compaction_thresholds(
        cfg, rm_role, log=loop_log
    )
    cfg = role_providers.resolve_decompose(cfg, rm_role, log=loop_log)
    access = command_access or prepare_local_execution(
        cfg,
        cwd,
        isolation,
        name=layout.session_id,
        session_net=session_net,
        worktree_git_dir=worktree_git_dir,
    )
    stamps.track_execution_targets(layout.session_dir, events, note=loop_log)
    try:
        dispatcher = dispatch.ToolDispatcher(
            root=cwd,
            config=cfg,
            isolation=isolation,
            prompts=prompts,
            events=events,
            curator=curator,
            run_root_node_id=None,  # Harness seeds the root + calls set_run_root_node_id
            mcp_manager=mcp_manager,
            mode=mode,
            state_dir=state_dir,
            session_dir=layout.session_dir,
            findings=functools.partial(review_actions.findings_result, layout.session_dir),
            command_access=access,
            execution_id=execution_id,
        )
    except BaseException:
        access.close()
        raise
    return SessionTools(
        curator=curator,
        dispatcher=dispatcher,
        compact_drop_at_chars=compact_drop,
        compact_summarise_at_chars=compact_summarise,
        keep_recent_chars=keep_recent,
        cfg=cfg,
    )


def session_facts_provider(
    budget: agent6_budget.BudgetTracker,
    model: str,
    run_commands: str,
    isolation: str,
    network: str,
) -> Callable[[], frontend.SessionFacts]:
    """Return the live-facts thunk the pause banner reads; spend reads live, the rest binds."""

    def facts() -> frontend.SessionFacts:
        spend, partial = budget.estimate_usd()
        return frontend.SessionFacts(
            spend_usd=spend,
            spend_partial=partial,
            model=model,
            run_commands=run_commands,
            isolation=isolation,
            network=network,
        )

    return facts
