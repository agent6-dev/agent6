# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Assemble the session pieces `run_task` and `resume_task` share.

The isolation preflight and the provider, dispatcher and tools build. The lifecycles keep
their own workspace steps and their Harness wiring.
"""

from __future__ import annotations

import dataclasses
import pathlib
from collections.abc import Callable

import agent6
from agent6 import budget as agent6_budget
from agent6 import event_log, kinds
from agent6.app import _setup, confine, frontend, preflight, providers
from agent6.app import reporter as app_reporter
from agent6.config import ClaudeCodeProviderEntry, Config, RoleModel, RoleName
from agent6.graph import curator as graph_curator
from agent6.harness import _compaction, _reviewer
from agent6.providers import Provider, TranscriptSink
from agent6.sandbox import detect, jail
from agent6.sessions import layout as sessions_layout
from agent6.tools import dispatch, mcp_client, operator_prompts


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

    Claude Code persists a result above its threshold and serves a preview, so its bound is
    tighter than the loop's default.
    """
    rm = cfg.models.resolve(role)
    entry = cfg.providers.get(rm.provider) if rm is not None else None
    if isinstance(entry, ClaudeCodeProviderEntry):
        return _compaction.CLAUDE_CODE_RESULT_CAP_BYTES
    return _compaction.TOOL_RESULT_CAP_BYTES


def select_isolation(
    cfg: Config,
    *,
    cwd: pathlib.Path,
    confirm_unconfined: Callable[[kinds.IsolationLevel, Config], bool],
    reporter: app_reporter.Reporter,
    explicit_leaves: frozenset[str] = frozenset(),
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
        explicit_leaves: The config leaves the operator wrote, which refuse rather than degrade.
        worktree_git_dir: The linked worktree's git dir, when the run is in one.

    Returns:
        The isolation level.

    Raises:
        SessionRefusedError: The host, the config or the operator refused the run.
    """
    try:
        env = _setup.detect_env()
    except jail.JailUnavailableError as exc:
        # The strict probe could not run the jail binary; no command will run under it either.
        reporter.refuse(str(exc))
        raise preflight.SessionRefusedError(2) from exc
    selected = resolve_isolation_or_refuse(cfg, env, reporter=reporter)
    try:
        confine.warn_sandbox_gaps(
            selected, env, cfg, root=cwd, worktree_git_dir=worktree_git_dir, reporter=reporter
        )
    except jail.JailUnavailableError as exc:
        # The hardened exposure scan builds the run's policy, which creates the jail's HOME.
        reporter.refuse(str(exc))
        raise preflight.SessionRefusedError(2) from exc
    confine.warn_cleartext_credential_endpoints(cfg, reporter=reporter)
    if not confirm_unconfined(selected, cfg):
        reporter.note("aborted.")
        raise preflight.SessionRefusedError(2)
    net_err = confine.check_network_support(cfg, selected)
    if net_err is not None:
        reporter.refuse(net_err)
        raise preflight.SessionRefusedError(2)
    # A default this host cannot honour degraded with a warning above; a written value refuses.
    cfg_err = confine.config_refusal(
        cfg, selected, cwd, explicit_leaves=explicit_leaves, worktree_git_dir=worktree_git_dir
    )
    if cfg_err is not None:
        reporter.refuse(cfg_err)
        raise preflight.SessionRefusedError(2)
    budget_err = preflight.budget_preflight(cfg, reporter=reporter)
    if budget_err is not None:
        reporter.refuse(budget_err)
        raise preflight.SessionRefusedError(2)
    return selected


def install_inside_workspace(cwd: pathlib.Path) -> pathlib.Path | None:
    """Return agent6's install root when it sits inside the workspace, else None.

    An in-tree install is inside the jail's writable workspace, so a jailed command can
    rewrite the running agent.
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
    review_seats: list[_reviewer.ReviewSeat]

    def close(self) -> None:
        """Release every provider's held process (a `claude_code` session)."""
        providers.close_provider(self.provider)
        if self.summariser_provider is not None:
            providers.close_provider(self.summariser_provider)
        for seat in self.review_seats:
            providers.close_provider(seat.provider)


def build_session_providers(
    cfg: Config,
    *,
    role: RoleName,
    events: event_log.EventSink,
    transcript_sink: TranscriptSink,
    stream_text: bool,
    reporter: app_reporter.Reporter = app_reporter.STDIO_REPORTER,
) -> SessionProviders:
    """Build the driving role's provider and the summariser and review seats.

    Args:
        cfg: The resolved config.
        role: The driving role.
        events: The run's event sink.
        transcript_sink: Where every provider call is transcribed.
        stream_text: Whether the provider streams text into the sink.
        reporter: Where the prompt-override warning goes.

    Returns:
        The providers, metering into one tracker.
    """
    budget = _setup.budget_tracker(cfg)
    inner = providers.build_role_provider(cfg, role, transcript_sink=transcript_sink, budget=budget)
    rm_role = cfg.models.resolve(role)
    assert rm_role is not None  # require_runnable validated this
    preflight.warn_if_prompt_override_incomplete(cfg, reporter=reporter)
    provider: Provider = providers.InstrumentedProvider(
        inner=inner,
        role=role,
        model=rm_role.model,
        provider_name=rm_role.provider,
        events=events,
        budget=budget,
        stream_text=stream_text,
    )
    summariser_provider = providers.reviewer_seat_provider(
        cfg, "summariser", transcript_sink=transcript_sink, budget=budget, events=events
    )
    # The panel is the in-loop review: a trigger with no seats builds one seat on the reviewer.
    review_seats = (
        providers.build_review_seats(
            cfg, transcript_sink=transcript_sink, budget=budget, n=1, events=events
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
    dispatcher = dispatch.ToolDispatcher(
        root=cwd,
        config=cfg,
        isolation=isolation,
        prompts=prompts,
        events=events,
        curator=curator,
        run_root_node_id=None,  # Harness seeds the root + calls set_run_root_node_id
        mcp_manager=mcp_manager,
        worktree_git_dir=worktree_git_dir,
        mode=mode,
        state_dir=state_dir,
        session_dir=layout.session_dir,
        use_jail_session=True,
        session_net=session_net,
    )
    compact_drop, compact_summarise, keep_recent = providers.resolve_compaction_thresholds(
        cfg, rm_role, log=loop_log
    )
    cfg = providers.resolve_decompose(cfg, rm_role, log=loop_log)
    return SessionTools(
        curator=curator,
        dispatcher=dispatcher,
        compact_drop_at_chars=compact_drop,
        compact_summarise_at_chars=compact_summarise,
        keep_recent_chars=keep_recent,
        cfg=cfg,
    )


def session_facts_provider(
    budget: agent6_budget.BudgetTracker, model: str, run_commands: str, isolation: str
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
        )

    return facts
