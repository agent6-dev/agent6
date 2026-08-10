# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""One owner for the session assembly `run_task` and `resume_task` share: the
isolation preflight, the egress/landlock start, and the provider/dispatcher
build. The lifecycles keep their own workspace steps (branch cut + manifest
vs snapshot guards) and their Workflow wiring."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import agent6
from agent6.app._setup import SandboxOverrides, detect_env
from agent6.app.confine import (
    check_hide_paths_support,
    check_mcp_network_support,
    check_network_support,
    check_protect_git_support,
    check_workspace_outside_private_dirs,
    warn_sandbox_gaps,
)
from agent6.app.preflight import budget_preflight, warn_if_prompt_override_incomplete
from agent6.app.providers import (
    InstrumentedProvider,
    build_critic_provider,
    build_review_seats,
    build_role_provider,
    build_summariser_provider,
    resolve_compaction_thresholds,
    resolve_decompose,
    review_panel_configured,
)
from agent6.app.reporter import Reporter
from agent6.budget import BudgetTracker
from agent6.config import Config, RoleModel, RoleName
from agent6.events import EventSink
from agent6.graph.curator import GraphCurator
from agent6.providers import Provider, TranscriptSink
from agent6.sandbox.detect import IsolationUnavailableError, resolve_isolation
from agent6.sandbox.jail import SessionNetwork
from agent6.sessions.layout import SessionLayout
from agent6.tools.dispatch import Approver, ToolDispatcher
from agent6.tools.mcp_client import MCPManager
from agent6.tools.schema import UserQuestion
from agent6.types import IsolationLevel, session_kind
from agent6.workflows.review import ReviewSeat


class SessionRefused(Exception):
    """A preflight refusal already reported through the Reporter; the caller
    returns ``rc`` as the process exit code."""

    def __init__(self, rc: int) -> None:
        super().__init__(f"session refused (exit {rc})")
        self.rc = rc


def select_isolation(
    cfg: Config,
    *,
    confirm_unconfined: Callable[[IsolationLevel, Config], bool],
    reporter: Reporter,
    explicit_leaves: frozenset[str] = frozenset(),
) -> IsolationLevel:
    """The isolation preflight: pick the sandbox isolation for this environment,
    confirm an unconfined autorun, and refuse configs the isolation cannot honor
    (network mode, strict egress, budget) or a workspace no tool could read.
    Raises :class:`SessionRefused`."""
    env = detect_env()
    try:
        selected = resolve_isolation(cfg.sandbox.isolation, env)
    except IsolationUnavailableError as exc:
        reporter.err(f"REFUSING: {exc}")
        raise SessionRefused(2) from exc
    warn_sandbox_gaps(selected, env, cfg, reporter=reporter)
    if not confirm_unconfined(selected, cfg):
        reporter.err("[agent6] aborted.")
        raise SessionRefused(1)
    net_err = check_network_support(cfg, selected)
    if net_err is not None:
        reporter.err(f"REFUSING: {net_err}")
        raise SessionRefused(2)
    mcp_net_err = check_mcp_network_support(cfg, selected)
    if mcp_net_err is not None:
        reporter.err(f"REFUSING: {mcp_net_err}")
        raise SessionRefused(2)
    hide_err = check_hide_paths_support(cfg, selected)
    if hide_err is not None:
        reporter.err(f"REFUSING: {hide_err}")
        raise SessionRefused(2)
    ws_err = check_workspace_outside_private_dirs(Path.cwd())
    if ws_err is not None:
        reporter.err(f"REFUSING: {ws_err}")
        raise SessionRefused(2)
    # A DEFAULT degrades with the warning above; a value the operator wrote
    # down refuses, because they asked for something this host cannot give.
    git_err = check_protect_git_support(
        cfg, selected, explicitly_set="sandbox.protect_git" in explicit_leaves
    )
    if git_err is not None:
        reporter.err(f"REFUSING: {git_err}")
        raise SessionRefused(2)
    budget_err = budget_preflight(cfg)
    if budget_err is not None:
        reporter.err(f"REFUSING: {budget_err}")
        raise SessionRefused(2)
    return selected


def install_inside_workspace(cwd: Path) -> Path | None:
    """agent6's own install root when it sits inside *cwd*, else None.

    An in-tree install (pip into the project's venv) is inside the jail's
    writable workspace, so a jailed command can rewrite the running agent.
    """
    root = Path(agent6.__file__).resolve().parent
    return root if root.is_relative_to(cwd.resolve()) else None


def warn_install_inside_workspace(cwd: Path, *, reporter: Reporter) -> None:
    """Warn when agent6 is installed inside the workspace a jailed command can
    write (never refuse it: agent6 developing agent6 is exactly that shape)."""
    if (root := install_inside_workspace(cwd)) is not None:
        reporter.err(
            f"[agent6] WARNING: agent6 is installed inside this workspace ({root});"
            " a jailed command can rewrite the running agent. Install it outside"
            " the project (pipx / uv tool)."
        )


@dataclass(frozen=True, slots=True)
class SessionProviders:
    """The per-run provider battery: the driving role's instrumented provider
    plus the critic/summariser/review seats, all metering into one tracker."""

    budget: BudgetTracker
    rm_role: RoleModel
    provider: Provider
    critic_provider: Provider | None
    summariser_provider: Provider | None
    review_seats: list[ReviewSeat]


def build_session_providers(
    cfg: Config,
    *,
    role: RoleName,
    events: EventSink,
    transcript_sink: TranscriptSink,
    stream_text: bool,
) -> SessionProviders:
    budget = BudgetTracker(
        max_usd=cfg.budget.max_usd,
        max_tokens_fallback=cfg.budget.max_tokens_fallback,
    )
    inner = build_role_provider(cfg, role, transcript_sink=transcript_sink, budget=budget)
    rm_role = cfg.models.resolve(role)
    assert rm_role is not None  # require_runnable validated this
    warn_if_prompt_override_incomplete(cfg)
    provider: Provider = InstrumentedProvider(
        inner=inner,
        role=role,
        model=rm_role.model,
        provider_name=rm_role.provider,
        events=events,
        budget=budget,
        stream_text=stream_text,
    )
    critic_provider = build_critic_provider(
        cfg, transcript_sink=transcript_sink, budget=budget, events=events
    )
    summariser_provider = build_summariser_provider(
        cfg, transcript_sink=transcript_sink, budget=budget, events=events
    )
    # The grounded review panel runs at the critic trigger WHEN explicitly
    # configured (any review_* key); otherwise critic!=off keeps the legacy
    # single critic, so a pre-panel before_finish/periodic config still gates.
    review_seats = (
        build_review_seats(cfg, transcript_sink=transcript_sink, budget=budget, n=1, events=events)
        if cfg.review.trigger != "off" and review_panel_configured(cfg)
        else []
    )
    return SessionProviders(
        budget=budget,
        rm_role=rm_role,
        provider=provider,
        critic_provider=critic_provider,
        summariser_provider=summariser_provider,
        review_seats=review_seats,
    )


@dataclass(frozen=True, slots=True)
class SessionTools:
    """The curator + dispatcher pair and the model-derived loop knobs.
    ``cfg`` is the decompose-resolved config the Workflow must be built with."""

    curator: GraphCurator
    dispatcher: ToolDispatcher
    compact_drop_at_chars: int
    compact_summarise_at_chars: int
    cfg: Config


def build_session_tools(
    cfg: Config,
    *,
    cwd: Path,
    state_dir: Path,
    layout: SessionLayout,
    isolation: IsolationLevel,
    mode: Literal["run", "plan", "ask"],
    events: EventSink,
    approver: Approver,
    questioner: Callable[[tuple[UserQuestion, ...]], tuple[str, ...]],
    loop_log: Callable[[str], None],
    mcp_manager: MCPManager | None,
    rm_role: RoleModel,
    session_net: SessionNetwork | None = None,
) -> SessionTools:
    # The DAG curator runs in-process: the run's worker.lock already makes
    # this the sole writer, so no subprocess or socket is needed.
    curator = GraphCurator(layout)
    dispatcher = ToolDispatcher(
        root=cwd,
        config=cfg,
        isolation=isolation,
        approver=approver,
        questioner=questioner,
        events=events,
        curator=curator,
        run_root_node_id=None,  # Workflow seeds the root + calls set_run_root_node_id
        mcp_manager=mcp_manager,
        mode=mode,
        state_dir=state_dir,
        session_dir=layout.session_dir,
        # One jail process for this run's commands.
        use_jail_session=True,
        session_net=session_net,
    )
    compact_drop, compact_summarise = resolve_compaction_thresholds(cfg, rm_role, log=loop_log)
    cfg = resolve_decompose(cfg, rm_role, log=loop_log)
    return SessionTools(
        curator=curator,
        dispatcher=dispatcher,
        compact_drop_at_chars=compact_drop,
        compact_summarise_at_chars=compact_summarise,
        cfg=cfg,
    )


def session_config(cfg: Config, mode: str, overrides: SandboxOverrides | None = None) -> Config:
    """The effective config for a session of *mode*.

    Both lifecycles call this before anything reads a knob, so a fresh ask and
    a resumed one are governed identically. Today it is the ask clamp; anything
    else mode-dependent belongs here rather than at one call site.

    *overrides* are the operator's per-invocation flags, and they land LAST:
    the most specific layer, and the one the LLM cannot reach. The ask clamp
    exists to catch a STANDING ``run_commands = "yes"`` that nobody is watching,
    not an explicit ``--auto-approve`` on this invocation -- clamping that made
    the flag inert and every headless `ask --auto-approve` refused. Tightening
    still wins outright: ``--no-commands`` pins "no", and ``--auto-approve``
    never resurrects a withheld one.
    """
    clamped = cfg.clamped_for_ask() if session_kind(mode).clamps_commands else cfg
    return clamped if overrides is None else overrides.apply(clamped)
