# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The `agent6 run` lifecycle and its plan and ask modes.

Preflight, branch cut, manifest, loop construction, finalize. Everything that
touches the terminal is injected through `SessionFrontend`, so this module never
imports `agent6.ui`.
"""

from __future__ import annotations

import functools
import os
import pathlib
from collections.abc import Callable, Sequence

from agent6 import budget as agent6_budget
from agent6 import directive, kinds
from agent6.app import (
    execution,
    execution_finish,
    fork_worktrees,
    preflight,
    run_open,
    run_preflight,
)
from agent6.app import frontend as app_frontend
from agent6.app import reporter as app_reporter
from agent6.app.session import prepare as session_prepare
from agent6.app.session import setup as session_setup
from agent6.config import Config, layer_data
from agent6.sessions import process


def no_gate(
    cfg: Config, budget: agent6_budget.BudgetTracker, available: Callable[[str], bool]
) -> Config:
    """Return the config as it is: a bounded conversation never verifies."""
    # No inference, no gate rows in its journal.
    del budget, available
    return cfg


def infer_gate(
    cfg: Config,
    budget: agent6_budget.BudgetTracker,
    available: Callable[[str], bool],
    *,
    start: run_preflight.RunStart,
    opened: run_open.Opened,
    reporter: app_reporter.Reporter,
) -> Config:
    """Run a fresh execution's gate step: infer an unset gate, then pin what it runs with.

    Args:
        cfg: The execution's config.
        budget: The run's budget, which the inference spends from.
        available: Says whether the jail can run an executable.
        start: The start in its checkout.
        opened: The session's channels and its gate's origin.
        reporter: Receives the inference's notes.

    Returns:
        The config with the gate this execution runs; dropped last when commands are withheld.
    """
    configured_gate = bool(cfg.harness.verify.command)
    # In the repository, never the run's worktree: a signal such as `.venv` is absent
    # there until the setup command builds it, and the gate is carried.
    cfg = preflight.infer_verify_if_unset(
        cfg,
        start.cwd if start.review_world is not None else start.repo,
        mode=start.mode,
        available=available,
        events=opened.channels.events,
        transcript_sink=opened.channels.transcript_sink,
        budget=budget,
        reporter=reporter,
    )
    source = opened.child_gate_source or ("configured" if configured_gate else "inferred")
    cfg, _withheld = session_prepare.settle_gate(
        cfg, start.layout, source=source, events=opened.channels.events, reporter=reporter
    )
    return cfg


def run_inputs(
    start: run_preflight.RunStart,
    isolation: kinds.IsolationLevel,
    held: session_prepare.Held,
    opened: run_open.Opened,
    *,
    interactive: bool,
    standing_goal: str,
    budget_overrides: session_setup.BudgetOverrides | None,
    sandbox_overrides: session_setup.SandboxOverrides | None,
    pins: Sequence[str],
    explicit_leaves: frozenset[str],
    config_paths: tuple[pathlib.Path, ...],
    effective: layer_data.EffectiveConfig | None,
    reporter: app_reporter.Reporter,
    contract: kinds.ConversationContract | None,
    opening: directive.Segment | tuple[directive.Segment, ...] | None,
) -> execution.ExecutionInputs:
    """Build what a fresh execution hands the body.

    Args:
        start: The start in its checkout.
        isolation: The isolation its commands run under.
        held: The teardown record, for the operator's untracked files and the writer lock
            the auto-merge shares.
        opened: The session's channels and its gate's origin.
        interactive: Whether the ask REPL drives the session.
        standing_goal: The standing goal, when set.
        budget_overrides: The budget flags a child inherits.
        sandbox_overrides: The sandbox flags a child inherits.
        pins: The pinned paths.
        explicit_leaves: The config leaves the operator wrote.
        config_paths: The `--config` files an `/undo` fork is stamped under.
        effective: The config and its file origins before execution policy.
        reporter: Receives the gate's and the fork's notes.
        contract: The bounded conversation a `reviewer` session drives.
        opening: The tasks the run dispatches in place of its first model call.

    Returns:
        The inputs.
    """
    layout, review_world = start.layout, start.review_world
    return execution.ExecutionInputs(
        session_id=layout.session_id,
        mode=start.mode,
        role=kinds.session_kind(start.mode).role,
        explicit_leaves=explicit_leaves,
        effective=effective,
        isolation=isolation,
        tui_enabled=start.tui_enabled,
        interactive=interactive,
        task=start.task,
        operator_task=start.operator_task,
        gate=(
            functools.partial(infer_gate, start=start, opened=opened, reporter=reporter)
            if contract is None
            else no_gate
        ),
        contract=contract,
        opening=(opening,) if isinstance(opening, directive.Segment) else opening or (),
        chain_branch=start.run_branch,
        base_sha=start.base_sha,
        untracked_at_start=held.untracked_at_start,
        # Written for every mode: `agent6 resume` reaches an ask too.
        resume_state_path=layout.session_dir / "loop_state.json",
        undo_forker=session_prepare.undo_forker(
            config_paths, layout.session_id, cwd=start.repo, reporter=reporter
        ),
        prompts=opened.channels.prompts,
        # The REPL prints and saves each turn itself.
        ask_transcript_task=None if interactive else start.task,
        budget_overrides=budget_overrides,
        sandbox_overrides=sandbox_overrides,
        standing_goal=standing_goal,
        pins=tuple(pins),
        worktree_git_dir=start.worktree_git_dir,
        mirror_into=fork_worktrees.mirror_origin(start.repo, start.cwd),
        worker_lock_fd=held.worker_lock_fd,
        base_rules=review_world.rules if review_world is not None else None,
        review_target=(
            (review_world.target, review_world.diff) if review_world is not None else None
        ),
    )


def run_task(
    cfg: Config,
    task: str,
    *,
    frontend: app_frontend.SessionFrontend,
    cwd: pathlib.Path | None = None,
    started_at: float,
    operator_task: str | None = None,
    session_id: str = "",
    source_session_id: str | None = None,
    interactive: bool = False,
    tui: bool = False,
    mode: kinds.LoopMode = "run",
    metric: directive.MetricSpec | None = None,
    standing_goal: str = "",
    budget_overrides: session_setup.BudgetOverrides | None = None,
    sandbox_overrides: session_setup.SandboxOverrides | None = None,
    preset: str = "",
    initial_steer: str = "",
    pins: Sequence[str] = (),
    preset_stamp: tuple[str, bool] | None = None,
    model: str | kinds.ModelRoute | None = None,
    explicit_leaves: frozenset[str] = frozenset(),
    worktree: bool = False,
    config_paths: tuple[pathlib.Path, ...] = (),
    effective: layer_data.EffectiveConfig | None = None,
    reporter: app_reporter.Reporter = app_reporter.STDIO_REPORTER,
    contract: kinds.ConversationContract | None = None,
    opening: directive.Segment | tuple[directive.Segment, ...] | None = None,
) -> int:
    """Run one session from preflight to finalize; the sole `agent6 run` path.

    The front-end has built the config, resolved the task, checked the git-repo wall
    and the runnable roles, and routed `--parallel` away. Plan mode drives the same
    harness with the planning prompt, no edit tools, `finish_planning` and no
    auto-commit; its markdown lands at `<session-dir>/plan.md`.
    Each stage reports its refusal and raises it; the stages run in preflight order, and
    `hold` releases what they took.

    Args:
        cfg: The run's config, overrides applied.
        task: The task text.
        operator_task: The words before composition; None means `task` is uncomposed.
        frontend: The surface's seam: console, confirms, approvals, questions.
        cwd: The repository directory; None uses the process cwd.
        started_at: The instant this execution began; the stale-state clear keeps
            bridge markers written since.
        session_id: An explicit id; empty allocates one.
        source_session_id: The plan session a `run --from` came from.
        interactive: Whether the ask REPL drives the session.
        tui: Whether `--tui` was asked for.
        mode: The session mode.
        metric: The operator's metric for this run, or None.
        standing_goal: The standing goal, when set.
        budget_overrides: The budget flags the lifecycle re-reads.
        sandbox_overrides: The sandbox flags the lifecycle re-reads.
        preset: The `--preset` flag's value.
        initial_steer: An operator follow-up queued for the loop's first boundary,
            seeded after the stale-state clear so it survives it, with the images a
            front-end recorded for it.
        pins: The pinned paths.
        preset_stamp: `(name, from_flag)` to stamp in place of deriving from `agent6.preset`,
            so a parked resume replays the original submission's precedence.
        model: The `--model` value that set the driving role.
        explicit_leaves: The config leaves the operator wrote, as dotted paths; a
            default this host cannot honour degrades, a written value refuses.
        worktree: Whether the run works in its own linked worktree, added at HEAD's
            commit under `[git.worktree].base_dir`; the checkout is then never touched.
        config_paths: The `--config` files, which an `/undo` fork is stamped under.
        effective: The config and its file origins before execution policy.
        reporter: Receives every refusal and note.
        contract: The bounded conversation a `reviewer` session drives; None for every
            other mode.
        opening: A single `--parallel` task or the tasks parsed from a composer line.
            The run ranks the attempts for each task in place of its first model call.
            None opens with the model.

    Returns:
        The process exit code.

    Raises:
        ValueError: `reviewer` without a contract, or a contract under another mode.
    """
    if (mode == "reviewer") != (contract is not None):
        raise ValueError("mode 'reviewer' is the one mode that drives a conversation contract")
    try:
        start, isolation = run_preflight.check_run(
            cfg,
            task,
            frontend=frontend,
            repo=cwd if cwd is not None else pathlib.Path.cwd(),
            operator_task=task if operator_task is None else operator_task,
            session_id=session_id,
            interactive=interactive,
            tui=tui,
            mode=mode,
            sandbox_overrides=sandbox_overrides,
            model=model,
            worktree=worktree,
            opening=opening,
            reporter=reporter,
        )
        with session_prepare.hold(start.layout, frontend=frontend, reporter=reporter) as held:
            start, isolation = run_preflight.claim_checkout(
                start, isolation, held, frontend=frontend, reporter=reporter
            )
            held.in_session = True
            opened = run_open.open_run(
                start,
                isolation,
                frontend=frontend,
                started_at=started_at,
                source_session_id=source_session_id,
                metric=metric,
                budget_overrides=budget_overrides,
                preset=preset,
                initial_steer=initial_steer,
                preset_stamp=preset_stamp,
                model=model,
                effective=effective,
                reporter=reporter,
            )
            # A live worker from here, so the start questions the settle asks reach
            # `agent6 answer`.
            process.write_worker_pid(start.layout.session_dir, os.getpid())
            run_open.settle_run_tree(start, held, opened, reporter=reporter)
            end = execution.run_execution(
                start.cfg,
                start.layout,
                run_inputs(
                    start,
                    isolation,
                    held,
                    opened,
                    interactive=interactive,
                    standing_goal=standing_goal,
                    budget_overrides=budget_overrides,
                    sandbox_overrides=sandbox_overrides,
                    pins=pins,
                    explicit_leaves=explicit_leaves,
                    config_paths=config_paths,
                    effective=effective,
                    reporter=reporter,
                    contract=contract,
                    opening=opening,
                ),
                frontend=frontend,
                reporter=reporter,
                events=opened.channels.events,
                transcript_sink=opened.channels.transcript_sink,
                cwd=start.cwd,
                state_dir=start.state,
            )
            held.detached = end.detach_requested
    except preflight.SessionRefusedError as refusal:
        return refusal.rc
    if end.detach_requested:
        # The worker lock is released, so the detached `resume` can acquire it.
        execution_finish.detach_to_background(
            frontend=frontend,
            cfg=start.cfg,
            layout=start.layout,
            cwd=start.repo,
            flags=session_setup.override_flags(
                budget_overrides,
                sandbox_overrides,
                session_setup.flag_route(start.cfg, mode, model),
            ),
            reporter=reporter,
        )
    return end.rc
