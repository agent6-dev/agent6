# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The `agent6 resume` lifecycle.

Picks a paused or crashed session back up from its snapshot, over the same
`SessionFrontend` seam `run_task` uses.
"""

from __future__ import annotations

import contextlib
import functools
import os
import pathlib
from collections.abc import Callable

from agent6 import budget, kinds
from agent6.app import (
    execution,
    fork_worktrees,
    preflight,
    resume_chain,
    resume_record,
    resume_setup,
    resume_tree,
)
from agent6.app import frontend as app_frontend
from agent6.app import reporter as app_reporter
from agent6.app.session import prepare as session_prepare
from agent6.app.session import setup as session_setup
from agent6.config import Config
from agent6.harness import loop_snapshot
from agent6.harness.prompt import repo_context
from agent6.sessions import event_log, process
from agent6.sessions import layout as sessions_layout
from agent6.sessions import manifest as sessions_manifest
from agent6.vcs.git import ops as git_ops


def replay_gate(
    cfg: Config,
    _budget: budget.BudgetTracker,
    _available: Callable[[str], bool],
    *,
    layout: sessions_layout.SessionLayout,
    snapshot: loop_snapshot.SessionSnapshot,
    events: event_log.EventSink,
    reporter: app_reporter.Reporter,
) -> Config:
    """Run a resumed execution's gate step: reuse the run's gate, then pin what it runs with.

    Args:
        cfg: The config with the execution's own gate, when it configures one.
        layout: The session's directory layout.
        snapshot: The run's last snapshot, which names the gate before any pin resolved.
        events: The run's event sink.
        reporter: Receives the reuse and change notes.

    Returns:
        The config with the gate this execution runs.
    """
    pinned = sessions_manifest.GateStamp(availability="pending")
    with contextlib.suppress(sessions_manifest.ManifestError, OSError):
        pinned = sessions_manifest.read_manifest(layout.session_dir).harness.gate
    # A resolved pin outranks a snapshot that predates adoption or un-adoption.
    replayed = pinned.command if pinned.availability != "pending" else snapshot.verify_command
    execution_configured = bool(cfg.harness.verify.command)
    reused = not execution_configured and bool(replayed)
    if reused:
        cfg = cfg.with_verify_command(replayed)
    # Dropped last when commands are withheld, so nothing hands the gate back.
    cfg, withheld = session_prepare.settle_gate(
        cfg,
        layout,
        source="configured" if execution_configured else pinned.source or "inferred",
        events=events,
        reporter=reporter,
    )
    # A withheld gate is neither a reuse nor a change.
    if reused and not withheld:
        reporter.note(f"reusing this run's verify command: {preflight.gate_text(replayed)}")
    # The manifest says which gate this execution used.
    if pinned.command != cfg.harness.verify.command and not withheld:
        # Both directions: the frozen system prompt names the old gate either way.
        reporter.note(
            "this run's verify gate changed:"
            f" was {preflight.gate_text(pinned.command)},"
            f" now {preflight.gate_text(cfg.harness.verify.command)}"
        )
    return cfg


def resume_inputs(
    target: resume_record.ResumeTarget,
    snapshot: loop_snapshot.SessionSnapshot,
    start: resume_setup.ResumeStart,
    tree: resume_tree.TreeVerdict,
    held: session_prepare.Held,
    *,
    base_sha: str,
    kept: tuple[str, ...],
    merged_in: git_ops.ChainMerge | None,
    config_paths: tuple[pathlib.Path, ...],
    steer: str,
    interactive: bool,
    sandbox_overrides: session_setup.SandboxOverrides | None,
    reporter: app_reporter.Reporter,
) -> execution.ExecutionInputs:
    """Build what a resumed execution hands the body.

    Args:
        target: The session.
        snapshot: The run's last snapshot.
        start: The config and policy the execution starts under.
        tree: Whose the working tree's changes are.
        held: The teardown record, for the writer lock the auto-merge shares.
        base_sha: The run's base.
        kept: The outside paths left in the tree for the run.
        merged_in: The branch merged into the run's chain.
        config_paths: The `--config` files an `/undo` fork is stamped under.
        steer: The `--steer` text.
        interactive: Whether the ask REPL drives the session.
        sandbox_overrides: The sandbox flags.
        reporter: Receives the gate's and the fork's notes.

    Returns:
        The inputs.
    """
    layout, manifest, cfg = target.layout, target.manifest, start.cfg
    return execution.ExecutionInputs(
        session_id=layout.session_id,
        mode=target.mode,
        role=target.role,
        explicit_leaves=start.effective.explicit_leaves,
        effective=start.effective,
        isolation=start.isolation,
        tui_enabled=start.tui_enabled,
        interactive=interactive,
        task=None,
        gate=functools.partial(
            replay_gate,
            layout=layout,
            snapshot=snapshot,
            events=start.channels.events,
            reporter=reporter,
        ),
        chain_branch=manifest.run_branch or None,
        base_sha=base_sha,
        untracked_at_start=tree.untracked_at_start,
        resume_state_path=target.snapshot_path,
        undo_forker=session_prepare.undo_forker(
            config_paths, layout.session_id, cwd=target.repo, reporter=reporter
        ),
        prompts=start.channels.prompts,
        # The follow-up this execution answered, not the original task.
        ask_transcript_task=steer.strip() or manifest.user_task,
        resume_steer=steer,
        operator_edits=kept,
        merged_in=merged_in,
        budget_overrides=target.budget,
        sandbox_overrides=sandbox_overrides,
        resuming=True,
        worktree_git_dir=manifest.worktree_git_dir,
        mirror_into=fork_worktrees.mirror_origin(target.repo, target.cwd),
        worker_lock_fd=held.worker_lock_fd,
        base_rules=(
            repo_context.review_rules(
                target.cwd, manifest.review_base_tree, cfg.sandbox.hide_paths
            ).text
            if manifest.review_base_tree
            else None
        ),
    )


def resume_task(
    config_paths: tuple[pathlib.Path, ...],
    session_id: str,
    *,
    frontend: app_frontend.SessionFrontend,
    cwd: pathlib.Path | None = None,
    force: bool,
    started_at: float,
    tui: bool = False,
    budget_overrides: session_setup.BudgetOverrides | None = None,
    sandbox_overrides: session_setup.SandboxOverrides | None = None,
    preset: str = "",
    steer: str = "",
    interactive: bool = False,
    model: str = "",
    effort: kinds.EffortLevel | None = None,
    merge_from: str = "",
    reporter: app_reporter.Reporter = app_reporter.STDIO_REPORTER,
) -> int:
    """Resume a paused or crashed session from its snapshot.

    The budget is a fresh ceiling per execution.
    A fork's execution drives the fork's own worktree.
    Each stage reports its refusal and raises it; the stages run in preflight order, and
    `hold` releases what they took.

    Args:
        config_paths: The `--config` files in command-line order; none means discovery alone.
        session_id: The session's id or prefix.
        frontend: The surface's seam: console, confirms, approvals, questions.
        cwd: The repository directory; None uses the process cwd.
        force: Whether to resume onto a chain tip that diverged from the snapshot.
        started_at: The instant this execution began; the stale-state clear keeps
            bridge markers written since.
        tui: Whether `--tui` was asked for.
        budget_overrides: The budget flags the lifecycle re-reads.
        sandbox_overrides: The sandbox flags the lifecycle re-reads.
        preset: The `--preset` flag's value.
        steer: An operator follow-up queued for the loop's first boundary.
        interactive: Whether the ask REPL drives the session.
        model: The `--model` value for this execution.
        effort: The reasoning effort this execution's role runs at; None keeps the config's.
        merge_from: A branch to merge into the run's chain before the first turn, its
            conflicts left as markers for the run to resolve; "" merges nothing.
        reporter: Receives every refusal and note.

    Returns:
        The process exit code.
    """
    repo = cwd if cwd is not None else pathlib.Path.cwd()
    try:
        target = resume_record.find_resume(
            repo,
            session_id,
            steer=steer,
            model=model,
            budget_overrides=budget_overrides,
            reporter=reporter,
        )
        layout = target.layout
        with session_prepare.hold(layout, frontend=frontend, reporter=reporter) as held:
            # One writer per session dir, taken before any shared state is touched.
            held.worker_lock_fd = session_prepare.take_single_writer(layout, reporter=reporter)
            scan = resume_record.reopen_record(
                target, steer=steer, merge_from=merge_from, reporter=reporter
            )
            if target.manifest.parked_task:
                # Before the bridge reset: a steer seeded here would stay beside run_task's own.
                held.in_session = True
                return resume_setup.start_parked(
                    target,
                    held,
                    config_paths=config_paths,
                    frontend=frontend,
                    started_at=started_at,
                    sandbox_overrides=sandbox_overrides,
                    preset=preset,
                    steer=steer,
                    effort=effort,
                    reporter=reporter,
                )
            carried = resume_record.reset_bridge(
                target, scan, steer=steer, started_at=started_at, reporter=reporter
            )
            held.in_session = True
            held.repo_lock_fd = resume_setup.take_checkout(target, reporter=reporter)
            snapshot = resume_record.read_snapshot(target, reporter=reporter)
            merged = resume_record.check_chain(target, force=force, reporter=reporter)
            start = resume_setup.resume_start(
                target,
                snapshot,
                carried,
                config_paths=config_paths,
                frontend=frontend,
                tui=tui,
                interactive=interactive,
                sandbox_overrides=sandbox_overrides,
                preset=preset,
                effort=effort,
                reporter=reporter,
            )
            tree = resume_tree.classify(
                target.cwd,
                layout,
                target.manifest,
                start.cfg,
                mode=target.mode,
                snapshot=snapshot,
                scan=scan,
                merged=merged,
                can_ask=frontend.capabilities.can_ask,
                reporter=reporter,
            )
            held.untracked_at_start = tree.untracked_at_start
            # Written once the preflight passed: a refused resume never had a live worker, and
            # the question the settle asks reaches `agent6 answer`.
            process.write_worker_pid(layout.session_dir, os.getpid())
            kept = resume_setup.settle_tree(target, start, tree, held, reporter=reporter)
            base_sha = resume_chain.move_base(
                target.cwd,
                layout,
                target.manifest,
                merged,
                start.identity,
                mode=target.mode,
                reporter=reporter,
            )
            merged_in = resume_setup.merge_in(
                target, base_sha, start.identity, merge_from=merge_from, reporter=reporter
            )
            resume_setup.stamp_resume(
                target,
                start,
                tree,
                base_sha,
                preset=preset,
                model=model,
                steer=steer,
                reporter=reporter,
            )
            end = execution.run_execution(
                start.cfg,
                layout,
                resume_inputs(
                    target,
                    snapshot,
                    start,
                    tree,
                    held,
                    base_sha=base_sha,
                    kept=kept,
                    merged_in=merged_in,
                    config_paths=config_paths,
                    steer=steer,
                    interactive=interactive,
                    sandbox_overrides=sandbox_overrides,
                    reporter=reporter,
                ),
                frontend=frontend,
                reporter=reporter,
                events=start.channels.events,
                transcript_sink=start.channels.transcript_sink,
                cwd=target.cwd,
                state_dir=target.state,
            )
            held.detached = end.detach_requested
    except preflight.SessionRefusedError as refusal:
        return refusal.rc
    if end.detach_requested:
        execution.detach_to_background(
            frontend=frontend,
            cfg=start.cfg,
            layout=layout,
            cwd=repo,
            flags=session_setup.override_flags(
                target.budget,
                sandbox_overrides,
                session_setup.flag_route(start.cfg, target.mode, target.route),
            ),
            reporter=reporter,
        )
    return end.rc
