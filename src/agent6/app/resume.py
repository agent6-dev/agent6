# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The `agent6 resume` lifecycle.

Picks a paused or crashed session back up from its snapshot, over the same
`SessionFrontend` seam `run_task` uses.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import os
import pathlib
from collections.abc import Callable

from agent6 import budget, errors, kinds, terminal
from agent6.app import (
    execution,
    fork_worktrees,
    preflight,
    resume_chain,
    resume_questions,
    resume_record,
    resume_tree,
    run,
    stamps,
    worktree_setup,
)
from agent6.app import frontend as app_frontend
from agent6.app import reporter as app_reporter
from agent6.app.session import prepare as session_prepare
from agent6.app.session import setup as session_setup
from agent6.config import Config, ConfigError, layer_data
from agent6.harness import loop_snapshot
from agent6.harness.prompt import repo_context
from agent6.sessions import event_log, ipc, journal, lock, process
from agent6.sessions import layout as sessions_layout
from agent6.sessions import manifest as sessions_manifest
from agent6.vcs.git import ops as git_ops
from agent6.vcs.git import read as git_read


@dataclasses.dataclass(frozen=True, slots=True)
class ResumeStart:
    """The config and policy a resumed execution starts under.

    Attributes:
        effective: The loaded config with its origins.
        cfg: The execution's config: the setup command read, the worktree's hidden paths added.
        isolation: The resolved isolation level.
        identity: The identity the run's commits carry.
        channels: The journal, the transcripts and the gate to the operator.
        tui_enabled: Whether the TUI owns the terminal.
    """

    effective: layer_data.EffectiveConfig
    cfg: Config
    isolation: kinds.IsolationLevel
    identity: git_read.CommitIdentity
    channels: session_prepare.Channels
    tui_enabled: bool


def _load_config(
    target: resume_record.ResumeTarget,
    config_paths: tuple[pathlib.Path, ...],
    *,
    sandbox_overrides: session_setup.SandboxOverrides | None,
    preset: str,
    effort: kinds.EffortLevel | None,
    reporter: app_reporter.Reporter,
) -> layer_data.EffectiveConfig:
    """Load the config this execution runs under, the recorded choices under its flags.

    Args:
        target: The session.
        config_paths: The `--config` files.
        sandbox_overrides: The sandbox flags.
        preset: The `--preset` flag's value.
        effort: The reasoning effort of this execution's role.
        reporter: Receives the refusal.

    Returns:
        The effective config, checked runnable for the mode's role.

    Raises:
        preflight.SessionRefusedError: The config does not load or its role cannot run.
    """
    try:
        # replay_preset: a config-selected preset re-resolves rather than ranking as a flag.
        return session_setup.load_session_config(
            target.repo,
            config_paths,
            mode=target.mode,
            preset=preset or target.manifest.harness.replay_preset,
            budget_overrides=target.budget,
            sandbox_overrides=sandbox_overrides,
            model=target.route,
            effort=effort,
        )
    except ConfigError as exc:
        reporter.error(str(exc))
        raise preflight.SessionRefusedError(2) from exc


def _note_a_changed_driver(
    cfg: Config, target: resume_record.ResumeTarget, *, reporter: app_reporter.Reporter
) -> None:
    """Say which model drives this execution when it is not the recorded one."""
    recorded = target.manifest.models.driver
    driver = cfg.models.resolve(target.role)
    if (
        recorded is not None
        and driver is not None
        and (driver.provider, driver.model) != (recorded.provider, recorded.model)
    ):
        reporter.note(
            f"resuming with {driver.provider}/{driver.model}"
            f" (recorded: {recorded.provider}/{recorded.model})"
        )


def start_parked(
    target: resume_record.ResumeTarget,
    held: session_prepare.Held,
    *,
    config_paths: tuple[pathlib.Path, ...],
    frontend: app_frontend.SessionFrontend,
    started_at: float,
    sandbox_overrides: session_setup.SandboxOverrides | None,
    preset: str,
    steer: str,
    effort: kinds.EffortLevel | None,
    reporter: app_reporter.Reporter,
) -> int:
    """Start a parked submission: nothing ever ran, so `run_task` starts it under the same id.

    Args:
        target: The session.
        held: The teardown record; the writer lock goes before `run_task` takes its own.
        config_paths: The `--config` files.
        frontend: The surface's seam.
        started_at: The instant this execution began.
        sandbox_overrides: The sandbox flags.
        preset: The `--preset` flag's value.
        steer: The `--steer` text, which rides the start as its initial steer.
        effort: The reasoning effort of this execution's role.
        reporter: Receives the refusal and the notes.

    Returns:
        The exit code of the run.

    Raises:
        preflight.SessionRefusedError: The config does not load or its role cannot run.
    """
    manifest = target.manifest
    effective = _load_config(
        target,
        config_paths,
        sandbox_overrides=sandbox_overrides,
        preset=preset,
        effort=effort,
        reporter=reporter,
    )
    _note_a_changed_driver(effective.config, target, reporter=reporter)
    # run_task takes both locks itself, so ours is released first; the pid is its own too.
    lock.release_single_writer(held.worker_lock_fd)
    held.worker_lock_fd = None
    held.handed_off = True
    return run.run_task(
        effective.config,
        manifest.parked_task,
        operator_task=manifest.parked_operator_task,
        frontend=frontend,
        cwd=target.repo,
        started_at=started_at,
        session_id=target.layout.session_id,
        config_paths=config_paths,
        mode=target.mode,
        budget_overrides=target.budget,
        sandbox_overrides=sandbox_overrides,
        preset=preset,
        model=target.route,
        explicit_leaves=effective.explicit_leaves,
        effective=effective,
        # The original stamp survives only for a flag-selected preset this resume
        # does not override; a config-selected one re-derives.
        preset_stamp=(
            (manifest.harness.preset, True)
            if (not preset and manifest.harness.preset_from_flag)
            else None
        ),
        # run_task resets the bridge and seeds the steer itself, with the steer's images.
        initial_steer=steer,
        reporter=reporter,
    )


def take_checkout(
    target: resume_record.ResumeTarget, *, reporter: app_reporter.Reporter
) -> int | None:
    """Take the checkout's writer lock for a run.

    Args:
        target: The session.
        reporter: Receives the refusal.

    Returns:
        The lock's file descriptor, released by the teardown; None outside run mode.

    Raises:
        preflight.SessionRefusedError: Another run drives the checkout.
    """
    # One run-mode worker per checkout; a fork's worktree never contends with the repo's.
    if target.mode != "run":
        return None
    fd, holder = lock.take_checkout_writer(target.cwd, target.layout.session_id)
    if fd is None:
        reporter.refuse(
            f"run {holder!r} is already driving this checkout; a"
            " second run-mode worker would interleave auto-commits on the"
            " one working tree. Wait for it, or stop it first:\n"
            f"    agent6 stop {holder}"
        )
        raise preflight.SessionRefusedError(2)
    return fd


def resume_start(
    target: resume_record.ResumeTarget,
    snapshot: loop_snapshot.SessionSnapshot,
    carried: resume_questions.CarriedQuestions,
    *,
    config_paths: tuple[pathlib.Path, ...],
    frontend: app_frontend.SessionFrontend,
    tui: bool,
    interactive: bool,
    sandbox_overrides: session_setup.SandboxOverrides | None,
    preset: str,
    effort: kinds.EffortLevel | None,
    reporter: app_reporter.Reporter,
) -> ResumeStart:
    """Load the config and resolve the policy this execution starts under.

    Args:
        target: The session.
        snapshot: The run's last snapshot, for the questions it already delivered.
        carried: The questions the journal carries over.
        config_paths: The `--config` files.
        frontend: The surface's seam.
        tui: Whether `--tui` was asked for.
        interactive: Whether the ask REPL drives the session.
        sandbox_overrides: The sandbox flags.
        preset: The `--preset` flag's value.
        effort: The reasoning effort of this execution's role.
        reporter: Receives the refusal and the notices.

    Returns:
        The config, the isolation, the commit identity and the channels.

    Raises:
        preflight.SessionRefusedError: The config, the route, the isolation, the git identity
            or the headless check refused.
    """
    repo, layout, manifest, cwd = target.repo, target.layout, target.manifest, target.cwd
    effective = _load_config(
        target,
        config_paths,
        sandbox_overrides=sandbox_overrides,
        preset=preset,
        effort=effort,
        reporter=reporter,
    )
    cfg = effective.config
    if worktree_setup.session_needs_setup(layout.session_dir):
        cfg = worktree_setup.read_if_unset(cfg, repo, reporter=reporter, at=manifest.base_sha)
    # Needs the config: the away grant is per scope, one per configured MCP server.
    app_frontend.settle_away_mode(layout.session_dir, cfg, can_ask=frontend.capabilities.can_ask)
    if not preflight.route_preflight(
        cfg, target.role, reporter=reporter, model_flag=session_setup.route_text(target.route)
    ):
        raise preflight.SessionRefusedError(2)
    if manifest.worktree is not None:
        cfg = cfg.with_hidden_in_checkout(repo, cwd)
    isolation = session_prepare.hold_workspace_and_select_isolation(
        cfg, cwd, frontend=frontend, reporter=reporter, worktree_git_dir=manifest.worktree_git_dir
    )
    identity = git_read.CommitIdentity(name=cfg.git.commit.name, email=cfg.git.commit.email)
    # The no-repo guard ran in `resume_record.read_snapshot`.
    if target.mode != "ask":
        try:
            git_read.verify_git_identity(cwd, identity)
        except errors.GitError as exc:
            reporter.error(str(exc))
            raise preflight.SessionRefusedError(2) from exc
    channels = session_prepare.operator_channels(
        layout, frontend, carried=carried, delivered=snapshot.delivered_questions
    )
    session_prepare.workspace_notices(cwd, cfg, reporter=reporter)
    tui_enabled = session_prepare.headless_check(
        cfg,
        frontend,
        tui=tui,
        interactive=interactive,
        mode=target.mode,
        # The raw env first, so a typo refuses here as on run; else the recorded choice.
        away=os.environ.get("AGENT6_DETACHED_AWAY", "") or ipc.effective_away(layout.session_dir),
        parking_away=ipc.effective_away(layout.session_dir),
        reporter=reporter,
    )
    return ResumeStart(
        effective=effective,
        cfg=cfg,
        isolation=isolation,
        identity=identity,
        channels=channels,
        tui_enabled=tui_enabled,
    )


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


def settle_tree(
    target: resume_record.ResumeTarget,
    start: ResumeStart,
    tree: resume_tree.TreeVerdict,
    held: session_prepare.Held,
    *,
    reporter: app_reporter.Reporter,
) -> tuple[str, ...]:
    """Apply `[git].dirty_tree` to the edits made outside the run.

    Args:
        target: The session.
        start: The config and the gate to the operator.
        tree: Whose the working tree's changes are.
        held: The teardown record, which takes the stash.
        reporter: Receives the refusal and the stash's note.

    Returns:
        The outside paths left in the tree for the run.

    Raises:
        preflight.SessionRefusedError: The operator cancelled, or the stash was refused.
    """
    manifest, prompts = target.manifest, start.channels.prompts
    settled = resume_tree.settle(
        target.cwd,
        tree,
        policy=start.cfg.git.dirty_tree,
        run_branch=manifest.run_branch,
        session_id=target.layout.session_id,
        ask=lambda question: prompts.ask((question,)).answers[0],
        reporter=reporter,
    )
    if settled.stashed:
        held.stash = session_prepare.AutoStash(
            target.cwd,
            base_branch=manifest.base_branch,
            run_branch=manifest.run_branch,
            auto_pop=start.cfg.git.auto_stash_pop or settled.pop,
            in_worktree=manifest.worktree is not None,
        )
        # After the record: the teardown ends the stash even when this line cannot be written.
        reporter.note(resume_tree.stashed_note(tree))
    return settled.kept


def merge_in(
    target: resume_record.ResumeTarget,
    base_sha: str,
    identity: git_read.CommitIdentity,
    *,
    merge_from: str,
    reporter: app_reporter.Reporter,
) -> git_ops.ChainMerge | None:
    """Merge the `--merge` branch into the run's chain, its conflicts left as markers.

    Args:
        target: The session.
        base_sha: The run's base, the merge's parent when the chain is unborn.
        identity: The author of the merge commit.
        merge_from: The branch; "" merges nothing.
        reporter: Receives the refusal and the merge's line.

    Returns:
        The merge, or None without a branch.

    Raises:
        preflight.SessionRefusedError: The session is no run, or git could not merge.
    """
    if not merge_from:
        return None
    cwd, session_id = target.cwd, target.layout.session_id
    if fork_worktrees.mirror_origin(target.repo, cwd) is not None:
        # The branch is the repository's; a clone's copy of it is as old as the clone.
        with contextlib.suppress(errors.GitError):
            git_ops.fetch_branch(
                cwd, target.repo, f"+refs/heads/{merge_from}:refs/heads/{merge_from}", sender=None
            )
    if target.mode != "run":
        reporter.refuse(f"--merge continues a run's work; {session_id} is a {target.mode}")
        raise preflight.SessionRefusedError(2)
    try:
        merged_in = git_ops.chain_merge_with_markers(
            cwd,
            merge_from,
            f"agent6: merge {merge_from} into the run's work",
            ref=git_read.chain_ref_for(session_id),
            fallback_parent=base_sha,
            identity=identity,
            also_branch=target.manifest.run_branch or None,
        )
    except errors.GitError as exc:
        reporter.error(f"cannot merge {merge_from} into this run's work: {exc}")
        raise preflight.SessionRefusedError(2) from exc
    reporter.note(resume_chain.merged_in_line(merged_in))
    return merged_in


def stamp_resume(
    target: resume_record.ResumeTarget,
    start: ResumeStart,
    tree: resume_tree.TreeVerdict,
    base_sha: str,
    *,
    preset: str,
    model: str,
    steer: str,
    reporter: app_reporter.Reporter,
) -> None:
    """Record the operator's choices for this execution, once every preflight passed.

    Args:
        target: The session.
        start: The config and policy the execution starts under.
        tree: Whose the working tree's changes are.
        base_sha: The run's base.
        preset: The `--preset` flag's value.
        model: The `--model` value.
        steer: The `--steer` text.
        reporter: Receives the notes.
    """
    layout, manifest, mode, cfg = target.layout, target.manifest, target.mode, start.cfg
    _note_a_changed_driver(cfg, target, reporter=reporter)
    # Every preflight passed: the operator's new choices are recorded from here.
    if preset:
        stamps.stamp_preset(layout.session_dir, preset)
    if (flagged := session_setup.flag_route(cfg, mode, model)) is not None:
        stamps.stamp_model(layout.session_dir, flagged)
    if target.budget.stamp() != manifest.budget:
        # Re-read: the stamps above rewrote the file. A later resume replays these caps.
        current = sessions_manifest.read_manifest(layout.session_dir)
        stamps.write_manifest(
            layout.manifest_path, current.model_copy(update={"budget": target.budget.stamp()})
        )
    if caps := target.budget.argv():
        reporter.note(f"budget caps in force from this run's flags: {' '.join(caps)}")
    if steer.strip() and manifest.parent_session_id:
        stamps.stamp_fork_task(
            layout.session_dir,
            steer.strip(),
            source_dir=layout.session_dir.parent / manifest.parent_session_id,
        )
    else:
        stamps.stamp_continuation(layout.session_dir, steer.strip())
    # This execution's models and policy, for `agent6 exec` and the policy surfaces.
    stamps.stamp_execution(
        layout.session_dir,
        cfg,
        mode,
        start.isolation,
        # An effort this execution was started with reads as its own, not a layer's.
        config_sources={**start.effective.sources, **start.effective.applied_sources},
    )
    if tree.carried:
        note = f"uncommitted from this run's last execution: {resume_tree.first_four(tree.carried)}"
        reporter.note(note)
        start.channels.events.emit(journal.LoopNote(text=note))
    if mode == "run" and cfg.git.control != "model":
        _note_uncommitted_changes(target, tree, base_sha, reporter=reporter)


def _note_uncommitted_changes(
    target: resume_record.ResumeTarget,
    tree: resume_tree.TreeVerdict,
    base_sha: str,
    *,
    reporter: app_reporter.Reporter,
) -> None:
    """Name the changes no commit has, which the next auto-commit takes."""
    # The previous execution's tail and any operator edit since; the operator's untracked
    # files stay out.
    with contextlib.suppress(errors.GitError, OSError):
        dirty = git_read.chain_dirty_paths(
            target.cwd,
            git_read.chain_ref_for(target.layout.session_id),
            base_sha,
            5,
            exclude=tree.untracked_at_start | tree.hidden,
        )
        if dirty:
            shown = map(terminal.printable_name, dirty[:4])
            named = ", ".join(shown) + (", ..." if len(dirty) > 4 else "")
            reporter.note(
                f"the tree holds changes no commit has ({named});"
                " this execution's next commit takes them"
            )


def resume_inputs(
    target: resume_record.ResumeTarget,
    snapshot: loop_snapshot.SessionSnapshot,
    start: ResumeStart,
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
                return start_parked(
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
            held.repo_lock_fd = take_checkout(target, reporter=reporter)
            snapshot = resume_record.read_snapshot(target, reporter=reporter)
            merged = resume_record.check_chain(target, force=force, reporter=reporter)
            start = resume_start(
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
            kept = settle_tree(target, start, tree, held, reporter=reporter)
            base_sha = resume_chain.move_base(
                target.cwd,
                layout,
                target.manifest,
                merged,
                start.identity,
                mode=target.mode,
                reporter=reporter,
            )
            merged_in = merge_in(
                target, base_sha, start.identity, merge_from=merge_from, reporter=reporter
            )
            stamp_resume(
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
