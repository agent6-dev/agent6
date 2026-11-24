# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Prepare the checkout, config and channels for a resumed execution."""

from __future__ import annotations

import contextlib
import dataclasses
import os
import pathlib

from agent6 import errors, kinds, terminal
from agent6.app import (
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
from agent6.config import layer_data, validation
from agent6.config import model as config_model
from agent6.harness import loop_snapshot
from agent6.sessions import ipc, journal, lock
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
    cfg: config_model.Config
    isolation: kinds.IsolationLevel
    identity: git_read.CommitIdentity
    channels: session_prepare.Channels
    tui_enabled: bool


def load_config(
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
    except validation.ConfigError as exc:
        reporter.error(str(exc))
        raise preflight.SessionRefusedError(2) from exc


def note_changed_driver(
    cfg: config_model.Config, target: resume_record.ResumeTarget, *, reporter: app_reporter.Reporter
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
    effective = load_config(
        target,
        config_paths,
        sandbox_overrides=sandbox_overrides,
        preset=preset,
        effort=effort,
        reporter=reporter,
    )
    note_changed_driver(effective.config, target, reporter=reporter)
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
    effective = load_config(
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
    note_changed_driver(cfg, target, reporter=reporter)
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
        note_uncommitted_changes(target, tree, base_sha, reporter=reporter)


def note_uncommitted_changes(
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
