# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Open the session record and settle the working tree under the writer lock."""

from __future__ import annotations

import contextlib
import dataclasses
import pathlib
import shutil
from typing import NoReturn

from agent6 import directive, errors, kinds, pack
from agent6.app import frontend as app_frontend
from agent6.app import (
    preflight,
    run_preflight,
    stamps,
)
from agent6.app import reporter as app_reporter
from agent6.app.session import prepare as session_prepare
from agent6.app.session import setup as session_setup
from agent6.config import layer_data
from agent6.sessions import ipc, journal, lock
from agent6.sessions import layout as sessions_layout
from agent6.sessions import manifest as sessions_manifest
from agent6.tools import operator_prompts
from agent6.vcs.git import ops as git_ops
from agent6.vcs.git import read as git_read
from agent6.viewmodel.session import log as session_log


def discard_husk_dir(session_dir: pathlib.Path) -> None:
    """Remove a session dir a preflight refused before a manifest or log was written."""
    if sessions_layout.session_has_record(session_dir):
        return
    with contextlib.suppress(OSError):
        shutil.rmtree(session_dir)


@dataclasses.dataclass(frozen=True, slots=True)
class DirtyTree:
    """The operator's uncommitted changes to tracked files at a run's start.

    Attributes:
        modified: The changed tracked paths outside the hidden ones; empty outside run mode.
        hidden: The run's hidden paths inside the repository.
        unmerged_run: The earlier run whose unmerged work the changes match; "" for none.
    """

    modified: tuple[str, ...]
    hidden: frozenset[str]
    unmerged_run: str


@dataclasses.dataclass(frozen=True, slots=True)
class Opened:
    """The session a fresh run opened.

    Attributes:
        channels: The journal, the transcripts and the gate to the operator.
        dirty: The operator's uncommitted changes to tracked files.
        parked_reason: Why the submission this start takes up was parked; None for a new one.
        child_gate_source: The origin of the gate a child's config carries; "" outside a child.
    """

    channels: session_prepare.Channels
    dirty: DirtyTree
    parked_reason: str | None
    child_gate_source: sessions_manifest.GateSource


def read_dirty_tree(
    start: run_preflight.RunStart,
    *,
    frontend: app_frontend.SessionFrontend,
    reporter: app_reporter.Reporter,
) -> DirtyTree:
    """Read the operator's uncommitted changes to tracked files.

    Args:
        start: The start in its checkout.
        frontend: The surface, which says whether it can ask.
        reporter: Receives the refusal.

    Returns:
        The changes outside the hidden paths, and the earlier run they match.

    Raises:
        preflight.SessionRefusedError: The run would have to ask about them and cannot.
    """
    cfg, cwd, layout = start.cfg, start.cwd, start.layout
    hidden = git_read.hidden_in_repo(cwd, cfg.sandbox.hide_paths)
    # A hidden path's change is the operator's business: never asked about, never stashed.
    modified = (
        [p for p in git_read.modified_paths(cwd) if not git_read.excluded(p, hidden)]
        if start.mode == "run"
        else []
    )
    must_ask = bool(modified) and cfg.git.dirty_tree == "ask"
    answerable = frontend.capabilities.can_ask or ipc.away_mode(layout.session_dir) == "wait"
    unmerged_run = (
        preflight.unmerged_run_holding_the_tree(
            cwd, start.state, except_id=layout.session_id, modified=modified
        )
        if modified and cfg.git.dirty_tree != "stash"
        else ""
    )
    if must_ask and not answerable:
        reporter.refuse(preflight.dirty_tree_refusal(modified, unmerged_run=unmerged_run))
        discard_husk_dir(layout.session_dir)
        raise preflight.SessionRefusedError(2)
    return DirtyTree(modified=tuple(modified), hidden=hidden, unmerged_run=unmerged_run)


def seed_steer(
    layout: sessions_layout.SessionLayout, steer: str, *, reporter: app_reporter.Reporter
) -> None:
    """Queue the steer a parked resume hands the start, with the images sent with it.

    Args:
        layout: The session's directory layout.
        steer: The steer as the operator gave it.
        reporter: Receives the refusal.

    Raises:
        preflight.SessionRefusedError: The steer could not be written.
    """
    # The steer as a front-end recorded it before it launched the resume: its images and its id.
    launch = ipc.take_launch_input(layout.session_dir, steer)
    if not ipc.submit_steer(
        layout.session_dir,
        steer.strip(),
        images=launch.images,
        message_id=launch.message_id,
        # A plain steer takes the free slot; one with an id cannot, so it takes the inbox's head.
        first=bool(launch.message_id),
    ):
        reporter.error("could not write the initial steer request")
        raise preflight.SessionRefusedError(2)


def open_run(
    start: run_preflight.RunStart,
    isolation: kinds.IsolationLevel,
    *,
    frontend: app_frontend.SessionFrontend,
    started_at: float,
    source_session_id: str | None,
    metric: directive.MetricSpec | None,
    budget_overrides: session_setup.BudgetOverrides | None,
    preset: str,
    initial_steer: str,
    preset_stamp: tuple[str, bool] | None,
    model: str | kinds.ModelRoute | None,
    effective: layer_data.EffectiveConfig | None,
    reporter: app_reporter.Reporter,
) -> Opened:
    """Open the session: reset its bridge, open its channels and write its manifest.

    Args:
        start: The start in its checkout.
        isolation: The isolation its commands run under.
        frontend: The surface's seam.
        started_at: The instant this execution began; the stale-state clear keeps bridge
            markers written since.
        source_session_id: The plan session a `run --from` came from.
        metric: The operator's metric for this run, or None.
        budget_overrides: The budget flags.
        preset: The `--preset` flag's value.
        initial_steer: An operator follow-up queued for the loop's first boundary.
        preset_stamp: `(name, from_flag)` to stamp in place of deriving from `agent6.preset`.
        model: The `--model` value that set the driving role.
        effective: The config and its file origins before execution policy.
        reporter: Receives the refusal and the notices.

    Returns:
        The channels and what the start found.

    Raises:
        preflight.SessionRefusedError: The steer could not be written, or the run would have
            to ask about tracked changes and cannot.
    """
    layout, cfg, mode = start.layout, start.cfg, start.mode
    # A reused dir carries the previous execution's bridge state.
    ipc.clear_pending_answers(layout.session_dir, started_at=started_at)
    if initial_steer.strip():
        seed_steer(layout, initial_steer, reporter=reporter)
    app_frontend.settle_away_mode(layout.session_dir, cfg, can_ask=frontend.capabilities.can_ask)
    # A run that would have to ask about tracked changes but cannot refuses here, before the
    # manifest.
    dirty = read_dirty_tree(start, frontend=frontend, reporter=reporter)
    channels = session_prepare.operator_channels(layout, frontend)
    session_prepare.workspace_notices(start.cwd, cfg, reporter=reporter)
    # Written before the gates of `settle_run_tree`, which park rather than refuse; the
    # rewrite keeps the park, which the execution's start clears.
    parked = stamps.parked_stamp(layout.session_dir)
    # A child's config carries the gate its parent resolved, origin and all.
    child_gate_source = stamps.child_gate_source()
    review_world = start.review_world
    stamps.write_session_manifest(
        layout,
        session_id=layout.session_id,
        source_session_id=source_session_id,
        user_task=start.task,
        operator_task=start.operator_task,
        base_sha=start.base_sha,
        base_branch=start.base_branch,
        run_branch=start.run_branch,
        cfg=cfg,
        # An effort this execution was started with reads as its own, not a layer's.
        config_sources=({**effective.sources, **effective.applied_sources} if effective else None),
        mode=mode,
        gate=(
            sessions_manifest.GateStamp(
                command=cfg.harness.verify.command,
                source=child_gate_source,
                availability="pending",
            )
            if child_gate_source
            else None
        ),
        effective_preset=(preset_stamp[0] if preset_stamp else (preset or cfg.agent6.preset)),
        preset_from_flag=(preset_stamp[1] if preset_stamp else bool(preset)),
        driver_from_flag=bool(model),
        isolation=isolation,
        worktree=start.cwd if start.worktree else None,
        worktree_git_dir=start.worktree_git_dir,
        review_base_tree=review_world.target.base_tree if review_world is not None else "",
        budget=budget_overrides.stamp() if budget_overrides is not None else None,
    )
    if metric is not None and mode == "run":
        previous = session_log.read_metric(layout.logs_path) if parked else None
        channels.events.emit(
            journal.MetricSet(revision=previous.revision + 1 if previous else 1, spec=metric.text())
        )
    if parked is not None:
        stamps.stamp_parked(
            layout.session_dir, task=parked[0], reason=parked[1], operator_task=parked[2]
        )
    return Opened(
        channels=channels,
        dirty=dirty,
        parked_reason=parked[1] if parked is not None else None,
        child_gate_source=child_gate_source,
    )


def park(
    start: run_preflight.RunStart,
    reason: str,
    detail: str,
    *,
    hint: str = "",
    reporter: app_reporter.Reporter,
) -> NoReturn:
    """Save the submission as a parked run, and refuse the start.

    Args:
        start: The start whose task is saved.
        reason: The short cause listings show.
        detail: The sentence read now.
        hint: Further ways to run the task, printed under the resume line.
        reporter: Receives the park.

    Raises:
        preflight.SessionRefusedError: Always, with exit 2.
    """
    session_id = start.layout.session_id
    stamps.stamp_parked(
        start.layout.session_dir,
        task=start.operator_task if start.review_world is not None else start.task,
        reason=reason,
        operator_task=start.operator_task,
    )
    # No loop read the steer this start was given: it waits with the task, in the inbox.
    ipc.demote_steer(start.layout.session_dir)
    reporter.err(
        f"PARKED: {detail}. Your task is saved as run {session_id!r}:\n"
        f"    agent6 resume {session_id}    (starts it)" + (f"\n{hint}" if hint else "")
    )
    raise preflight.SessionRefusedError(2)


def settle_dirty_tree(
    start: run_preflight.RunStart,
    held: session_prepare.Held,
    dirty: DirtyTree,
    prompts: operator_prompts.OperatorPrompts,
    *,
    reporter: app_reporter.Reporter,
) -> None:
    """Apply `[git].dirty_tree` to the operator's changes, asking when the policy says so.

    Args:
        start: The start in its checkout.
        held: The teardown record, which takes the stash.
        dirty: The operator's uncommitted changes.
        prompts: The gate to the operator.
        reporter: Receives a park.

    Raises:
        preflight.SessionRefusedError: The operator cancelled, a bare `/review` would stash
            what it reviews, or the stash failed; each parks the run.
    """
    cfg, review_world = start.cfg, start.review_world
    # Whether the stash is applied back at run end: config, or "stash" chosen at the start.
    stash_pop = cfg.git.auto_stash_pop
    choice: preflight.DirtyTreeChoice
    if cfg.git.dirty_tree == "stash":
        choice = "stash"
    elif cfg.git.dirty_tree == "include":
        choice = "include"
    else:
        question = preflight.dirty_tree_question(dirty.modified, unmerged_run=dirty.unmerged_run)
        choice = preflight.dirty_tree_choice(prompts.ask((question,)).answers[0])
        stash_pop = stash_pop or choice == "stash"
    if choice == "cancel":
        settle = (
            f"`agent6 sessions merge {dirty.unmerged_run}` merges them (they match the"
            f" unmerged work of run {dirty.unmerged_run})"
            if dirty.unmerged_run
            else "commit or stash them"
        )
        park(
            start,
            "uncommitted changes",
            f"the working tree has uncommitted changes to tracked files; {settle},"
            " then start the run",
            reporter=reporter,
        )
    if choice != "stash":
        return
    if review_world is not None and review_world.target.source == "worktree":
        park(
            start,
            "review needs uncommitted changes",
            "a bare /review cannot stash the changes it reviews; use agent6 ask "
            '"/review", or set git.dirty_tree to include and retry',
            reporter=reporter,
        )
    try:
        git_ops.stash_tracked_changes(
            start.cwd,
            git_ops.auto_stash_message(start.layout.session_id),
            paths=dirty.modified if dirty.hidden else (),
        )
    except errors.GitError as exc:
        park(
            start,
            "stash failed",
            f"stashing the working tree's changes failed: {exc}",
            reporter=reporter,
        )
    held.stash = session_prepare.AutoStash(
        start.cwd,
        base_branch=start.base_branch,
        run_branch=start.run_branch,
        auto_pop=stash_pop,
        in_worktree=start.worktree,
    )


def settle_checkout(
    start: run_preflight.RunStart,
    held: session_prepare.Held,
    dirty: DirtyTree,
    prompts: operator_prompts.OperatorPrompts,
    *,
    reporter: app_reporter.Reporter,
) -> None:
    """Take a run's checkout and settle what its commits leave out.

    Args:
        start: The start in its checkout.
        held: The teardown record, which takes the lock, the stash and the untracked files.
        dirty: The operator's uncommitted changes.
        prompts: The gate to the operator.
        reporter: Receives a park and the notes.

    Raises:
        preflight.SessionRefusedError: Another run drives the checkout, or the operator's
            changes could not be settled; each parks the run.
    """
    cwd, layout = start.cwd, start.layout
    # One run-mode worker per checkout, taken before any tree mutation.
    held.repo_lock_fd, holder = lock.take_checkout_writer(cwd, layout.session_id)
    if held.repo_lock_fd is None:
        park(
            start,
            "checkout busy",
            f"run {holder!r} is already driving this checkout, and a second run-mode"
            " worker would interleave auto-commits on the one working tree",
            hint=(
                "or run it now in its own worktree, beside the live run:\n"
                '    agent6 run --worktree "<the same task>"\n'
                f"or hand it to the live run as an isolated child by steering"
                f" {holder!r} with:\n    /parallel 1 -- <the same task>"
            ),
            reporter=reporter,
        )
    # Settle the operator's uncommitted changes before the first commit can sweep them up.
    if dirty.modified:
        settle_dirty_tree(start, held, dirty, prompts, reporter=reporter)
    # agent6's own directory stays out of the run's commits, a pack a model writes too.
    try:
        excluded = pack.keep_out_of_git(cwd)
    except (errors.PackError, errors.GitError) as exc:
        reporter.warn(f"could not exclude .agent6/ from git: {exc}")
    else:
        if excluded is not None:
            reporter.note(f"excluded .agent6/ in {excluded}; .gitignore is untouched")
    # The operator's files, untracked after any stash; every commit leaves them out.
    held.untracked_at_start = git_read.untracked_paths(cwd)
    sessions_layout.write_untracked_at_start(layout.session_dir, held.untracked_at_start)
    if not dirty.unmerged_run or held.stash is not None:
        return
    # A merge includes the earlier run only if its entire tip matches this tree.
    included = preflight.included_run(
        cwd,
        dirty.unmerged_run,
        base_sha=start.base_sha,
        exclude=held.untracked_at_start | dirty.hidden,
    )
    if included is not None:
        current = sessions_manifest.read_manifest(layout.session_dir)
        stamps.write_manifest(
            layout.manifest_path, current.model_copy(update={"includes": included})
        )


def settle_run_tree(
    start: run_preflight.RunStart,
    held: session_prepare.Held,
    opened: Opened,
    *,
    reporter: app_reporter.Reporter,
) -> None:
    """Settle the checkout, before the first commit can sweep anything up.

    Args:
        start: The start in its checkout.
        held: The teardown record, which takes the checkout lock, the stash and the
            operator's untracked files.
        opened: The session's channels and the operator's uncommitted changes.
        reporter: Receives a park and the notes.

    Raises:
        preflight.SessionRefusedError: The start parked: the checkout is busy, or the
            operator's changes could not be settled.
    """
    layout = start.layout
    if start.mode == "run":
        settle_checkout(start, held, opened.dirty, opened.channels.prompts, reporter=reporter)
    if opened.parked_reason is not None:
        why = f" ({opened.parked_reason})" if opened.parked_reason else ""
        reporter.note(f"run {layout.session_id!r} was parked at submission{why}; starting it now.")
