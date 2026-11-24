# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The `agent6 run` lifecycle and its plan and ask modes.

Preflight, branch cut, manifest, loop construction, finalize. Everything that
touches the terminal is injected through `SessionFrontend`, so this module never
imports `agent6.ui`.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import os
import pathlib
import shutil
from collections.abc import Callable, Sequence
from typing import NoReturn

from agent6 import budget as agent6_budget
from agent6 import directive, errors, kinds, pack
from agent6.app import (
    execution,
    fork_worktrees,
    preflight,
    run_preflight,
    stamps,
)
from agent6.app import frontend as app_frontend
from agent6.app import reporter as app_reporter
from agent6.app.session import prepare as session_prepare
from agent6.app.session import setup as session_setup
from agent6.config import Config, layer_data
from agent6.sessions import ipc, journal, lock, process
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


def _dirty_tree(
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


def _seed_steer(
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
        _seed_steer(layout, initial_steer, reporter=reporter)
    app_frontend.settle_away_mode(layout.session_dir, cfg, can_ask=frontend.capabilities.can_ask)
    # A run that would have to ask about tracked changes but cannot refuses here, before the
    # manifest.
    dirty = _dirty_tree(start, frontend=frontend, reporter=reporter)
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


def _settle_dirty_tree(
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


def _settle_checkout(
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
        _settle_dirty_tree(start, held, dirty, prompts, reporter=reporter)
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
        _settle_checkout(start, held, opened.dirty, opened.channels.prompts, reporter=reporter)
    if opened.parked_reason is not None:
        why = f" ({opened.parked_reason})" if opened.parked_reason else ""
        reporter.note(f"run {layout.session_id!r} was parked at submission{why}; starting it now.")


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
    opened: Opened,
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
    opened: Opened,
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
            opened = open_run(
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
            settle_run_tree(start, held, opened, reporter=reporter)
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
        execution.detach_to_background(
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
