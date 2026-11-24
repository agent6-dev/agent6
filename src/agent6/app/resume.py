# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The `agent6 resume` lifecycle.

Picks a paused or crashed session back up from its snapshot, over the same
`SessionFrontend` seam `run_task` uses.
"""

from __future__ import annotations

import contextlib
import json
import os
import pathlib
from collections.abc import Callable, Collection

from agent6 import budget, directive, errors, kinds, terminal
from agent6 import paths as agent6_paths
from agent6.app import (
    child_launch,
    child_resume,
    execution,
    finalize,
    fork_worktrees,
    preflight,
    resume_chain,
    resume_questions,
    resume_tree,
    run,
    stamps,
    worktree_setup,
)
from agent6.app import frontend as app_frontend
from agent6.app import reporter as app_reporter
from agent6.app.session import prepare as session_prepare
from agent6.app.session import setup as session_setup
from agent6.config import (
    Config,
    ConfigError,
)
from agent6.harness import loop_snapshot, repo_context
from agent6.sessions import child_dispatch, id, ipc, journal, lock, process
from agent6.sessions import layout as sessions_layout
from agent6.sessions import manifest as sessions_manifest
from agent6.vcs.git import ops as git_ops
from agent6.vcs.git import read as git_read
from agent6.viewmodel import listing
from agent6.viewmodel.session import log as session_log


def resumable_bucket_dirs(state_dir: pathlib.Path) -> list[pathlib.Path]:
    """List the bucket dirs holding sessions `agent6 resume` can pick up.

    Args:
        state_dir: The repo's state directory.

    Returns:
        One dir per resumable session kind.
    """
    return [
        sessions_layout.bucket_dir(state_dir, kinds.session_bucket(kind.name))
        for kind in kinds.SESSION_KINDS.values()
        if kind.resumable
    ]


def _paths_the_run_wrote(logs_path: pathlib.Path) -> frozenset[str]:
    """Collect every path a `tool.result` of the run names as written.

    Args:
        logs_path: The run's event log.

    Returns:
        The paths over every execution; a torn line contributes none.
    """
    paths: set[str] = set()
    try:
        with logs_path.open(encoding="utf-8", errors="replace") as lines:
            for line in lines:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if isinstance(event, dict) and event.get("type") == "tool.result":
                    named = event.get("paths", ())
                    if isinstance(named, list):
                        paths.update(str(p) for p in named)
    except OSError:
        return frozenset()
    return frozenset(paths)


def _first_four(paths: Collection[str]) -> str:
    """Return the first four paths in order, then how many more there are."""
    named = sorted(paths)
    more = f", +{len(named) - 4} more" if len(named) > 4 else ""
    return ", ".join(map(terminal.printable_name, named[:4])) + more


def covering_stamp(
    repo: pathlib.Path, manifest: sessions_manifest.SessionManifest
) -> sessions_manifest.MergeStamp | None:
    """Return the merge stamp when it covers every commit the run made.

    A resumed run commits past its stamp, and no merge covers those commits.

    Args:
        repo: The repository.
        manifest: The session's manifest.

    Returns:
        The stamp, or None when it covers less than the chain.
    """
    stamp = manifest.merged
    if stamp is None or not stamp.into:
        return None
    holds = git_read.merge_stamp_holds(
        repo, manifest.session_id, manifest.run_branch or "", stamp.tip
    )
    return stamp if holds else None


def commits_note(repo: pathlib.Path, manifest: sessions_manifest.SessionManifest) -> str:
    """Say where a session's commits are, for a refusal that points at them.

    Args:
        repo: The repository.
        manifest: The session's manifest.

    Returns:
        The run branch, the merge that included them, the chain ref, or "it recorded
        no commits".
    """
    if manifest.run_branch and git_read.branch_exists(repo, manifest.run_branch):
        return f"its commits are on {manifest.run_branch}"
    stamp = covering_stamp(repo, manifest)
    if stamp is not None:
        return f"its commits are {stamp.description()}"
    chain = git_read.chain_ref_for(manifest.session_id)
    if git_read.chain_tip(repo, chain) is not None:
        return f"its commits are on {chain}"
    return "it recorded no commits"


def snapshot_head_mismatch(
    snapshot_path: pathlib.Path, repo_root: pathlib.Path, *, chain_ref: str
) -> tuple[str, str] | None:
    """Detect a chain tip that diverged from the run's last snapshot.

    Forward movement on the same line is the run's own per-turn commits and
    resumes cleanly; only a tip that is not a descendant of the snapshot head
    counts. Committed history only. A snapshot with no recorded head, a corrupt
    snapshot or an unborn ref is skipped.

    Args:
        snapshot_path: The run's snapshot.
        repo_root: The checkout.
        chain_ref: The chain ref resume commits on top of.

    Returns:
        `(snapshot head, resume-onto head)` on divergence, else None.
    """
    snap_head = ""
    with contextlib.suppress(OSError, ValueError):
        loaded = json.loads(snapshot_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            # A raw peek that must not raise; the key is `SessionSnapshot.head_sha`.
            snap_head = str(loaded.get("head_sha") or "")
    if not snap_head:
        return None
    try:
        current_head = git_read.chain_tip(repo_root, chain_ref)
    except errors.GitError:
        return None
    if current_head is None or current_head == snap_head:
        return None
    if git_read.is_ancestor(repo_root, snap_head, current_head):
        # The tip moved forward on the same line: the run's own commits.
        return None
    return (snap_head, current_head)


def resume_task(  # noqa: C901, PLR0911, PLR0912, PLR0915  # every way a resume is refused, in preflight order
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
    launch = child_launch.read()
    # A continued child writes to its bound record but executes with its clone's state.
    if launch is not None and launch.continuation:
        try:
            state = launch.checked_clone(repo).state
        except child_launch.LaunchError as exc:
            reporter.refuse(str(exc))
            return 2
        record = launch.record_state
    else:
        record = state = agent6_paths.state_dir(repo)
    if steer.strip() and (problem := directive.steer_problem(steer)) is not None:
        reporter.error(problem)
        return 2
    # One resolver across buckets: a runs/-only fallback would pick a run over an ask.
    try:
        layout = id.resolve_session(record, session_id)
    except id.SessionIdError as exc:
        reporter.error(str(exc))
        return 2
    session_id = layout.session_id
    # The manifest first: clearing state before it would clobber a session resume refuses.
    try:
        manifest = sessions_manifest.read_manifest(layout.session_dir)
        # A resume keeps the recorded route unless it sets its own.
        recorded = manifest.models.driver
        route: str | kinds.ModelRoute | None = model or (
            kinds.ModelRoute(recorded.provider, recorded.model) if recorded is not None else None
        )
        mode = manifest.session_mode()
        # A `--max-*` set on the run replays the same way, field by field.
        budget_overrides = (budget_overrides or session_setup.BudgetOverrides()).over(
            manifest.budget
        )
    except sessions_manifest.ManifestError as exc:
        reporter.error(f"cannot resume {session_id}: {exc}")
        return 2
    owned_child = child_resume.authorized(launch, manifest, session_id=session_id, root=repo)
    if launch is not None and launch.continuation and not owned_child:
        reporter.refuse("the private launch does not authorize this child continuation")
        return 2
    if manifest.parallel is not None and not owned_child:
        reporter.refuse(listing.manifest_resume_blocker(manifest, session_id))
        return 2
    cwd = launch.worktree if owned_child and launch is not None else manifest.worktree or repo
    if missing := sessions_manifest.worktree_blocker(manifest.worktree):
        reporter.error(
            f"cannot resume {session_id}: {missing}; {commits_note(repo, manifest)};"
            f" `agent6 fork {session_id}` continues it in a new worktree."
        )
        return 2
    if (
        manifest.worktree is not None
        and manifest.worktree_git_dir is None
        and not fork_worktrees.is_clone(manifest.worktree)
    ):
        reporter.error(
            f"cannot resume {session_id}: its manifest names a worktree but not the repository"
            f" git dir it points into, so its jail cannot grant one; `agent6 fork {session_id}`"
            " continues its commits in a new worktree."
        )
        return 2
    # One writer per session dir, taken before any shared state is touched.
    try:
        worker_lock_fd: int | None = session_prepare.take_single_writer(layout, reporter=reporter)
    except preflight.SessionRefusedError as refusal:
        return refusal.rc
    if fork_worktrees.mirror_origin(repo, cwd) is not None:
        # The repository's refs are the record between executions: a fork's seed, a merge.
        try:
            git_ops.mirror_session(
                cwd, repo, git_read.chain_ref_for(session_id), manifest.run_branch or None
            )
        except errors.GitError as exc:
            reporter.error(
                f"cannot resume {session_id}: its worktree {cwd} has no copy of its"
                f" commits and git could not fetch them: {exc}"
            )
            lock.release_single_writer(worker_lock_fd)
            return 1
    # A finished run has nothing to continue without --steer; every other ending resumes.
    scan = session_log.scan_session_log(layout.logs_path)
    new_work = listing.needs_new_work(
        finished=scan.finished,
        stop_reason=scan.stop_reason,
        verification=scan.verification,
        saved_answers=len(ipc.saved_question_answers(layout.session_dir)),
    )
    if scan.finished and scan.stop_reason == "undone":
        # The fork is the continuation; resuming here would redo the taken-back edit.
        reporter.refuse(listing.undone_refusal(session_id, scan.undone_to))
        lock.release_single_writer(worker_lock_fd)
        return 2
    if not steer.strip() and not merge_from and new_work:
        reporter.refuse(listing.needs_new_work_refusal(session_id))
        lock.release_single_writer(worker_lock_fd)
        return 2
    carried = resume_questions.carried_questions(layout.logs_path)
    # Approval ids reset on resume, so their stale answers go; a pending question's stays.
    ipc.clear_pending_answers(
        layout.session_dir,
        started_at=started_at,
        keep_questions=[request.id for request in carried.pending],
        keep_steer=scan.stop_reason == "setup_command_failed",
    )
    # The images a front-end stored for this steer before it launched the resume.
    images = ipc.take_launch_images(layout.session_dir, steer)
    if steer.strip() and not ipc.submit_steer(layout.session_dir, steer, images=images):
        reporter.error("could not write the initial steer request")
        lock.release_single_writer(worker_lock_fd)
        return 2

    detach_requested = False
    stashed = resume_tree.Settled()  # the outside edits this execution set aside
    untracked_at_start: frozenset[str] = frozenset()
    handed_to_run_task = False  # a parked submission: run_task owns its lifecycle
    cfg: Config | None = None  # the finally reads it for a detach
    repo_lock_fd: int | None = None
    try:
        role = kinds.session_kind(mode).role

        if manifest.parked_task:
            # Nothing ever ran: run_task starts it fresh under the same id and takes both
            # locks itself, so ours is released first.
            try:
                # replay_preset: a config-selected preset re-resolves rather than ranking as a flag.
                effective = session_setup.load_session_config(
                    repo,
                    config_paths,
                    mode=mode,
                    preset=preset or manifest.harness.replay_preset,
                    budget_overrides=budget_overrides,
                    sandbox_overrides=sandbox_overrides,
                    model=route,
                    effort=effort,
                )
                cfg = effective.config
            except ConfigError as exc:
                reporter.error(str(exc))
                return 2
            driver = cfg.models.resolve(role)
            if (
                recorded is not None
                and driver is not None
                and (driver.provider, driver.model) != (recorded.provider, recorded.model)
            ):
                reporter.note(
                    f"resuming with {driver.provider}/{driver.model}"
                    f" (recorded: {recorded.provider}/{recorded.model})"
                )
            saved_task = manifest.parked_task
            lock.release_single_writer(worker_lock_fd)
            worker_lock_fd = None
            handed_to_run_task = True
            return run.run_task(
                cfg,
                saved_task,
                operator_task=manifest.parked_operator_task,
                frontend=frontend,
                cwd=repo,
                started_at=started_at,
                session_id=session_id,
                config_paths=config_paths,
                mode=mode,
                budget_overrides=budget_overrides,
                sandbox_overrides=sandbox_overrides,
                preset=preset,
                model=route,
                explicit_leaves=effective.explicit_leaves,
                effective=effective,
                # The original stamp survives only for a flag-selected preset this resume
                # does not override; a config-selected one re-derives.
                preset_stamp=(
                    (manifest.harness.preset, True)
                    if (not preset and manifest.harness.preset_from_flag)
                    else None
                ),
                # run_task seeds its own initial steer over the files seeded above.
                initial_steer=steer,
                reporter=reporter,
            )

        # One run-mode worker per checkout; a fork's worktree never contends with the repo's.
        if mode == "run":
            repo_lock_fd, holder = lock.take_checkout_writer(cwd, session_id)
            if repo_lock_fd is None:
                reporter.refuse(
                    f"run {holder!r} is already driving this checkout; a"
                    " second run-mode worker would interleave auto-commits on the"
                    " one working tree. Wait for it, or stop it first:\n"
                    f"    agent6 stop {holder}"
                )
                return 2
        if manifest.worktree is not None:
            reporter.note(f"working in the worktree {cwd}")

        snapshot_path = layout.session_dir / "loop_state.json"
        if not snapshot_path.is_file():
            reporter.error(
                f"{session_id} never reached its first turn: no resume snapshot was written,"
                " so there is nothing to resume. Start the task again with a new run."
            )
            return 2

        # An ask is read-only and may run outside a git repo, so it skips the git preflight.
        writes_code = mode != "ask"
        # The no-repo guard runs before any git-touching check.
        if writes_code and not preflight.require_git_repo(cwd, reporter=reporter):
            return 2

        # An unreadable or outdated snapshot refuses here, before anything is touched.
        try:
            snapshot = loop_snapshot.load_session_snapshot(snapshot_path)
        except (ValueError, OSError) as exc:
            reporter.error(str(exc))
            return 1
        # The loop settles children from these records at every turn boundary.
        try:
            child_dispatch.read_records(layout.session_dir)
        except child_dispatch.DispatchRecordError as exc:
            reporter.error(
                f"{session_id} cannot be resumed: a dispatch record is unreadable."
                f" Start a new run.\n{exc}"
            )
            return 1
        manifest_preset = manifest.harness.replay_preset
        resume_base_sha = manifest.base_sha
        run_branch = manifest.run_branch or ""

        # A rewritten chain ref would leave the model reasoning about a changed record.
        mismatch = (
            snapshot_head_mismatch(snapshot_path, cwd, chain_ref=git_read.chain_ref_for(session_id))
            if writes_code
            else None
        )
        if mismatch is not None:
            snap_head, onto_head = mismatch
            reporter.err(
                "GUARD: the code this run would resume onto diverged from its last snapshot."
            )
            reporter.err(f"  snapshot head: {snap_head}")
            reporter.err(f"  resume onto:   {onto_head}")
            if not force:
                reporter.err("REFUSING to resume. Re-run with --force to override.")
                return 2

        # A merged run can continue from HEAD. Otherwise, check for commits off the chain.
        merged: resume_chain.MergedStart | None = None
        if mode == "run":
            try:
                merged = resume_chain.merged_start(cwd, manifest)
                off = None if merged else resume_chain.branch_off_the_chain(cwd, manifest)
            except errors.GitError as exc:
                reporter.error(str(exc))
                return 2
            if off is not None:
                branch, branch_tip, ahead = off
                if not force:
                    reporter.refuse(
                        resume_chain.off_chain_refusal(session_id, branch, branch_tip, ahead=ahead)
                    )
                    return 2
                reporter.note(
                    f"continuing on the chain; {branch} stays at {branch_tip[:12]}, off the run"
                )

        try:
            effective = session_setup.load_session_config(
                repo,
                config_paths,
                mode=mode,
                preset=preset or manifest_preset,
                budget_overrides=budget_overrides,
                sandbox_overrides=sandbox_overrides,
                model=route,
                effort=effort,
            )
        except ConfigError as exc:
            reporter.error(str(exc))
            return 2
        cfg, explicit_leaves = effective.config, effective.explicit_leaves
        if worktree_setup.session_needs_setup(layout.session_dir):
            cfg = worktree_setup.read_if_unset(cfg, repo, reporter=reporter, at=resume_base_sha)

        # Needs the config: the away grant is per scope, one per configured MCP server.
        app_frontend.settle_away_mode(
            layout.session_dir, cfg, can_ask=frontend.capabilities.can_ask
        )

        if not preflight.route_preflight(
            cfg, role, reporter=reporter, model_flag=session_setup.route_text(route)
        ):
            return 2

        if manifest.worktree is not None:
            cfg = cfg.with_hidden_in_checkout(repo, cwd)
        try:
            isolation = session_prepare.hold_workspace_and_select_isolation(
                cfg,
                cwd,
                frontend=frontend,
                reporter=reporter,
                worktree_git_dir=manifest.worktree_git_dir,
            )
        except preflight.SessionRefusedError as refusal:
            return refusal.rc

        identity = git_read.CommitIdentity(name=cfg.git.commit.name, email=cfg.git.commit.email)
        # The no-repo guard ran above.
        if writes_code:
            try:
                git_read.verify_git_identity(cwd, identity)
            except errors.GitError as exc:
                reporter.error(str(exc))
                return 2

        channels = session_prepare.operator_channels(
            layout, frontend, carried=carried, delivered=snapshot.delivered_questions
        )
        events, prompts = channels.events, channels.prompts
        transcript_sink = channels.transcript_sink
        session_prepare.workspace_notices(cwd, cfg, reporter=reporter)

        try:
            tui_enabled = session_prepare.headless_check(
                cfg,
                frontend,
                tui=tui,
                interactive=interactive,
                mode=mode,
                # The raw env first, so a typo refuses here as on run; else the recorded choice.
                away=os.environ.get("AGENT6_DETACHED_AWAY", "")
                or ipc.effective_away(layout.session_dir),
                parking_away=ipc.effective_away(layout.session_dir),
                reporter=reporter,
            )
        except preflight.SessionRefusedError as refusal:
            return refusal.rc

        def _gate(
            cfg: Config, _budget: budget.BudgetTracker, _available: Callable[[str], bool]
        ) -> Config:
            pinned = sessions_manifest.GateStamp(availability="pending")
            with contextlib.suppress(sessions_manifest.ManifestError, OSError):
                pinned = sessions_manifest.read_manifest(layout.session_dir).harness.gate
            # A resolved pin outranks a snapshot that predates adoption or un-adoption.
            replay_gate = (
                pinned.command if pinned.availability != "pending" else snapshot.verify_command
            )
            execution_configured = bool(cfg.harness.verify.command)
            reused = not execution_configured and bool(replay_gate)
            if reused:
                cfg = cfg.with_verify_command(replay_gate)
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
                reporter.note(
                    f"reusing this run's verify command: {preflight.gate_text(replay_gate)}"
                )
            # The manifest says which gate this execution used.
            if pinned.command != cfg.harness.verify.command and not withheld:
                # Both directions: the frozen system prompt names the old gate either way.
                reporter.note(
                    "this run's verify gate changed:"
                    f" was {preflight.gate_text(pinned.command)},"
                    f" now {preflight.gate_text(cfg.harness.verify.command)}"
                )
            return cfg

        untracked_at_start = sessions_layout.read_untracked_at_start(layout.session_dir)
        hidden = git_read.hidden_in_repo(cwd, cfg.sandbox.hide_paths)
        outside: list[str] = []
        carried_edits: frozenset[str] = frozenset()
        end_tree = ""
        seed: str | None = None
        if mode == "run":
            # An untracked file the run did not leave, the chain does not hold and no tool wrote
            # arrived between executions, so it is the operator's; a git failure refuses, since
            # this decides what the run may commit. After a death mid-turn a call of that turn
            # may have written it with no result to say so, and it stays the run's.
            try:
                flight = loop_snapshot.read_turn_marker(
                    layout.session_dir / loop_snapshot.TURN_IN_FLIGHT_NAME
                )
            except ValueError:
                flight = None  # the loop names and drops it when it resumes
            died_in_turn = flight is not None and flight.turn == snapshot.next_turn
            chain_ref = git_read.chain_ref_for(session_id)
            end_tree = scan.end_tree
            model_end = manifest.git_control == "model" and bool(end_tree)
            record_ref = (
                (scan.index_tree or snapshot.head_sha or end_tree) if model_end else chain_ref
            )
            seed = (
                record_ref
                if model_end
                else git_read.chain_tip(cwd, chain_ref)
                or snapshot.head_sha  # the model's git keeps no chain
                or resume_base_sha
                or None
            )
            written = _paths_the_run_wrote(layout.logs_path)
            if merged is not None:
                # The run's work is in HEAD, so what differs from HEAD is the operator's.
                end_tree, seed = f"{merged.head}^{{tree}}", merged.head
            try:
                stopped_paths = git_read.tree_paths(cwd, end_tree) if end_tree else frozenset()
                arrived = (
                    git_read.untracked_paths(cwd)
                    - untracked_at_start
                    - git_read.tree_paths(cwd, record_ref)
                    - stopped_paths
                    - written
                )
                if arrived:
                    # A file replacing a recorded directory is a change to that tree.
                    arrived -= {
                        str(p)
                        for name in stopped_paths
                        for p in pathlib.PurePosixPath(name).parents
                    }
                shown = _first_four(arrived)
                if arrived and died_in_turn:
                    them = "it" if len(arrived) == 1 else "them"
                    reporter.note(
                        f"taken as this run's: {shown} (untracked since the turn the last"
                        f" execution died in, whose calls may have written {them}; the run's"
                        f" next commit holds {them})"
                    )
                elif arrived:
                    untracked_at_start = untracked_at_start | arrived
                    sessions_layout.write_untracked_at_start(layout.session_dir, untracked_at_start)
                    reporter.note(
                        f"left out of this run's commits as yours: {shown} (arrived"
                        " between executions, unwritten by any tool of the run)"
                    )
                # Tracked content that changed since the last end is the operator's too; a
                # file taken as the run's above stays its own.
                runs_own = written | arrived if died_in_turn else written
                exclude = untracked_at_start | hidden | {".agent6"}
                outside = resume_tree.outside_edits(cwd, end_tree, seed, exclude, written=runs_own)
                if model_end:
                    committed = snapshot.head_sha or git_read.empty_tree(cwd)
                    staged = git_read.index_tree(cwd)
                    if any(
                        code != "??"
                        and code[1] != " "
                        and git_read.excluded(p, {".agent6"})
                        and not git_read.excluded(p, hidden)
                        for code, p in git_read.porcelain_entries(cwd)
                    ):
                        reporter.refuse(
                            "a tracked file under .agent6 has working-copy edits."
                            " The run did not measure this directory."
                        )
                        return 2
                    staged_changes = resume_tree.visible_changes(
                        cwd, scan.index_tree or committed, staged, hidden
                    )
                    if staged_changes and not scan.index_tree:
                        note = (
                            "the run's last stop has no staged-tree record."
                            f" Unconfirmed staged edits: {_first_four(staged_changes)}"
                        )
                        if cfg.git.dirty_tree != "include":
                            reporter.refuse(note)
                            return 2
                        reporter.note(note)
                    else:
                        outside = sorted(set(outside).union(staged_changes))
                    carried_edits = (
                        frozenset(resume_tree.visible_changes(cwd, committed, end_tree, exclude))
                        .union(resume_tree.visible_changes(cwd, committed, staged, exclude))
                        .difference(outside, staged_changes)
                    )
                unsettled = resume_tree.refusal(
                    cwd,
                    policy=cfg.git.dirty_tree,
                    outside=outside,
                    end_tree=end_tree,
                    seed=seed,
                    run_branch=manifest.run_branch,
                    answerable=frontend.capabilities.can_ask
                    or ipc.away_mode(layout.session_dir) == "wait",
                )
            except (errors.GitError, OSError) as exc:
                reporter.error(f"cannot tell the run's files from the operator's: {exc}")
                return 2
            if unsettled is not None:
                reporter.refuse(unsettled)
                return 2
        # Written once the preflight passed: a refused resume never had a live worker, and the
        # question below reaches `agent6 answer`.
        process.write_worker_pid(layout.session_dir, os.getpid())
        settled = resume_tree.settle(
            cwd,
            policy=cfg.git.dirty_tree,
            outside=outside,
            end_tree=end_tree,
            seed=seed,
            run_branch=manifest.run_branch,
            session_id=session_id,
            ask=lambda question: prompts.ask((question,)).answers[0],
            reporter=reporter,
        )
        if settled is None:
            return 2
        stashed = settled
        if merged is not None:
            try:
                resume_base_sha = resume_chain.continue_from_head(cwd, merged, manifest, identity)
            except errors.GitError as exc:
                reporter.error(f"cannot continue this run from HEAD: {exc}")
                return 2
            stamps.stamp_continued_from(layout.session_dir, resume_base_sha, merged.branch)
            reporter.note(
                f"this run's earlier work was merged into {merged.into}; continuing from HEAD"
                f" ({merged.head[:12]})"
            )
        elif mode == "run" and (moved := resume_chain.unmerged_base(cwd, layout, manifest)):
            resume_base_sha = moved
            # The run's base moves to where its unmerged work starts; its chain stays.
            current = sessions_manifest.read_manifest(layout.session_dir)
            stamps.write_manifest(
                layout.manifest_path, current.model_copy(update={"base_sha": moved})
            )
            reporter.note(
                f"this run's earlier work was merged into {manifest.base_branch}; its diff and"
                f" next merge start at {moved[:12]}"
            )
        merged_in: git_ops.ChainMerge | None = None
        if merge_from and fork_worktrees.mirror_origin(repo, cwd) is not None:
            # The branch is the repository's; a clone's copy of it is as old as the clone.
            with contextlib.suppress(errors.GitError):
                git_ops.fetch_branch(cwd, repo, f"+refs/heads/{merge_from}:refs/heads/{merge_from}")
        if merge_from:
            if mode != "run":
                reporter.refuse(f"--merge continues a run's work; {session_id} is a {mode}")
                return 2
            try:
                merged_in = git_ops.chain_merge_with_markers(
                    cwd,
                    merge_from,
                    f"agent6: merge {merge_from} into the run's work",
                    ref=git_read.chain_ref_for(session_id),
                    fallback_parent=resume_base_sha,
                    identity=identity,
                    also_branch=run_branch or None,
                )
            except errors.GitError as exc:
                reporter.error(f"cannot merge {merge_from} into this run's work: {exc}")
                return 2
            reporter.note(resume_chain.merged_in_line(merged_in))
        driver = cfg.models.resolve(role)
        if (
            recorded is not None
            and driver is not None
            and (driver.provider, driver.model) != (recorded.provider, recorded.model)
        ):
            reporter.note(
                f"resuming with {driver.provider}/{driver.model}"
                f" (recorded: {recorded.provider}/{recorded.model})"
            )
        # Every preflight passed: the operator's new choices are recorded from here.
        if preset:
            stamps.stamp_preset(layout.session_dir, preset)
        if (flagged := session_setup.flag_route(cfg, mode, model)) is not None:
            stamps.stamp_model(layout.session_dir, flagged)
        if budget_overrides.stamp() != manifest.budget:
            # Re-read: the stamps above rewrote the file. A later resume replays these caps.
            current = sessions_manifest.read_manifest(layout.session_dir)
            stamps.write_manifest(
                layout.manifest_path,
                current.model_copy(update={"budget": budget_overrides.stamp()}),
            )
        if caps := budget_overrides.argv():
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
            isolation,
            # An effort this execution was started with reads as its own, not a layer's.
            config_sources={**effective.sources, **effective.applied_sources},
        )
        if carried_edits:
            note = f"uncommitted from this run's last execution: {_first_four(carried_edits)}"
            reporter.note(note)
            events.emit(journal.LoopNote(text=note))
        if mode == "run" and cfg.git.control != "model":
            # The next auto-commit takes the previous execution's tail and any operator edit
            # since; the operator's untracked files stay out.
            with contextlib.suppress(errors.GitError, OSError):
                dirty = git_read.chain_dirty_paths(
                    cwd,
                    git_read.chain_ref_for(session_id),
                    resume_base_sha,
                    5,
                    exclude=untracked_at_start | hidden,
                )
                if dirty:
                    shown = map(terminal.printable_name, dirty[:4])
                    named = ", ".join(shown) + (", ..." if len(dirty) > 4 else "")
                    reporter.note(
                        f"the tree holds changes no commit has ({named});"
                        " this execution's next commit takes them"
                    )
        end = execution.run_execution(
            cfg,
            layout,
            execution.ExecutionInputs(
                session_id=session_id,
                mode=mode,
                role=role,
                explicit_leaves=explicit_leaves,
                effective=effective,
                isolation=isolation,
                tui_enabled=tui_enabled,
                interactive=interactive,
                task=None,
                gate=_gate,
                chain_branch=run_branch or None,
                base_sha=resume_base_sha,
                untracked_at_start=untracked_at_start,
                resume_state_path=snapshot_path,
                undo_forker=session_prepare.undo_forker(
                    config_paths, session_id, cwd=repo, reporter=reporter
                ),
                prompts=prompts,
                # The follow-up this execution answered, not the original task.
                ask_transcript_task=steer.strip() or manifest.user_task,
                resume_steer=steer,
                operator_edits=stashed.kept,
                merged_in=merged_in,
                budget_overrides=budget_overrides,
                sandbox_overrides=sandbox_overrides,
                resuming=True,
                worktree_git_dir=manifest.worktree_git_dir,
                mirror_into=fork_worktrees.mirror_origin(repo, cwd),
                worker_lock_fd=worker_lock_fd,
                base_rules=(
                    repo_context.review_rules(
                        cwd, manifest.review_base_tree, cfg.sandbox.hide_paths
                    ).text
                    if manifest.review_base_tree
                    else None
                ),
            ),
            frontend=frontend,
            reporter=reporter,
            events=events,
            transcript_sink=transcript_sink,
            cwd=cwd,
            state_dir=state,
        )
        detach_requested = end.detach_requested
        return end.rc
    finally:
        # The one owner of worker.pid and both locks on every exit path; a detach keeps the
        # pid until the background `resume` claims it. Nested so a teardown raise strands no lock.
        try:
            try:
                frontend.close_console_view()  # stops the heartbeat, clears the spinner
            finally:
                if not detach_requested and not handed_to_run_task:
                    # run_task's own teardown owns the pid when it ran.
                    process.clear_worker_pid(layout.session_dir)
                if stashed.stashed and cfg is not None:
                    finalize.end_auto_stash(
                        cwd,
                        detached=detach_requested,
                        base_branch=manifest.base_branch,
                        run_branch=manifest.run_branch,
                        auto_pop=cfg.git.auto_stash_pop or stashed.pop,
                        session_id=session_id,
                        in_worktree=manifest.worktree is not None,
                        exclude=untracked_at_start,
                        reporter=reporter,
                    )
        finally:
            lock.release_single_writer(repo_lock_fd)
            lock.release_single_writer(worker_lock_fd)
        if detach_requested and cfg is not None:
            execution.detach_to_background(
                frontend=frontend,
                cfg=cfg,
                layout=layout,
                cwd=repo,
                flags=session_setup.override_flags(
                    budget_overrides, sandbox_overrides, session_setup.flag_route(cfg, mode, route)
                ),
                reporter=reporter,
            )
