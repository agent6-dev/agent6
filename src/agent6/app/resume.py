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

from agent6 import budget, directive, errors, event_log, git_ops, kinds, terminal
from agent6 import paths as agent6_paths
from agent6.app import (
    _execution,
    _resume_chain,
    _resume_tree,
    _session,
    _setup,
    finalize,
    preflight,
    run,
    stamps,
    workspace_setup,
)
from agent6.app import frontend as app_frontend
from agent6.app import reporter as app_reporter
from agent6.config import (
    Config,
    ConfigError,
)
from agent6.harness import _context, _snapshot
from agent6.providers import (
    TranscriptSink,
)
from agent6.sessions import id, ipc, lock, process
from agent6.sessions import layout as sessions_layout
from agent6.sessions import manifest as sessions_manifest
from agent6.tools import operator_prompts
from agent6.viewmodel import listing


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


def _first_four(paths: frozenset[str]) -> str:
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
    holds = git_ops.merge_stamp_holds(
        repo, manifest.session_id, manifest.run_branch or "", stamp.tip
    )
    return stamp if holds else None


def commits_note(repo: pathlib.Path, manifest: sessions_manifest.SessionManifest) -> str:
    """Say where a session's commits are, for a refusal that points at them.

    Args:
        repo: The repository.
        manifest: The session's manifest.

    Returns:
        The run branch, the merge that landed them, the chain ref, or "it recorded
        no commits".
    """
    if manifest.run_branch and git_ops.branch_exists(repo, manifest.run_branch):
        return f"its commits are on {manifest.run_branch}"
    stamp = covering_stamp(repo, manifest)
    if stamp is not None:
        return f"its commits are {stamp.landed()}"
    chain = git_ops.chain_ref_for(manifest.session_id)
    if git_ops.chain_tip(repo, chain) is not None:
        return f"its commits are on {chain}"
    return "it recorded no commits"


def snapshot_head_mismatch(
    snapshot_path: pathlib.Path, repo_root: pathlib.Path, *, chain_ref: str
) -> tuple[str, str] | None:
    """Detect a chain tip that diverged from the run's last snapshot.

    Forward movement on the same line is the run's own per-step commits and
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
        current_head = git_ops.chain_tip(repo_root, chain_ref)
    except errors.GitError:
        return None
    if current_head is None or current_head == snap_head:
        return None
    if git_ops.is_ancestor(repo_root, snap_head, current_head):
        # The tip moved forward on the same line: the run's own commits.
        return None
    return (snap_head, current_head)


def execution_gate_origin(*, configured: bool, has_gate: bool, pinned: str) -> str:
    """Name where this execution's gate came from.

    Config outranks the run's pin, the pin stands when reused, and a gateless
    execution claims nothing.

    Args:
        configured: Whether config names the gate.
        has_gate: Whether the execution runs a gate at all.
        pinned: The origin the run's manifest pins.

    Returns:
        "configured", the pinned origin, "inferred", or "" for no gate.
    """
    if not has_gate:
        return ""
    if configured:
        return "configured"
    return "inferred" if pinned in ("", "pending") else pinned


def resume_task(  # noqa: C901, PLR0911, PLR0912, PLR0915  # every way a resume is refused, in preflight order
    config_paths: tuple[pathlib.Path, ...],
    session_id: str,
    *,
    frontend: app_frontend.SessionFrontend,
    force: bool,
    started_at: float,
    tui: bool = False,
    budget_overrides: _setup.BudgetOverrides | None = None,
    sandbox_overrides: _setup.SandboxOverrides | None = None,
    preset: str = "",
    steer: str = "",
    interactive: bool = False,
    model: str = "",
    reporter: app_reporter.Reporter = app_reporter.STDIO_REPORTER,
) -> int:
    """Resume a paused or crashed session from its snapshot.

    The budget is a fresh ceiling per execution. The process cwd is the
    repository; a fork's execution drives the fork's own worktree.

    Args:
        config_paths: The `--config` files in command-line order; none means discovery alone.
        session_id: The session's id or prefix.
        frontend: The surface's seam: console, confirms, approvals, questions.
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
        reporter: Receives every refusal and note.

    Returns:
        The process exit code.
    """
    repo = pathlib.Path.cwd()
    state = agent6_paths.state_dir(repo)
    if steer.strip() and (problem := directive.steer_problem(steer)) is not None:
        reporter.error(f"--steer: {problem}")
        return 2
    # One resolver across buckets: a runs/-only fallback would pick a run over an ask.
    try:
        layout = id.resolve_session(state, session_id)
    except id.SessionIdError as exc:
        reporter.error(str(exc))
        return 2
    session_id = layout.session_id
    # The manifest first: clearing state before it would clobber a session resume refuses.
    try:
        manifest = sessions_manifest.read_manifest(layout.session_dir)
        # A `--model` set on the run replays unless this resume sets its own.
        recorded = manifest.models.replay_driver
        route: str | kinds.ModelRoute | None = model or (
            kinds.ModelRoute(recorded.provider, recorded.model) if recorded is not None else None
        )
        mode = manifest.session_mode()
        # A `--max-*` set on the run replays the same way, field by field.
        budget_overrides = (budget_overrides or _setup.BudgetOverrides()).over(manifest.budget)
    except sessions_manifest.ManifestError as exc:
        reporter.error(f"cannot resume {session_id}: {exc}")
        return 2
    if manifest.fanout is not None:
        reporter.error(listing.fanout_resume_refusal(session_id))
        return 2
    cwd = manifest.worktree or repo
    if manifest.worktree is not None and manifest.worktree_git_dir is None:
        reporter.error(
            f"cannot resume {session_id}: its manifest names a worktree but not the repository"
            f" git dir it points into, so its jail cannot grant one; `agent6 fork {session_id}`"
            " continues its commits in a new worktree."
        )
        return 2
    if manifest.worktree is not None and not (cwd / ".git").exists():
        reporter.error(
            f"cannot resume {session_id}: its worktree {cwd} is gone (pruned or removed);"
            f" {commits_note(repo, manifest)}; `agent6 fork {session_id}` continues it in a"
            " new worktree."
        )
        return 2
    # One writer per session dir, taken before any shared state is touched.
    worker_lock_fd = lock.acquire_single_writer(layout.session_dir)
    if worker_lock_fd is None:
        reporter.err(lock.SINGLE_WRITER_BUSY.format(rid=session_id))
        return 2
    # A finished run has nothing to continue without --steer; every other ending resumes.
    scan = listing.scan_session_log(layout.logs_path)
    new_work = listing.needs_new_work(
        finished=scan.finished, end_reason=scan.end_reason, all_passed=scan.all_passed
    )
    if scan.finished and scan.end_reason == "undone":
        # The fork is the continuation; resuming here would redo the taken-back edit.
        reporter.refuse(listing.undone_refusal(session_id, scan.undone_to))
        lock.release_single_writer(worker_lock_fd)
        return 2
    if not steer.strip() and new_work:
        reporter.refuse(listing.needs_new_work_refusal(session_id))
        lock.release_single_writer(worker_lock_fd)
        return 2
    # The prompt ids reset on resume, so stale answers go; markers since this start stay.
    ipc.clear_pending_answers(layout.session_dir, started_at=started_at)
    # Seeded after the clear, which drops steer files.
    if steer.strip() and not ipc.submit_steer(layout.session_dir, steer.strip()):
        reporter.error("could not write the initial steer request")
        lock.release_single_writer(worker_lock_fd)
        return 2

    detach_requested = False
    stashed = _resume_tree.Settled()  # the outside edits this execution set aside
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
                effective = _setup.load_session_config(
                    repo,
                    config_paths,
                    mode=mode,
                    preset=preset or manifest.harness.replay_preset,
                    budget_overrides=budget_overrides,
                    sandbox_overrides=sandbox_overrides,
                    model=route,
                )
                cfg = effective.config
            except ConfigError as exc:
                reporter.error(str(exc))
                return 2
            saved_task = manifest.parked_task
            lock.release_single_writer(worker_lock_fd)
            worker_lock_fd = None
            handed_to_run_task = True
            return run.run_task(
                cfg,
                saved_task,
                frontend=frontend,
                started_at=started_at,
                session_id=session_id,
                config_paths=config_paths,
                mode=mode,
                budget_overrides=budget_overrides,
                sandbox_overrides=sandbox_overrides,
                preset=preset,
                model=route,
                explicit_leaves=effective.explicit_leaves,
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
            repo_lock_fd = lock.acquire_repo_writer(state, cwd, session_id)
            if repo_lock_fd is None:
                holder = lock.repo_writer_holder(state, cwd) or "another run"
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
            snapshot = _snapshot.load_session_snapshot(snapshot_path)
        except (ValueError, OSError) as exc:
            reporter.error(str(exc))
            return 1
        manifest_preset = manifest.harness.replay_preset
        resume_base_sha = manifest.base_sha
        run_branch = manifest.run_branch or ""

        # A rewritten chain ref would leave the model reasoning about a changed record.
        mismatch = (
            snapshot_head_mismatch(snapshot_path, cwd, chain_ref=git_ops.chain_ref_for(session_id))
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

        # A run whose work landed continues from HEAD; any other run's next commits would
        # leave behind a commit the operator made on its branch.
        landed: _resume_chain.LandedStart | None = None
        if mode == "run":
            try:
                landed = _resume_chain.landed_start(cwd, manifest)
                off = None if landed else _resume_chain.branch_off_the_chain(cwd, manifest)
            except errors.GitError as exc:
                reporter.error(str(exc))
                return 2
            if off is not None:
                branch, branch_tip, ahead = off
                if not force:
                    reporter.refuse(
                        _resume_chain.off_chain_refusal(session_id, branch, branch_tip, ahead=ahead)
                    )
                    return 2
                reporter.note(
                    f"continuing on the chain; {branch} stays at {branch_tip[:12]}, off the run"
                )

        try:
            effective = _setup.load_session_config(
                repo,
                config_paths,
                mode=mode,
                preset=preset or manifest_preset,
                budget_overrides=budget_overrides,
                sandbox_overrides=sandbox_overrides,
                model=route,
            )
        except ConfigError as exc:
            reporter.error(str(exc))
            return 2
        cfg, explicit_leaves = effective.config, effective.explicit_leaves
        # A lane's record is imported into the origin's state with its branch; its tree is no
        # checkout's. Only its own clone, under the parallel workdir, may continue it.
        lane = manifest.parallel
        workdir = agent6_paths.parallel_base(cfg.parallel.workdir).resolve()
        if lane is not None and not cwd.resolve().is_relative_to(workdir):
            reporter.refuse(
                f"{session_id} is a lane of the fan-out {lane.coordinator}: its work is on its"
                " own branch, and this checkout holds another tree, so it cannot continue here."
                " Continue it in a worktree of its own:\n"
                f'    agent6 fork {session_id} --steer "..."'
            )
            return 2
        if workspace_setup.session_needs_setup(layout.session_dir):
            cfg = workspace_setup.infer_if_unset(cfg, repo, reporter=reporter)

        # Needs the config: the away grant is per scope, one per configured MCP server.
        app_frontend.settle_away_mode(
            layout.session_dir, cfg, can_ask=frontend.capabilities.can_ask
        )

        if not preflight.route_preflight(
            cfg, role, reporter=reporter, model_flag=_setup.route_text(route)
        ):
            return 2

        try:
            isolation = _session.select_isolation(
                cfg,
                cwd=cwd,
                confirm_unconfined=frontend.confirm_unconfined_autorun,
                reporter=reporter,
                explicit_leaves=explicit_leaves,
                worktree_git_dir=manifest.worktree_git_dir,
            )
        except preflight.SessionRefusedError as refusal:
            return refusal.rc

        identity = git_ops.CommitIdentity(name=cfg.git.commit.name, email=cfg.git.commit.email)
        # The no-repo guard ran above.
        if writes_code:
            try:
                git_ops.verify_git_identity(cwd, identity)
            except errors.GitError as exc:
                reporter.error(str(exc))
                return 2

        transcript_sink = TranscriptSink(layout.transcripts_dir)
        events = event_log.EventSink(layout.logs_path)
        # The execution's one gate to the operator, whichever front-end answers.
        prompts = operator_prompts.OperatorPrompts(
            approver=frontend.build_approver(layout.session_dir),
            questioner=frontend.build_questioner(layout.session_dir),
            journal=events.emit,
            session_dir=layout.session_dir,
        )

        _session.warn_install_inside_workspace(cwd, reporter=reporter)
        for line in _context.agents_md_notices(cwd, cfg.sandbox.hide_paths):
            reporter.note(line)

        tui_enabled = frontend.should_spawn_tui(tui, interactive, mode)
        refusal = preflight.headless_approval_refusal(
            cfg,
            tui_enabled=tui_enabled,
            # The raw env first, so a typo refuses here as on run; else the recorded choice.
            away=os.environ.get("AGENT6_DETACHED_AWAY", "")
            or ipc.effective_away(layout.session_dir),
            can_ask=frontend.capabilities.can_ask,
            clamped=kinds.session_kind(mode).clamps_commands,
        )
        if refusal is not None:
            reporter.refuse(refusal)
            return 2
        parking = preflight.headless_parking_note(
            cfg,
            tui_enabled=tui_enabled,
            away=ipc.effective_away(layout.session_dir),
            can_ask=frontend.capabilities.can_ask,
        )
        if parking is not None:
            reporter.note(parking)

        def _gate(cfg: Config, _budget: budget.BudgetTracker) -> Config:
            # The run's resolved gate is reused, not re-inferred: the snapshot's, unless a
            # newer adopted or unadopted pin outranks it; config outranks both.
            pinned_origin, pinned_gate = "", ()
            with contextlib.suppress(sessions_manifest.ManifestError, OSError):
                pinned = sessions_manifest.read_manifest(layout.session_dir).harness
                pinned_origin, pinned_gate = pinned.verify_origin, pinned.verify_command
            replay_gate = (
                pinned_gate
                if pinned_origin in ("adopted", "unadopted")
                else snapshot.verify_command
            )
            execution_configured = bool(cfg.harness.verify_command)
            reused = not execution_configured and bool(replay_gate)
            if reused:
                cfg = cfg.with_verify_command(replay_gate)
            # Dropped last when commands are withheld, so nothing hands the gate back.
            gate_before = cfg.harness.verify_command
            cfg = preflight.drop_gate_if_unrunnable(
                cfg, session_dir=layout.session_dir, reporter=reporter
            )
            # A withheld gate is neither a reuse nor a change.
            withheld = bool(gate_before) and not cfg.harness.verify_command
            if reused and not withheld:
                reporter.note(
                    f"reusing this run's verify command: {preflight.gate_text(replay_gate)}"
                )
            # The manifest says which gate this execution used.
            if tuple(pinned_gate) != cfg.harness.verify_command and not withheld:
                # Both directions: the frozen system prompt names the old gate either way.
                reporter.note(
                    "this run's verify gate changed:"
                    f" was {preflight.gate_text(tuple(pinned_gate))},"
                    f" now {preflight.gate_text(cfg.harness.verify_command)}"
                )
            stamps.pin_gate(
                layout.session_dir,
                cfg.harness.verify_command,
                execution_gate_origin(
                    configured=execution_configured,
                    has_gate=bool(cfg.harness.verify_command),
                    pinned=pinned_origin,
                ),
                events=events,
                reporter=reporter,
            )
            return cfg

        def _undo_forker() -> tuple[str, str] | None:
            # Lazy: app.fork imports this module.
            from agent6.app import undo  # noqa: PLC0415  # noqa: PLC0415

            return undo.undo_fork(config_paths, session_id, cwd=repo, reporter=reporter)

        untracked_at_start = sessions_layout.read_untracked_at_start(layout.session_dir)
        outside: list[str] = []
        end_tree = ""
        seed: str | None = None
        if mode == "run":
            # An untracked file the run did not leave, the chain does not hold and no tool wrote
            # arrived between executions, so it is the operator's; a git failure refuses, since
            # this decides what the run may commit. After a death mid-turn a call of that turn
            # may have written it with no result to say so, and it stays the run's.
            flight = _snapshot.read_turn_marker(layout.session_dir / _snapshot.TURN_IN_FLIGHT_NAME)
            died_in_turn = flight is not None and flight.iteration == snapshot.next_iteration
            chain_ref = git_ops.chain_ref_for(session_id)
            end_tree = scan.end_tree
            seed = git_ops.chain_tip(cwd, chain_ref) or resume_base_sha or None
            written = _paths_the_run_wrote(layout.logs_path)
            if landed is not None:
                # The run's work is in HEAD, so what differs from HEAD is the operator's.
                end_tree, seed = f"{landed.head}^{{tree}}", landed.head
            try:
                arrived = (
                    git_ops.untracked_paths(cwd)
                    - untracked_at_start
                    - git_ops.tree_paths(cwd, chain_ref)
                    - (git_ops.tree_paths(cwd, end_tree) if end_tree else frozenset())
                    - written
                )
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
                outside = _resume_tree.outside_edits(
                    cwd, end_tree, seed, untracked_at_start, written=runs_own
                )
                unsettled = _resume_tree.refusal(
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
        settled = _resume_tree.settle(
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
        if landed is not None:
            try:
                resume_base_sha = _resume_chain.continue_from_head(cwd, landed, manifest, identity)
            except errors.GitError as exc:
                reporter.error(f"cannot continue this run from HEAD: {exc}")
                return 2
            stamps.stamp_continued_from(layout.session_dir, resume_base_sha, landed.branch)
            reporter.note(
                f"this run's earlier work landed on {landed.into}; continuing from HEAD"
                f" ({landed.head[:12]})"
            )
        elif mode == "run" and (moved := _resume_chain.unlanded_base(cwd, layout, manifest)):
            resume_base_sha = moved
            stamps.stamp_base(layout.session_dir, moved)
            reporter.note(
                f"this run's earlier work landed on {manifest.base_branch}; its diff and"
                f" next merge start at {moved[:12]}"
            )
        # Every preflight passed: the operator's new choices are recorded from here.
        if preset:
            stamps.stamp_preset(layout.session_dir, preset)
        if (flagged := _setup.flag_route(cfg, mode, model)) is not None:
            stamps.stamp_model(layout.session_dir, flagged)
        if budget_overrides.stamp() != manifest.budget:
            stamps.stamp_budget(layout.session_dir, budget_overrides.stamp())
        if caps := budget_overrides.argv():
            reporter.note(f"budget caps in force from this run's flags: {' '.join(caps)}")
        if steer.strip():
            # A steer that is the work names the run, for a finished run or a fork still
            # carrying its source's task.
            if new_work:
                stamps.stamp_task(layout.session_dir, steer.strip())
            elif manifest.parent_session_id:
                stamps.stamp_fork_task(
                    layout.session_dir,
                    steer.strip(),
                    source_dir=layout.session_dir.parent / manifest.parent_session_id,
                )
        # This execution's models and policy, for `agent6 exec` and the policy surfaces.
        stamps.stamp_execution(layout.session_dir, cfg, mode, isolation)
        if mode == "run":
            # The next auto-commit takes the previous execution's tail and any operator edit
            # since; the operator's untracked files stay out.
            with contextlib.suppress(errors.GitError, OSError):
                dirty = git_ops.chain_dirty_paths(
                    cwd,
                    git_ops.chain_ref_for(session_id),
                    resume_base_sha,
                    5,
                    exclude=untracked_at_start,
                )
                if dirty:
                    shown = map(terminal.printable_name, dirty[:4])
                    named = ", ".join(shown) + (", ..." if len(dirty) > 4 else "")
                    reporter.note(
                        f"the tree holds changes no commit has ({named});"
                        " this execution's next commit takes them"
                    )
        end = _execution.run_execution(
            cfg,
            layout,
            _execution.ExecutionInputs(
                session_id=session_id,
                mode=mode,
                role=role,
                explicit_leaves=explicit_leaves,
                isolation=isolation,
                tui_enabled=tui_enabled,
                interactive=interactive,
                task=None,
                gate=_gate,
                chain_branch=run_branch or None,
                base_sha=resume_base_sha,
                untracked_at_start=untracked_at_start,
                resume_state_path=snapshot_path,
                undo_forker=_undo_forker,
                prompts=prompts,
                # The follow-up this execution answered, not the original task.
                ask_transcript_task=steer.strip() or manifest.user_task,
                resume_steer=steer.strip(),
                budget_overrides=budget_overrides,
                sandbox_overrides=sandbox_overrides,
                resuming=True,
                worktree_git_dir=manifest.worktree_git_dir,
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
            _execution.detach_to_background(
                frontend=frontend,
                cfg=cfg,
                layout=layout,
                cwd=repo,
                flags=_setup.override_flags(
                    budget_overrides, sandbox_overrides, _setup.flag_route(cfg, mode, route)
                ),
                reporter=reporter,
            )
