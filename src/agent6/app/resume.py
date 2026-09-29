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
from collections.abc import Callable
from pathlib import Path

from agent6.app._execution import ExecutionInputs, detach_to_background, run_execution
from agent6.app._session import (
    select_isolation,
    warn_install_inside_workspace,
)
from agent6.app._setup import (
    BudgetOverrides,
    SandboxOverrides,
    flag_route,
    load_session_config,
    override_flags,
    route_text,
)
from agent6.app.frontend import (
    SessionFrontend,
    settle_away_mode,
)
from agent6.app.manifest import (
    pin_gate,
    stamp_execution,
    stamp_fork_task,
    stamp_model,
    stamp_preset,
    stamp_task,
)
from agent6.app.preflight import (
    SessionRefusedError,
    drop_gate_if_unrunnable,
    gate_text,
    headless_approval_refusal,
    headless_parking_note,
    require_git_repo,
    route_preflight,
)
from agent6.app.reporter import STDIO_REPORTER, Reporter
from agent6.app.run import run_task
from agent6.budget import BudgetTracker
from agent6.config import (
    Config,
    ConfigError,
)
from agent6.directive import steer_problem
from agent6.events import EventSink
from agent6.git_ops import (
    CommitIdentity,
    GitError,
    branch_exists,
    chain_dirty_paths,
    chain_ref_for,
    chain_tip,
    is_ancestor,
    merge_stamp_holds,
    tree_paths,
    untracked_paths,
    verify_git_identity,
)
from agent6.harness._context import agents_md_notices
from agent6.harness._snapshot import (
    TURN_IN_FLIGHT_NAME,
    clear_turn_marker,
    load_session_snapshot,
    read_turn_marker,
)
from agent6.kinds import SESSION_KINDS, ModelRoute, session_bucket, session_kind
from agent6.paths import state_dir
from agent6.providers import (
    TranscriptSink,
)
from agent6.sessions.id import SessionIdError, resolve_session
from agent6.sessions.ipc import (
    clear_pending_answers,
    clear_worker_pid,
    effective_away,
    submit_steer,
    write_worker_pid,
)
from agent6.sessions.layout import (
    bucket_dir,
    read_untracked_at_start,
    write_untracked_at_start,
)
from agent6.sessions.lock import (
    SINGLE_WRITER_BUSY,
    acquire_repo_writer,
    acquire_single_writer,
    release_single_writer,
    repo_writer_holder,
)
from agent6.sessions.manifest import ManifestError, MergeStamp, SessionManifest, read_manifest
from agent6.tools.operator_prompts import OperatorPrompts
from agent6.viewmodel.listing import finished_needs_new_work, needs_new_work_refusal


def resumable_bucket_dirs(state_dir: Path) -> list[Path]:
    """List the bucket dirs holding sessions `agent6 resume` can pick up.

    Args:
        state_dir: The repo's state directory.

    Returns:
        One dir per resumable session kind.
    """
    return [
        bucket_dir(state_dir, session_bucket(kind.name))
        for kind in SESSION_KINDS.values()
        if kind.resumable
    ]


def _paths_the_run_wrote(logs_path: Path) -> frozenset[str]:
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


def covering_stamp(repo: Path, manifest: SessionManifest) -> MergeStamp | None:
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
    holds = merge_stamp_holds(repo, manifest.session_id, manifest.run_branch or "", stamp.tip)
    return stamp if holds else None


def commits_note(repo: Path, manifest: SessionManifest) -> str:
    """Say where a session's commits are, for a refusal that points at them.

    Args:
        repo: The repository.
        manifest: The session's manifest.

    Returns:
        The run branch, the merge that landed them, the chain ref, or "it recorded
        no commits".
    """
    if manifest.run_branch and branch_exists(repo, manifest.run_branch):
        return f"its commits are on {manifest.run_branch}"
    stamp = covering_stamp(repo, manifest)
    if stamp is not None:
        return f"its commits are {stamp.landed()}"
    chain = chain_ref_for(manifest.session_id)
    if chain_tip(repo, chain) is not None:
        return f"its commits are on {chain}"
    return "it recorded no commits"


def turn_replay_allowed(
    session_dir: Path,
    next_iteration: int,
    confirm: Callable[[int, tuple[str, ...]], bool],
) -> bool:
    """Decide whether resume may proceed given the mid-turn-crash marker.

    A stale marker (its turn completed) is cleared silently. A marker for the turn
    about to re-run is a real mid-turn crash whose tools may have partially
    applied, so the front-end decides; approval leaves the marker for the replayed
    turn to supersede.

    Args:
        session_dir: The session's directory.
        next_iteration: The turn the snapshot resumes at.
        confirm: Asked, with the crashed turn and its tools, whether to replay.

    Returns:
        Whether to proceed.
    """
    marker_path = session_dir / TURN_IN_FLIGHT_NAME
    marker = read_turn_marker(marker_path)
    if marker is None:
        return True
    iteration, tools = marker
    if iteration < next_iteration:
        clear_turn_marker(marker_path)  # stale: the turn completed, never ask
        return True
    return confirm(iteration, tools)


def snapshot_head_mismatch(
    snapshot_path: Path, repo_root: Path, *, chain_ref: str
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
        current_head = chain_tip(repo_root, chain_ref)
    except GitError:
        return None
    if current_head is None or current_head == snap_head:
        return None
    if is_ancestor(repo_root, snap_head, current_head):
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
    return pinned or "inferred"


def resume_task(  # noqa: C901, PLR0911, PLR0912, PLR0915  # every way a resume is refused, in preflight order
    config_path: Path | None,
    session_id: str,
    *,
    frontend: SessionFrontend,
    force: bool,
    started_at: float,
    tui: bool = False,
    budget_overrides: BudgetOverrides | None = None,
    sandbox_overrides: SandboxOverrides | None = None,
    preset: str = "",
    steer: str = "",
    interactive: bool = False,
    model: str = "",
    reporter: Reporter = STDIO_REPORTER,
) -> int:
    """Resume a paused or crashed session from its snapshot.

    The budget is a fresh ceiling per execution. The process cwd is the
    repository; a fork's execution drives the fork's own worktree.

    Args:
        config_path: An explicit config file, or None for discovery.
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
    repo = Path.cwd()
    state = state_dir(repo)
    if steer.strip() and (problem := steer_problem(steer)) is not None:
        reporter.error(f"--steer: {problem}")
        return 2
    # One resolver across buckets: a runs/-only fallback would pick a run over an ask.
    try:
        layout = resolve_session(state, session_id)
    except SessionIdError as exc:
        reporter.error(str(exc))
        return 2
    session_id = layout.session_id
    # The manifest first: clearing state before it would clobber a session resume refuses.
    try:
        manifest = read_manifest(layout.session_dir)
        # A `--model` set on the run replays unless this resume sets its own.
        recorded = manifest.models.replay_driver
        route: str | ModelRoute | None = model or (
            ModelRoute(recorded.provider, recorded.model) if recorded is not None else None
        )
        mode = manifest.session_mode()
    except ManifestError as exc:
        reporter.error(f"cannot resume {session_id}: {exc}")
        return 2
    if manifest.fanout is not None:
        reporter.error(
            f"cannot resume {session_id}: a fan-out coordinator is not resumable;"
            f" `agent6 sessions show {session_id}` lists its lanes."
        )
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
    worker_lock_fd = acquire_single_writer(layout.session_dir)
    if worker_lock_fd is None:
        reporter.err(SINGLE_WRITER_BUSY.format(rid=session_id))
        return 2
    # A finished run has nothing to continue without --steer; every other ending resumes.
    new_work = finished_needs_new_work(layout.session_dir)
    if not steer.strip() and new_work:
        reporter.refuse(needs_new_work_refusal(session_id))
        release_single_writer(worker_lock_fd)
        return 2
    # The prompt ids reset on resume, so stale answers go; markers since this start stay.
    clear_pending_answers(layout.session_dir, started_at=started_at)
    # Seeded after the clear, which drops steer files.
    if steer.strip() and not submit_steer(layout.session_dir, steer.strip()):
        reporter.error("could not write the initial steer request")
        release_single_writer(worker_lock_fd)
        return 2

    detach_requested = False
    handed_to_run_task = False  # a parked submission: run_task owns its lifecycle
    cfg: Config | None = None  # the finally reads it for a detach
    repo_lock_fd: int | None = None
    try:
        role = session_kind(mode).role

        if manifest.parked_task:
            # Nothing ever ran: run_task starts it fresh under the same id and takes both
            # locks itself, so ours is released first.
            try:
                # replay_preset: a config-selected preset re-resolves rather than ranking as a flag.
                effective = load_session_config(
                    repo,
                    config_path,
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
            release_single_writer(worker_lock_fd)
            worker_lock_fd = None
            handed_to_run_task = True
            return run_task(
                cfg,
                saved_task,
                frontend=frontend,
                started_at=started_at,
                session_id=session_id,
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
            repo_lock_fd = acquire_repo_writer(state, cwd, session_id)
            if repo_lock_fd is None:
                holder = repo_writer_holder(state, cwd) or "another run"
                reporter.refuse(
                    f"run {holder!r} is already driving this checkout; a"
                    " second run-mode worker would interleave auto-commits on the"
                    " one working tree. Wait for it, or stop it first:\n"
                    f"    agent6 stop {holder}"
                )
                return 2

        snapshot_path = layout.session_dir / "loop_state.json"
        if not snapshot_path.is_file():
            reporter.error(f"no resume snapshot at {snapshot_path}; nothing to resume.")
            return 2

        # An ask is read-only and may run outside a git repo, so it skips the git preflight.
        writes_code = mode != "ask"
        # The no-repo guard runs before any git-touching check.
        if writes_code and not require_git_repo(cwd, reporter=reporter):
            return 2

        # An unreadable or outdated snapshot refuses here, before anything is touched.
        try:
            snapshot = load_session_snapshot(snapshot_path)
        except (ValueError, OSError) as exc:
            reporter.error(str(exc))
            return 1
        if not turn_replay_allowed(
            layout.session_dir, snapshot.next_iteration, frontend.confirm_replay_after_crash
        ):
            reporter.err(
                "Not resuming: inspect the working tree and the run log first"
                " (the marker stays; resume again and answer yes to replay)."
            )
            return 2
        manifest_preset = manifest.harness.replay_preset
        resume_base_sha = manifest.base_sha
        run_branch = manifest.run_branch or ""

        # A rewritten chain ref would leave the model reasoning about a changed record.
        mismatch = (
            snapshot_head_mismatch(snapshot_path, cwd, chain_ref=chain_ref_for(session_id))
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

        try:
            effective = load_session_config(
                repo,
                config_path,
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

        # Needs the config: the away grant is per scope, one per configured MCP server.
        settle_away_mode(layout.session_dir, cfg)

        if not route_preflight(cfg, role, reporter=reporter, model_flag=route_text(route)):
            return 2

        try:
            isolation = select_isolation(
                cfg,
                cwd=cwd,
                confirm_unconfined=frontend.confirm_unconfined_autorun,
                reporter=reporter,
                explicit_leaves=explicit_leaves,
                worktree_git_dir=manifest.worktree_git_dir,
            )
        except SessionRefusedError as refusal:
            return refusal.rc

        identity = CommitIdentity(name=cfg.git.commit.name, email=cfg.git.commit.email)
        # The no-repo guard ran above.
        if writes_code:
            try:
                verify_git_identity(cwd, identity)
            except GitError as exc:
                reporter.error(str(exc))
                return 2

        transcript_sink = TranscriptSink(layout.transcripts_dir)
        events = EventSink(layout.logs_path)
        # The execution's one gate to the operator, whichever front-end answers.
        prompts = OperatorPrompts(
            approver=frontend.build_approver(layout.session_dir),
            questioner=frontend.build_questioner(layout.session_dir),
            journal=events.emit,
            session_dir=layout.session_dir,
        )

        warn_install_inside_workspace(cwd, reporter=reporter)
        for line in agents_md_notices(cwd):
            reporter.note(line)

        tui_enabled = frontend.should_spawn_tui(tui, interactive, mode)
        refusal = headless_approval_refusal(
            cfg,
            tui_enabled=tui_enabled,
            # The raw env first, so a typo refuses here as on run; else the recorded choice.
            away=os.environ.get("AGENT6_DETACHED_AWAY", "") or effective_away(layout.session_dir),
            can_ask=frontend.capabilities.can_ask,
            clamped=session_kind(mode).clamps_commands,
        )
        if refusal is not None:
            reporter.refuse(refusal)
            return 2
        parking = headless_parking_note(
            cfg,
            tui_enabled=tui_enabled,
            away=effective_away(layout.session_dir),
            can_ask=frontend.capabilities.can_ask,
        )
        if parking is not None:
            reporter.note(parking)

        def _gate(cfg: Config, _budget: BudgetTracker) -> Config:
            # The run's resolved gate is reused, not re-inferred: the snapshot's, unless a
            # newer adopted or unadopted pin outranks it; config outranks both.
            pinned_origin, pinned_gate = "", ()
            with contextlib.suppress(ManifestError, OSError):
                pinned = read_manifest(layout.session_dir).harness
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
            cfg = drop_gate_if_unrunnable(cfg, session_dir=layout.session_dir, reporter=reporter)
            # A withheld gate is neither a reuse nor a change.
            withheld = bool(gate_before) and not cfg.harness.verify_command
            if reused and not withheld:
                reporter.note(f"reusing this run's verify command: {gate_text(replay_gate)}")
            # The manifest says which gate this execution used.
            if tuple(pinned_gate) != cfg.harness.verify_command and not withheld:
                # Both directions: the frozen system prompt names the old gate either way.
                reporter.note(
                    "this run's verify gate changed:"
                    f" was {gate_text(tuple(pinned_gate))},"
                    f" now {gate_text(cfg.harness.verify_command)}"
                )
            pin_gate(
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
            from agent6.app.undo import undo_fork  # noqa: PLC0415

            return undo_fork(config_path, session_id, cwd=repo, reporter=reporter)

        untracked_at_start = read_untracked_at_start(layout.session_dir)
        if mode == "run":
            # An untracked file the chain does not hold and no tool wrote arrived between
            # executions, so it is the operator's; a git failure refuses, since this decides
            # what the run may commit.
            try:
                arrived = (
                    untracked_paths(cwd)
                    - untracked_at_start
                    - tree_paths(cwd, chain_ref_for(session_id))
                    - _paths_the_run_wrote(layout.logs_path)
                )
                if arrived:
                    untracked_at_start = untracked_at_start | arrived
                    write_untracked_at_start(layout.session_dir, untracked_at_start)
                    named = sorted(arrived)
                    shown = ", ".join(named[:4])
                    if len(named) > 4:
                        shown += f", +{len(named) - 4} more"
                    reporter.note(
                        f"left out of this run's commits as yours: {shown} (arrived"
                        " between executions, unwritten by any tool of the run)"
                    )
            except (GitError, OSError) as exc:
                reporter.error(f"cannot tell the run's files from the operator's: {exc}")
                return 2
        # Every preflight passed: the operator's new choices are recorded from here.
        if preset:
            stamp_preset(layout.session_dir, preset)
        if (flagged := flag_route(cfg, mode, model)) is not None:
            stamp_model(layout.session_dir, flagged)
        if steer.strip():
            # A steer that is the work names the run, for a finished run or a fork still
            # carrying its source's task.
            if new_work:
                stamp_task(layout.session_dir, steer.strip())
            elif manifest.parent_session_id:
                stamp_fork_task(
                    layout.session_dir,
                    steer.strip(),
                    source_dir=layout.session_dir.parent / manifest.parent_session_id,
                )
        # Written once the preflight passed: a refused resume never had a live worker.
        write_worker_pid(layout.session_dir, os.getpid())
        # This execution's models and policy, for `agent6 exec` and the policy surfaces.
        stamp_execution(layout.session_dir, cfg, mode, isolation)
        if mode == "run":
            # The next auto-commit takes the previous execution's tail and any operator edit
            # since; the operator's untracked files stay out.
            with contextlib.suppress(GitError, OSError):
                dirty = chain_dirty_paths(
                    cwd,
                    chain_ref_for(session_id),
                    resume_base_sha,
                    5,
                    exclude=untracked_at_start,
                )
                if dirty:
                    named = ", ".join(dirty[:4]) + (", ..." if len(dirty) > 4 else "")
                    reporter.note(
                        f"the tree holds changes no commit has ({named});"
                        " this execution's next commit takes them"
                    )
        end = run_execution(
            cfg,
            layout,
            ExecutionInputs(
                session_id=session_id,
                mode=mode,
                role=role,
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
                    clear_worker_pid(layout.session_dir)
        finally:
            release_single_writer(repo_lock_fd)
            release_single_writer(worker_lock_fd)
        if detach_requested and cfg is not None:
            detach_to_background(
                frontend=frontend,
                cfg=cfg,
                layout=layout,
                cwd=repo,
                flags=override_flags(
                    budget_overrides, sandbox_overrides, flag_route(cfg, mode, route)
                ),
                reporter=reporter,
            )
