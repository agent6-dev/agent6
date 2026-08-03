# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The `agent6 resume` lifecycle: pick a paused or crashed run back up from its
snapshot. `ui/cli/resume.py` adapts argv and injects the same
:class:`agent6.app.run.RunFrontend` seam `run_task` uses."""

from __future__ import annotations

import contextlib
import json
import os
import sys
from pathlib import Path

from agent6.app._session import (
    SessionRefused,
    build_session_providers,
    build_session_tools,
    select_isolation,
    start_isolation,
)
from agent6.app._setup import (
    BudgetOverrides,
    SandboxOverrides,
    check_provider_keys,
    start_mcp_manager_if_enabled,
)
from agent6.app.egress import (
    EgressGuard,
    lane_launcher,
    spawn_detached,
    stop_egress,
)
from agent6.app.finalize import (
    finalize_auto_merge,
    fire_notify_hook,
    print_interrupt_end,
    print_run_end,
    run_exit_code,
)
from agent6.app.manifest import stamp_verify_gate
from agent6.app.preflight import (
    infer_verify_if_unset,
    require_git_repo,
    warn_if_headless_ask,
)
from agent6.app.providers import (
    role_temperature,
)
from agent6.app.reporter import STDIO_REPORTER, Reporter
from agent6.app.run import RunFacts, RunFrontend, apply_spawned_away_default, run_task
from agent6.config import (
    Config,
    ConfigError,
    role_for_mode,
)
from agent6.config.layer import (
    load_effective,
    resolved_state_dir,
)
from agent6.events import EventSink, EventWriteError
from agent6.git_ops import (
    CommitIdentity,
    GitError,
    branch_tip_sha,
    create_branch,
    is_ancestor,
    set_repo_hook_policy,
    verify_git_identity,
)
from agent6.git_ops import (
    status as git_status,
)
from agent6.paths import (
    chown_to_real_user,
)
from agent6.providers import (
    TranscriptSink,
)
from agent6.runs.id import RunIdError, resolve_run_id
from agent6.runs.ipc import (
    clear_away_mode,
    clear_compact_request,
    clear_pending_answers,
    clear_stop_request,
    clear_worker_pid,
    read_compact_request,
    request_steer,
    session_allow_set,
    stop_request_pending,
    write_steer_answer,
    write_worker_pid,
)
from agent6.runs.layout import RunLayout
from agent6.runs.lock import (
    SINGLE_WRITER_BUSY,
    acquire_repo_writer,
    acquire_single_writer,
    release_single_writer,
    repo_writer_holder,
)
from agent6.runs.manifest import ManifestError, read_manifest
from agent6.tools.dispatch import ToolDispatcher
from agent6.viewmodel import newest_run_dir
from agent6.workflows._run_state import RunReason, load_run_snapshot
from agent6.workflows.loop import ResumeError, Workflow


def ensure_on_run_branch(cwd: Path, layout: RunLayout) -> str | None:
    """Check out the run's branch if HEAD isn't already on it.

    The loop's per-step commits land on whatever branch HEAD points at, so a
    resume must be on the run's branch. ``run_task`` checks it out up front, but
    two paths reach resume off the run branch: ``agent6 fork`` cuts
    ``agent6/<id>`` additively (never switching to it), and an operator may have
    moved branches since the original run. Either way, without this the work
    silently lands on the operator's current branch and the run branch stays
    empty (so ``runs diff`` shows nothing).

    Reads ``run_branch`` from the manifest. Returns None when there's nothing to
    do (no branch recorded, or already on it) or after a clean checkout; returns
    an error string when a switch is needed but the working tree is dirty.
    """
    try:
        manifest = read_manifest(layout.run_dir)
    except ManifestError:
        return None
    run_branch = manifest.run_branch
    try:
        st = git_status(cwd)
    except GitError:
        st = None
    # Nothing to do: branch_per_run was off (no run_branch), git unreadable, or
    # already on the run branch. Commits then land on HEAD as before.
    if not run_branch or st is None or st.branch == run_branch:
        return None
    # Only MODIFIED tracked files block the switch; untracked files are carried
    # across a checkout fine (and a rare untracked-vs-target collision is caught
    # by the create_branch error below), so don't refuse on those.
    if st.modified_count > 0:
        return (
            f"ERROR: resume needs to switch to this run's branch {run_branch!r}, but the "
            "working tree has uncommitted changes to tracked files. Commit or stash them "
            f"(or run `git checkout {run_branch}` yourself), then resume."
        )
    try:
        create_branch(cwd, run_branch)  # idempotent: checks out the existing branch
    except GitError as exc:
        return f"ERROR: could not switch to run branch {run_branch!r}: {exc}"
    return None


def snapshot_head_mismatch(
    snapshot_path: Path, repo_root: Path, *, run_branch: str = ""
) -> tuple[str, str] | None:
    """(snapshot head, resume-onto head) when the code resume would continue on
    DIVERGED from the run's last snapshot, else None.

    The head compared is the one resume will commit on top of: the run branch's
    tip when *run_branch* resolves, so the guard needs NO checkout and runs
    before any workspace mutation; the current HEAD otherwise (branch_per_run
    off, or a deleted branch the checkout step re-cuts at HEAD).

    Divergence, not mere movement: the run's own per-step commits advance the
    branch forward from the snapshot between snapshot writes (a turn commits,
    then a critic/metric call runs before the next snapshot), so a kill in that
    window leaves the tip ahead of the recorded head_sha on the SAME line. That
    must resume cleanly. Only refuse when the tip is not a descendant of the
    snapshot head -- an operator commit on another line, a rebase, a reset, or a
    snapshot commit that git-gc made unreachable -- i.e. the model would resume
    against code that changed under it. Working-tree (uncommitted) divergence
    is not checked; only committed history.

    Best-effort: the snapshot records head_sha as "" when git was unreadable at
    write time (skip), a corrupt snapshot file is left for the loud
    resume-snapshot load to report (skip), and a non-repo raises nothing here
    (the caller's require_git_repo already ran).
    """
    snap_head = ""
    with contextlib.suppress(OSError, ValueError):
        loaded = json.loads(snapshot_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            # Raw single-key peek (must not raise); "head_sha" is
            # RunSnapshot.head_sha -- keep in sync on a field rename.
            snap_head = str(loaded.get("head_sha") or "")
    if not snap_head:
        return None
    current_head = branch_tip_sha(repo_root, run_branch) if run_branch else None
    if current_head is None:
        try:
            current_head = git_status(repo_root).head_sha
        except GitError:
            return None
    if not current_head or current_head == snap_head:
        return None
    if is_ancestor(repo_root, snap_head, current_head):
        # The tip moved forward from the snapshot on the same line (the run's
        # own commits): not divergence.
        return None
    return (snap_head, current_head)


def leg_gate_origin(*, configured: bool, has_gate: bool, pinned: str) -> str:
    """Where THIS leg's gate came from: config outranks the run's pin, the pin
    stands when the leg reused it (an adopted gate stays adopted), and a leg
    that had to re-infer says so."""
    if configured:
        return "configured"
    if not has_gate:
        return ""
    return pinned or "inferred"


def resume_task(  # noqa: PLR0911, PLR0912, PLR0915
    config_path: Path | None,
    run_id: str,
    *,
    frontend: RunFrontend,
    force: bool,
    tui: bool = False,
    budget_overrides: BudgetOverrides | None = None,
    sandbox_overrides: SandboxOverrides | None = None,
    preset: str = "",
    steer: str = "",
    reporter: Reporter = STDIO_REPORTER,
) -> int:
    """Resume a paused/crashed run from its snapshot.

    Mirrors ``run_task`` setup but uses the existing run id, refuses
    if no ``loop_state.json`` snapshot exists, and calls ``wf.resume()``
    instead of ``wf.run(task)``. A safety check refuses when the
    workspace HEAD DIVERGED from the snapshot (a rebase/reset/commit on
    another line); plain forward movement on the same line resumes
    cleanly. ``--force`` overrides the refusal.

    NOTE: token budget on resume is a FRESH ceiling, not a continuation
    of the prior run's accounting. Each ``agent6 resume`` invocation
    starts at 0 against the ``[budget]`` ledgers. This is by design - the budget is a per-
    invocation runaway-cost circuit breaker.
    """
    cwd = Path.cwd()
    state_dir = resolved_state_dir(cwd)
    runs_dir = state_dir / "runs"
    if not run_id:
        # "resume my last run" -- the common recovery case, matching `runs *`.
        latest = newest_run_dir([runs_dir])
        if latest is None:
            reporter.err(f"ERROR: no runs under {runs_dir}; nothing to resume.")
            return 2
        run_id = latest.name
        reporter.err(f"[agent6] resuming most recent run: {run_id}")
    try:
        resolved = resolve_run_id(runs_dir, run_id)
    except RunIdError as exc:
        reporter.err(f"ERROR: {exc}")
        return 2
    run_id = resolved
    layout = RunLayout(state_dir=state_dir, run_id=run_id)
    if not layout.run_dir.is_dir():
        reporter.err(f"ERROR: no such run dir: {layout.run_dir}")
        return 2
    # One authoritative writer per run dir (see acquire_single_writer). Refuse a
    # second resume of a still-live run before touching any shared state.
    worker_lock_fd = acquire_single_writer(layout.run_dir)
    if worker_lock_fd is None:
        reporter.err(SINGLE_WRITER_BUSY.format(rid=run_id))
        return 2
    # Drop a prior session's stale answer files (the id counters reset
    # on resume, an old answer must not be read instead of re-prompting).
    clear_pending_answers(layout.run_dir)
    if steer.strip():
        # --steer: queue the operator's follow-up as the first steering
        # instruction. Seeded AFTER the stale-state clear (which drops steer
        # files), so the loop's steer poll injects it at its first boundary.
        request_steer(layout.run_dir)
        write_steer_answer(layout.run_dir, steer.strip())
    if sys.stdin.isatty():  # a foreground start clears a stale detach away-mode
        clear_away_mode(layout.run_dir)
    else:
        # A front-end (web/TUI) or a detach spawns resume with no terminal; honor
        # AGENT6_DETACHED_AWAY so ask_user/approvals WAIT for a viewer instead of
        # fabricating an empty answer. Mirrors run_task; a pure headless resume
        # (CI) sets no env, so this is a no-op and keeps the non-hanging default.
        apply_spawned_away_default(layout.run_dir)
    # Record this worker's pid so `agent6 runs show` can probe liveness even while
    # the worker is blocked in a long provider call (which emits no events).
    write_worker_pid(layout.run_dir, os.getpid())

    detach_requested = False
    cfg: Config | None = None  # bound below; the finally reads it (detach away-mode)
    guard = EgressGuard()  # replaced at egress start; the finally tears it down
    repo_lock_fd: int | None = None
    try:
        # The original run's manifest drives resume: `mode` (a plan run resumes
        # read-only with the plan tools, never as a write run), `preset` (resume
        # has no --preset flag), `base_sha` (the review-panel diff base), and
        # `run_branch` (the head guard + the checkout below). Read FIRST: a
        # PARKED run (manifest carries parked_task, no snapshot exists) is
        # started fresh below instead of hitting the no-snapshot refusal.
        # `mode` is security-relevant: a damaged run dir (unreadable, corrupt, or
        # an unknown mode value) must NOT fall open to the more-privileged "run"
        # (write) mode. read_manifest / validated_mode fail loud on any of those --
        # the underlying cause carries in the ManifestError detail -- rather than
        # silently escalating a plan run to a write run.
        try:
            manifest = read_manifest(layout.run_dir)
            mode = manifest.validated_mode()
        except ManifestError as exc:
            reporter.err(f"ERROR: cannot read run manifest {layout.manifest_path}: {exc}")
            return 2
        role = role_for_mode(mode)

        if manifest.parked_task:
            # Parked at submission (the checkout was busy): nothing ever ran, so
            # "resume" is its fresh start. Hand the verbatim saved task to
            # run_task under the same run id; it re-acquires both locks itself
            # (and re-parks with a fresh message if the checkout is STILL busy),
            # so release ours first. Its manifest rewrite clears parked_task.
            try:
                # replay_preset, not the raw stamped name: a config-selected
                # preset re-resolves from the same files, and handing its name
                # back would make _select_preset rank it as a flag (the same
                # rule as the snapshot-resume path below).
                cfg = load_effective(
                    cwd, config_path, preset=preset or manifest.workflow.replay_preset
                ).config
                set_repo_hook_policy(cfg.git.run_repo_hooks)
                if budget_overrides is not None:
                    cfg = budget_overrides.apply(cfg)
                if sandbox_overrides is not None:
                    cfg = sandbox_overrides.apply(cfg)
                cfg.require_runnable(role)
            except ConfigError as exc:
                reporter.err(f"CONFIG ERROR:\n{exc}")
                return 2
            reporter.err(
                f"[agent6] run {run_id!r} was parked at submission (the checkout was"
                " busy); starting it now."
            )
            saved_task = manifest.parked_task
            clear_worker_pid(layout.run_dir)
            release_single_writer(worker_lock_fd)
            worker_lock_fd = None
            return run_task(
                cfg,
                saved_task,
                frontend=frontend,
                run_id=run_id,
                mode=mode,
                budget_overrides=budget_overrides,
                sandbox_overrides=sandbox_overrides,
                preset=preset,
                # Pin the ORIGINAL stamp ONLY for a FLAG-selected preset whose
                # veto must survive, and only when this resume sets no --preset
                # of its own. A CONFIG-selected preset (from_flag False) re-
                # resolves from the CURRENT config below, so pinning the manifest's
                # old NAME would show a stale preset if the config changed since;
                # pass None and let run_task derive it from the re-resolved cfg,
                # like a fresh run. A resume that DOES set --preset is a fresh
                # flag choice, so run_task's own derivation stamps it.
                preset_stamp=(
                    (manifest.workflow.preset, True)
                    if (not preset and manifest.workflow.preset_from_flag)
                    else None
                ),
                # Hand --steer through: the bridge files seeded above are
                # wiped by run_task's own stale-state clear, so a parked
                # resume's follow-up was silently lost.
                initial_steer=steer,
                reporter=reporter,
            )

        # One live run-mode worker per CHECKOUT (see acquire_repo_writer): a
        # resumed run drives the shared working tree exactly like a fresh one.
        if mode == "run":
            repo_lock_fd = acquire_repo_writer(state_dir, run_id)
            if repo_lock_fd is None:
                holder = repo_writer_holder(state_dir) or "another run"
                reporter.err(
                    f"REFUSING: run {holder!r} is already driving this checkout; a"
                    " second run-mode worker would interleave auto-commits on the"
                    " one working tree. Wait for it, or stop it first:\n"
                    f"    agent6 runs stop {holder}"
                )
                return 2

        snapshot_path = layout.run_dir / "loop_state.json"
        if not snapshot_path.is_file():
            reporter.err(f"ERROR: no resume snapshot at {snapshot_path}; nothing to resume.")
            return 2

        # Friendly no-repo guard BEFORE any git-touching check (which would
        # otherwise print zeroed-out heads first, then the real error).
        if not require_git_repo(cwd):
            return 2

        # Snapshot version guard, BEFORE maybe_start_egress spawns a broker +
        # netns: a v1 snapshot cannot be resumed, and the refusal must land in
        # preflight (like `fork`) with the checkout and the network untouched --
        # not after a broker + preamble already printed. wf.resume() re-validates
        # the same snapshot; a corrupt/old file refuses identically here (exit 1).
        try:
            load_run_snapshot(snapshot_path)
        except (ValueError, OSError) as exc:
            reporter.err(f"ERROR: {exc}")
            return 1
        manifest_preset = manifest.workflow.replay_preset
        resume_base_sha = manifest.base_sha
        run_branch = manifest.run_branch or ""

        # Safety check: refuse when the code resume would continue on DIVERGED
        # from the run's last snapshot (a rebase, reset, or a commit on another
        # line would leave the model reasoning about code that changed under
        # it). Compared against the run branch's tip, so no checkout is needed;
        # plain forward movement on the same line -- the run's own per-step
        # commits -- resumes cleanly. The snapshot records head_sha best-effort
        # ("" when git was unreadable at write time); skip the check then, and
        # let the loud snapshot load below handle a corrupt file.
        mismatch = snapshot_head_mismatch(snapshot_path, cwd, run_branch=run_branch)
        if mismatch is not None:
            snap_head, onto_head = mismatch
            reporter.err(
                "GUARD: the code this run would resume onto diverged from its last snapshot."
            )
            reporter.err(f"  snapshot head: {snap_head}")
            reporter.err(f"  resume onto:   {onto_head}")
            if not force:
                reporter.err("REFUSING to resume. Re-run with --force to override.")
                return 1

        try:
            cfg = load_effective(Path.cwd(), config_path, preset=preset or manifest_preset).config
            set_repo_hook_policy(cfg.git.run_repo_hooks)
            if budget_overrides is not None:
                cfg = budget_overrides.apply(cfg)
            if sandbox_overrides is not None:
                cfg = sandbox_overrides.apply(cfg)
            cfg.require_runnable(role)
        except ConfigError as exc:
            reporter.err(f"CONFIG ERROR:\n{exc}")
            return 2

        try:
            isolation = select_isolation(
                cfg, confirm_unconfined=frontend.confirm_unconfined_autorun, reporter=reporter
            )
        except SessionRefused as refusal:
            return refusal.rc

        missing = check_provider_keys(cfg)
        if missing is not None:
            reporter.err(missing)
            return 2

        identity = CommitIdentity(
            name=cfg.git.commit.name,
            email=cfg.git.commit.email,
            coauthor=cfg.git.commit.coauthor,
        )
        # (no-repo guard already ran above, before the resume head guard)
        try:
            verify_git_identity(cwd, identity)
        except GitError as exc:
            reporter.err(f"ERROR: {exc}")
            return 2

        transcript_sink = TranscriptSink(layout.transcripts_dir)
        events = EventSink(layout.logs_path)

        try:
            guard = start_isolation(
                cfg, isolation, agent6_exe=frontend.agent6_exe, reporter=reporter
            )
        except SessionRefused as refusal:
            return refusal.rc

        # Get onto the run's branch so the loop's commits land there (a fork
        # cuts agent6/<id> without checking it out; the operator may have moved
        # branches since the original run). This is the ONLY workspace mutation
        # in preflight, and deliberately the LAST step: every refusal above
        # exits with the operator's checkout untouched. From here on a failure
        # is a failed RUN, parked on the run branch like any crashed run (the
        # end-of-run banner says how to switch back).
        branch_err = ensure_on_run_branch(cwd, layout)
        if branch_err is not None:
            reporter.err(branch_err)
            return 2

        tui_enabled = frontend.should_spawn_tui(tui, False, mode)
        warn_if_headless_ask(cfg, tui_enabled=tui_enabled)
        stream_text, console_stream = frontend.stream_modes(tui_enabled)
        if console_stream:
            frontend.attach_console_view(events)
        session = build_session_providers(
            cfg, role=role, events=events, transcript_sink=transcript_sink, stream_text=stream_text
        )
        budget = session.budget
        # Resume reuses the verify command the ORIGINAL run resolved (stored in the
        # snapshot), so the tool list, prompt, and commit branch stay consistent with
        # the frozen system prompt -- never re-inferring (which could flip and
        # diverge). Re-infer only when the snapshot can't be read (missing/corrupt/an
        # old format resume refuses loudly right after), and only when the operator
        # hasn't since pinned a command in config.
        leg_configured = bool(cfg.workflow.verify_command)
        if not cfg.workflow.verify_command:
            snap_verify: tuple[str, ...] | None = None
            if snapshot_path.is_file():
                try:
                    snap_verify = load_run_snapshot(snapshot_path).verify_command
                except (ValueError, OSError, KeyError):
                    snap_verify = None
            if snap_verify is None:  # snapshot unreadable: re-infer as the original did
                cfg = infer_verify_if_unset(
                    cfg,
                    cwd,
                    mode=mode,
                    events=events,
                    transcript_sink=transcript_sink,
                    budget=budget,
                )
            elif snap_verify:  # () means the original run was gateless: stay gateless
                cfg = cfg.with_inferred_verify(snap_verify)
                reporter.err(f"[agent6] reusing this run's verify command: {' '.join(snap_verify)}")
        # Re-pin for this leg: config outranks the pin, the pin outranks a
        # re-inference, and the manifest has to say which one this leg used.
        with contextlib.suppress(ManifestError, OSError):
            pinned = read_manifest(layout.run_dir).workflow
            stamp_verify_gate(
                layout.run_dir,
                cfg.workflow.verify_command,
                leg_gate_origin(
                    configured=leg_configured,
                    has_gate=bool(cfg.workflow.verify_command),
                    pinned=pinned.verify_origin,
                ),
            )

        # Bound now, not read in the handler: these never change for the leg.
        facts_model, facts_commands = session.rm_role.model, cfg.sandbox.run_commands

        def _run_facts() -> RunFacts:
            spend, partial = budget.estimate_usd()
            return RunFacts(
                spend_usd=spend,
                spend_partial=partial,
                model=facts_model,
                run_commands=facts_commands,
                isolation=isolation,
            )

        steer_state = frontend.make_steer_state(events, layout.run_dir, _run_facts)

        result = None
        interrupted = False
        dispatcher: ToolDispatcher | None = None
        # Spawned inside the try so the finally below tears it down even if a
        # spawn (MCP) fails.
        mcp_manager = None
        try:
            reporter.err(f"[agent6] resume run id: {run_id}")

            mcp_manager = start_mcp_manager_if_enabled(cfg, reporter=reporter)

            loop_log = frontend.loop_logger(mode)
            tools = build_session_tools(
                cfg,
                cwd=cwd,
                state_dir=state_dir,
                layout=layout,
                isolation=isolation,
                mode=mode,
                events=events,
                approver=frontend.build_approver(layout.run_dir, events),
                questioner=frontend.build_questioner(layout.run_dir, events),
                loop_log=loop_log,
                mcp_manager=mcp_manager,
                rm_role=session.rm_role,
            )
            curator = tools.curator
            dispatcher = tools.dispatcher
            cfg = tools.cfg
            wf = Workflow(
                root=cwd,
                config=cfg,
                provider=session.provider,
                dispatcher=dispatcher,
                logger=loop_log,
                events=events,
                curator=curator,
                steer_requested=steer_state.requested,
                steer_clear=steer_state.clear,
                steer_reset=steer_state.reset_stage,
                steer_prompt=steer_state.prompt,
                # "Compact now" from a front-end: the same file-bridge
                # pattern as steer, honored at the next pre-call boundary.
                compact_requested=lambda: read_compact_request(layout.run_dir),
                compact_clear=lambda: clear_compact_request(layout.run_dir),
                stop_requested=lambda: stop_request_pending(layout.run_dir),
                stop_clear=lambda: clear_stop_request(layout.run_dir),
                should_abort=steer_state.abort_pending,
                should_interrupt=steer_state.interrupt,
                # `/parallel` steer dispatch: the coordinator's group spawner
                # (None in plan resume, and inside a lane -- depth 1). Under a
                # strict egress netns, lane_launcher(guard) hands lanes the same
                # host-spawner escape this resume's own detach uses.
                lane_spawner=frontend.build_coordinator_spawner(
                    cfg,
                    cwd,
                    state_dir,
                    mode,
                    run_id,
                    budget_overrides.max_usd if budget_overrides is not None else None,
                    sandbox_overrides.auto_approve if sandbox_overrides is not None else False,
                    lane_launcher(guard),
                ),
                budget=budget,
                resume_state_path=snapshot_path,
                mode=mode,
                plan_output_path=(layout.run_dir / "plan.md" if mode == "plan" else None),
                critic_provider=session.critic_provider,
                critic_mode=cfg.review.trigger,
                critic_period=cfg.review.period,
                review_seats=session.review_seats,
                review_decision=cfg.review.decision,
                review_quorum=cfg.review.quorum,
                review_max_total_rejections=cfg.review.max_total_rejections,
                review_budget_fraction=cfg.review.budget_fraction,
                review_concurrency=cfg.review.concurrency,
                base_sha=resume_base_sha,
                temperature=role_temperature(cfg, role),
                critic_temperature=role_temperature(cfg, "reviewer"),
                summariser_provider=session.summariser_provider,
                compact_drop_at_chars=tools.compact_drop_at_chars,
                compact_summarise_at_chars=tools.compact_summarise_at_chars,
                context_summary_max_tokens=cfg.context.summary_max_tokens,
                compact_elision_gists=cfg.context.elision_gists,
            )
            try:
                with frontend.tui_session(layout.run_dir, tui_enabled):
                    result = wf.resume()
            except ResumeError as exc:
                reporter.err(f"ERROR: {exc}")
                return 1
            except KeyboardInterrupt:
                interrupted = True
                reporter.err("\n[agent6] resume interrupted")
                # suppress: the interrupt exit (130 + resume hint) must not be
                # masked by a dead journal.
                reason: RunReason = "interrupted"
                with contextlib.suppress(EventWriteError):
                    events.emit(
                        "run.end",
                        reason=reason,
                        iterations=wf.iterations_reached,
                        all_passed=False,
                    )
            except Exception:
                # Any other escape (a broken stdout pipe from `| head`, an
                # unexpected fault) also leaves the loop without a run.end,
                # and the outer finally then clears worker.pid -- the only
                # immediate liveness evidence -- so every surface read the
                # dead run as "running" until the silence window expired.
                # Record the end, then let the error surface as before.
                with contextlib.suppress(EventWriteError):
                    events.emit(
                        "run.end",
                        reason="crashed",
                        iterations=wf.iterations_reached,
                        all_passed=False,
                    )
                raise
        finally:
            steer_state.restore()
            if dispatcher is not None:
                dispatcher.close()
            if mcp_manager is not None:
                mcp_manager.close()
            # Egress teardown is owned by the outer finally (a single call).
            # Doing it here too would reap the broker pid, then the auto-merge
            # git subprocesses and the notify hook below could recycle it before
            # the outer close() signalled the pid again.
            if (
                not interrupted
                and result is not None
                and result.completed
                # Never auto-merge a red gate: `completed` means the agent
                # stopped deliberately, not that the work verified.
                and result.verified != "failed"
                and cfg.git.auto_merge
            ):
                finalize_auto_merge(cwd, layout=layout, cfg=cfg)
            # Never leave root-owned run state in the user's repo (sudo case).
            chown_to_real_user(state_dir)

        if interrupted:
            # Same close the run path prints: the leg's spend, the cross-leg
            # run total, the resume hint, the on-the-run-branch note.
            print_interrupt_end(layout=layout, budget=budget)
            return 130
        if result is None:
            return 1

        if result.reason == "detached":
            detach_requested = True
            reporter.out(f"\n[agent6] detached: {layout.run_id} continues in the background.")
            reporter.out(f"          reattach:  agent6 attach {layout.run_id}")
            return 0

        print_run_end(result, layout=layout, budget=budget, console_stream=console_stream)
        fire_notify_hook(
            cfg.notify,
            run_id=layout.run_id,
            run_dir=layout.run_dir,
            ok=result.completed,
            reason=result.reason,
            verified=result.verified,
        )
        return run_exit_code(result)
    finally:
        # Single owner of worker.pid + egress teardown for every resume exit
        # path, refusals and Ctrl-C during verify inference included.
        frontend.close_console_view()  # stop the heartbeat thread, clear any spinner line
        clear_worker_pid(layout.run_dir)
        stop_egress(guard)
        release_single_writer(repo_lock_fd)
        release_single_writer(worker_lock_fd)
        if detach_requested and cfg is not None:
            if cfg.sandbox.run_commands == "ask" and not session_allow_set(layout.run_dir):
                frontend.prompt_detach_away_mode(layout.run_dir)
            err = spawn_detached(guard, cwd, layout.run_id, fallback=frontend.spawn_detached_resume)
            if err:
                reporter.err(f"[agent6] {err}")
        if guard.detach_spawner is not None:
            guard.detach_spawner.close()
