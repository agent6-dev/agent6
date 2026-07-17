# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The `agent6 resume` lifecycle: pick a paused or crashed run back up from its
snapshot. `ui/cli/resume.py` adapts argv and injects the same
:class:`agent6.app.run.RunFrontend` seam `run_task` uses."""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Literal

from agent6.app._setup import (
    BudgetOverrides,
    SandboxOverrides,
    detect_env,
)
from agent6.app._setup import (
    check_provider_keys as _check_provider_keys,
)
from agent6.app._setup import (
    explicit_usd_flag_error as _explicit_usd_flag_error,
)
from agent6.app._setup import (
    start_mcp_manager_if_enabled as _start_mcp_manager_if_enabled,
)
from agent6.app.egress import (
    EgressGuard,
    check_network_profile,
    maybe_apply_agent_landlock,
    maybe_start_egress,
    resolve_strict_egress_viability,
    spawn_detached,
    stop_egress,
    warn_if_unsandboxed,
)
from agent6.app.finalize import (
    finalize_auto_merge as _finalize_auto_merge,
)
from agent6.app.finalize import (
    fire_notify_hook as _fire_notify_hook,
)
from agent6.app.finalize import (
    print_run_end as _print_run_end,
)
from agent6.app.finalize import (
    run_exit_code as _run_exit_code,
)
from agent6.app.preflight import (
    infer_verify_if_unset as _infer_verify_if_unset,
)
from agent6.app.preflight import (
    require_git_repo as _require_git_repo,
)
from agent6.app.preflight import (
    warn_if_headless_ask as _warn_if_headless_ask,
)
from agent6.app.preflight import (
    warn_if_prompt_override_incomplete as _warn_if_prompt_override_incomplete,
)
from agent6.app.preflight import (
    warn_if_usd_unenforceable as _warn_if_usd_unenforceable,
)
from agent6.app.providers import (
    InstrumentedProvider as _InstrumentedProvider,
)
from agent6.app.providers import (
    build_critic_provider as _build_critic_provider,
)
from agent6.app.providers import (
    build_review_seats,
    resolve_compaction_thresholds,
    resolve_decompose,
    review_panel_configured,
)
from agent6.app.providers import (
    build_role_provider as _build_role_provider,
)
from agent6.app.providers import (
    build_summariser_provider as _build_summariser_provider,
)
from agent6.app.providers import (
    role_temperature as _role_temperature,
)
from agent6.app.run import RunFrontend
from agent6.budget import BudgetTracker
from agent6.config import (
    Config,
    ConfigError,
)
from agent6.config.layer import (
    load_effective,
    resolved_state_dir,
)
from agent6.events import EventSink
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
from agent6.graph.client import CuratorClientError, GraphClient, spawn_curator
from agent6.paths import (
    chown_to_real_user,
)
from agent6.providers import (
    Provider,
    TranscriptSink,
)
from agent6.runs.bridge import (
    clear_away_mode,
    clear_compact_request,
    clear_pending_answers,
    clear_stop_request,
    clear_worker_pid,
    compact_request_pending,
    request_steer,
    session_allow_set,
    stop_request_pending,
    write_steer_answer,
    write_worker_pid,
)
from agent6.runs.id import RunIdError, resolve_run_id
from agent6.runs.layout import RunLayout
from agent6.runs.lock import (
    SINGLE_WRITER_BUSY as _SINGLE_WRITER_BUSY,
)
from agent6.runs.lock import (
    acquire_single_writer as _acquire_single_writer,
)
from agent6.runs.lock import (
    release_single_writer as _release_single_writer,
)
from agent6.sandbox.detect import ProfileUnavailableError, select_profile
from agent6.tools.dispatch import ToolDispatcher
from agent6.viewmodel import most_recent_run_id as _most_recent_run_id
from agent6.workflows._run_state import load_resume_snapshot
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
        manifest = json.loads(layout.manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    run_branch = manifest.get("run_branch")
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


def resume_task(  # noqa: PLR0911, PLR0912, PLR0915
    config_path: Path | None,
    run_id: str,
    *,
    frontend: RunFrontend,
    force: bool,
    tui: bool = False,
    budget_overrides: BudgetOverrides | None = None,
    sandbox_overrides: SandboxOverrides | None = None,
    profile: str = "",
    steer: str = "",
) -> int:
    """Resume a paused/crashed run from its snapshot.

    Mirrors ``run_task`` setup but uses the existing run id, refuses
    if no ``loop_state.json`` snapshot exists, and calls ``wf.resume()``
    instead of ``wf.run(task)``. A safety check refuses when the
    workspace HEAD DIVERGED from the snapshot (a rebase/reset/commit on
    another line); plain forward movement on the same line resumes
    cleanly. ``--force-resume`` overrides the refusal.

    NOTE: token budget on resume is a FRESH ceiling, not a continuation
    of the prior run's accounting. Each ``agent6 resume`` invocation
    starts at 0 tokens against ``[budget].max_input_tokens`` /
    ``max_output_tokens``. This is by design - the budget is a per-
    invocation runaway-cost circuit breaker.
    """
    cwd = Path.cwd()
    state_dir = resolved_state_dir(cwd)
    runs_dir = state_dir / "runs"
    if not run_id:
        # "resume my last run" -- the common recovery case, matching `runs *`.
        latest = _most_recent_run_id(runs_dir)
        if latest is None:
            print(f"ERROR: no runs under {runs_dir}; nothing to resume.", file=sys.stderr)
            return 2
        run_id = latest
        print(f"[agent6] resuming most recent run: {run_id}", file=sys.stderr)
    try:
        resolved = resolve_run_id(runs_dir, run_id)
    except RunIdError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    run_id = resolved
    layout = RunLayout(state_dir=state_dir, run_id=run_id)
    if not layout.run_dir.is_dir():
        print(f"ERROR: no such run dir: {layout.run_dir}", file=sys.stderr)
        return 2
    # One authoritative writer per run dir (see acquire_single_writer). Refuse a
    # second resume of a still-live run before touching any shared state.
    worker_lock_fd = _acquire_single_writer(layout.run_dir)
    if worker_lock_fd is None:
        print(_SINGLE_WRITER_BUSY.format(rid=run_id), file=sys.stderr)
        return 2
    # Drop a prior session's stale answer files + frontend.pid (the id counters reset
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
    # Record this worker's pid so `agent6 runs show` can probe liveness even while
    # the worker is blocked in a long provider call (which emits no events).
    write_worker_pid(layout.run_dir, os.getpid())

    detach_requested = False
    cfg: Config | None = None  # bound below; the finally reads it (detach away-mode)
    guard = EgressGuard()  # replaced at egress start; the finally tears it down
    try:
        snapshot_path = layout.run_dir / "loop_state.json"
        if not snapshot_path.is_file():
            print(
                f"ERROR: no resume snapshot at {snapshot_path}; nothing to resume.",
                file=sys.stderr,
            )
            return 2

        # Friendly no-repo guard BEFORE any git-touching check (which would
        # otherwise print zeroed-out heads first, then the real error).
        if not _require_git_repo(cwd):
            return 2

        # The original run's manifest drives resume: `mode` (a plan run resumes
        # read-only with the plan tools, never as a write run), `profile` (resume
        # has no --profile flag), `base_sha` (the review-panel diff base), and
        # `run_branch` (the head guard + the checkout below).
        # `mode` is security-relevant: a missing/corrupt manifest must NOT fall
        # open to the more-privileged "run" (write) mode. A valid run always
        # wrote a manifest, so anything else here is a damaged run dir -- fail
        # loud rather than silently escalating a plan run to a write run.
        try:
            loaded = json.loads(layout.manifest_path.read_text(encoding="utf-8"))
        except OSError as exc:
            print(f"ERROR: cannot read run manifest {layout.manifest_path}: {exc}", file=sys.stderr)
            return 2
        except ValueError as exc:
            print(f"ERROR: run manifest {layout.manifest_path} is corrupt: {exc}", file=sys.stderr)
            return 2
        if not isinstance(loaded, dict):
            print(
                f"ERROR: run manifest {layout.manifest_path} is malformed"
                f" (expected a JSON object, got {type(loaded).__name__}).",
                file=sys.stderr,
            )
            return 2
        manifest: dict[str, Any] = loaded
        mode: Literal["run", "plan"] = "plan" if manifest.get("mode") == "plan" else "run"
        workflow_section = manifest.get("workflow")
        manifest_profile = (
            str(workflow_section.get("profile") or "") if isinstance(workflow_section, dict) else ""
        )
        resume_base_sha = str(manifest.get("base_sha") or "")
        run_branch = str(manifest.get("run_branch") or "")

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
            print(
                "GUARD: the code this run would resume onto diverged from its last snapshot.",
                file=sys.stderr,
            )
            print(f"  snapshot head: {snap_head}", file=sys.stderr)
            print(f"  resume onto:   {onto_head}", file=sys.stderr)
            if not force:
                print(
                    "REFUSING to resume. Re-run with --force-resume to override.",
                    file=sys.stderr,
                )
                return 1

        try:
            cfg = load_effective(
                Path.cwd(), config_path, profile=profile or manifest_profile
            ).config
            set_repo_hook_policy(cfg.git.run_repo_hooks)
            if budget_overrides is not None:
                cfg = budget_overrides.apply(cfg)
            if sandbox_overrides is not None:
                cfg = sandbox_overrides.apply(cfg)
            cfg.require_runnable("worker")
        except ConfigError as exc:
            print(f"CONFIG ERROR:\n{exc}", file=sys.stderr)
            return 2

        env = detect_env()
        try:
            selected_profile = select_profile(cfg.sandbox.profile, env)
        except ProfileUnavailableError as exc:
            print(f"REFUSING: {exc}", file=sys.stderr)
            return 2
        warn_if_unsandboxed(selected_profile)
        if not frontend.confirm_unconfined_autorun(selected_profile, cfg):
            print("[agent6] aborted.", file=sys.stderr)
            return 1

        net_err = check_network_profile(cfg, selected_profile)
        if net_err is not None:
            print(f"REFUSING: {net_err}", file=sys.stderr)
            return 2
        # strict can be selected because the jail launcher has userns, yet this
        # process can't create one for the egress broker (surgical AppArmor profile).
        # Downgrade auto->hardened, or refuse an explicit strict, with guidance.
        selected_profile, egress_err = resolve_strict_egress_viability(cfg, selected_profile)
        if egress_err is not None:
            print(egress_err, file=sys.stderr)
            return 2

        missing = _check_provider_keys(cfg)
        usd_err = _explicit_usd_flag_error(
            budget_overrides.max_usd if budget_overrides else None, cfg
        )
        if usd_err is not None:
            print(f"REFUSING: {usd_err}", file=sys.stderr)
            return 2
        if missing is not None:
            print(missing, file=sys.stderr)
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
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2

        transcript_sink = TranscriptSink(layout.transcripts_dir)
        events = EventSink(layout.logs_path)

        guard, egress_err = maybe_start_egress(
            cfg, selected_profile, detach_exe=frontend.agent6_exe()
        )
        if egress_err is not None:
            print(f"REFUSING: {egress_err}", file=sys.stderr)
            return 2
        if guard.broker is not None:
            print(
                f"[agent6] provider-only egress: confined to host network "
                f"namespace via broker pid {guard.broker.pid}",
                file=sys.stderr,
            )

        landlock_err = maybe_apply_agent_landlock(cfg, selected_profile, env)
        if landlock_err is not None:
            print(f"REFUSING: {landlock_err}", file=sys.stderr)
            # The egress broker is already running; the outer finally tears it
            # down (and its socket dir) so this refusal does not leak it.
            return 2

        # Get onto the run's branch so the loop's commits land there (a fork
        # cuts agent6/<id> without checking it out; the operator may have moved
        # branches since the original run). This is the ONLY workspace mutation
        # in preflight, and deliberately the LAST step: every refusal above
        # exits with the operator's checkout untouched. From here on a failure
        # is a failed RUN, parked on the run branch like any crashed run (the
        # end-of-run banner says how to switch back).
        branch_err = ensure_on_run_branch(cwd, layout)
        if branch_err is not None:
            print(branch_err, file=sys.stderr)
            return 2

        budget = BudgetTracker(
            max_input_tokens=cfg.budget.max_input_tokens,
            max_output_tokens=cfg.budget.max_output_tokens,
            max_usd=cfg.budget.best_effort_usd_limit,
        )

        worker_inner = _build_role_provider(
            cfg, "worker", transcript_sink=transcript_sink, budget=budget
        )
        rm_worker = cfg.models.resolve("worker")
        assert rm_worker is not None  # require_runnable validated this
        _warn_if_usd_unenforceable(cfg)
        _warn_if_prompt_override_incomplete(cfg)
        tui_enabled = frontend.should_spawn_tui(tui, False, mode)
        _warn_if_headless_ask(cfg, tui_enabled=tui_enabled)
        stream_text, console_stream = frontend.stream_modes(tui_enabled)
        if console_stream:
            frontend.attach_console_view(events)
        provider: Provider = _InstrumentedProvider(
            inner=worker_inner,
            role="worker",
            model=rm_worker.model,
            provider_name=rm_worker.provider,
            events=events,
            budget=budget,
            stream_text=stream_text,
        )

        critic_provider = _build_critic_provider(
            cfg, transcript_sink=transcript_sink, budget=budget, events=events
        )
        summariser_provider = _build_summariser_provider(
            cfg, transcript_sink=transcript_sink, budget=budget, events=events
        )
        review_seats = (
            build_review_seats(
                cfg,
                transcript_sink=transcript_sink,
                budget=budget,
                n=cfg.review.panel_size,
                personas=cfg.review.personas,
            )
            if cfg.review.trigger != "off" and review_panel_configured(cfg)
            else []
        )
        # Resume reuses the verify command the ORIGINAL run resolved (stored in the
        # snapshot), so the tool list, prompt, and commit branch stay consistent with
        # the frozen system prompt -- never re-inferring (which could flip and
        # diverge). Fall back to re-inference only for a pre-field snapshot, and only
        # when the operator hasn't since pinned a command in config.
        if not cfg.workflow.verify_command:
            snap_verify: tuple[str, ...] | None = None
            if snapshot_path.is_file():
                try:
                    snap_verify = load_resume_snapshot(snapshot_path).verify_command
                except (ValueError, OSError, KeyError):
                    snap_verify = None
            if snap_verify is None:  # older snapshot: re-infer as the original did
                cfg = _infer_verify_if_unset(
                    cfg,
                    cwd,
                    mode=mode,
                    events=events,
                    transcript_sink=transcript_sink,
                    budget=budget,
                )
            elif snap_verify:  # () means the original run was gateless: stay gateless
                cfg = cfg.with_inferred_verify(snap_verify)
                print(
                    f"[agent6] reusing this run's verify command: {' '.join(snap_verify)}",
                    file=sys.stderr,
                )

        sock_path = layout.run_dir / "curator.sock"  # rebound to the /tmp socket inside the try

        steer_state = frontend.make_steer_state(events, layout.run_dir)

        result = None
        interrupted = False
        dispatcher: ToolDispatcher | None = None
        # Spawned inside the try so the finally below always tears them down even
        # if a spawn itself fails (otherwise curator/MCP procs + the /tmp socket
        # dir leak past the only cleanup path).
        curator_proc: subprocess.Popen[bytes] | None = None
        sock_tmpdir: Path | None = None
        mcp_manager = None
        try:
            sock_tmpdir = Path(tempfile.mkdtemp(prefix="agent6-sock-"))
            sock_path = sock_tmpdir / "curator.sock"
            sock_link = layout.run_dir / "curator.sock"
            with contextlib.suppress(FileNotFoundError):
                sock_link.unlink()
            sock_link.symlink_to(sock_path)
            curator_proc = spawn_curator(state_dir, run_id, sock_path, subdir=layout.subdir)
            print(f"[agent6] resume run id: {run_id}", file=sys.stderr)

            mcp_manager = _start_mcp_manager_if_enabled(cfg)

            with GraphClient(sock_path, alive=lambda: curator_proc.poll() is None) as graph_client:
                dispatcher = ToolDispatcher(
                    root=cwd,
                    config=cfg,
                    sandbox_profile=selected_profile,
                    approver=frontend.build_approver(layout.run_dir, events),
                    questioner=frontend.build_questioner(layout.run_dir, events),
                    events=events,
                    graph_client=graph_client,
                    run_root_node_id=None,
                    mcp_manager=mcp_manager,
                    mode=mode,
                    state_dir=state_dir,
                )
                loop_log = frontend.loop_logger(mode)
                compact_drop, compact_summarise = resolve_compaction_thresholds(
                    cfg, rm_worker, log=loop_log
                )
                # Same pin the original run applied: the frozen system prompt
                # already carries its decompose block; this keeps the loop's
                # banner/hint reads consistent with it.
                cfg = resolve_decompose(cfg, rm_worker, log=loop_log)
                wf = Workflow(
                    root=cwd,
                    config=cfg,
                    provider=provider,
                    dispatcher=dispatcher,
                    logger=loop_log,
                    events=events,
                    graph_client=graph_client,
                    steer_requested=steer_state.requested,
                    steer_clear=steer_state.clear,
                    steer_prompt=steer_state.prompt,
                    # "Compact now" from a front-end: the same file-bridge
                    # pattern as steer, honored at the next pre-call boundary.
                    compact_requested=lambda: compact_request_pending(layout.run_dir),
                    compact_clear=lambda: clear_compact_request(layout.run_dir),
                    stop_requested=lambda: stop_request_pending(layout.run_dir),
                    stop_clear=lambda: clear_stop_request(layout.run_dir),
                    should_abort=steer_state.abort_pending,
                    should_interrupt=steer_state.interrupt,
                    # `/parallel` steer dispatch: the coordinator's group spawner
                    # (None in plan resume, and inside a lane -- depth 1).
                    lane_spawner=frontend.build_coordinator_spawner(
                        cfg,
                        cwd,
                        state_dir,
                        mode,
                        run_id,
                        budget_overrides.max_usd if budget_overrides is not None else None,
                        sandbox_overrides.auto_approve if sandbox_overrides is not None else False,
                    ),
                    budget=budget,
                    resume_state_path=snapshot_path,
                    mode=mode,
                    plan_output_path=(layout.run_dir / "plan.md" if mode == "plan" else None),
                    critic_provider=critic_provider,
                    critic_mode=cfg.review.trigger,
                    critic_period=cfg.review.period,
                    review_seats=review_seats,
                    review_decision=cfg.review.decision,
                    review_quorum=cfg.review.quorum,
                    review_max_total_rejections=cfg.review.max_total_rejections,
                    review_budget_fraction=cfg.review.budget_fraction,
                    review_concurrency=cfg.review.concurrency,
                    base_sha=resume_base_sha,
                    temperature=_role_temperature(cfg, "worker"),
                    critic_temperature=_role_temperature(cfg, "reviewer"),
                    summariser_provider=summariser_provider,
                    compact_drop_at_chars=compact_drop,
                    compact_summarise_at_chars=compact_summarise,
                    context_summary_max_tokens=cfg.context.summary_max_tokens,
                    compact_elision_gists=cfg.context.elision_gists,
                )
                try:
                    with frontend.tui_session(layout.run_dir, tui_enabled):
                        result = wf.resume()
                except ResumeError as exc:
                    print(f"ERROR: {exc}", file=sys.stderr)
                    return 1
                except KeyboardInterrupt:
                    interrupted = True
                    print("\n[agent6] resume interrupted", file=sys.stderr)
                    events.emit("run.end", reason="interrupted", all_passed=False)
        except CuratorClientError as exc:
            print(f"ERROR: curator failed to start: {exc}", file=sys.stderr)
            return 1
        finally:
            if curator_proc is not None:
                curator_proc.terminate()
                try:
                    curator_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    curator_proc.kill()
            steer_state.restore()
            with contextlib.suppress(FileNotFoundError):
                (layout.run_dir / "curator.sock").unlink()
            if sock_tmpdir is not None:
                shutil.rmtree(sock_tmpdir, ignore_errors=True)
            if dispatcher is not None:
                dispatcher.close()
            if mcp_manager is not None:
                mcp_manager.close()
            # Egress teardown is owned by the outer finally (a single call).
            # Doing it here too would reap the broker pid, then the auto-merge
            # git subprocesses and the notify hook below could recycle it before
            # the outer close() signalled the pid again.
            if not interrupted and result is not None and result.completed and cfg.git.auto_merge:
                _finalize_auto_merge(cwd, layout=layout, cfg=cfg)
            # Never leave root-owned run state in the user's repo (sudo case).
            chown_to_real_user(state_dir)

        if interrupted:
            return 130
        if result is None:
            return 1

        if result.reason == "detached":
            detach_requested = True
            print(f"\n[agent6] detached: {layout.run_id} continues in the background.")
            print(f"          reattach:  agent6 attach {layout.run_id}")
            return 0

        _print_run_end(result, layout=layout, budget=budget, console_stream=console_stream)
        _fire_notify_hook(
            cfg.notify,
            run_id=layout.run_id,
            run_dir=layout.run_dir,
            ok=result.completed,
            reason=result.reason,
        )
        return _run_exit_code(result)
    finally:
        # Single owner of worker.pid + egress teardown for every resume exit
        # path; refusals and Ctrl-C during verify inference used to leak both.
        frontend.close_console_view()  # stop the heartbeat thread, clear any spinner line
        clear_worker_pid(layout.run_dir)
        stop_egress(guard)
        _release_single_writer(worker_lock_fd)
        if detach_requested and cfg is not None:
            if cfg.sandbox.run_commands == "ask" and not session_allow_set(layout.run_dir):
                frontend.prompt_detach_away_mode(layout.run_dir)
            err = spawn_detached(guard, cwd, layout.run_id, fallback=frontend.spawn_detached_resume)
            if err:
                print(f"[agent6] {err}", file=sys.stderr)
        if guard.detach_spawner is not None:
            guard.detach_spawner.close()
