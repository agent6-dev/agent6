# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The `agent6 run` lifecycle (and its plan/ask modes): preflight, branch cut,
manifest, loop construction, finalize. `ui/cli/run.py` adapts argv, builds the
:class:`SessionFrontend` seam, and calls :func:`run_task`; everything that touches
the terminal is injected through that seam so this module never imports
`agent6.ui` (mirrors `LaneRuntime` in `app.parallel`)."""

from __future__ import annotations

import contextlib
import os
import shutil
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

from agent6.app._session import (
    build_session_providers,
    build_session_tools,
    select_isolation,
    session_config,
    session_facts_provider,
    warn_install_inside_workspace,
)
from agent6.app._setup import (
    BudgetOverrides,
    SandboxOverrides,
    start_mcp_manager_if_enabled,
    wants_session_network,
)
from agent6.app.finalize import (
    auto_merge_eligible,
    finalize_auto_merge,
    finalize_auto_stash,
    fire_notify_hook,
    print_interrupt_end,
    print_session_end,
    session_exit_code,
    stash_recovery_hint,
    stranded_edits,
)
from agent6.app.frontend import (
    SessionFrontend,
    apply_spawned_away_default,
    approval_scopes,
)
from agent6.app.manifest import (
    pin_gate,
    write_session_manifest,
)
from agent6.app.preflight import (
    SessionRefused,
    drop_gate_if_unrunnable,
    git_preflight,
    headless_approval_refusal,
    infer_verify_if_unset,
)
from agent6.app.providers import (
    build_prompt_reviser_provider,
    role_temperature,
)
from agent6.app.reporter import STDIO_REPORTER, Reporter
from agent6.config import Config
from agent6.config.layer import resolved_state_dir
from agent6.events import EventSink, EventWriteError
from agent6.git_ops import (
    GitError,
    auto_stash_message,
    chain_ref_for,
    dirty_paths,
    render_commit_trailer,
    stash_all,
)
from agent6.paths import chown_to_real_user, mkdir_for_real_user
from agent6.providers import TranscriptSink
from agent6.sandbox.jail import SessionNetwork
from agent6.sessions.id import (
    SessionIdError,
    session_id_bucket,
    unused_session_id,
    validate_explicit_session_id,
)
from agent6.sessions.ipc import (
    COMMAND_SCOPE,
    clear_away_mode,
    clear_compact_request,
    clear_pending_answers,
    clear_session_netns_pid,
    clear_stop_request,
    clear_worker_pid,
    read_compact_request,
    request_steer,
    session_allow_set,
    stop_request_pending,
    write_session_netns_pid,
    write_steer_answer,
    write_worker_pid,
)
from agent6.sessions.layout import LOGS_NAME, SessionLayout
from agent6.sessions.lock import (
    SINGLE_WRITER_BUSY,
    acquire_repo_writer,
    acquire_single_writer,
    release_single_writer,
    repo_writer_holder,
)
from agent6.sessions.manifest import ManifestError, read_manifest
from agent6.tools.dispatch import ToolDispatcher
from agent6.types import session_bucket, session_kind
from agent6.workflows._context import agents_md_notices
from agent6.workflows._session_state import SessionEndReason
from agent6.workflows.loop import SessionResult, Workflow


def discard_husk_dir(session_dir: Path) -> None:
    """Remove a run dir a preflight refused before any real content was written
    (no manifest, no logs). Otherwise a refused start (e.g. dirty worktree)
    leaves an empty husk that `agent6 sessions` lists as '(no logs)' forever. Guarded
    on the manifest/logs check so a real run's dir is never removed."""
    if (session_dir / "manifest.json").exists() or (session_dir / LOGS_NAME).exists():
        return
    with contextlib.suppress(OSError):
        shutil.rmtree(session_dir)


def run_task(  # noqa: PLR0911, PLR0912, PLR0915
    cfg: Config,
    task: str,
    *,
    frontend: SessionFrontend,
    session_id: str = "",
    interactive: bool = False,
    tui: bool = False,
    mode: Literal["run", "plan", "ask"] = "run",
    standing_goal: str = "",
    budget_overrides: BudgetOverrides | None = None,
    sandbox_overrides: SandboxOverrides | None = None,
    preset: str = "",
    initial_steer: str = "",
    pins: Sequence[str] = (),
    preset_stamp: tuple[str, bool] | None = None,
    # Which config leaves the operator actually WROTE, as dotted paths. A
    # default that this host cannot honour degrades with a warning; a value
    # they wrote down refuses, because they asked for something specific.
    explicit_leaves: frozenset[str] = frozenset(),
    reporter: Reporter = STDIO_REPORTER,
) -> int:
    """Single-loop agent: one provider, one LLM driving via tool
    calls over the fixed tool surface, deterministic harness (jail +
    budget + verify timeout + DAG curator for persistence/resume).
    Sole `agent6 run` path; returns the process exit code.

    `initial_steer` queues an operator follow-up for the loop's first
    boundary, seeded AFTER this function's own stale-state clear -- the
    parked-resume delegation passes `resume --steer` through it (a pre-seeded
    bridge file would be wiped by that clear and silently lost).

    The caller (`ui/cli/run.py`) has already built *cfg* (config + overrides),
    resolved the task text, checked the git-repo wall / runnable roles /
    provider keys, and routed `--parallel` away. *budget_overrides* /
    *sandbox_overrides* are passed through for the flags the lifecycle re-reads
    (`--max-usd` enforcement, lane dispatch).

    *preset_stamp* `(name, from_flag)` overrides the manifest's stamped
    preset instead of deriving it from *preset*. A parked resume has no
    `--preset` flag but must record the ORIGINAL submission's stamp so a
    later resume/fork replays the same precedence (fork carries it likewise);
    deriving it from the empty *preset* would drop the stamp, and the flag's
    veto with it, on the next leg.

    When `mode="plan"` the same harness drives a planning
    pass instead of an execution pass: planning system prompt,
    edit-tools filtered out, `finish_planning` instead of
    `finish_session`, no auto-commit. The plan markdown lands at
    `<run-dir>/plan.md` and is consumed by `agent6 run --from-plan`.
    The `planner` model role drives plan mode (falls back to `worker`).
    """
    role = session_kind(mode).role

    # Before anything reads a knob (see session_config): an interactive session
    # (ask / plan) never runs a command unwatched, whether it is starting here
    # or resuming -- unless the operator granted this invocation, which lands
    # after the clamp.
    cfg = session_config(cfg, mode, sandbox_overrides)
    # Refuse an unanswerable run BEFORE anything is created: refusing after
    # the session dir and its manifest exist would leave a never-started run
    # listed forever and poison its id (`--session-id` retries answer "already
    # exists, use resume", and resume finds no snapshot). Everything this needs
    # is known here; the clamp above is the last thing that can change
    # `run_commands`.
    tui_enabled = frontend.should_spawn_tui(tui, interactive, mode)
    refusal = headless_approval_refusal(
        cfg,
        tui_enabled=tui_enabled,
        away=os.environ.get("AGENT6_DETACHED_AWAY", ""),
        can_ask=frontend.capabilities.can_ask,
    )
    if refusal is not None:
        reporter.err(f"REFUSING: {refusal}")
        return 2
    try:
        isolation = select_isolation(
            cfg,
            confirm_unconfined=frontend.confirm_unconfined_autorun,
            reporter=reporter,
            explicit_leaves=explicit_leaves,
        )
    except SessionRefused as refusal:
        return refusal.rc

    cwd = Path.cwd()
    try:
        git = git_preflight(
            cwd,
            cfg,
            mode,
            confirm_run_on_run_branch=frontend.confirm_run_on_run_branch,
            reporter=reporter,
        )
    except SessionRefused as refusal:
        return refusal.rc
    base_sha, base_branch, pre_status = git.base_sha, git.base_branch, git.pre_status

    # Layout: standard run-dir scaffolding for transcripts + logs. ask sessions
    # live under the per-repo state dir (asks subdir) to stay separate from real runs.
    if session_id:
        try:
            validate_explicit_session_id(session_id)
        except SessionIdError as exc:
            reporter.err(f"ERROR: {exc}")
            return 2
    state_dir = resolved_state_dir(cwd)
    bucket = session_bucket(mode)
    # Same-bucket reuse is the resume/park flow below; another bucket's id is
    # a collision every surface would see as ambiguous.
    if session_id and (held := session_id_bucket(state_dir, session_id)) not in (None, bucket):
        reporter.err(
            f"ERROR: --session-id {session_id!r} already names a session under {held}/;"
            " ids are unique across every bucket. Pick another id."
        )
        return 2
    effective_session_id = session_id or unused_session_id(state_dir, bucket)
    layout = SessionLayout(
        state_dir=state_dir,
        session_id=effective_session_id,
        subdir=bucket,
    )
    # An explicit --session-id that already has a session is a resume, not a fresh start:
    # reusing the dir would write a new manifest + loop_state beside the old run's
    # graph/checkpoints/transcripts (mixed state). Refuse and point at resume.
    # (ask sessions are transient Q&A, so reusing their dir is fine.) The one
    # reusable dir is a PARKED run (manifest carries parked_task, nothing else
    # ever ran): starting it IS its fresh start, and the manifest rewrite below
    # un-parks it.
    if session_id and mode != "ask" and layout.manifest_path.exists():
        try:
            parked = read_manifest(layout.session_dir).parked_task
        except ManifestError:
            parked = ""
        if not parked:
            reporter.err(
                f"ERROR: run {session_id!r} already exists. Use `agent6 resume {session_id}` to "
                "continue it, or choose a different --session-id."
            )
            return 2
    # Under sudo the first run on a machine creates the whole state ancestry;
    # hand the created dirs back NOW, not at teardown -- a killed run must not
    # leave a root-owned base that blocks every other repo's non-root runs.
    mkdir_for_real_user(layout.session_dir)
    layout.ensure()
    # One authoritative writer per run dir. Acquire BEFORE touching any shared
    # run state (clearing answers, the worker pid, the curator) so a second
    # process refuses cleanly instead of clobbering the live run.
    worker_lock_fd = acquire_single_writer(layout.session_dir)
    if worker_lock_fd is None:
        reporter.err(SINGLE_WRITER_BUSY.format(rid=effective_session_id))
        return 2
    repo_lock_fd: int | None = None
    # Bound before the try so its finally can report on the run whatever went
    # wrong inside it.
    result: SessionResult | None = None
    stashed = False
    run_branch: str | None = None
    detach_requested = False
    # /undo's outcome, captured by the injected forker so the undone-reason
    # handling below can name the fork and hand the text back.
    undo_outcome: list[tuple[str, str]] = []

    def _undo_forker() -> tuple[str, str] | None:
        # Lazy: app.fork imports app.resume, which imports this module.
        from agent6.app.fork import undo_fork  # noqa: PLC0415

        got = undo_fork(None, effective_session_id, cwd=cwd, reporter=reporter)
        if got is not None:
            undo_outcome.append(got)
        return got

    try:
        # Drop stale approve/ask/steer answers from a prior session (the
        # id counters reset on resume, so an old answer must not be read instead of
        # re-prompting; dead front-end claims are pruned by the liveness probe).
        clear_pending_answers(layout.session_dir)
        if initial_steer.strip():
            request_steer(layout.session_dir)
            write_steer_answer(layout.session_dir, initial_steer.strip())
        if sys.stdin.isatty():  # a foreground start clears a stale detach away-mode
            clear_away_mode(layout.session_dir)
        else:
            apply_spawned_away_default(layout.session_dir, approval_scopes(cfg))
        # Record this worker's pid so `agent6 sessions show` can probe liveness even while
        # the worker is blocked in a long provider call (which emits no events).
        write_worker_pid(layout.session_dir, os.getpid())

        # One live run-mode worker per CHECKOUT, not just per run dir: two runs
        # share one worktree, so each would commit the other's in-flight edits
        # into its own chain. Taken BEFORE
        # any tree mutation (auto-stash, branch cut). plan/ask are read-only and
        # skip it. A refused submission is PARKED, not dropped: the manifest saves
        # the verbatim task and `agent6 resume <id>` starts it once the checkout
        # frees up.
        if mode == "run":
            repo_lock_fd = acquire_repo_writer(layout.state_dir, effective_session_id)
            if repo_lock_fd is None:
                holder = repo_writer_holder(layout.state_dir) or "another run"
                write_session_manifest(
                    layout,
                    session_id=effective_session_id,
                    user_task=task,
                    base_sha=base_sha,
                    base_branch=pre_status.branch if pre_status is not None else "",
                    run_branch=None,
                    cfg=cfg,
                    mode=mode,
                    # The CONFIG preset, not the sandbox one: resume feeds this
                    # back to load_effective, and a sandbox word ("strict") there
                    # made every parked resume die with "unknown preset".
                    effective_preset=(preset_stamp[0] if preset_stamp else (preset or cfg.preset)),
                    preset_from_flag=(preset_stamp[1] if preset_stamp else bool(preset)),
                    parked_task=task,
                )
                reporter.err(
                    f"REFUSING: run {holder!r} is already driving this checkout; a second"
                    " run-mode worker would interleave auto-commits on the one working"
                    f" tree. Your task was parked as run {effective_session_id!r}:\n"
                    f"    agent6 resume {effective_session_id}    (start it once the checkout"
                    " is free)\n"
                    f"or hand it to the live run as an isolated lane by steering"
                    f" {holder!r} with:\n"
                    "    /parallel 1 <the same task>"
                )
                return 2

        # Enforce the dirty-tree policy BEFORE cutting the run branch, so the
        # branch is cut from a clean tree and the agent's per-step auto-commits
        # (`git add -A`) never swallow the user's pre-existing uncommitted work.
        # Only `run` makes commits; `plan`/`ask` are read-only (matching the
        # branch_per_run guard below).
        if mode == "run" and pre_status is not None and not pre_status.is_clean:
            if cfg.git.auto_stash:
                try:
                    stash_all(cwd, auto_stash_message(effective_session_id))
                    stashed = True
                except GitError as exc:
                    reporter.err(f"ERROR: could not auto-stash before run: {exc}")
                    discard_husk_dir(layout.session_dir)
                    return 2
            elif cfg.git.require_clean_worktree:
                dirty = dirty_paths(cwd)
                listed = "\n".join(f"    {p}" for p in dirty)
                more = "\n    ..." if len(dirty) >= 10 else ""
                reporter.err(
                    "REFUSING: working tree is not clean:\n"
                    f"{listed}{more}\n"
                    "Commit, stash, or discard your changes, set [git].auto_stash=true, "
                    "or set [git].require_clean_worktree=false to override."
                )
                discard_husk_dir(layout.session_dir)
                return 2

        # A visible branch named after the run id is 1:1 with the run (find it
        # from any run id, `agent6 sessions diff <id>`, or delete the branch to
        # drop the pointer). The name is the unique run id. Only real `run`
        # mode branches: `plan`/`ask` make no commits. The ref itself is
        # advanced by the first chain commit; nothing is cut or checked out.
        if cfg.git.branch_per_run and mode == "run":
            run_branch = f"agent6/{effective_session_id}"

        transcript_sink = TranscriptSink(layout.transcripts_dir)
        events = EventSink(layout.logs_path)

        warn_install_inside_workspace(cwd, reporter=reporter)
        for line in agents_md_notices(cwd):
            reporter.err(f"[agent6] {line}")

        # Write the run manifest. This is the canonical record of where the
        # run started (base_sha + base_branch), which model+provider drove
        # it, and the user_task it was given. `agent6 sessions diff <run-id>` and
        # any future tooling that wants to reproduce a run reads from here.
        write_session_manifest(
            layout,
            session_id=effective_session_id,
            user_task=task,
            base_sha=base_sha,
            base_branch=base_branch,
            run_branch=run_branch,
            cfg=cfg,
            mode=mode,
            effective_preset=(preset_stamp[0] if preset_stamp else (preset or cfg.preset)),
            preset_from_flag=(preset_stamp[1] if preset_stamp else bool(preset)),
            isolation=isolation,
        )

        # The interactive revision prompt reads the terminal; with the TUI owning
        # it the prompt would land invisibly in the console log and contend for
        # stdin. Skip revision for this run instead.
        effective_revise_prompt = cfg.prompt.revise_prompt
        if effective_revise_prompt == "interactive" and tui_enabled:
            reporter.err(
                "[agent6] prompt.revise_prompt='interactive' needs the terminal; the TUI"
                " owns it. Skipping prompt revision for this run."
            )
            effective_revise_prompt = "off"
        stream_text, console_stream = frontend.stream_modes(tui_enabled)
        if console_stream:
            frontend.attach_console_view(events)
        session = build_session_providers(
            cfg, role=role, events=events, transcript_sink=transcript_sink, stream_text=stream_text
        )
        budget = session.budget
        prompt_reviser_provider = build_prompt_reviser_provider(
            cfg, transcript_sink=transcript_sink, budget=budget, events=events
        )

        # Verify is optional: if unset, infer one for this run (AGENTS.md -> repo
        # signals -> a cheap LLM call) and inject it in-memory. Never persisted.
        # The drop comes LAST so nothing hands the gate back: a leg that cannot
        # run a command is gateless, whatever inference found.
        configured_gate = bool(cfg.workflow.verify_command)
        cfg = infer_verify_if_unset(
            cfg, cwd, mode=mode, events=events, transcript_sink=transcript_sink, budget=budget
        )
        cfg = drop_gate_if_unrunnable(cfg, session_dir=layout.session_dir, reporter=reporter)
        # After resolution, never before: preflight can DROP the gate (a run
        # that cannot run commands), and an empty gate with an origin of
        # "configured" is a self-contradiction the next leg reads back.
        gate_origin = ""
        if cfg.workflow.verify_command:
            gate_origin = "configured" if configured_gate else "inferred"
        # Pin it: from here the run is judged by THIS gate, whatever the file it
        # was inferred from says later.
        pin_gate(
            layout.session_dir,
            cfg.workflow.verify_command,
            gate_origin,
            events=events,
            reporter=reporter,
        )

        # Steering (mid-run Ctrl-C -> the pause menu) needs the terminal; the
        # console view's heartbeat spinner is suspended for the prompt so its
        # line-erase cannot wipe the pause-menu line.
        steer_state = frontend.make_steer_state(
            events,
            layout.session_dir,
            session_facts_provider(
                budget, session.rm_role.model, cfg.sandbox.run_commands, isolation
            ),
        )

        interrupted = False
        dispatcher: ToolDispatcher | None = None
        # Spawned inside the try so the finally below tears it down even if a
        # spawn (MCP) fails.
        mcp_manager = None
        session_net: SessionNetwork | None = None
        try:
            reporter.err(f"[agent6] session id: {effective_session_id}")

            # Spawn any configured MCP servers BEFORE the workflow
            # starts so their tools are visible from iteration 1. The manager
            # owns its subprocesses; the finally block below closes it.
            # The run's session network, before its first member: the
            # commands and any server that joins it share this one.
            if wants_session_network(cfg, isolation):
                session_net = SessionNetwork.open()
                # Published so `agent6 exec`/`forward` can join it: a separate
                # process names a namespace only through a live /proc entry.
                write_session_netns_pid(layout.session_dir, session_net.holder_pid)
            mcp_manager = start_mcp_manager_if_enabled(
                cfg, cwd, isolation, reporter=reporter, events=events, session_net=session_net
            )

            loop_log = frontend.loop_logger(mode)
            tools = build_session_tools(
                cfg,
                cwd=cwd,
                state_dir=state_dir,
                layout=layout,
                isolation=isolation,
                mode=mode,
                events=events,
                approver=frontend.build_approver(layout.session_dir, events),
                questioner=frontend.build_questioner(layout.session_dir, events),
                loop_log=loop_log,
                mcp_manager=mcp_manager,
                session_net=session_net,
                rm_role=session.rm_role,
            )
            curator = tools.curator
            dispatcher = tools.dispatcher
            cfg = tools.cfg
            after_auto_commit: Callable[[int, str], Literal["continue", "stop"]] = (
                frontend.build_repl_hook(cwd, budget, effective_session_id, mcp_manager)
                if interactive and mode == "run"
                else (lambda _i, _s: "continue")
            )
            wf = Workflow(
                root=cwd,
                config=cfg,
                standing_goal=standing_goal,
                interactive=interactive and mode == "run",
                initial_pins=tuple(pins),
                commit_trailer=render_commit_trailer(
                    cfg.git.commit.trailer, models=(session.rm_role.model,)
                ),
                chain_ref=chain_ref_for(effective_session_id) if mode == "run" else None,
                chain_branch=run_branch,
                chain_fallback_parent=base_sha or None,
                commit_per_step=cfg.git.commit_per_step,
                provider=session.provider,
                dispatcher=dispatcher,
                logger=loop_log,
                events=events,
                curator=curator,
                steer_requested=steer_state.requested,
                steer_clear=steer_state.clear,
                steer_prompt=steer_state.prompt,
                steer_reset=steer_state.reset_stage,
                # "Compact now" from a front-end: the same file-bridge
                # pattern as steer, honored at the next pre-call boundary.
                compact_requested=lambda: read_compact_request(layout.session_dir),
                compact_clear=lambda: clear_compact_request(layout.session_dir),
                stop_requested=lambda: stop_request_pending(layout.session_dir),
                stop_clear=lambda: clear_stop_request(layout.session_dir),
                should_abort=steer_state.abort_pending,
                undo_forker=_undo_forker,
                should_interrupt=steer_state.interrupt,
                # `/parallel` steer dispatch: the coordinator's group spawner
                # (None in plan/ask, and inside a lane -- depth 1).
                lane_spawner=frontend.build_coordinator_spawner(
                    cfg,
                    cwd,
                    state_dir,
                    mode,
                    effective_session_id,
                    budget_overrides.max_usd if budget_overrides is not None else None,
                    sandbox_overrides.auto_approve if sandbox_overrides is not None else False,
                ),
                budget=budget,
                state_dir=state_dir,
                # Written for every mode: `agent6 resume` reaches an ask too.
                resume_state_path=layout.session_dir / "loop_state.json",
                mode=mode,
                plan_output_path=(layout.session_dir / "plan.md" if mode == "plan" else None),
                after_auto_commit=after_auto_commit,
                review_trigger=cfg.review.trigger,
                review_period=cfg.review.period,
                review_seats=session.review_seats,
                review_decision=cfg.review.decision,
                review_quorum=cfg.review.quorum,
                review_max_total_rejections=cfg.review.max_total_rejections,
                review_budget_fraction=cfg.review.budget_fraction,
                review_concurrency=cfg.review.concurrency,
                base_sha=base_sha,
                prompt_reviser_provider=prompt_reviser_provider,
                revise_prompt=effective_revise_prompt,
                temperature=role_temperature(cfg, role),
                prompt_reviser_temperature=role_temperature(cfg, "reviewer"),
                prompt_revision_selector=(
                    frontend.select_revised_prompt
                    if effective_revise_prompt == "interactive"
                    else None
                ),
                summariser_provider=session.summariser_provider,
                compact_drop_at_chars=tools.compact_drop_at_chars,
                compact_summarise_at_chars=tools.compact_summarise_at_chars,
                context_summary_max_tokens=cfg.context.summary_max_tokens,
                keep_recent_chars=cfg.context.keep_recent_chars,
                keep_thinking_turns=cfg.context.keep_thinking_turns,
                compact_elision_gists=cfg.context.elision_gists,
            )
            try:
                with frontend.tui_session(layout.session_dir, tui_enabled):
                    if mode == "ask" and interactive:
                        result = frontend.run_ask_repl(wf, budget, layout, task)
                    else:
                        result = wf.run(task)
            except KeyboardInterrupt:
                interrupted = True
                reporter.err("\n[agent6] run interrupted")
                # The loop was cut mid-step, so it never emitted session.end; do it
                # here so an attached watcher/TUI stops instead of hanging. Carry
                # the iteration the loop reached so session.end keeps one shape.
                # suppress: the interrupt exit (130 + resume hint) must not be
                # masked by a dead journal.
                reason: SessionEndReason = "interrupted"
                with contextlib.suppress(EventWriteError):
                    events.emit(
                        "session.end",
                        reason=reason,
                        iterations=wf.iterations_reached,
                        all_passed=False,
                    )
            except Exception:
                # Any other escape (a broken stdout pipe from `| head`, an
                # unexpected fault) also leaves the loop without a session.end,
                # and the outer finally then clears worker.pid -- the only
                # immediate liveness evidence -- so every surface would read
                # the dead run as "running" until the silence window expires.
                # Record the end, then re-raise.
                with contextlib.suppress(EventWriteError):
                    events.emit(
                        "session.end",
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
            if session_net is not None:
                # The last handles on the run's network: closing them is what
                # lets the kernel reclaim it.
                session_net.close()
                clear_session_netns_pid(layout.session_dir)
            if (
                not interrupted
                and result is not None
                and auto_merge_eligible(result)
                and cfg.git.auto_merge
            ):
                finalize_auto_merge(
                    cwd, layout=layout, cfg=cfg, reporter=reporter, budget=budget, events=events
                )
            # Never leave root-owned run state in the user's repo (sudo case).
            chown_to_real_user(state_dir)

        if interrupted:
            print_interrupt_end(layout=layout, budget=budget, reporter=reporter)
            return 130
        if result is None:
            return 1

        if mode == "ask":
            # The answer IS result.summary (kept whole in ask mode). stdout gets
            # just the answer (clean for piping); cost + saved-path go to stderr.
            # The REPL already printed + saved each turn, so only the one-shot path
            # prints/saves here.
            if not interactive:
                reporter.out(result.summary)
                frontend.save_ask_transcript(layout, task, result.summary)
                reporter.err(f"\n[agent6] answer saved to {layout.session_dir / 'transcript.md'}")
            reporter.err(budget.format_summary())
            return 0 if result.completed else 1

        if result.reason == "undone" and undo_outcome:
            new_id, undone_text = undo_outcome[-1]
            reporter.out(f"\n[agent6] undone: continue as {new_id} with your message back to edit:")
            reporter.out(f"    agent6 resume {new_id} --steer {undone_text!r}")
            return 0
        if result.reason == "detached":
            # Keep going in the background: the outer finally releases this run's
            # worker lock, then spawns a detached `resume` that picks it up.
            detach_requested = True
            reporter.out(f"\n[agent6] detached: {layout.session_id} continues in the background.")
            reporter.out(f"          reattach:  agent6 attach {layout.session_id}")
            return 0

        print_session_end(
            result,
            layout=layout,
            budget=budget,
            console_stream=console_stream,
            reporter=reporter,
        )
        fire_notify_hook(
            cfg.notify,
            session_id=layout.session_id,
            session_dir=layout.session_dir,
            ok=result.completed,
            reason=result.reason,
            verified=result.verified,
            reporter=reporter,
        )
        return session_exit_code(result, stranded=stranded_edits(result, layout))
    finally:
        # Single owner of worker.pid, both writer locks, and auto-stash
        # finalization, for EVERY exit path: preflight refusals, Ctrl-C during
        # verify inference, and setup-window crashes included. worker.pid and
        # the stash pop happen UNDER the locks, so the releases come after --
        # nested, because they must survive a teardown raise: the ACP front-end
        # calls run_task in-process, where a leaked flock refuses every later
        # run on the session until the server restarts.
        try:
            frontend.close_console_view()  # stop the heartbeat thread, clear any spinner line
            clear_worker_pid(layout.session_dir)
            if stashed:
                if detach_requested:
                    # The run is NOT over: popping the stash now would feed the
                    # user's pre-run files into the detached continuation's
                    # auto-commits. Leave the stash and say so.
                    # By sha, never by position: this hint has the LONGEST window of
                    # any -- the operator reads it now and runs it after a
                    # background run that may take hours, by which point a
                    # positional pop restores whatever else was stashed meanwhile.
                    hint = stash_recovery_hint(
                        cwd, session_id=effective_session_id, base_branch=base_branch
                    )
                    reporter.err(
                        "[agent6] pre-run changes remain stashed while the run continues"
                        " in the background; after it ends, restore them with:"
                        f" {hint}"
                        if hint
                        else "[agent6] pre-run changes remain stashed while the run continues"
                        " in the background, but the stash could not be located; check"
                        " `git stash list`"
                    )
                else:
                    finalize_auto_stash(
                        cwd,
                        base_branch=base_branch,
                        run_branch=run_branch,
                        auto_pop=cfg.git.auto_stash_pop,
                        session_id=layout.session_id,
                        reporter=reporter,
                    )
        finally:
            release_single_writer(repo_lock_fd)
            release_single_writer(worker_lock_fd)
        if detach_requested:
            # Ask how to handle approvals while away BEFORE spawning, so the marker is
            # set when the background run reads it. The worker lock is released now, so
            # the detached `resume` acquires it.
            if cfg.sandbox.run_commands == "ask" and not session_allow_set(
                layout.session_dir, COMMAND_SCOPE
            ):
                frontend.prompt_detach_away_mode(layout.session_dir, approval_scopes(cfg))
            err = frontend.spawn_detached_resume(cwd, layout.session_id)
            if err:
                reporter.err(f"[agent6] {err}")
