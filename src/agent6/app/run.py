# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The `agent6 run` lifecycle (and its plan/ask modes): preflight, branch cut,
manifest, loop construction, finalize. `ui/cli/run.py` adapts argv, builds the
:class:`RunFrontend` seam, and calls :func:`run_task`; everything that touches
the terminal is injected through that seam so this module never imports
`agent6.ui` (mirrors `LaneRuntime` in `app.parallel`)."""

from __future__ import annotations

import contextlib
import os
import shutil
import sys
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from agent6.app._session import (
    SessionRefused,
    build_session_providers,
    build_session_tools,
    select_isolation,
    session_config,
    start_isolation,
)
from agent6.app._setup import (
    BudgetOverrides,
    SandboxOverrides,
    start_mcp_manager_if_enabled,
)
from agent6.app.finalize import (
    finalize_auto_merge,
    finalize_auto_stash,
    fire_notify_hook,
    print_interrupt_end,
    print_run_end,
    run_exit_code,
    stash_recovery_hint,
)
from agent6.app.manifest import (
    pin_gate,
    write_run_manifest,
)
from agent6.app.preflight import (
    BranchChoice,
    drop_gate_if_unrunnable,
    headless_approval_refusal,
    infer_verify_if_unset,
)
from agent6.app.providers import (
    build_prompt_reviser_provider,
    role_temperature,
)
from agent6.app.reporter import STDIO_REPORTER, Reporter
from agent6.budget import BudgetTracker
from agent6.config import Config, role_for_mode
from agent6.config.layer import resolved_state_dir
from agent6.events import EventSink, EventWriteError
from agent6.git_ops import (
    CommitIdentity,
    GitError,
    auto_stash_message,
    create_branch,
    dirty_paths,
    stash_all,
    verify_git_identity,
)
from agent6.git_ops import (
    status as git_status,
)
from agent6.paths import chown_to_real_user, mkdir_for_real_user
from agent6.providers import TranscriptSink
from agent6.runs.id import RunIdError, new_friendly_id, validate_explicit_run_id
from agent6.runs.ipc import (
    away_mode,
    clear_away_mode,
    clear_compact_request,
    clear_pending_answers,
    clear_stop_request,
    clear_worker_pid,
    read_compact_request,
    request_steer,
    session_allow_set,
    set_away_mode,
    set_session_allow,
    stop_request_pending,
    write_steer_answer,
    write_worker_pid,
)
from agent6.runs.layout import LOGS_NAME, RunLayout
from agent6.runs.lock import (
    SINGLE_WRITER_BUSY,
    acquire_repo_writer,
    acquire_single_writer,
    release_single_writer,
    repo_writer_holder,
)
from agent6.runs.manifest import ManifestError, read_manifest
from agent6.tools.dispatch import Approver, ToolDispatcher
from agent6.tools.mcp_client import MCPManager
from agent6.tools.schema import UserQuestion
from agent6.types import IsolationLevel
from agent6.workflows._run_state import RunReason
from agent6.workflows.loop import RunResult, Workflow
from agent6.workflows.subrun import GroupLaneSpawner


@dataclass(frozen=True, slots=True)
class RunFacts:
    """The live facts the CLI pause banner shows, so an operator deciding
    whether to interrupt can see what this run is doing without the widgets a
    TUI/web viewer has. Built by the lifecycle (which holds the tracker and the
    resolved config) and rendered by the front-end; read inside a signal
    handler, so every field is already in memory -- no file read, no fold."""

    spend_usd: float
    spend_partial: bool  # a model with no price data contributed: a lower bound
    model: str
    run_commands: str
    isolation: str


class SteerHooks(Protocol):
    """What the lifecycle needs of the front-end's steer state (the SIGINT
    pause menu or the file-bridge steer); `ui/cli/_steer.SteerState` satisfies
    it structurally."""

    requested: Callable[[], bool]
    clear: Callable[[], None]
    prompt: Callable[[], str | None]
    restore: Callable[[], None]
    abort_pending: Callable[[], bool]
    interrupt: Callable[[], bool]
    reset_stage: Callable[[], None]


def apply_spawned_away_default(run_dir: Path) -> None:
    """Honor AGENT6_DETACHED_AWAY, set by a front-end launcher (web/TUI hub) that
    spawns a run detached and drives it over the bridge. Without it a spawned run
    with no terminal fabricates empty ask_user answers when no viewer is live;
    'wait' makes approvals and questions block for a front-end. A pure headless
    run (no launcher) sets no env, so this is a no-op and it keeps its default.

    A DEFAULT: an away mode already on the run dir is the operator's own detach
    answer, and the resume this spawns carries 'wait' regardless -- overwriting
    silently upgraded a chosen 'deny' to 'wait', so the run blocked on an
    approval nobody was there to give instead of denying and carrying on."""
    away = os.environ.get("AGENT6_DETACHED_AWAY", "")
    if not away or away_mode(run_dir):
        return
    if away == "approve":
        # approve is never stored in away.mode (deny|wait): like the interactive
        # detach prompt, approve-all reuses the session-allow marker.
        set_session_allow(run_dir)
    elif away in ("wait", "deny"):
        set_away_mode(run_dir, away)


@dataclass(frozen=True, slots=True)
class FrontendCapabilities:
    """What this surface can actually do, declared once at wiring.

    Every one of these was answered at CALL time by the callable failing, so
    `/btw` was offered to a run that could never spawn one and answered "needs
    a live run with a terminal" only once the operator asked. A surface that
    knows what it cannot do never offers it.
    """

    # A pane that renders as it happens (a console view, the TUI, the web).
    live_view: bool = True
    # Approvals and ask_user reach a human. False for a headless run with no
    # away-mode -- which is exactly what `headless_approval_refusal` computes.
    can_ask: bool = True
    # A pause menu exists: mid-run steering, /compact, /pin.
    can_steer: bool = True
    # May start sibling sessions: `/btw`, `/parallel`.
    can_spawn: bool = True


@dataclass(frozen=True, slots=True)
class RunFrontend:
    """The presentation + process-spawn callables `ui/cli` injects into the
    run/resume lifecycle: the live console view (held cli-side; the lifecycle
    only signals attach/close), the interactive prompts, and the REPLs. The
    lifecycle owns egress itself (`app.egress`) and the run-dir bridge
    (`runs.ipc`); only the exe-spawn primitives it can't reach stay injected.
    One value serves both `run_task` and `resume_task`; resume simply never
    calls the run-only fields."""

    # What this surface can do at all. Read before offering something, rather
    # than discovered by trying it.
    capabilities: FrontendCapabilities
    # live view: the console-view instance lives cli-side; builders that need it
    # (approver/questioner/steer/logger) close over it there.
    should_spawn_tui: Callable[[bool, bool, str], bool]
    stream_modes: Callable[[bool], tuple[bool, bool]]
    attach_console_view: Callable[[EventSink], None]
    close_console_view: Callable[[], None]
    loop_logger: Callable[[str], Callable[[str], None]]
    tui_session: Callable[[Path, bool], AbstractContextManager[None]]
    # operator interaction
    build_approver: Callable[[Path, EventSink], Approver]
    build_questioner: Callable[
        [Path, EventSink], Callable[[tuple[UserQuestion, ...]], tuple[str, ...]]
    ]
    make_steer_state: Callable[[EventSink, Path, Callable[[], RunFacts]], SteerHooks]
    confirm_unconfined_autorun: Callable[[IsolationLevel, Config], bool]
    confirm_run_on_run_branch: Callable[[str], bool]
    choose_branch_start_point: Callable[[Config, Path, str], BranchChoice]
    prompt_detach_away_mode: Callable[[Path], None]
    select_revised_prompt: Callable[[str, str, tuple[str, ...]], str | None]
    # `run -i` / `ask -i`
    build_repl_hook: Callable[
        [Path, BudgetTracker, str, MCPManager | None],
        Callable[[int, str], Literal["continue", "stop"]],
    ]
    run_ask_repl: Callable[[Workflow, BudgetTracker, RunLayout, str], RunResult]
    save_ask_transcript: Callable[[RunLayout, str, str], None]
    # `/parallel` coordinator dispatch (the cli builds LaneRuntime + spawner).
    build_coordinator_spawner: Callable[
        [Config, Path, Path, str, str, float | None, bool],
        GroupLaneSpawner | None,
    ]
    # process-spawn primitives the front-end owns (`ui.spawn`, mirroring
    # LaneRuntime's injected spawner).
    agent6_exe: Callable[[], str]
    spawn_detached_resume: Callable[[Path, str], str]


def discard_husk_dir(run_dir: Path) -> None:
    """Remove a run dir a preflight refused before any real content was written
    (no manifest, no logs). Otherwise a refused start (e.g. dirty worktree)
    leaves an empty husk that `agent6 runs` lists as '(no logs)' forever. Guarded
    on the manifest/logs check so a real run's dir is never removed."""
    if (run_dir / "manifest.json").exists() or (run_dir / LOGS_NAME).exists():
        return
    with contextlib.suppress(OSError):
        shutil.rmtree(run_dir)


def run_task(  # noqa: PLR0911, PLR0912, PLR0915
    cfg: Config,
    task: str,
    *,
    frontend: RunFrontend,
    run_id: str = "",
    interactive: bool = False,
    tui: bool = False,
    mode: Literal["run", "plan", "ask"] = "run",
    budget_overrides: BudgetOverrides | None = None,
    sandbox_overrides: SandboxOverrides | None = None,
    preset: str = "",
    initial_steer: str = "",
    pins: Sequence[str] = (),
    preset_stamp: tuple[str, bool] | None = None,
    reporter: Reporter = STDIO_REPORTER,
) -> int:
    """Single-loop agent: one provider, one LLM driving via tool
    calls over the fixed tool surface, deterministic harness (jail +
    budget + verify timeout + DAG curator for persistence/resume).
    Sole ``agent6 run`` path; returns the process exit code.

    ``initial_steer`` queues an operator follow-up for the loop's first
    boundary, seeded AFTER this function's own stale-state clear -- the
    parked-resume delegation passes `resume --steer` through it (a pre-seeded
    bridge file would be wiped by that clear and silently lost).

    The caller (`ui/cli/run.py`) has already built *cfg* (config + overrides),
    resolved the task text, checked the git-repo wall / runnable roles /
    provider keys, and routed ``--parallel`` away. *budget_overrides* /
    *sandbox_overrides* are passed through for the flags the lifecycle re-reads
    (`--max-usd` enforcement, lane dispatch).

    *preset_stamp* ``(name, from_flag)`` overrides the manifest's stamped
    preset instead of deriving it from *preset*. A parked resume has no
    ``--preset`` flag but must record the ORIGINAL submission's stamp so a
    later resume/fork replays the same precedence (fork carries it likewise);
    deriving from the empty *preset* dropped it, and the veto a flag-selected
    preset carried vanished on the next leg.

    When ``mode="plan"`` the same harness drives a planning
    pass instead of an execution pass: planning system prompt,
    edit-tools filtered out, ``finish_planning`` instead of
    ``finish_run``, no auto-commit. The plan markdown lands at
    ``<run-dir>/plan.md`` and is consumed by ``agent6 run --from-plan``.
    The ``planner`` model role drives plan mode (falls back to ``worker``).
    """
    role = role_for_mode(mode)

    # Before anything reads a knob (see session_config): an ask never runs a
    # command unwatched, whether it is starting here or resuming.
    cfg = session_config(cfg, mode)
    try:
        isolation = select_isolation(
            cfg, confirm_unconfined=frontend.confirm_unconfined_autorun, reporter=reporter
        )
    except SessionRefused as refusal:
        return refusal.rc

    # Git pre-flight (verify identity).
    # The auto-commit-on-verify-pass behaviour requires a clean working tree,
    # so the same git assumptions apply. Skipping these left first-time runs
    # crashing on dirty-tree or missing-identity errors deep into a paid run.
    cwd = Path.cwd()
    identity = CommitIdentity(
        name=cfg.git.commit.name,
        email=cfg.git.commit.email,
        coauthor=cfg.git.commit.coauthor,
    )
    # ask is read-only and may run outside a git repo (e.g. agent6 self-help),
    # so it skips the commit-oriented git pre-flight entirely.
    base_sha = ""
    base_branch = ""
    pre_status = None  # set below for run/plan; stays None for read-only ask
    if mode != "ask":
        # The not-a-git-repo guard already ran up front, before require_runnable.
        try:
            verify_git_identity(cwd, identity)
        except GitError as exc:
            reporter.err(f"ERROR: {exc}")
            return 2

        # Capture base sha + branch BEFORE we (optionally) cut a run branch
        # so `agent6 runs diff <run-id>` knows where the run started.
        try:
            pre_status = git_status(cwd)
        except GitError as exc:
            reporter.err(f"ERROR: {exc}")
            return 2
        base_sha = pre_status.head_sha
        base_branch = pre_status.branch
        # Starting a run while checked out on ANOTHER run's branch (agent6/<id>) is
        # usually a slip -- the operator forgot to merge or switch back -- so the new
        # run would pile on top of an unmerged one. Confirm; they may instead intend
        # to continue that line with a fresh session, in which case proceed.
        if (
            mode == "run"
            and base_branch.startswith("agent6/")
            and not frontend.confirm_run_on_run_branch(base_branch)
        ):
            reporter.err(
                "[agent6] aborted. Merge (agent6 runs merge) or switch branches first, then re-run."
            )
            return 2

    # Layout: standard run-dir scaffolding for transcripts + logs. ask sessions
    # live under the per-repo state dir (asks subdir) to stay separate from real runs.
    if run_id:
        try:
            validate_explicit_run_id(run_id)
        except RunIdError as exc:
            reporter.err(f"ERROR: {exc}")
            return 2
    effective_run_id = run_id or new_friendly_id()
    state_dir = resolved_state_dir(cwd)
    layout = RunLayout(
        state_dir=state_dir,
        run_id=effective_run_id,
        subdir="asks" if mode == "ask" else "runs",
    )
    # An explicit --run-id that already has a run is a resume, not a fresh start:
    # reusing the dir would write a new manifest + loop_state beside the old run's
    # graph/checkpoints/transcripts (mixed state). Refuse and point at resume.
    # (ask sessions are transient Q&A, so reusing their dir is fine.) The one
    # reusable dir is a PARKED run (manifest carries parked_task, nothing else
    # ever ran): starting it IS its fresh start, and the manifest rewrite below
    # un-parks it.
    if run_id and mode != "ask" and layout.manifest_path.exists():
        try:
            parked = read_manifest(layout.run_dir).parked_task
        except ManifestError:
            parked = ""
        if not parked:
            reporter.err(
                f"ERROR: run {run_id!r} already exists. Use `agent6 resume {run_id}` to "
                "continue it, or choose a different --run-id."
            )
            return 2
    # Under sudo the first run on a machine creates the whole state ancestry;
    # hand the created dirs back NOW, not at teardown -- a killed run must not
    # leave a root-owned base that blocks every other repo's non-root runs.
    mkdir_for_real_user(layout.run_dir)
    layout.ensure()
    # One authoritative writer per run dir. Acquire BEFORE touching any shared
    # run state (clearing answers, the worker pid, the curator) so a second
    # process refuses cleanly instead of clobbering the live run.
    worker_lock_fd = acquire_single_writer(layout.run_dir)
    if worker_lock_fd is None:
        reporter.err(SINGLE_WRITER_BUSY.format(rid=effective_run_id))
        return 2
    # Drop stale approve/ask/steer answers from a prior session (the
    # id counters reset on resume, so an old answer must not be read instead of
    # re-prompting; dead front-end claims are pruned by the liveness probe).
    clear_pending_answers(layout.run_dir)
    if initial_steer.strip():
        request_steer(layout.run_dir)
        write_steer_answer(layout.run_dir, initial_steer.strip())
    if sys.stdin.isatty():  # a foreground start clears a stale detach away-mode
        clear_away_mode(layout.run_dir)
    else:
        apply_spawned_away_default(layout.run_dir)
    # Record this worker's pid so `agent6 runs show` can probe liveness even while
    # the worker is blocked in a long provider call (which emits no events).
    write_worker_pid(layout.run_dir, os.getpid())

    # One live run-mode worker per CHECKOUT, not just per run dir: auto-commits
    # are `git add -A` on whatever HEAD points at, so a second concurrent run
    # that checks out its own branch makes both workers commit each other's
    # in-flight edits onto whichever branch won the last checkout. Taken BEFORE
    # any tree mutation (auto-stash, branch cut). plan/ask are read-only and
    # skip it. A refused submission is PARKED, not dropped: the manifest saves
    # the verbatim task and `agent6 resume <id>` starts it once the checkout
    # frees up.
    repo_lock_fd: int | None = None
    # Bound before the lock scope so the teardown can report on the run whatever
    # went wrong inside it.
    result: RunResult | None = None
    if mode == "run":
        repo_lock_fd = acquire_repo_writer(layout.state_dir, effective_run_id)
        if repo_lock_fd is None:
            holder = repo_writer_holder(layout.state_dir) or "another run"
            write_run_manifest(
                layout,
                run_id=effective_run_id,
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
                f" tree. Your task was parked as run {effective_run_id!r}:\n"
                f"    agent6 resume {effective_run_id}    (start it once the checkout"
                " is free)\n"
                f"or hand it to the live run as an isolated lane by steering"
                f" {holder!r} with:\n"
                "    /parallel 1 <the same task>"
            )
            clear_worker_pid(layout.run_dir)
            release_single_writer(worker_lock_fd)
            return 2

    # Enforce the dirty-tree policy BEFORE cutting the run branch, so the
    # branch is cut from a clean tree and the agent's per-step auto-commits
    # (`git add -A`) never swallow the user's pre-existing uncommitted work.
    # Only `run` makes commits; `plan`/`ask` are read-only (matching the
    # branch_per_run guard below).
    # Track an auto-stash so the run-end finalizer can restore or at least report
    # it; otherwise the user's stashed pre-run work is silently left behind.
    stashed = False
    base_branch = pre_status.branch if pre_status is not None else ""
    if mode == "run" and pre_status is not None and not pre_status.is_clean:
        if cfg.git.auto_stash:
            try:
                stash_all(cwd, auto_stash_message(effective_run_id))
                stashed = True
            except GitError as exc:
                reporter.err(f"ERROR: could not auto-stash before run: {exc}")
                clear_worker_pid(layout.run_dir)
                release_single_writer(repo_lock_fd)
                release_single_writer(worker_lock_fd)
                discard_husk_dir(layout.run_dir)
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
            clear_worker_pid(layout.run_dir)
            release_single_writer(repo_lock_fd)
            release_single_writer(worker_lock_fd)
            discard_husk_dir(layout.run_dir)
            return 2

    run_branch: str | None = None
    branch_start_point: str | None = None
    detach_requested = False
    try:
        # A fresh branch named after the run id is 1:1 with the run (find it
        # from any run id, `agent6 runs diff <id>`, or just delete the branch to
        # discard everything the agent did). The name is the unique run id,
        # never a timestamp+task-slug that collides into a pile of near-
        # duplicate `agent6/<ts>-<same-task>` branches on re-runs. Only real
        # `run` mode branches: `plan`/`ask` make no commits, so a branch for
        # them is pure litter. Decided here; the CUT happens below, after every
        # refusal-capable preflight step.
        if cfg.git.branch_per_run and mode == "run":
            run_branch = f"agent6/{effective_run_id}"
            # git.branch_from decides whether to cut from HEAD (stack) or from the
            # base line when you are on a previous run's branch (see BranchChoice).
            branch_choice = frontend.choose_branch_start_point(cfg, layout.state_dir, base_branch)
            if branch_choice.abort:
                reporter.err("[agent6] aborted; nothing was started.")
                clear_worker_pid(layout.run_dir)
                release_single_writer(repo_lock_fd)
                release_single_writer(worker_lock_fd)
                discard_husk_dir(layout.run_dir)
                return 0
            branch_start_point = branch_choice.start_point

        transcript_sink = TranscriptSink(layout.transcripts_dir)
        events = EventSink(layout.logs_path)

        try:
            start_isolation(cfg, isolation, reporter=reporter)
        except SessionRefused as refusal:
            # Nothing ran, so leave no run dir behind: every other refusal
            # discards its husk, and one that survives is listed forever as a
            # run that produced nothing.
            discard_husk_dir(layout.run_dir)
            return refusal.rc

        # Cut the run branch, then write the manifest that records it. The cut
        # is the ONLY workspace mutation in preflight and deliberately its LAST
        # step (mirroring resume): every refusal above -- and a failed cut
        # itself -- exits with the operator's checkout untouched and the run
        # dir still a discardable husk, not a manifest'd "(no logs)" ghost.
        if run_branch is not None:
            try:
                create_branch(cwd, run_branch, start_point=branch_start_point)
            except GitError as exc:
                reporter.err(f"ERROR: could not cut run branch {run_branch}: {exc}")
                discard_husk_dir(layout.run_dir)
                return 2

        # Write the run manifest. This is the canonical record of where the
        # run started (base_sha + base_branch), which model+provider drove
        # it, and the user_task it was given. `agent6 runs diff <run-id>` and
        # any future tooling that wants to reproduce a run reads from here.
        write_run_manifest(
            layout,
            run_id=effective_run_id,
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

        tui_enabled = frontend.should_spawn_tui(tui, interactive, mode)
        refusal = headless_approval_refusal(
            cfg, tui_enabled=tui_enabled, away=os.environ.get("AGENT6_DETACHED_AWAY", "")
        )
        if refusal is not None:
            reporter.err(f"REFUSING: {refusal}")
            return 2
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
        configured_gate = bool(cfg.workflow.verify_command)
        cfg = drop_gate_if_unrunnable(cfg, run_dir=layout.run_dir, reporter=reporter)
        cfg = infer_verify_if_unset(
            cfg, cwd, mode=mode, events=events, transcript_sink=transcript_sink, budget=budget
        )
        # After resolution, never before: preflight can DROP the gate (a run
        # that cannot run commands), and an empty gate with an origin of
        # "configured" is a self-contradiction the next leg reads back.
        gate_origin = ""
        if cfg.workflow.verify_command:
            gate_origin = "configured" if configured_gate else "inferred"
        # Pin it: from here the run is judged by THIS gate, whatever the file it
        # was inferred from says later.
        pin_gate(
            layout.run_dir,
            cfg.workflow.verify_command,
            gate_origin,
            events=events,
            reporter=reporter,
        )

        # Steering (mid-run Ctrl-C -> the pause menu) needs the terminal; the
        # console view's heartbeat spinner is suspended for the prompt so its
        # line-erase cannot wipe the pause-menu line.
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

        interrupted = False
        dispatcher: ToolDispatcher | None = None
        # Spawned inside the try so the finally below tears it down even if a
        # spawn (MCP) fails.
        mcp_manager = None
        try:
            reporter.err(f"[agent6] run id: {effective_run_id}")

            # Spawn any configured MCP servers BEFORE the workflow
            # starts so their tools are visible from iteration 1. The manager
            # owns its subprocesses; we close it in the finally block.
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
            after_auto_commit: Callable[[int, str], Literal["continue", "stop"]] = (
                frontend.build_repl_hook(cwd, budget, effective_run_id, mcp_manager)
                if interactive and mode == "run"
                else (lambda _i, _s: "continue")
            )
            wf = Workflow(
                root=cwd,
                config=cfg,
                initial_pins=tuple(pins),
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
                compact_requested=lambda: read_compact_request(layout.run_dir),
                compact_clear=lambda: clear_compact_request(layout.run_dir),
                stop_requested=lambda: stop_request_pending(layout.run_dir),
                stop_clear=lambda: clear_stop_request(layout.run_dir),
                should_abort=steer_state.abort_pending,
                should_interrupt=steer_state.interrupt,
                # `/parallel` steer dispatch: the coordinator's group spawner
                # (None in plan/ask, and inside a lane -- depth 1).
                lane_spawner=frontend.build_coordinator_spawner(
                    cfg,
                    cwd,
                    state_dir,
                    mode,
                    effective_run_id,
                    budget_overrides.max_usd if budget_overrides is not None else None,
                    sandbox_overrides.auto_approve if sandbox_overrides is not None else False,
                ),
                budget=budget,
                state_dir=state_dir,
                # Written for every mode: `agent6 resume` reaches an ask too.
                resume_state_path=layout.run_dir / "loop_state.json",
                mode=mode,
                plan_output_path=(layout.run_dir / "plan.md" if mode == "plan" else None),
                after_auto_commit=after_auto_commit,
                critic_provider=session.critic_provider,
                critic_mode=cfg.review.trigger,
                critic_period=cfg.review.period,
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
                critic_temperature=role_temperature(cfg, "reviewer"),
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
                compact_elision_gists=cfg.context.elision_gists,
            )
            try:
                with frontend.tui_session(layout.run_dir, tui_enabled):
                    if mode == "ask" and interactive:
                        result = frontend.run_ask_repl(wf, budget, layout, task)
                    else:
                        result = wf.run(task)
            except KeyboardInterrupt:
                interrupted = True
                reporter.err("\n[agent6] run interrupted")
                # The loop was cut mid-step, so it never emitted run.end; do it
                # here so an attached watcher/TUI stops instead of hanging. Carry
                # the iteration the loop reached so run.end keeps one shape.
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
            print_interrupt_end(layout=layout, budget=budget)
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
                reporter.err(f"\n[agent6] answer saved to {layout.run_dir / 'transcript.md'}")
            reporter.err(budget.format_summary())
            return 0 if result.completed else 1

        if result.reason == "detached":
            # Keep going in the background: the outer finally releases this run's
            # worker lock, then spawns a detached `resume` that picks it up.
            detach_requested = True
            reporter.out(f"\n[agent6] detached: {layout.run_id} continues in the background.")
            reporter.out(f"          reattach:  agent6 attach {layout.run_id}")
            return 0

        print_run_end(
            result,
            layout=layout,
            budget=budget,
            console_stream=console_stream,
        )
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
        # Single owner of worker.pid, egress-broker, and auto-stash
        # finalization, for EVERY exit path: refusal returns, Ctrl-C during
        # verify inference, and setup-window crashes included.
        frontend.close_console_view()  # stop the heartbeat thread, clear any spinner line
        clear_worker_pid(layout.run_dir)
        if stashed:
            if detach_requested:
                # The run is NOT over: the detached resume needs the checkout
                # left on the run branch, and popping the stash now would
                # either kill it (tracked dirt -> ensure_on_run_branch refuses
                # with stderr at /dev/null: a silent dead continuation) or,
                # for untracked-only dirt, feed the user's pre-run files into
                # the agent's auto-commits. Leave the stash and say so.
                # By sha, never by position: this hint has the LONGEST window of
                # any -- the operator reads it now and runs it after a
                # background run that may take hours, by which point a
                # positional pop restores whatever else was stashed meanwhile.
                hint = stash_recovery_hint(cwd, run_id=effective_run_id, base_branch=base_branch)
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
                    run_id=layout.run_id,
                )
        release_single_writer(repo_lock_fd)
        release_single_writer(worker_lock_fd)
        if detach_requested:
            # Ask how to handle approvals while away BEFORE spawning, so the marker is
            # set when the background run reads it. The worker lock is released now, so
            # the detached `resume` acquires it.
            if cfg.sandbox.run_commands == "ask" and not session_allow_set(layout.run_dir):
                frontend.prompt_detach_away_mode(layout.run_dir)
            err = frontend.spawn_detached_resume(cwd, layout.run_id)
            if err:
                reporter.err(f"[agent6] {err}")
