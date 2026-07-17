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
import subprocess
import sys
import tempfile
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from agent6.app._setup import (
    BudgetOverrides,
    SandboxOverrides,
    detect_env,
)
from agent6.app._setup import (
    explicit_usd_flag_error as _explicit_usd_flag_error,
)
from agent6.app._setup import (
    start_mcp_manager_if_enabled as _start_mcp_manager_if_enabled,
)
from agent6.app.finalize import (
    finalize_auto_merge as _finalize_auto_merge,
)
from agent6.app.finalize import (
    finalize_auto_stash as _finalize_auto_stash,
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
from agent6.app.manifest import (
    write_run_manifest as _write_run_manifest,
)
from agent6.app.preflight import (
    BranchChoice,
)
from agent6.app.preflight import (
    infer_verify_if_unset as _infer_verify_if_unset,
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
    build_prompt_reviser_provider as _build_prompt_reviser_provider,
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
from agent6.budget import BudgetTracker
from agent6.config import Config, RoleName
from agent6.config.layer import resolved_state_dir
from agent6.events import EventSink
from agent6.git_ops import (
    CommitIdentity,
    GitError,
    create_branch,
    dirty_paths,
    stash_all,
    verify_git_identity,
)
from agent6.git_ops import (
    status as git_status,
)
from agent6.graph.client import CuratorClientError, GraphClient, spawn_curator
from agent6.paths import chown_to_real_user
from agent6.providers import Provider, TranscriptSink
from agent6.runs.id import RunIdError, new_friendly_id, validate_explicit_run_id
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
from agent6.sandbox.detect import Environment, ProfileUnavailableError, select_profile
from agent6.tools.dispatch import ToolDispatcher
from agent6.tools.mcp_client import MCPManager
from agent6.tools.schema import UserQuestion
from agent6.types import SandboxProfile
from agent6.workflows.loop import RunResult, Workflow
from agent6.workflows.subrun import GroupLaneSpawner

# Default USD ceiling for `agent6 ask` when no budget is configured, so an
# exploratory question can't quietly run up a bill.
_ASK_DEFAULT_MAX_USD = 0.50


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


@dataclass(frozen=True, slots=True)
class RunBridge:
    """The run-dir file-bridge contract (`ui.bridge.approval`), injected so the
    lifecycle never imports `agent6.ui` (the same seam `LaneRuntime` uses for
    its liveness probe)."""

    clear_pending_answers: Callable[[Path], None]
    clear_away_mode: Callable[[Path], None]
    apply_spawned_away_default: Callable[[Path], None]
    write_worker_pid: Callable[[Path, int], None]
    clear_worker_pid: Callable[[Path], None]
    compact_request_pending: Callable[[Path], bool]
    clear_compact_request: Callable[[Path], None]
    stop_request_pending: Callable[[Path], bool]
    clear_stop_request: Callable[[Path], None]
    session_allow_set: Callable[[Path], bool]
    # resume --steer seeds the first steering instruction over the same bridge.
    request_steer: Callable[[Path], None]
    write_steer_answer: Callable[[Path, str], None]


@dataclass(frozen=True, slots=True)
class EgressHooks:
    """Interim egress seam (the guard moves into `app` with `ui/cli/egress.py`):
    the `EgressGuard` lives cli-side behind these closures, so the lifecycle
    drives WHEN egress starts/stops without holding the guard value. The
    ``start`` hook also prints the broker banner (presentation)."""

    warn_if_unsandboxed: Callable[[SandboxProfile], None]
    check_network_profile: Callable[[Config, SandboxProfile], str | None]
    resolve_strict_viability: Callable[[Config, SandboxProfile], tuple[SandboxProfile, str | None]]
    start: Callable[[Config, SandboxProfile], str | None]
    apply_agent_landlock: Callable[[Config, SandboxProfile, Environment], str | None]
    stop: Callable[[], None]
    spawn_detached: Callable[[Path, str], str]
    close_detach_spawner: Callable[[], None]


@dataclass(frozen=True, slots=True)
class RunFrontend:
    """The presentation callables `ui/cli` injects into the run/resume
    lifecycle: the live console view (held cli-side; the lifecycle only signals
    attach/close), the interactive prompts, the REPLs, and the bridge/egress
    sub-seams. One value serves both `run_task` and `resume_task`; resume simply
    never calls the run-only fields."""

    # live view: the console-view instance lives cli-side; builders that need it
    # (approver/questioner/steer/logger) close over it there.
    should_spawn_tui: Callable[[bool, bool, str], bool]
    stream_modes: Callable[[bool], tuple[bool, bool]]
    attach_console_view: Callable[[EventSink], None]
    close_console_view: Callable[[], None]
    loop_logger: Callable[[str], Callable[[str], None]]
    tui_session: Callable[[Path, bool], AbstractContextManager[None]]
    # operator interaction
    build_approver: Callable[[Path, EventSink], Callable[[str], bool]]
    build_questioner: Callable[
        [Path, EventSink], Callable[[tuple[UserQuestion, ...]], tuple[str, ...]]
    ]
    make_steer_state: Callable[[EventSink, Path], SteerHooks]
    confirm_unconfined_autorun: Callable[[SandboxProfile, Config], bool]
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
    # `/parallel` coordinator dispatch (the cli builds LaneRuntime + spawner)
    build_coordinator_spawner: Callable[
        [Config, Path, Path, str, str, float | None, bool], GroupLaneSpawner | None
    ]
    # sub-seams
    bridge: RunBridge
    egress: EgressHooks


def discard_husk_dir(run_dir: Path) -> None:
    """Remove a run dir a preflight refused before any real content was written
    (no manifest, no logs). Otherwise a refused start (e.g. dirty worktree)
    leaves an empty husk that `agent6 runs` lists as '(no logs)' forever. Guarded
    on the manifest/logs check so a real run's dir is never removed."""
    if (run_dir / "manifest.json").exists() or (run_dir / "logs.jsonl").exists():
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
    profile: str = "",
) -> int:
    """Single-loop agent: one provider, one LLM driving via tool
    calls over the fixed tool surface, deterministic harness (jail +
    budget + verify timeout + DAG curator for persistence/resume).
    Sole ``agent6 run`` path; returns the process exit code.

    The caller (`ui/cli/run.py`) has already built *cfg* (config + overrides),
    resolved the task text, checked the git-repo wall / runnable roles /
    provider keys, and routed ``--parallel`` away. *budget_overrides* /
    *sandbox_overrides* are passed through for the flags the lifecycle re-reads
    (`--max-usd` enforcement, lane dispatch).

    When ``mode="plan"`` the same harness drives a planning
    pass instead of an execution pass: planning system prompt,
    edit-tools filtered out, ``finish_planning`` instead of
    ``finish_run``, no auto-commit. The plan markdown lands at
    ``<run-dir>/plan.md`` and is consumed by ``agent6 run --from-plan``.
    The ``planner`` model role drives plan mode (falls back to ``worker``).
    """
    role: RoleName = "planner" if mode == "plan" else "worker"

    env = detect_env()
    try:
        selected_profile = select_profile(cfg.sandbox.profile, env)
    except ProfileUnavailableError as exc:
        print(f"REFUSING: {exc}", file=sys.stderr)
        return 2
    frontend.egress.warn_if_unsandboxed(selected_profile)
    if not frontend.confirm_unconfined_autorun(selected_profile, cfg):
        print("[agent6] aborted.", file=sys.stderr)
        return 1

    net_err = frontend.egress.check_network_profile(cfg, selected_profile)
    if net_err is not None:
        print(f"REFUSING: {net_err}", file=sys.stderr)
        return 2
    # strict can be selected because the jail launcher has userns, yet this
    # process can't create one for the egress broker (surgical AppArmor profile).
    # Downgrade auto->hardened, or refuse an explicit strict, with guidance.
    selected_profile, egress_err = frontend.egress.resolve_strict_viability(cfg, selected_profile)
    if egress_err is not None:
        print(egress_err, file=sys.stderr)
        return 2

    usd_err = _explicit_usd_flag_error(budget_overrides.max_usd if budget_overrides else None, cfg)
    if usd_err is not None:
        print(f"REFUSING: {usd_err}", file=sys.stderr)
        return 2

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
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2

        # Capture base sha + branch BEFORE we (optionally) cut a run branch
        # so `agent6 runs diff <run-id>` knows where the run started.
        try:
            pre_status = git_status(cwd)
        except GitError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
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
            print(
                "[agent6] aborted. Merge (agent6 runs merge) or switch branches first,"
                " then re-run.",
                file=sys.stderr,
            )
            return 2

    # Layout: standard run-dir scaffolding for transcripts + logs. ask sessions
    # live under the per-repo state dir (asks subdir) to stay separate from real runs.
    if run_id:
        try:
            validate_explicit_run_id(run_id)
        except RunIdError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
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
    # (ask sessions are transient Q&A, so reusing their dir is fine.)
    if run_id and mode != "ask" and layout.manifest_path.exists():
        print(
            f"ERROR: run {run_id!r} already exists. Use `agent6 resume {run_id}` to "
            "continue it, or choose a different --run-id.",
            file=sys.stderr,
        )
        return 2
    layout.ensure()
    # One authoritative writer per run dir. Acquire BEFORE touching any shared
    # run state (clearing answers, the worker pid, the curator) so a second
    # process refuses cleanly instead of clobbering the live run.
    worker_lock_fd = _acquire_single_writer(layout.run_dir)
    if worker_lock_fd is None:
        print(_SINGLE_WRITER_BUSY.format(rid=effective_run_id), file=sys.stderr)
        return 2
    # Drop stale approve/ask/steer answers + frontend.pid from a prior session (the
    # id counters reset on resume, so an old answer must not be read instead of
    # re-prompting; a stale frontend.pid would otherwise stall the answer-poll).
    frontend.bridge.clear_pending_answers(layout.run_dir)
    if sys.stdin.isatty():  # a foreground start clears a stale detach away-mode
        frontend.bridge.clear_away_mode(layout.run_dir)
    else:
        # A front-end launcher (web/TUI hub) spawns this run detached and drives
        # it over the bridge, but with no controlling terminal ask_user would
        # otherwise fabricate an empty answer when no viewer happens to be
        # connected. The launcher sets AGENT6_DETACHED_AWAY so approvals AND
        # questions WAIT for a front-end instead. A pure headless run (CI, no
        # launcher) sets no env, so it keeps its non-hanging default.
        frontend.bridge.apply_spawned_away_default(layout.run_dir)
    # Record this worker's pid so `agent6 runs show` can probe liveness even while
    # the worker is blocked in a long provider call (which emits no events).
    frontend.bridge.write_worker_pid(layout.run_dir, os.getpid())

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
                stash_all(cwd, f"agent6 auto-stash before run {effective_run_id}")
                stashed = True
            except GitError as exc:
                print(f"ERROR: could not auto-stash before run: {exc}", file=sys.stderr)
                frontend.bridge.clear_worker_pid(layout.run_dir)
                _release_single_writer(worker_lock_fd)
                discard_husk_dir(layout.run_dir)
                return 2
        elif cfg.git.require_clean_worktree:
            dirty = dirty_paths(cwd)
            listed = "\n".join(f"    {p}" for p in dirty)
            more = "\n    ..." if len(dirty) >= 10 else ""
            print(
                "REFUSING: working tree is not clean:\n"
                f"{listed}{more}\n"
                "Commit, stash, or discard your changes, set [git].auto_stash=true, "
                "or set [git].require_clean_worktree=false to override.",
                file=sys.stderr,
            )
            frontend.bridge.clear_worker_pid(layout.run_dir)
            _release_single_writer(worker_lock_fd)
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
                print("[agent6] aborted; nothing was started.", file=sys.stderr)
                frontend.bridge.clear_worker_pid(layout.run_dir)
                _release_single_writer(worker_lock_fd)
                discard_husk_dir(layout.run_dir)
                return 0
            branch_start_point = branch_choice.start_point

        transcript_sink = TranscriptSink(layout.transcripts_dir)
        events = EventSink(layout.logs_path)

        egress_err = frontend.egress.start(cfg, selected_profile)
        if egress_err is not None:
            print(f"REFUSING: {egress_err}", file=sys.stderr)
            return 2

        landlock_err = frontend.egress.apply_agent_landlock(cfg, selected_profile, env)
        if landlock_err is not None:
            print(f"REFUSING: {landlock_err}", file=sys.stderr)
            return 2

        # Cut the run branch, then write the manifest that records it. The cut
        # is the ONLY workspace mutation in preflight and deliberately its LAST
        # step (mirroring resume): every refusal above -- and a failed cut
        # itself -- exits with the operator's checkout untouched and the run
        # dir still a discardable husk, not a manifest'd "(no logs)" ghost.
        if run_branch is not None:
            try:
                create_branch(cwd, run_branch, start_point=branch_start_point)
            except GitError as exc:
                print(f"ERROR: could not cut run branch {run_branch}: {exc}", file=sys.stderr)
                discard_husk_dir(layout.run_dir)
                return 2

        # Write the run manifest. This is the canonical record of where the
        # run started (base_sha + base_branch), which model+provider drove
        # it, and the user_task it was given. `agent6 runs diff <run-id>` and
        # any future tooling that wants to reproduce a run reads from here.
        _write_run_manifest(
            layout,
            run_id=effective_run_id,
            user_task=task,
            base_sha=base_sha,
            base_branch=base_branch,
            run_branch=run_branch,
            cfg=cfg,
            mode=mode,
            effective_profile=profile or cfg.profile,
        )

        # ask gets a small default USD ceiling so an exploratory question can't run
        # away; an explicit [budget].best_effort_usd_limit or --max-usd overrides it.
        usd_limit = cfg.budget.best_effort_usd_limit
        ask_max_usd = usd_limit or (_ASK_DEFAULT_MAX_USD if mode == "ask" else 0.0)
        budget = BudgetTracker(
            max_input_tokens=cfg.budget.max_input_tokens,
            max_output_tokens=cfg.budget.max_output_tokens,
            max_usd=ask_max_usd,
        )

        # Workflow uses ONE provider for everything (the worker role, or the
        # planner role in plan mode). No critic/triage/planner/reviewer/escalation
        # cascade inside the loop.
        worker_inner = _build_role_provider(
            cfg, role, transcript_sink=transcript_sink, budget=budget
        )
        rm_worker = cfg.models.resolve(role)
        assert rm_worker is not None  # require_runnable validated this
        _warn_if_usd_unenforceable(cfg)
        _warn_if_prompt_override_incomplete(cfg)
        # Enable SSE streaming when stderr is a TTY (covers TUI
        # and interactive shell use). Bench/CI runs pipe stderr, so they
        # stay on the audited non-streaming code path UNLESS the operator
        # sets AGENT6_FORCE_STREAM=1, the Kimi/OpenRouter bench needs
        # streaming on because the gateway emits SSE keep-alive comment
        # heartbeats during long requests, which corrupt the non-streaming
        # response body (resp.json() blows up with JSONDecodeError).
        tui_enabled = frontend.should_spawn_tui(tui, interactive, mode)
        _warn_if_headless_ask(cfg, tui_enabled=tui_enabled)
        # The interactive revision prompt reads the terminal; with the TUI owning
        # it the prompt would land invisibly in the console log and contend for
        # stdin. Skip revision for this run instead.
        effective_revise_prompt = cfg.prompt.revise_prompt
        if effective_revise_prompt == "interactive" and tui_enabled:
            print(
                "[agent6] prompt.revise_prompt='interactive' needs the terminal; the TUI"
                " owns it. Skipping prompt revision for this run.",
                file=sys.stderr,
            )
            effective_revise_prompt = "off"
        stream_text, console_stream = frontend.stream_modes(tui_enabled)
        if console_stream:
            frontend.attach_console_view(events)
        provider: Provider = _InstrumentedProvider(
            inner=worker_inner,
            role=role,
            model=rm_worker.model,
            provider_name=rm_worker.provider,
            events=events,
            budget=budget,
            stream_text=stream_text,
        )

        critic_provider = _build_critic_provider(
            cfg, transcript_sink=transcript_sink, budget=budget, events=events
        )
        prompt_reviser_provider = _build_prompt_reviser_provider(
            cfg, transcript_sink=transcript_sink, budget=budget, events=events
        )
        summariser_provider = _build_summariser_provider(
            cfg, transcript_sink=transcript_sink, budget=budget, events=events
        )
        # The grounded review panel runs at the critic trigger WHEN explicitly
        # configured (any review_* key); otherwise critic!=off keeps the legacy single
        # critic, so a pre-panel before_finish/periodic config still gates as before.
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

        # Verify is optional: if unset, infer one for this run (AGENTS.md -> repo
        # signals -> a cheap LLM call) and inject it in-memory. Never persisted.
        cfg = _infer_verify_if_unset(
            cfg, cwd, mode=mode, events=events, transcript_sink=transcript_sink, budget=budget
        )

        # AF_UNIX paths have a 108-char limit (Linux sun_path), which
        # bench setups with long BENCH_ROOT (and any future overlay-mount
        # paths) blew through. Bind the socket under a short /tmp dir and
        # leave a symlink under run_dir for observability. Cleaned up in
        # the finally block. See bench/improvement_plan.md audit cross-cutting.
        sock_path = layout.run_dir / "curator.sock"  # rebound to the /tmp socket inside the try

        # Steering (mid-run Ctrl-C -> the pause menu) needs the terminal; the
        # console view's heartbeat spinner is suspended for the prompt so its
        # line-erase cannot wipe the pause-menu line.
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
            # Spawn the curator + connect a GraphClient so the agent
            # has access to the DAG-as-tool surface.
            sock_tmpdir = Path(tempfile.mkdtemp(prefix="agent6-sock-"))
            sock_path = sock_tmpdir / "curator.sock"
            sock_link = layout.run_dir / "curator.sock"
            with contextlib.suppress(FileNotFoundError):
                sock_link.unlink()
            sock_link.symlink_to(sock_path)
            curator_proc = spawn_curator(
                state_dir, effective_run_id, sock_path, subdir=layout.subdir
            )
            print(f"[agent6] run id: {effective_run_id}", file=sys.stderr)

            # Spawn any configured MCP servers BEFORE the workflow
            # starts so their tools are visible from iteration 1. The manager
            # owns its subprocesses; we close it in the finally block.
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
                    run_root_node_id=None,  # Workflow seeds the root + calls set_run_root_node_id
                    mcp_manager=mcp_manager,
                    mode=mode,
                    state_dir=state_dir,
                )
                loop_log = frontend.loop_logger(mode)
                compact_drop, compact_summarise = resolve_compaction_thresholds(
                    cfg, rm_worker, log=loop_log
                )
                cfg = resolve_decompose(cfg, rm_worker, log=loop_log)
                after_auto_commit: Callable[[int, str], Literal["continue", "stop"]] = (
                    frontend.build_repl_hook(cwd, budget, effective_run_id, mcp_manager)
                    if interactive and mode == "run"
                    else (lambda _i, _s: "continue")
                )
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
                    compact_requested=lambda: frontend.bridge.compact_request_pending(
                        layout.run_dir
                    ),
                    compact_clear=lambda: frontend.bridge.clear_compact_request(layout.run_dir),
                    stop_requested=lambda: frontend.bridge.stop_request_pending(layout.run_dir),
                    stop_clear=lambda: frontend.bridge.clear_stop_request(layout.run_dir),
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
                    # `agent6 ask` (under asks/) is not resumable -- `agent6 resume`
                    # only looks under runs/ -- so don't write an orphan snapshot.
                    resume_state_path=(
                        None if mode == "ask" else layout.run_dir / "loop_state.json"
                    ),
                    mode=mode,
                    plan_output_path=(layout.run_dir / "plan.md" if mode == "plan" else None),
                    after_auto_commit=after_auto_commit,
                    critic_provider=critic_provider,
                    critic_mode=cfg.review.trigger,
                    critic_period=cfg.review.period,
                    review_seats=review_seats,
                    review_decision=cfg.review.decision,
                    review_quorum=cfg.review.quorum,
                    review_max_total_rejections=cfg.review.max_total_rejections,
                    review_budget_fraction=cfg.review.budget_fraction,
                    review_concurrency=cfg.review.concurrency,
                    base_sha=base_sha,
                    prompt_reviser_provider=prompt_reviser_provider,
                    revise_prompt=effective_revise_prompt,
                    temperature=_role_temperature(cfg, role),
                    critic_temperature=_role_temperature(cfg, "reviewer"),
                    prompt_reviser_temperature=_role_temperature(cfg, "reviewer"),
                    prompt_revision_selector=(
                        frontend.select_revised_prompt
                        if effective_revise_prompt == "interactive"
                        else None
                    ),
                    summariser_provider=summariser_provider,
                    compact_drop_at_chars=compact_drop,
                    compact_summarise_at_chars=compact_summarise,
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
                    print("\n[agent6] run interrupted", file=sys.stderr)
                    # The loop was cut mid-step, so it never emitted run.end; do it
                    # here so an attached watcher/TUI stops instead of hanging.
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
            # Clean up the /tmp socket dir + symlink under run_dir.
            with contextlib.suppress(FileNotFoundError):
                (layout.run_dir / "curator.sock").unlink()
            if sock_tmpdir is not None:
                shutil.rmtree(sock_tmpdir, ignore_errors=True)
            if dispatcher is not None:
                dispatcher.close()
            if mcp_manager is not None:
                mcp_manager.close()
            if not interrupted and result is not None and result.completed and cfg.git.auto_merge:
                _finalize_auto_merge(cwd, layout=layout, cfg=cfg)
            # Never leave root-owned run state in the user's repo (sudo case).
            chown_to_real_user(state_dir)

        if interrupted:
            return 130
        if result is None:
            return 1

        if mode == "ask":
            # The answer IS result.summary (kept whole in ask mode). stdout gets
            # just the answer (clean for piping); cost + saved-path go to stderr.
            # The REPL already printed + saved each turn, so only the one-shot path
            # prints/saves here.
            if not interactive:
                print(result.summary)
                frontend.save_ask_transcript(layout, task, result.summary)
                print(
                    f"\n[agent6] answer saved to {layout.run_dir / 'transcript.md'}",
                    file=sys.stderr,
                )
            print(budget.format_summary(), file=sys.stderr)
            return 0 if result.completed else 1

        if result.reason == "detached":
            # Keep going in the background: the outer finally releases this run's
            # worker lock, then spawns a detached `resume` that picks it up.
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
        # Single owner of worker.pid, egress-broker, and auto-stash
        # finalization. Refusal returns, Ctrl-C during verify inference, and
        # setup-window crashes used to skip these, leaving a stale pid, a
        # leaked broker process, and the user's stashed work silently hidden.
        frontend.close_console_view()  # stop the heartbeat thread, clear any spinner line
        frontend.bridge.clear_worker_pid(layout.run_dir)
        frontend.egress.stop()
        if stashed:
            _finalize_auto_stash(
                cwd,
                base_branch=base_branch,
                run_branch=run_branch,
                auto_pop=cfg.git.auto_stash_pop,
            )
        _release_single_writer(worker_lock_fd)
        if detach_requested:
            # Ask how to handle approvals while away BEFORE spawning, so the marker is
            # set when the background run reads it. The worker lock is released now, so
            # the detached `resume` acquires it.
            if cfg.sandbox.run_commands == "ask" and not frontend.bridge.session_allow_set(
                layout.run_dir
            ):
                frontend.prompt_detach_away_mode(layout.run_dir)
            err = frontend.egress.spawn_detached(cwd, layout.run_id)
            if err:
                print(f"[agent6] {err}", file=sys.stderr)
        frontend.egress.close_detach_spawner()
