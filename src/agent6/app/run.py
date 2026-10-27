# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The `agent6 run` lifecycle and its plan and ask modes.

Preflight, branch cut, manifest, loop construction, finalize. Everything that
touches the terminal is injected through `SessionFrontend`, so this module never
imports `agent6.ui`.
"""

from __future__ import annotations

import contextlib
import os
import pathlib
import shutil
from collections.abc import Sequence

from agent6 import budget as agent6_budget
from agent6 import directive, errors, event_log, kinds, paths
from agent6.app import (
    _execution,
    _session,
    _setup,
    _workdirs,
    finalize,
    fork_worktrees,
    preflight,
    stamps,
    workspace_setup,
)
from agent6.app import frontend as app_frontend
from agent6.app import reporter as app_reporter
from agent6.config import Config
from agent6.harness import _context
from agent6.providers import TranscriptSink
from agent6.sessions import id, ipc, lock, process
from agent6.sessions import layout as sessions_layout
from agent6.sessions import manifest as sessions_manifest
from agent6.tools import operator_prompts
from agent6.vcs import git_ops, git_read
from agent6.viewmodel import listing


def discard_husk_dir(session_dir: pathlib.Path) -> None:
    """Remove a session dir a preflight refused before a manifest or log was written."""
    if sessions_layout.session_has_record(session_dir):
        return
    with contextlib.suppress(OSError):
        shutil.rmtree(session_dir)


def run_task(  # noqa: C901, PLR0911, PLR0912, PLR0915  # every way a run is refused, in preflight order
    cfg: Config,
    task: str,
    *,
    frontend: app_frontend.SessionFrontend,
    started_at: float,
    session_id: str = "",
    source_session_id: str | None = None,
    interactive: bool = False,
    tui: bool = False,
    mode: kinds.ResumableMode = "run",
    standing_goal: str = "",
    budget_overrides: _setup.BudgetOverrides | None = None,
    sandbox_overrides: _setup.SandboxOverrides | None = None,
    preset: str = "",
    initial_steer: str = "",
    pins: Sequence[str] = (),
    preset_stamp: tuple[str, bool] | None = None,
    model: str | kinds.ModelRoute | None = None,
    explicit_leaves: frozenset[str] = frozenset(),
    worktree: bool = False,
    config_paths: tuple[pathlib.Path, ...] = (),
    reporter: app_reporter.Reporter = app_reporter.STDIO_REPORTER,
) -> int:
    """Run one session from preflight to finalize; the sole `agent6 run` path.

    The front-end has built the config, resolved the task, checked the git-repo wall
    and the runnable roles, and routed `--parallel` away. Plan mode drives the same
    harness with the planning prompt, no edit tools, `finish_planning` and no
    auto-commit; its markdown lands at `<session-dir>/plan.md`.

    Args:
        cfg: The run's config, overrides applied.
        task: The task text.
        frontend: The surface's seam: console, confirms, approvals, questions.
        started_at: The instant this execution began; the stale-state clear keeps
            bridge markers written since.
        session_id: An explicit id; empty allocates one.
        source_session_id: The plan session a `run --from` came from.
        interactive: Whether the ask REPL drives the session.
        tui: Whether `--tui` was asked for.
        mode: The session mode.
        standing_goal: The standing goal, when set.
        budget_overrides: The budget flags the lifecycle re-reads.
        sandbox_overrides: The sandbox flags the lifecycle re-reads.
        preset: The `--preset` flag's value.
        initial_steer: An operator follow-up queued for the loop's first boundary,
            seeded after the stale-state clear so it survives it.
        pins: The pinned paths.
        preset_stamp: `(name, from_flag)` to stamp in place of deriving from `preset`,
            so a parked resume replays the original submission's precedence.
        model: The `--model` value that set the driving role.
        explicit_leaves: The config leaves the operator wrote, as dotted paths; a
            default this host cannot honour degrades, a written value refuses.
        worktree: Whether the run works in its own linked worktree, added at HEAD's
            commit under `[workspace].base_dir`; the checkout is then never touched.
        config_paths: The `--config` files, which an `/undo` fork is stamped under.
        reporter: Receives every refusal and note.

    Returns:
        The process exit code.
    """
    role = kinds.session_kind(mode).role
    repo = pathlib.Path.cwd()
    # The run's checkout: the repository, or the worktree `worktree` adds below.
    cwd = repo
    # A first prompt is a task, not a composer line.
    if (problem := directive.steer_problem(task)) is not None:
        reporter.error(problem)
        return 2
    if session_id:
        try:
            id.validate_explicit_session_id(session_id)
        except id.SessionIdError as exc:
            reporter.error(str(exc))
            return 2

    # The clamp comes before anything reads a knob; the invocation's grant lands after it.
    cfg = _setup.session_config(cfg, mode, sandbox_overrides)
    if worktree and cfg.git.control == "model":
        reporter.refuse(
            "--worktree runs in a linked worktree whose .git is read-only in the jail; under"
            ' [git].control = "model" the model could not commit there. Set [git].control ='
            ' "agent6" (`agent6 config set --repo git.control agent6`), or run without'
            " --worktree."
        )
        return 2
    if worktree and (top := paths.checkout_root(repo)) != repo.resolve():
        reporter.refuse(
            f"--worktree starts at the repository's top level, and this is a subdirectory of"
            f" {top}. Run it from there, or run without --worktree."
        )
        return 2
    # Refuse an unanswerable run before anything is created, so its id stays free.
    tui_enabled = frontend.should_spawn_tui(tui, interactive, mode)
    refusal = preflight.headless_approval_refusal(
        cfg,
        tui_enabled=tui_enabled,
        # A fresh run has no dir yet, so the env is the only away answer.
        away=os.environ.get("AGENT6_DETACHED_AWAY", ""),
        can_ask=frontend.capabilities.can_ask,
        clamped=kinds.session_kind(mode).clamps_commands,
    )
    if refusal is not None:
        reporter.refuse(refusal)
        return 2
    parking = preflight.headless_parking_note(
        cfg,
        tui_enabled=tui_enabled,
        away=os.environ.get("AGENT6_DETACHED_AWAY", ""),
        can_ask=frontend.capabilities.can_ask,
    )
    if parking is not None:
        reporter.note(parking)
    # Before isolation: its budget preflight prices the model from the refreshed listing.
    if not preflight.route_preflight(
        cfg, role, reporter=reporter, model_flag=_setup.route_text(model)
    ):
        return 2
    # A worktree run selects below, once the worktree its commands run in exists.
    isolation: kinds.IsolationLevel | None = None
    if not worktree:
        try:
            isolation = _session.select_isolation(
                cfg,
                cwd=cwd,
                confirm_unconfined=frontend.confirm_unconfined_autorun,
                reporter=reporter,
            )
        except preflight.SessionRefusedError as refusal:
            return refusal.rc

    try:
        git = preflight.git_preflight(
            repo,
            cfg,
            mode,
            confirm_run_on_run_branch=frontend.confirm_run_on_run_branch,
            reporter=reporter,
        )
    except preflight.SessionRefusedError as refusal:
        return refusal.rc
    base_sha, base_branch = git.base_sha, git.base_branch
    if worktree and not base_sha:
        reporter.refuse(
            "--worktree starts from HEAD's commit, and this repository has none. Commit first"
            ' (`git add -A && git commit -m "initial commit"`), or run without --worktree.'
        )
        return 2

    state = paths.state_dir(repo)
    bucket = kinds.session_bucket(mode)
    # Same-bucket reuse is the resume or park flow below; another bucket's id is a collision.
    if session_id and (held := id.session_id_bucket(state, session_id)) not in (None, bucket):
        reporter.error(
            f"--session-id {session_id!r} already names a session under {held}/;"
            " ids are unique across every bucket. Pick another id."
        )
        return 2
    effective_session_id = session_id or id.unused_session_id(state, bucket)
    layout = sessions_layout.SessionLayout(
        state_dir=state,
        session_id=effective_session_id,
        subdir=bucket,
    )
    # An existing session under the id is a resume, not a fresh start; a parked run and an
    # ask are the reusable dirs.
    if session_id and mode != "ask" and layout.manifest_path.exists():
        try:
            parked = sessions_manifest.read_manifest(layout.session_dir).parked_task
        except sessions_manifest.ManifestError:
            parked = ""
            resume = ""
        else:
            resume = (
                f'`agent6 resume {session_id} --steer "<what to do next>"` to give it new work'
                if listing.finished_needs_new_work(layout.session_dir)
                else f"`agent6 resume {session_id}` to continue it"
            )
        if not parked:
            next_step = (
                f" Use {resume}, or choose a different --session-id."
                if resume
                else " Choose a different --session-id."
            )
            reporter.error(f"run {session_id!r} already exists.{next_step}")
            return 2

    if worktree or stamps.parallel_lineage() is not None:
        cfg = workspace_setup.infer_if_unset(cfg, repo, reporter=reporter, at=base_sha)
    if worktree and (
        gate_refusal := workspace_setup.gate_refusal(
            cfg,
            repo,
            workspace_setup.origin_gate(cfg, repo),
            workspace="a new worktree",
            at=base_sha,
        )
    ):
        reporter.refuse(gate_refusal)
        return 2

    checkout: fork_worktrees.Checkout | None = None

    def _drop_unrecorded_worktree() -> None:
        # The manifest names a worktree as agent6's, so a start refused before it leaves none.
        if checkout is None:
            return
        with contextlib.suppress(sessions_manifest.ManifestError):
            if sessions_manifest.read_manifest(layout.session_dir).worktree == checkout.worktree:
                return
        gone, why = fork_worktrees.remove_fork_worktree(repo, checkout.worktree, (base_sha,))
        if not gone:
            reporter.note(
                f"the worktree {checkout.worktree} stays: it {why}. No session names it; remove"
                f" it with `git worktree remove {checkout.worktree}`"
            )

    if worktree:
        path = _workdirs.subordinate_workdir_root(cfg, repo, effective_session_id)
        if path.is_relative_to(repo.resolve()):
            reporter.refuse(
                f"[workspace].base_dir puts the worktree inside this checkout ({path}), where it"
                " would show as an untracked change. Set [workspace].base_dir to a directory"
                " outside the repository, or run without --worktree."
            )
            return 2
        try:
            checkout = fork_worktrees.add_session_worktree(repo, path, base_sha)
        except errors.GitError as exc:
            reporter.error(f"could not add the run's worktree at {path}: {exc}")
            return 1
        cwd = checkout.worktree
    worktree_git_dir = checkout.git_dir if checkout is not None else None
    worker_lock_fd: int | None = None
    try:
        if checkout is not None:
            isolation = _session.select_isolation(
                cfg,
                cwd=cwd,
                confirm_unconfined=frontend.confirm_unconfined_autorun,
                reporter=reporter,
                worktree_git_dir=worktree_git_dir,
            )
        assert isolation is not None  # selected before the git preflight, or on the worktree
        layout.ensure()
        # One writer per session dir, taken before any shared state is touched.
        worker_lock_fd = lock.acquire_single_writer(layout.session_dir)
        if worker_lock_fd is None:
            reporter.err(lock.SINGLE_WRITER_BUSY.format(rid=effective_session_id))
            return 2
    except preflight.SessionRefusedError as refusal:
        return refusal.rc
    finally:
        # Refused, interrupted or failed before the run's own cleanup below takes over.
        if worker_lock_fd is None:
            _drop_unrecorded_worktree()
    if checkout is not None:
        reporter.note(
            f"working in the worktree {cwd} (at {base_sha[:12]}); this checkout and its"
            " uncommitted changes stay as they are"
        )
    repo_lock_fd: int | None = None
    stashed = False
    # Whether the stash is applied back at run end: config, or "stash" chosen at the start.
    stash_pop = cfg.git.auto_stash_pop
    untracked_at_start: frozenset[str] = frozenset()
    run_branch: str | None = None
    detach_requested = False

    def _undo_forker() -> tuple[str, str] | None:
        # Lazy: app.fork imports app.resume, which imports this module.
        from agent6.app import undo  # noqa: PLC0415  # noqa: PLC0415

        return undo.undo_fork(config_paths, effective_session_id, cwd=repo, reporter=reporter)

    try:
        # A reused dir carries the previous execution's bridge state.
        ipc.clear_pending_answers(layout.session_dir, started_at=started_at)
        if initial_steer.strip() and not ipc.submit_steer(
            layout.session_dir, initial_steer.strip()
        ):
            reporter.error("could not write the initial steer request")
            return 2
        app_frontend.settle_away_mode(
            layout.session_dir, cfg, can_ask=frontend.capabilities.can_ask
        )
        # The branch is advanced by the first chain commit; nothing is cut or checked out.
        if cfg.git.branch_per_run and mode == "run" and cfg.git.control != "model":
            run_branch = git_read.run_branch_for(effective_session_id)

        # A run that would have to ask about tracked changes but cannot refuses here.
        modified = git_read.modified_paths(cwd) if mode == "run" else []
        must_ask = bool(modified) and cfg.git.dirty_tree == "ask"
        answerable = frontend.capabilities.can_ask or ipc.away_mode(layout.session_dir) == "wait"
        unmerged_run = (
            preflight.unmerged_run_holding_the_tree(
                cwd, state, except_id=effective_session_id, modified=modified
            )
            if modified and cfg.git.dirty_tree != "stash"
            else ""
        )
        if must_ask and not answerable:
            reporter.refuse(preflight.dirty_tree_refusal(modified, unmerged_run=unmerged_run))
            discard_husk_dir(layout.session_dir)
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

        # Written before the gates below, which park rather than refuse; the rewrite keeps
        # the park, which the execution's start clears.
        parked = stamps.parked_stamp(layout.session_dir)
        # A lane's config carries the gate its coordinator resolved, origin and all.
        lane_gate_origin = stamps.lane_verify_origin()
        stamps.write_session_manifest(
            layout,
            session_id=effective_session_id,
            source_session_id=source_session_id,
            user_task=task,
            base_sha=base_sha,
            base_branch=base_branch,
            run_branch=run_branch,
            cfg=cfg,
            mode=mode,
            gate=(cfg.harness.verify_command, lane_gate_origin) if lane_gate_origin else None,
            effective_preset=(preset_stamp[0] if preset_stamp else (preset or cfg.preset)),
            preset_from_flag=(preset_stamp[1] if preset_stamp else bool(preset)),
            driver_from_flag=bool(model),
            isolation=isolation,
            worktree=checkout.worktree if checkout is not None else None,
            worktree_git_dir=worktree_git_dir,
            budget=budget_overrides.stamp() if budget_overrides is not None else None,
        )
        if parked is not None:
            stamps.stamp_parked(layout.session_dir, task=parked[0], reason=parked[1])

        def _park(reason: str, detail: str, *, hint: str = "") -> int:
            # `reason` is the short cause listings show; `detail` is the sentence read now.
            stamps.stamp_parked(layout.session_dir, task=task, reason=reason)
            reporter.err(
                f"PARKED: {detail}. Your task is saved as run {effective_session_id!r}:\n"
                f"    agent6 resume {effective_session_id}    (starts it)"
                + (f"\n{hint}" if hint else "")
            )
            return 2

        # A live worker from here, so the start questions below reach `agent6 answer`.
        process.write_worker_pid(layout.session_dir, os.getpid())

        if mode == "run":
            # One run-mode worker per checkout, taken before any tree mutation.
            repo_lock_fd = lock.acquire_repo_writer(layout.state_dir, cwd, effective_session_id)
            if repo_lock_fd is None:
                holder = lock.repo_writer_holder(layout.state_dir, cwd) or "another run"
                return _park(
                    "checkout busy",
                    f"run {holder!r} is already driving this checkout, and a second run-mode"
                    " worker would interleave auto-commits on the one working tree",
                    hint=(
                        "or run it now in its own worktree, beside the live run:\n"
                        '    agent6 run --worktree "<the same task>"\n'
                        f"or hand it to the live run as an isolated lane by steering"
                        f" {holder!r} with:\n    /parallel 1 -- <the same task>"
                    ),
                )
            # Settle the operator's uncommitted changes before the first commit can sweep them up.
            if modified:
                choice: preflight.DirtyTreeChoice
                if cfg.git.dirty_tree == "stash":
                    choice = "stash"
                elif cfg.git.dirty_tree == "include":
                    choice = "include"
                else:
                    question = preflight.dirty_tree_question(modified, unmerged_run=unmerged_run)
                    answer = prompts.ask((question,))
                    choice = preflight.dirty_tree_choice(answer.answers[0])
                    stash_pop = stash_pop or choice == "stash"
                if choice == "cancel":
                    settle = (
                        f"`agent6 sessions merge {unmerged_run}` lands them (the unmerged work"
                        f" of run {unmerged_run})"
                        if unmerged_run
                        else "commit or stash them"
                    )
                    return _park(
                        "uncommitted changes",
                        f"the working tree has uncommitted changes to tracked files; {settle},"
                        " then start the run",
                    )
                if choice == "stash":
                    try:
                        git_ops.stash_tracked_changes(
                            cwd, git_ops.auto_stash_message(effective_session_id)
                        )
                        stashed = True
                    except errors.GitError as exc:
                        return _park(
                            "stash failed", f"stashing the working tree's changes failed: {exc}"
                        )
            # The operator's files, untracked after any stash; every commit leaves them out.
            untracked_at_start = git_read.untracked_paths(cwd)
            sessions_layout.write_untracked_at_start(layout.session_dir, untracked_at_start)
            if unmerged_run and not stashed:
                # Included: when the tree is exactly that run's tip, this run's merge lands it.
                included = preflight.included_run(
                    cwd,
                    unmerged_run,
                    base_sha=base_sha,
                    exclude=untracked_at_start
                    | git_read.hidden_in_repo(cwd, cfg.sandbox.hide_paths),
                )
                if included is not None:
                    stamps.stamp_includes(layout.session_dir, included)

        def _gate(cfg: Config, budget: agent6_budget.BudgetTracker) -> Config:
            # Infer an unset gate, then drop it last when commands are withheld.
            configured_gate = bool(cfg.harness.verify_command)
            # In the repository, never the run's worktree: a signal such as `.venv` is absent
            # there until the setup command builds it, and the gate is carried.
            cfg = preflight.infer_verify_if_unset(
                cfg,
                repo,
                mode=mode,
                events=events,
                transcript_sink=transcript_sink,
                budget=budget,
                reporter=reporter,
            )
            gate_before = cfg.harness.verify_command
            cfg = preflight.drop_gate_if_unrunnable(
                cfg, session_dir=layout.session_dir, reporter=reporter
            )
            # The origin follows the resolved gate, never the configured one.
            pinned_gate, gate_origin = cfg.harness.verify_command, ""
            if cfg.harness.verify_command:
                gate_origin = lane_gate_origin or ("configured" if configured_gate else "inferred")
            elif gate_before:
                # Withheld with commands: the execution is gateless, the manifest names the gate.
                pinned_gate, gate_origin = gate_before, "withheld"
            # From here the run is judged by this gate, whatever its source says later.
            stamps.pin_gate(
                layout.session_dir,
                pinned_gate,
                gate_origin,
                events=events,
                reporter=reporter,
            )
            return cfg

        if parked is not None:
            why = f" ({parked[1]})" if parked[1] else ""
            reporter.note(
                f"run {effective_session_id!r} was parked at submission{why}; starting it now."
            )
        end = _execution.run_execution(
            cfg,
            layout,
            _execution.ExecutionInputs(
                session_id=effective_session_id,
                mode=mode,
                role=role,
                explicit_leaves=explicit_leaves,
                isolation=isolation,
                tui_enabled=tui_enabled,
                interactive=interactive,
                task=task,
                gate=_gate,
                chain_branch=run_branch,
                base_sha=base_sha,
                untracked_at_start=untracked_at_start,
                # Written for every mode: `agent6 resume` reaches an ask too.
                resume_state_path=layout.session_dir / "loop_state.json",
                undo_forker=_undo_forker,
                prompts=prompts,
                # The REPL prints and saves each turn itself.
                ask_transcript_task=None if interactive else task,
                budget_overrides=budget_overrides,
                sandbox_overrides=sandbox_overrides,
                standing_goal=standing_goal,
                pins=tuple(pins),
                worktree_git_dir=worktree_git_dir,
                worker_lock_fd=worker_lock_fd,
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
        # The one owner of worker.pid, both locks and the stash, on every exit path; the
        # releases come last and survive a teardown raise (ACP runs this in-process).
        try:
            try:
                frontend.close_console_view()  # stops the heartbeat, clears the spinner
            finally:
                if not detach_requested:
                    # A detach keeps the pid until the background `resume` claims it.
                    process.clear_worker_pid(layout.session_dir)
                if stashed:
                    finalize.end_auto_stash(
                        cwd,
                        detached=detach_requested,
                        base_branch=base_branch,
                        run_branch=run_branch,
                        auto_pop=stash_pop,
                        session_id=layout.session_id,
                        in_worktree=checkout is not None,
                        exclude=untracked_at_start,
                        reporter=reporter,
                    )
        finally:
            lock.release_single_writer(repo_lock_fd)
            lock.release_single_writer(worker_lock_fd)
            _drop_unrecorded_worktree()
        if detach_requested:
            # The worker lock is released, so the detached `resume` can acquire it.
            _execution.detach_to_background(
                frontend=frontend,
                cfg=cfg,
                layout=layout,
                cwd=repo,
                flags=_setup.override_flags(
                    budget_overrides, sandbox_overrides, _setup.flag_route(cfg, mode, model)
                ),
                reporter=reporter,
            )
