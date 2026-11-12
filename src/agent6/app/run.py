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
from collections.abc import Callable, Mapping, Sequence

from agent6 import budget as agent6_budget
from agent6 import directive, errors, kinds, paths
from agent6.app import (
    execution,
    finalize,
    fork_worktrees,
    preflight,
    stamps,
    workdirs,
    workspace_setup,
)
from agent6.app import frontend as app_frontend
from agent6.app import reporter as app_reporter
from agent6.app.review import directive as review_directive
from agent6.app.session import prepare as session_prepare
from agent6.app.session import setup as session_setup
from agent6.config import Config
from agent6.sessions import id, ipc, lock, process
from agent6.sessions import layout as sessions_layout
from agent6.sessions import manifest as sessions_manifest
from agent6.vcs.git import ops as git_ops
from agent6.vcs.git import read as git_read
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
    cwd: pathlib.Path | None = None,
    started_at: float,
    operator_task: str | None = None,
    session_id: str = "",
    source_session_id: str | None = None,
    interactive: bool = False,
    tui: bool = False,
    mode: kinds.LoopMode = "run",
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
    config_sources: Mapping[str, str] | None = None,
    reporter: app_reporter.Reporter = app_reporter.STDIO_REPORTER,
    contract: kinds.ConversationContract | None = None,
) -> int:
    """Run one session from preflight to finalize; the sole `agent6 run` path.

    The front-end has built the config, resolved the task, checked the git-repo wall
    and the runnable roles, and routed `--parallel` away. Plan mode drives the same
    harness with the planning prompt, no edit tools, `finish_planning` and no
    auto-commit; its markdown lands at `<session-dir>/plan.md`.

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
        standing_goal: The standing goal, when set.
        budget_overrides: The budget flags the lifecycle re-reads.
        sandbox_overrides: The sandbox flags the lifecycle re-reads.
        preset: The `--preset` flag's value.
        initial_steer: An operator follow-up queued for the loop's first boundary,
            seeded after the stale-state clear so it survives it.
        pins: The pinned paths.
        preset_stamp: `(name, from_flag)` to stamp in place of deriving from `agent6.preset`,
            so a parked resume replays the original submission's precedence.
        model: The `--model` value that set the driving role.
        explicit_leaves: The config leaves the operator wrote, as dotted paths; a
            default this host cannot honour degrades, a written value refuses.
        worktree: Whether the run works in its own linked worktree, added at HEAD's
            commit under `[git.worktree].base_dir`; the checkout is then never touched.
        config_paths: The `--config` files, which an `/undo` fork is stamped under.
        config_sources: The layer that supplied each config leaf.
        reporter: Receives every refusal and note.
        contract: The bounded conversation a `read_only` session drives; None for every
            other mode.

    Returns:
        The process exit code.

    Raises:
        ValueError: `read_only` without a contract, or a contract under another mode.
    """
    if (mode == "read_only") != (contract is not None):
        raise ValueError("mode 'read_only' is the one mode that drives a conversation contract")
    if operator_task is None:
        operator_task = task
    role = kinds.session_kind(mode).role
    repo = cwd if cwd is not None else pathlib.Path.cwd()
    # The run's checkout: the repository, or the worktree `worktree` adds below.
    cwd = repo
    if (cwd / ".git").is_file() and not preflight.require_git_repo(cwd, reporter=reporter):
        return 2
    # A startup grant is recorded before model commands can rewrite the pointer under partial.
    worktree_git_dir = paths.linked_worktree_git_dir(cwd)
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
    cfg = session_setup.session_config(cfg, mode, sandbox_overrides)
    review_world: review_directive.ReviewWorld | None = None
    if (review_target := directive.parse_review(task)) is not None:
        try:
            review_world = review_directive.build(repo, review_target, hide=cfg.sandbox.hide_paths)
        except review_directive.ReviewDirectiveError as exc:
            reporter.refuse(str(exc))
            return 2
        if review_world.empty:
            reporter.out(review_world.message)
            return 0
        if worktree and not review_target:
            reporter.refuse("/review needs this working tree; drop --worktree or name a ref")
            return 2
        worktree = worktree or review_world.needs_worktree
        task = review_world.message
    if worktree and (top := paths.checkout_root(repo)) != repo.resolve():
        reporter.refuse(
            f"--worktree starts at the repository's top level, and this is a subdirectory of"
            f" {top}. Run it from there, or run without --worktree."
        )
        return 2
    # Refuse an unanswerable run before anything is created, so its id stays free. A fresh run
    # has no dir yet, so the env is the only away answer.
    try:
        tui_enabled = session_prepare.headless_check(
            cfg,
            frontend,
            tui=tui,
            interactive=interactive,
            mode=mode,
            away=os.environ.get("AGENT6_DETACHED_AWAY", ""),
            reporter=reporter,
        )
    except preflight.SessionRefusedError as refusal:
        return refusal.rc
    # Before isolation: its budget preflight prices the model from the refreshed listing.
    if not preflight.route_preflight(
        cfg, role, reporter=reporter, model_flag=session_setup.route_text(model)
    ):
        return 2
    # A worktree run selects below, once the worktree its commands run in exists.
    isolation: kinds.IsolationLevel | None = None
    if not worktree:
        try:
            isolation = session_prepare.hold_workspace_and_select_isolation(
                cfg, cwd, frontend=frontend, reporter=reporter, worktree_git_dir=worktree_git_dir
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
    if review_world is not None:
        base_sha = review_world.target.head_commit or review_world.target.base_commit or ""
        base_branch = review_world.head_branch
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
            cfg.harness.verify.command
            if review_world is not None
            else workspace_setup.origin_gate(cfg, repo),
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
        try:
            path = workdirs.place(cfg, repo, effective_session_id)
            checkout = fork_worktrees.add_session_worktree(
                repo, path, base_sha, cfg.git.worktree.kind
            )
        except errors.OperatorError as exc:
            reporter.refuse(str(exc))
            return 2
        except (errors.GitError, OSError) as exc:
            reporter.error(f"could not add the run's worktree: {exc}")
            return 1
        cwd = checkout.worktree
        cfg = cfg.with_hidden_in_checkout(repo, cwd)
        worktree_git_dir = checkout.git_dir
    worker_lock_fd: int | None = None
    try:
        if checkout is not None:
            isolation = session_prepare.hold_workspace_and_select_isolation(
                cfg, cwd, frontend=frontend, reporter=reporter, worktree_git_dir=worktree_git_dir
            )
            if cfg.git.control == "model" and isolation != "none" and worktree_git_dir is not None:
                # The repository git dir lies outside the worktree: read-only to jailed commands.
                reporter.refuse(
                    "--worktree runs in a linked worktree whose .git is read-only to commands"
                    f' under sandbox.isolation = {isolation!r}; under [git].control = "model"'
                    ' the model could not commit there. Set [git].control = "agent6"'
                    " (`agent6 config set --repo git.control agent6`), set git.worktree.kind ="
                    ' "clone", or run without --worktree.'
                )
                return 2
        assert isolation is not None  # selected before the git preflight, or on the worktree
        # One writer per session dir, taken before any shared state is touched.
        worker_lock_fd = session_prepare.take_single_writer(layout, reporter=reporter)
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
        hidden = git_read.hidden_in_repo(cwd, cfg.sandbox.hide_paths)
        # A hidden path's change is the operator's business: never asked about, never stashed.
        modified = (
            [p for p in git_read.modified_paths(cwd) if not git_read.excluded(p, hidden)]
            if mode == "run"
            else []
        )
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

        channels = session_prepare.operator_channels(layout, frontend)
        events, prompts = channels.events, channels.prompts
        transcript_sink = channels.transcript_sink
        session_prepare.workspace_notices(cwd, cfg, reporter=reporter)

        # Written before the gates below, which park rather than refuse; the rewrite keeps
        # the park, which the execution's start clears.
        parked = stamps.parked_stamp(layout.session_dir)
        # A lane's config carries the gate its coordinator resolved, origin and all.
        lane_gate_source = stamps.lane_gate_source()
        stamps.write_session_manifest(
            layout,
            session_id=effective_session_id,
            source_session_id=source_session_id,
            user_task=task,
            operator_task=operator_task,
            base_sha=base_sha,
            base_branch=base_branch,
            run_branch=run_branch,
            cfg=cfg,
            config_sources=config_sources,
            mode=mode,
            gate=(
                sessions_manifest.GateStamp(
                    command=cfg.harness.verify.command,
                    source=lane_gate_source,
                    availability="pending",
                )
                if lane_gate_source
                else None
            ),
            effective_preset=(preset_stamp[0] if preset_stamp else (preset or cfg.agent6.preset)),
            preset_from_flag=(preset_stamp[1] if preset_stamp else bool(preset)),
            driver_from_flag=bool(model),
            isolation=isolation,
            worktree=checkout.worktree if checkout is not None else None,
            worktree_git_dir=worktree_git_dir,
            review_base_tree=review_world.target.base_tree if review_world is not None else "",
            budget=budget_overrides.stamp() if budget_overrides is not None else None,
        )
        if parked is not None:
            stamps.stamp_parked(
                layout.session_dir, task=parked[0], reason=parked[1], operator_task=parked[2]
            )

        def _park(reason: str, detail: str, *, hint: str = "") -> int:
            # `reason` is the short cause listings show; `detail` is the sentence read now.
            stamps.stamp_parked(
                layout.session_dir,
                task=operator_task if review_world is not None else task,
                reason=reason,
                operator_task=operator_task,
            )
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
                    if review_world is not None and review_world.target.source == "worktree":
                        return _park(
                            "review needs uncommitted changes",
                            "a bare /review cannot stash the changes it reviews; use agent6 ask "
                            '"/review", or set git.dirty_tree to include and retry',
                        )
                    try:
                        git_ops.stash_tracked_changes(
                            cwd,
                            git_ops.auto_stash_message(effective_session_id),
                            paths=modified if hidden else (),
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
                    exclude=untracked_at_start | hidden,
                )
                if included is not None:
                    current = sessions_manifest.read_manifest(layout.session_dir)
                    stamps.write_manifest(
                        layout.manifest_path, current.model_copy(update={"includes": included})
                    )

        def _no_gate(
            cfg: Config, budget: agent6_budget.BudgetTracker, available: Callable[[str], bool]
        ) -> Config:
            # A bounded conversation never verifies: no inference, no gate rows in its journal.
            del budget, available
            return cfg

        def _gate(
            cfg: Config, budget: agent6_budget.BudgetTracker, available: Callable[[str], bool]
        ) -> Config:
            # Infer an unset gate, then drop it last when commands are withheld.
            configured_gate = bool(cfg.harness.verify.command)
            # In the repository, never the run's worktree: a signal such as `.venv` is absent
            # there until the setup command builds it, and the gate is carried.
            cfg = preflight.infer_verify_if_unset(
                cfg,
                cwd if review_world is not None else repo,
                mode=mode,
                available=available,
                events=events,
                transcript_sink=transcript_sink,
                budget=budget,
                reporter=reporter,
            )
            source = lane_gate_source or ("configured" if configured_gate else "inferred")
            cfg, _withheld = session_prepare.settle_gate(
                cfg, layout, source=source, events=events, reporter=reporter
            )
            return cfg

        if parked is not None:
            why = f" ({parked[1]})" if parked[1] else ""
            reporter.note(
                f"run {effective_session_id!r} was parked at submission{why}; starting it now."
            )
        end = execution.run_execution(
            cfg,
            layout,
            execution.ExecutionInputs(
                session_id=effective_session_id,
                mode=mode,
                role=role,
                explicit_leaves=explicit_leaves,
                isolation=isolation,
                tui_enabled=tui_enabled,
                interactive=interactive,
                task=task,
                operator_task=operator_task,
                gate=_gate if contract is None else _no_gate,
                contract=contract,
                chain_branch=run_branch,
                base_sha=base_sha,
                untracked_at_start=untracked_at_start,
                # Written for every mode: `agent6 resume` reaches an ask too.
                resume_state_path=layout.session_dir / "loop_state.json",
                undo_forker=session_prepare.undo_forker(
                    config_paths, effective_session_id, cwd=repo, reporter=reporter
                ),
                prompts=prompts,
                # The REPL prints and saves each turn itself.
                ask_transcript_task=None if interactive else task,
                budget_overrides=budget_overrides,
                sandbox_overrides=sandbox_overrides,
                standing_goal=standing_goal,
                pins=tuple(pins),
                worktree_git_dir=worktree_git_dir,
                mirror_into=fork_worktrees.mirror_origin(repo, cwd),
                worker_lock_fd=worker_lock_fd,
                base_rules=review_world.rules if review_world is not None else None,
                review_target=(
                    (review_world.target, review_world.diff) if review_world is not None else None
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
            execution.detach_to_background(
                frontend=frontend,
                cfg=cfg,
                layout=layout,
                cwd=repo,
                flags=session_setup.override_flags(
                    budget_overrides, sandbox_overrides, session_setup.flag_route(cfg, mode, model)
                ),
                reporter=reporter,
            )
