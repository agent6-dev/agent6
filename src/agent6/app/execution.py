# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Drive the one execution body a fresh run and a resumed execution share.

From the provider session to the end block: providers, the gate step, steer
state, the session network and MCP servers, the tool set, the Harness, its teardown, the
auto-merge and the end report. `run.py` and `resume.py` keep what differs before and after it
and hand the body its `ExecutionInputs`, so a knob wired into one lifecycle cannot be missing
from the other.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import pathlib
import sys
import uuid
from collections.abc import Callable, Sequence
from typing import Any

from agent6 import budget as agent6_budget
from agent6 import errors, kinds, paths, review_records
from agent6.app import (
    child_watch,
    conversation_state,
    finalize,
    lane_launch,
    notify_hook,
    review_actions,
    review_directive,
    role_providers,
    stamps,
    workspace_setup,
)
from agent6.app import frontend as app_frontend
from agent6.app import lanes as app_lanes
from agent6.app import reporter as app_reporter
from agent6.app.session import runtime as session_runtime
from agent6.app.session import setup as session_setup
from agent6.config import Config, RoleName
from agent6.harness import loop, loop_snapshot, operator_bridge, provider_call, run_chain
from agent6.harness.compaction import window as compaction_window
from agent6.harness.review import gate as review_gate
from agent6.providers import TranscriptSink
from agent6.sandbox import jail
from agent6.sessions import event_log, ipc, journal, process
from agent6.sessions import layout as sessions_layout
from agent6.sessions import manifest as sessions_manifest
from agent6.tools import command_tools, dispatch, operator_prompts, target_policy
from agent6.vcs import commit_message, git_ops, git_read
from agent6.viewmodel import policy, transcript_text


@dataclasses.dataclass(frozen=True, slots=True)
class ExecutionInputs:
    """Hold what a fresh execution and a resumed one hand the body differently.

    Attributes:
        session_id: The session's id.
        mode: The session mode.
        role: The role driving it.
        explicit_leaves: The dotted leaves the run's config layers set, inherited by lanes.
        isolation: The resolved isolation level.
        tui_enabled: Whether the TUI owns the terminal.
        interactive: Whether an operator is at the terminal.
        task: The task a fresh execution runs; None resumes the snapshot.
        operator_task: The words before composition; None means `task` is uncomposed.
        gate: The gate step: a fresh execution infers one, a resumed one reuses the snapshot's;
            both drop an unrunnable gate and pin the result. Runs once the budget exists.
        chain_branch: The chain the execution's commits advance.
        base_sha: The base the review panel and an unborn chain start from; "" without a head.
        untracked_at_start: The operator's untracked files, left out of every commit.
        resume_state_path: Where the snapshot is written.
        undo_forker: Forks back before the last message and rewinds the checkout.
        prompts: The one gate to the operator, so a question asked before the loop and the
            dispatcher's approvals share one journal and one id sequence.
        ask_transcript_task: The question a one-shot ask records its answer under; None when
            the REPL saved each turn.
        budget_overrides: The `--max-*` flags a `/parallel` lane inherits.
        sandbox_overrides: The `--auto-approve` a `/parallel` lane inherits.
        standing_goal: The standing goal.
        pins: The pinned notes.
        resuming: Whether this is a resumed execution.
        resume_steer: The steer that started this resumed execution.
        merged_in: A branch merged into the run's chain before this execution, for the
            model's first turn.
        operator_edits: The paths the operator changed between executions and left in the
            tree for the run; the model hears their names at its first turn.
        worktree_git_dir: The recorded repository git dir of a session in a linked worktree,
            granted to its jail beyond the workspace.
        mirror_into: The repository a session in a clone copies its commits into; None in
            the repository itself or a linked worktree of it.
        worker_lock_fd: The lifecycle's held session writer lock, shared with auto-merge.
        base_rules: A review session's base-tree AGENTS.md, in place of the checkout's.
        review_target: A review session's comparison and its diff, for the panel.
        contract: The bounded conversation a `read_only` session drives; None for a worker's.
    """

    session_id: str
    mode: kinds.LoopMode
    role: RoleName
    explicit_leaves: frozenset[str]
    isolation: kinds.IsolationLevel
    tui_enabled: bool
    interactive: bool
    task: str | None
    gate: Callable[[Config, agent6_budget.BudgetTracker, Callable[[str], bool]], Config]
    chain_branch: str | None
    base_sha: str
    untracked_at_start: frozenset[str]
    resume_state_path: pathlib.Path
    undo_forker: Callable[[], tuple[str, str] | None]
    prompts: operator_prompts.OperatorPrompts
    ask_transcript_task: str | None
    operator_task: str | None = None
    budget_overrides: session_setup.BudgetOverrides | None = None
    sandbox_overrides: session_setup.SandboxOverrides | None = None
    standing_goal: str = ""
    pins: tuple[str, ...] = ()
    resuming: bool = False
    resume_steer: str = ""
    operator_edits: tuple[str, ...] = ()
    merged_in: git_ops.ChainMerge | None = None
    worktree_git_dir: pathlib.Path | None = None
    mirror_into: pathlib.Path | None = None
    worker_lock_fd: int | None = None
    base_rules: str | None = None
    review_target: tuple[review_records.ReviewTarget, str] | None = None
    contract: kinds.ConversationContract | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class ExecutionEnd:
    """Record how the execution ended.

    Attributes:
        rc: The process exit code.
        detach_requested: The operator detached; the caller releases its locks and spawns the
            continuation.
    """

    rc: int
    detach_requested: bool = False


def headless_marker(event: dict[str, Any], *, log: Callable[[str], None]) -> None:
    """Print the degraded-jail marker a live view renders, through the run's text log.

    Args:
        event: One journal event.
        log: The run's text logger.
    """
    if event.get("type") == "jail.degraded":
        log(f"[agent6] {transcript_text.jail_degraded_body(event)}")


def detach_to_background(
    *,
    frontend: app_frontend.SessionFrontend,
    cfg: Config,
    layout: sessions_layout.SessionLayout,
    cwd: pathlib.Path,
    flags: Sequence[str],
    reporter: app_reporter.Reporter,
) -> None:
    """Hand a detached execution to a background `resume` under this invocation's flags.

    Called once the caller has released the run's locks. Asks how approvals are answered while
    nothing watches, spawns, then prints the reattach line only for a spawn that happened.

    Args:
        frontend: The injected front-end.
        cfg: The resolved config.
        layout: The session's directory layout.
        cwd: The workspace.
        flags: This invocation's overrides as CLI options.
        reporter: Where the reattach line goes.
    """
    if cfg.sandbox.run_commands == "ask" and not ipc.session_allow_set(
        layout.session_dir, ipc.COMMAND_SCOPE
    ):
        frontend.prompt_detach_away_mode(layout.session_dir, app_frontend.approval_scopes(cfg))
    err = frontend.spawn_detached_resume(cwd, layout.session_id, flags)
    if err:
        # The handoff failed, so the pid this process kept through the spawn goes now.
        process.clear_worker_pid(layout.session_dir)
        reporter.note(err)
        return
    reporter.out(f"\n[agent6] detached: {layout.session_id} continues in the background.")
    reporter.out(f"          reattach:  agent6 attach {layout.session_id}")


def _escape_reason(exc: BaseException) -> loop_snapshot.SessionEndReason:
    """Return the end reason an escaping exception stands for.

    Args:
        exc: The escaping exception.

    Returns:
        "interrupted" for Ctrl-C or a closed output pipe (`agent6 run ... | head`), the
        operator's side going away; else "crashed".
    """
    return "interrupted" if isinstance(exc, (KeyboardInterrupt, BrokenPipeError)) else "crashed"


def _journal_escape(
    events: event_log.EventSink,
    exc: BaseException,
    *,
    execution_id: str,
    turns: int,
    tree: str = "",
) -> loop_snapshot.SessionEndReason:
    """Journal the end an escape leaves.

    Without one every surface reads the dead run as running until the silence window expires;
    a dead journal must not mask the exit code.

    Args:
        events: The run's event sink.
        exc: The escaping exception.
        execution_id: The execution that stopped.
        turns: The turns reached.
        tree: The tree the run leaves, so the next resume tells its files from the operator's.

    Returns:
        The reason journaled.
    """
    reason = _escape_reason(exc)
    left: dict[str, Any] = {"tree": tree} if tree else {}
    with contextlib.suppress(event_log.EventWriteError):
        events.emit(
            journal.SessionStop(
                execution_id=execution_id,
                reason=reason,
                turns=turns,
                verification=None,
                **left,
            )
        )
    return reason


def _tree_left_before_the_loop(
    cwd: pathlib.Path, inputs: ExecutionInputs, hidden: frozenset[str]
) -> str:
    """Return the tree a run leaves when it escapes before its harness exists; "" outside run mode.

    Seeded on the chain tip like a chain commit, as the harness's own read is; the escape's
    end then tells the next resume the run's files from the operator's.

    Args:
        cwd: The workspace.
        inputs: The execution's inputs.
        hidden: The run's hidden paths inside the repository, left out like its untracked files.

    Returns:
        The tree's sha; "" outside run mode or when the read fails or is interrupted.
    """
    if inputs.mode != "run":
        return ""
    try:
        chain_tip = git_read.chain_tip(cwd, git_read.chain_ref_for(inputs.session_id))
        return git_read.worktree_tree(
            cwd, chain_tip or inputs.base_sha or None, inputs.untracked_at_start | hidden
        )
    except (errors.GitError, OSError, KeyboardInterrupt):
        return ""


def _tree_left(wf: loop.Harness, mode: str) -> str:
    """Return the tree a run leaves, for the end an escape journals; "" outside run mode.

    The read runs git inside the escape handler, where another Ctrl-C can land: the end is
    then written without the tree rather than not at all.

    Args:
        wf: The run's harness.
        mode: The session mode.

    Returns:
        The tree's sha; "" outside run mode or when the read fails or is interrupted.
    """
    if mode != "run":
        return ""
    try:
        return wf.chain.tree_sha()
    except KeyboardInterrupt:
        return ""


def run_execution(  # noqa: C901, PLR0911, PLR0912, PLR0915  # setup, run and teardown in reading order
    cfg: Config,
    layout: sessions_layout.SessionLayout,
    inputs: ExecutionInputs,
    *,
    frontend: app_frontend.SessionFrontend,
    reporter: app_reporter.Reporter,
    events: event_log.EventSink,
    transcript_sink: TranscriptSink,
    cwd: pathlib.Path,
    state_dir: pathlib.Path,
) -> ExecutionEnd:
    """Drive one execution to its end block.

    Args:
        cfg: The resolved config.
        layout: The session's directory layout.
        inputs: What the lifecycle hands the body.
        frontend: The injected front-end.
        reporter: The output channels.
        events: The run's event sink.
        transcript_sink: Where every provider call is transcribed.
        cwd: The workspace.
        state_dir: The repo's state directory.

    Returns:
        The exit code, and whether the operator detached.

    Raises:
        KeyboardInterrupt: An interrupt before the loop, journaled and re-raised.
        Exception: A crash before or after the loop, journaled and re-raised.
    """
    execution_id = uuid.uuid4().hex
    session_runtime.note_state_disk_space(state_dir, reporter=reporter)
    mode, role = inputs.mode, inputs.role
    label = "resume" if inputs.resuming else "run"
    session: session_runtime.SessionProviders | None = None
    access: command_tools.JailAccess | None = None
    session_net: jail.SessionNetwork | None = None
    hidden = git_read.hidden_in_repo(cwd, cfg.sandbox.hide_paths)
    try:
        stream_text, console_stream = frontend.stream_modes(inputs.tui_enabled)
        if console_stream:
            frontend.attach_console_view(events)
        conversations = conversation_state.Conversations(
            path=layout.session_dir / conversation_state.FILE_NAME,
            is_subagent=stamps.parallel_lineage() is not None,
        )
        session = session_runtime.build_session_providers(
            cfg,
            role=role,
            events=events,
            transcript_sink=transcript_sink,
            stream_text=stream_text,
            reporter=reporter,
            conversations=conversations,
        )
        budget = session.budget
        if session_setup.wants_session_network(cfg, inputs.isolation):
            session_net = jail.SessionNetwork.open()
            process.write_session_netns_pid(layout.session_dir, session_net.holder_pid)
        if cfg.sandbox.isolation == "container":
            mounted = session_runtime.mounted_root(cfg, cwd, inputs.worktree_git_dir)
            access = session_runtime.prepare_container_execution(
                cfg,
                mounted,
                session_runtime.container_options(cfg),
                # The lease lives with the mounted workspace, where its runs meet on it.
                state_dir=state_dir if mounted == cwd.resolve() else paths.state_dir(mounted),
                session_id=layout.session_id,
                execution_id=execution_id,
                checkout=cwd,
                worktree_git_dir=inputs.worktree_git_dir,
                controls=target_policy.container_controls(cfg),
                reporter=reporter,
                emit=events.emit,
            )
        else:
            access = session_runtime.prepare_local_execution(
                cfg,
                cwd,
                inputs.isolation,
                name=layout.session_id,
                session_net=session_net,
                worktree_git_dir=inputs.worktree_git_dir,
            )
        cfg = inputs.gate(cfg, budget, access.session.executable_available)

        steer_state = frontend.make_steer_state(
            events,
            layout.session_dir,
            session_runtime.session_facts_provider(
                budget,
                session.rm_role.model,
                cfg.sandbox.run_commands,
                inputs.isolation,
                target_policy.resolve_network(cfg, inputs.isolation),
            ),
        )
    except (KeyboardInterrupt, Exception) as exc:
        # A parked run whose start crashes here never ran: it stays parked with no session.stop.
        if stamps.parked_stamp(layout.session_dir) is not None:
            reporter.err(f"\n[agent6] {label} {_escape_reason(exc)}")
        else:
            tree = _tree_left_before_the_loop(cwd, inputs, hidden)
            reason = _journal_escape(events, exc, execution_id=execution_id, turns=0, tree=tree)
            reporter.err(f"\n[agent6] {label} {reason}")
        with contextlib.ExitStack() as cleanup:
            if session_net is not None:
                cleanup.callback(process.clear_session_netns_pid, layout.session_dir)
                cleanup.callback(session_net.close)
            if access is not None:
                cleanup.callback(access.close)
            if session is not None:
                cleanup.callback(session.close)
        raise

    interrupted = False
    result: loop_snapshot.SessionResult | None = None
    wf: loop.Harness | None = None
    escape_handled = False
    undo_outcome: list[tuple[str, str]] = []
    dispatcher: dispatch.ToolDispatcher | None = None
    mcp_manager = None
    try:
        reporter.note(f"{'resume ' if inputs.resuming else ''}session id: {inputs.session_id}")
        if not console_stream and not inputs.tui_enabled:
            for line in policy.session_policy(layout.session_dir).models:
                reporter.note(line)

        # A bounded conversation starts no server: its tool surface is the contract's.
        if inputs.contract is None:
            mcp_manager = session_setup.start_mcp_manager_if_enabled(
                cfg,
                cwd,
                inputs.isolation,
                reporter=reporter,
                events=events,
                session_net=session_net,
            )

        loop_log = frontend.loop_logger(mode)
        if not console_stream:
            # Headless, the fold's markers reach no view: a degraded jail is said once here.
            events.subscribe(functools.partial(headless_marker, log=loop_log))
        # A bounded conversation is its parent's evidence: the operator hears of the parent.
        if inputs.contract is None:
            events.subscribe(
                notify_hook.attention(
                    cfg.notify,
                    session_id=layout.session_id,
                    session_dir=layout.session_dir,
                    note=reporter.note,
                )
            )
        tools = session_runtime.build_session_tools(
            cfg,
            cwd=cwd,
            state_dir=state_dir,
            layout=layout,
            isolation=inputs.isolation,
            mode=mode,
            events=events,
            prompts=inputs.prompts,
            loop_log=loop_log,
            mcp_manager=mcp_manager,
            session_net=session_net,
            command_access=access,
            execution_id=execution_id,
            rm_role=session.rm_role,
            worktree_git_dir=inputs.worktree_git_dir,
        )
        curator = tools.curator
        dispatcher = tools.dispatcher
        cfg = tools.cfg
        if not console_stream and not inputs.tui_enabled:
            for line in policy.session_policy(layout.session_dir).environment:
                reporter.note(line)

        # Before the chain exists: what the setup command leaves untracked is the workspace's
        # environment, so every commit leaves it out as it does the operator's files.
        unready: str | None = None
        untracked_at_start = inputs.untracked_at_start
        if inputs.contract is None and workspace_setup.session_needs_setup(layout.session_dir):
            # The run's own earlier files read untracked too: the chain never moves the index.
            before = git_read.untracked_paths(cwd) if mode == "run" else frozenset[str]()
            prepared = workspace_setup.prepare(cfg, cwd, runner=dispatcher, reporter=reporter)
            unready = prepared.problem
            if prepared.ran:
                try:
                    current = sessions_manifest.read_manifest(layout.session_dir)
                    stamps.write_manifest(
                        layout.manifest_path, current.model_copy(update={"workspace_set_up": True})
                    )
                except (sessions_manifest.ManifestError, OSError) as exc:
                    reporter.note(f"could not record that this workspace is set up: {exc}")
                if mode == "run":
                    untracked_at_start |= git_read.untracked_paths(cwd) - before
                    sessions_layout.write_untracked_at_start(layout.session_dir, untracked_at_start)

        def _undo_forker() -> tuple[str, str] | None:
            """Return the `/undo` fork's outcome, kept for the end block.

            Returns:
                The fork's id and the text taken back, or None with the reason printed.
            """
            got = inputs.undo_forker()
            if got is not None:
                undo_outcome.append(got)
            return got

        after_auto_commit: Callable[[int, str], kinds.AutoCommitDirective] = (
            frontend.build_repl_hook(cwd, budget, inputs.session_id, mcp_manager)
            if inputs.interactive and mode == "run"
            else (lambda _i, _s: "continue")
        )

        def _steer_requested() -> bool:
            assert wf is not None
            review_actions.drain(
                events, execution_id=wf.execution_id, resume_steer=inputs.resume_steer
            )
            ipc.promote_queued_steer(layout.session_dir)
            return steer_state.requested()

        wf = loop.Harness(
            media_input=role_providers.input_capabilities(cfg, session.rm_role),
            chain=run_chain.RunChain(
                cwd,
                # `git.control = "model"` suspends the chain: the model's commits are the record.
                ref=git_read.chain_ref_for(inputs.session_id)
                if mode == "run" and cfg.git.control != "model"
                else None,
                branch=inputs.chain_branch,
                fallback_parent=inputs.base_sha or None,
                untracked_at_start=untracked_at_start,
                hidden=hidden,
                identity=run_chain.commit_identity(
                    cfg.git.commit,
                    commit_message.render_commit_trailer(
                        cfg.git.commit.trailer, models=(session.rm_role.model,)
                    ),
                ),
                per_turn=cfg.git.commit_per_turn,
                base_sha=inputs.base_sha,
                mirror_into=inputs.mirror_into,
            ),
            config=cfg,
            base_rules=inputs.base_rules,
            review_target=inputs.review_target,
            standing_goal=inputs.standing_goal,
            interactive=inputs.interactive and mode == "run",
            initial_pins=inputs.pins,
            max_turns=cfg.harness.max_turns,
            provider=session.provider,
            dispatcher=dispatcher,
            prompts=inputs.prompts,
            logger=loop_log,
            events=events,
            curator=curator,
            bridge=operator_bridge.OperatorBridge(
                steer_requested=_steer_requested,
                steer_clear=steer_state.clear,
                steer_prompt=steer_state.prompt,
                steer_reset=steer_state.reset_stage,
                compact_requested=lambda: ipc.read_compact_request(layout.session_dir),
                compact_clear=lambda: ipc.clear_compact_request(layout.session_dir),
                stop_requested=lambda: ipc.stop_request_pending(layout.session_dir),
                stop_is_now=lambda: ipc.stop_request_is_now(layout.session_dir),
                stop_clear=lambda: ipc.clear_stop_request(layout.session_dir),
                should_abort=steer_state.abort_pending,
                should_interrupt=steer_state.interrupt,
                take_requests=lambda: ipc.drain_requests(layout.session_dir),
                after_auto_commit=after_auto_commit,
                undo_forker=_undo_forker,
                review_preparer=functools.partial(
                    review_directive.live, root=cwd, hide=cfg.sandbox.hide_paths
                ),
                lane_authority=lane_launch.dispatch_authority(
                    cfg, mode=mode, session_id=inputs.session_id, root=cwd
                ),
                # Subordinate workers retain only harness-panel authority.
                lane_spawner=frontend.build_coordinator_spawner(
                    cfg,
                    inputs.explicit_leaves,
                    cwd,
                    state_dir,
                    mode,
                    inputs.session_id,
                    inputs.budget_overrides.max_usd
                    if inputs.budget_overrides is not None
                    else None,
                    inputs.sandbox_overrides.auto_approve
                    if inputs.sandbox_overrides is not None
                    else False,
                    child_watch.ChildWatch(inputs.prompts, layout.session_dir, budget, events.emit),
                ),
                lane_recoverer=app_lanes.build_lane_recoverer(
                    cfg, cwd, state_dir, mode=mode, reporter=reporter
                ),
            ),
            budget=budget,
            state_dir=state_dir,
            resume_state_path=inputs.resume_state_path,
            operator_edits=inputs.operator_edits,
            merged_in=inputs.merged_in,
            mode=mode,
            contract=inputs.contract,
            plan_output_path=(layout.session_dir / "plan.md" if mode == "plan" else None),
            review=review_gate.ReviewSettings(
                when=cfg.review.when,
                interval_turns=cfg.review.interval_turns,
                reviewers=session.review_seats,
                block_on=cfg.review.block_on,
                min_blocking_models=cfg.review.min_blocking_models,
                max_rejections=cfg.review.max_rejections,
                min_remaining_budget_fraction=cfg.review.min_remaining_budget_fraction,
                concurrency=cfg.review.concurrency,
                repo_max_turns=cfg.review.repo_max_turns,
            ),
            call=provider_call.CallSettings(temperature=role_providers.role_temperature(cfg, role)),
            compaction=compaction_window.CompactionSettings(
                drop_at_chars=tools.compact_drop_at_chars,
                summarise_at_chars=tools.compact_summarise_at_chars,
                tool_result_cap_bytes=session_runtime.tool_result_cap_bytes(cfg, role),
                keep_recent_chars=tools.keep_recent_chars,
                keep_thinking_turns=cfg.compaction.keep_thinking_turns,
                elision_gists=cfg.compaction.elision_gists,
                summary_max_tokens=cfg.compaction.summary_max_tokens,
                summariser=session.summariser_provider,
            ),
        )
        if inputs.task is not None:
            stamps.unpark(layout.session_dir, run_branch=inputs.chain_branch)
        try:
            with frontend.tui_session(layout.session_dir, inputs.tui_enabled):
                try:
                    if unready is not None:
                        # The model cannot fix the operator's setup command: a loud end before
                        # any turn, as an unexecutable gate's is.
                        loop_log(f"[agent6] LOOP: aborting -- {unready.splitlines()[0]}")
                        events.emit(
                            journal.SessionStop(
                                execution_id=dispatcher.execution_id,
                                reason="setup_command_failed",
                                summary=unready,
                                turns=0,
                                verification=None,
                            )
                        )
                        result = loop_snapshot.SessionResult(
                            completed=False,
                            reason="setup_command_failed",
                            summary=unready,
                            turns=0,
                            tool_calls=0,
                        )
                    elif inputs.task is None:
                        result = wf.resume(steer=inputs.resume_steer)
                    elif mode == "ask" and inputs.interactive:
                        result = frontend.run_ask_repl(
                            wf, budget, layout, inputs.task, inputs.operator_task
                        )
                    else:
                        result = wf.run(inputs.task, operator_task=inputs.operator_task)
                    # Settled now, not at teardown: a viewer left open reads `/shells` meanwhile.
                    dispatcher.settle_background()
                except (KeyboardInterrupt, Exception) as exc:
                    # Journaled inside the TUI scope: the dashboard leaves only on an end it sees.
                    escape_handled = True
                    if result is None:
                        _journal_escape(
                            events,
                            exc,
                            execution_id=wf.execution_id,
                            turns=wf.turns_reached,
                            tree=_tree_left(wf, mode),
                        )
                    raise
        except loop_snapshot.ResumeError as exc:
            reporter.error(str(exc))
            reporter.err(f"\n[agent6] {label} crashed")
            return ExecutionEnd(1)
        except KeyboardInterrupt:
            if result is not None:
                # After the run's own end an interrupt cuts only the background settle.
                reporter.err(f"\n[agent6] {label} had ended; its result stands")
            else:
                interrupted = True
                reporter.err(f"\n[agent6] {label} interrupted")
    except (KeyboardInterrupt, Exception) as exc:
        # An escape after the run ended is the execution's failure: the run's end stays its last.
        if (
            not escape_handled
            and result is None
            and stamps.parked_stamp(layout.session_dir) is None
        ):
            _journal_escape(
                events,
                exc,
                execution_id=wf.execution_id if wf else execution_id,
                turns=wf.turns_reached if wf else 0,
            )
        reporter.err(f"\n[agent6] {label} {_escape_reason(exc)}")
        raise
    finally:

        def _close_mcp() -> None:
            """Close the MCP servers and journal any that survived."""
            if mcp_manager is not None and (survivors := mcp_manager.close()):
                with contextlib.suppress(
                    event_log.EventWriteError
                ):  # a dead journal must not skip cleanup
                    events.emit(journal.JailDegraded(detail=jail.survivors_message(survivors)))

        # ExitStack runs every close even when one raises, and re-raises after: no merge lands
        # on an execution whose teardown failed.
        try:
            with contextlib.ExitStack() as cleanup:
                if session_net is not None:
                    cleanup.callback(process.clear_session_netns_pid, layout.session_dir)
                    cleanup.callback(session_net.close)
                cleanup.callback(_close_mcp)
                if dispatcher is not None:
                    cleanup.callback(dispatcher.close)
                else:
                    cleanup.callback(access.close)
                cleanup.callback(session.close)
                cleanup.callback(steer_state.restore)
                # A stop never read at a boundary would stop the next execution at its first.
                cleanup.callback(ipc.clear_stop_request, layout.session_dir)
            if inputs.mirror_into is not None and mode == "run":
                # Once more at the end: a merge taken in or an undo moves the chain with no commit;
                # under `git.control = "model"` the model's own commits are the record.
                try:
                    if cfg.git.control == "model":
                        git_ops.mirror_session(
                            inputs.mirror_into,
                            cwd,
                            git_read.chain_ref_for(inputs.session_id),
                            None,
                            head=True,
                        )
                    elif wf is not None:
                        wf.chain.mirror()
                except errors.GitError as exc:
                    reporter.warn(
                        f"the run's commits could not be copied into {inputs.mirror_into}: {exc};"
                        f" they are in its worktree {cwd}"
                    )
            # A lane's base branch is its throwaway clone's: its dispatcher lands the work.
            lands_itself = cfg.git.auto_merge and mode == "run" and not lane_launch.is_lane()
            if not interrupted and result is not None and lands_itself:
                if finalize.auto_merge_eligible(result):
                    finalize.finalize_auto_merge(
                        # A clone has a base branch of its own; the run lands on the repository's.
                        inputs.mirror_into or cwd,
                        layout=layout,
                        cfg=cfg,
                        reporter=reporter,
                        budget=budget,
                        events=events,
                        worker_lock_fd=inputs.worker_lock_fd,
                    )
                else:
                    reason = (
                        f"the run ended on {result.reason}, not finish_session"
                        if result.reason != "finish_session"
                        else f"the gate on the final tree is {result.verification}"
                    )
                    reporter.note(f"auto_merge skipped: {reason}")
        finally:
            # Never leave root-owned run state in the user's repo (the sudo case).
            paths.chown_to_real_user(state_dir)

    finalize.settle_budget(budget, events)
    if interrupted:
        finalize.print_interrupt_end(layout=layout, cwd=cwd, budget=budget, reporter=reporter)
        return ExecutionEnd(finalize.EXIT_OPERATOR_STOP)
    if result is None:
        return ExecutionEnd(1)

    # The operator's own ends come first: `/undo` and `/detach` end an ask as they end a run.
    if result.reason == "undone" and undo_outcome:
        new_id, undone_text = undo_outcome[-1]
        reporter.out(f"\n[agent6] undone: continue as {new_id} with your message back to edit:")
        reporter.out(f"    agent6 resume {new_id} --steer {undone_text!r}")
        return ExecutionEnd(0)
    if result.reason == "detached":
        # The caller releases the worker lock, then hands the run to `detach_to_background`.
        return ExecutionEnd(0, detach_requested=True)

    harness_gate = cfg.harness.verify.when != "never"
    if mode == "ask":
        # stdout gets the answer alone, for a pipe; when the stream (on stderr) and stdout are
        # the one terminal, the stream showed it. The REPL already printed and saved each turn.
        if inputs.ask_transcript_task is not None:
            if not (console_stream and sys.stdout.isatty() and sys.stderr.isatty()):
                reporter.out(result.summary)
            frontend.save_ask_transcript(layout, inputs.ask_transcript_task, result.summary)
            reporter.err(f"\n[agent6] answer saved to {layout.session_dir / 'transcript.md'}")
        reporter.err(budget.format_summary())
        return ExecutionEnd(finalize.session_exit_code(result, harness_gate=harness_gate))

    stranded = finalize.stranded_edits(result, layout, cwd, hidden=hidden)
    if stranded:
        # Journaled before the end block, whose headline and every listing read the journal.
        events.emit(journal.SessionStranded())
    finalize.print_session_end(
        result,
        layout=layout,
        cwd=cwd,
        budget=budget,
        console_stream=console_stream,
        reporter=reporter,
        hidden=hidden,
    )
    if inputs.contract is None:
        notify_hook.session_end(
            cfg.notify,
            session_id=layout.session_id,
            session_dir=layout.session_dir,
            ok=result.completed,
            reason=result.reason,
            verification=result.verification,
            note=reporter.note,
        )
    return ExecutionEnd(
        finalize.session_exit_code(result, harness_gate=harness_gate, stranded=stranded)
    )
