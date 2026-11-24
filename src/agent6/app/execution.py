# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Drive the one execution body a fresh run and a resumed execution share.

From the provider session to the end block: providers, the gate step, steer
state, the session network and MCP servers, the tool set, the Harness, its teardown, the
auto-merge and the end report. The run and resume lifecycles keep what differs before and after it
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
from collections.abc import Callable

from agent6 import budget as agent6_budget
from agent6 import directive, kinds, pack, paths, review_records
from agent6.app import (
    conversation_state,
    execution_finish,
    execution_harness,
    finalize,
    notify_hook,
    preflight,
    stamps,
    worktree_setup,
)
from agent6.app import frontend as app_frontend
from agent6.app import reporter as app_reporter
from agent6.app.parallel import children as app_children
from agent6.app.parallel import relay as child_watch
from agent6.app.session import runtime as session_runtime
from agent6.app.session import setup as session_setup
from agent6.config import Config, RoleName, layer_data
from agent6.harness import loop, loop_snapshot, run_chain
from agent6.providers import TranscriptSink
from agent6.sandbox import jail
from agent6.sessions import event_log, ipc, journal, process
from agent6.sessions import layout as sessions_layout
from agent6.sessions import manifest as sessions_manifest
from agent6.tools import command_tools, dispatch, operator_prompts, schema, target_policy
from agent6.tools import policy as tools_policy
from agent6.vcs.git import ops as git_ops
from agent6.vcs.git import read as git_read
from agent6.viewmodel.session import log as session_log


@dataclasses.dataclass(frozen=True, slots=True)
class ExecutionInputs:
    """Hold what a fresh execution and a resumed one hand the body differently.

    Attributes:
        session_id: The session's id.
        mode: The session mode.
        role: The role driving it.
        explicit_leaves: The dotted leaves the run's config layers set, inherited by children.
        isolation: The resolved isolation level.
        tui_enabled: Whether the TUI owns the terminal.
        interactive: Whether an operator is at the terminal.
        task: The task a fresh execution runs; None resumes the snapshot.
        effective: The config origins before execution policy.
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
        budget_overrides: The `--max-*` flags a `/parallel` child inherits.
        sandbox_overrides: The `--auto-approve` a `/parallel` child inherits.
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
        contract: The bounded conversation a `reviewer` session drives; None for a worker's.
        opening: The tasks a fresh run dispatches instead of a first model call.
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
    effective: layer_data.EffectiveConfig | None = None
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
    opening: tuple[directive.Segment, ...] = ()


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


def report_end(
    cfg: Config,
    layout: sessions_layout.SessionLayout,
    inputs: ExecutionInputs,
    *,
    result: loop_snapshot.SessionResult | None,
    interrupted: bool,
    undo_outcome: list[tuple[str, str]],
    budget: agent6_budget.BudgetTracker,
    console_stream: bool,
    hidden: frozenset[str],
    frontend: app_frontend.SessionFrontend,
    reporter: app_reporter.Reporter,
    events: event_log.EventSink,
    cwd: pathlib.Path,
) -> ExecutionEnd:
    """Settle the budget and print the end an execution reached.

    Args:
        cfg: The config the tools resolved.
        layout: The session's directory layout.
        inputs: What the lifecycle hands the body.
        result: How the run ended, or None when it escaped.
        interrupted: Whether the operator interrupted the run.
        undo_outcome: The `/undo` forks' outcomes, the last one named.
        budget: The run's budget.
        console_stream: Whether a console view showed the run.
        hidden: The run's hidden paths inside the repository.
        frontend: The injected front-end, which saves an ask's transcript.
        reporter: The output channels.
        events: The run's event sink.
        cwd: The workspace.

    Returns:
        The exit code, and whether the operator detached.
    """
    mode = inputs.mode
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
        # The caller releases the worker lock before spawning the background resume.
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
        finalize.print_session_usage(
            session_log.scan_session_log(layout.logs_path), emit=reporter.err
        )
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


def run_execution(  # noqa: C901, PLR0912, PLR0915  # setup, run and teardown in reading order
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
    preflight.note_unpriced_search(cfg, mode, reporter=reporter)
    label = "resume" if inputs.resuming else "run"
    session: session_runtime.SessionProviders | None = None
    access: command_tools.JailAccess | None = None
    session_net: jail.SessionNetwork | None = None
    mirror: run_chain.MirrorTarget | None = None
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
            actor=inputs.contract.actor if inputs.contract else "",
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
        if inputs.mirror_into is not None:
            # The clone's side of each copy into the repository runs under the run's own jail.
            mirror = run_chain.MirrorTarget(
                inputs.mirror_into, tools_policy.jail_policy(cwd, cfg, inputs.isolation, ())
            )

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
            reporter.err(
                f"\n[agent6] {label} {execution_finish.escape_reason(exc).replace('_', ' ')}"
            )
        else:
            tree = execution_finish.tree_left_before_the_loop(cwd, inputs, hidden)
            reason = execution_finish.journal_escape(
                events, exc, execution_id=execution_id, turns=0, tree=tree
            )
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
    spawner: app_children.GroupSpawner | None = None
    mcp_manager = None
    try:
        reporter.note(f"{'resume ' if inputs.resuming else ''}session id: {inputs.session_id}")

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
            # A headless run has no console view to print its markers.
            events.subscribe(functools.partial(execution_harness.headless_marker, log=loop_log))
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
        dispatcher = tools.dispatcher
        cfg = tools.cfg

        # Before the chain exists: what the setup command leaves untracked is the workspace's
        # environment, so every commit leaves it out as it does the operator's files.
        unready: str | None = None
        untracked_at_start = inputs.untracked_at_start
        # A reviewer handed `run_command` needs what git does not track, as any child does.
        runs_commands = inputs.contract is None or (
            schema.RunCommandInput.TOOL_NAME in inputs.contract.allowed_tools
            and dispatcher.command_policy() != "deny"
        )
        if runs_commands and worktree_setup.session_needs_setup(layout.session_dir):
            # The run's own earlier files read untracked too: the chain never moves the index.
            before = git_read.untracked_paths(cwd) if mode == "run" else frozenset[str]()
            prepared = worktree_setup.prepare(cfg, cwd, runner=dispatcher, reporter=reporter)
            unready = prepared.problem
            if prepared.ran:
                try:
                    current = sessions_manifest.read_manifest(layout.session_dir)
                    stamps.write_manifest(
                        layout.manifest_path, current.model_copy(update={"worktree_set_up": True})
                    )
                except (sessions_manifest.ManifestError, OSError) as exc:
                    reporter.note(f"could not record that this workspace is set up: {exc}")
                if mode == "run":
                    untracked_at_start |= git_read.untracked_paths(cwd) - before
                    sessions_layout.write_untracked_at_start(layout.session_dir, untracked_at_start)

        after_auto_commit: Callable[[int, str], kinds.AutoCommitDirective] = (
            frontend.build_repl_hook(cwd, budget, inputs.session_id, mcp_manager)
            if inputs.interactive and mode == "run"
            else (lambda _i, _s: "continue")
        )

        # A child worker keeps only panel authority; a child's record goes where this one's does.
        spawner = frontend.build_parent_spawner(
            cfg,
            inputs.explicit_leaves,
            cwd,
            layout.state_dir,
            mode,
            inputs.session_id,
            inputs.budget_overrides.max_usd if inputs.budget_overrides is not None else None,
            inputs.sandbox_overrides.auto_approve
            if inputs.sandbox_overrides is not None
            else False,
            events,
            child_watch.ChildWatch(inputs.prompts, layout.session_dir, budget, events.emit),
        )
        wf = execution_harness.build_harness(
            cfg,
            layout,
            inputs,
            session=session,
            tools=tools,
            steer_state=steer_state,
            spawner=spawner,
            after_auto_commit=after_auto_commit,
            undo_outcome=undo_outcome,
            untracked_at_start=untracked_at_start,
            hidden=hidden,
            mirror=mirror,
            loop_log=loop_log,
            console_stream=console_stream,
            events=events,
            reporter=reporter,
            cwd=cwd,
            state_dir=state_dir,
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
                        execution_finish.journal_escape(
                            events,
                            exc,
                            execution_id=wf.execution_id,
                            turns=wf.turns_reached,
                            tree=execution_finish.tree_left(wf, mode),
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
            execution_finish.journal_escape(
                events,
                exc,
                execution_id=wf.execution_id if wf else execution_id,
                turns=wf.turns_reached if wf else 0,
            )
        reporter.err(f"\n[agent6] {label} {execution_finish.escape_reason(exc).replace('_', ' ')}")
        raise
    finally:

        def _close_mcp() -> None:
            """Close the MCP servers and journal any that survived."""
            if mcp_manager is not None and (survivors := mcp_manager.close()):
                with contextlib.suppress(
                    event_log.EventWriteError
                ):  # a dead journal must not skip cleanup
                    events.emit(journal.JailDegraded(detail=jail.survivors_message(survivors)))

        # ExitStack runs every close even when one raises, and re-raises after: no merge runs
        # on an execution whose teardown failed.
        try:
            with contextlib.ExitStack() as cleanup:
                if session_net is not None:
                    cleanup.callback(process.clear_session_netns_pid, layout.session_dir)
                    cleanup.callback(session_net.close)
                cleanup.callback(_close_mcp)
                if spawner is not None:
                    # A child the loop's end did not stop (an escape) ends here, by signal.
                    cleanup.callback(spawner.close)
                if dispatcher is not None:
                    cleanup.callback(dispatcher.close)
                else:
                    cleanup.callback(access.close)
                cleanup.callback(session.close)
                cleanup.callback(steer_state.restore)
                # A stop never read at a boundary would stop the next execution at its first.
                cleanup.callback(ipc.clear_stop_request, layout.session_dir)
            execution_finish.finish_chain(
                cfg,
                layout,
                inputs,
                wf=wf,
                result=result,
                interrupted=interrupted,
                mirror=mirror,
                budget=budget,
                events=events,
                reporter=reporter,
                cwd=cwd,
            )
        finally:
            # Never leave root-owned run state in the user's repo (the sudo case).
            paths.chown_to_real_user(state_dir)
            # On every end, after the run's commands close: a model's tools can write a pack.json.
            pack.quarantine_tree(paths.checkout_root(cwd), warn=reporter.warn, events=events)

    return report_end(
        cfg,
        layout,
        inputs,
        result=result,
        interrupted=interrupted,
        undo_outcome=undo_outcome,
        budget=budget,
        console_stream=console_stream,
        hidden=hidden,
        frontend=frontend,
        reporter=reporter,
        events=events,
        cwd=cwd,
    )
