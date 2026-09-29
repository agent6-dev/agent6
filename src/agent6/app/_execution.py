# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Drive the one execution body a fresh run and a resumed execution share.

From the provider session to the end block: prompt revision, providers, the gate step, steer
state, the session network and MCP servers, the tool set, the Harness, its teardown, the
auto-merge and the end report. `run.py` and `resume.py` keep what differs before and after it
and hand the body its `ExecutionInputs`, so a knob wired into one lifecycle cannot be missing
from the other.
"""

from __future__ import annotations

import contextlib
import dataclasses
import pathlib
from collections.abc import Callable, Sequence

from agent6 import budget as agent6_budget
from agent6 import commit_message, git_ops, kinds, paths
from agent6 import events as agent6_events
from agent6.app import _session, _setup, finalize, manifest, providers
from agent6.app import frontend as app_frontend
from agent6.app import reporter as app_reporter
from agent6.config import Config, RoleName
from agent6.harness import (
    _chain,
    _compaction,
    _operator,
    _prompt_revision,
    _provider_call,
    _reviewer,
    _snapshot,
    loop,
)
from agent6.providers import Provider, TranscriptSink
from agent6.sandbox import jail
from agent6.sessions import ipc
from agent6.sessions import layout as sessions_layout
from agent6.tools import dispatch, operator_prompts


@dataclasses.dataclass(frozen=True, slots=True)
class ExecutionInputs:
    """Hold what a fresh execution and a resumed one hand the body differently.

    Attributes:
        session_id: The session's id.
        mode: The session mode.
        role: The role driving it.
        isolation: The resolved isolation level.
        tui_enabled: Whether the TUI owns the terminal.
        interactive: Whether an operator is at the terminal.
        task: The task a fresh execution runs; None resumes the snapshot.
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
        worktree_git_dir: A fork's repository git dir, the one grant its jail makes beyond the
            workspace.
    """

    session_id: str
    mode: kinds.ResumableMode
    role: RoleName
    isolation: kinds.IsolationLevel
    tui_enabled: bool
    interactive: bool
    task: str | None
    gate: Callable[[Config, agent6_budget.BudgetTracker], Config]
    chain_branch: str | None
    base_sha: str
    untracked_at_start: frozenset[str]
    resume_state_path: pathlib.Path
    undo_forker: Callable[[], tuple[str, str] | None]
    prompts: operator_prompts.OperatorPrompts
    ask_transcript_task: str | None
    budget_overrides: _setup.BudgetOverrides | None = None
    sandbox_overrides: _setup.SandboxOverrides | None = None
    standing_goal: str = ""
    pins: tuple[str, ...] = ()
    resuming: bool = False
    worktree_git_dir: pathlib.Path | None = None


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
        ipc.clear_worker_pid(layout.session_dir)
        reporter.note(err)
        return
    reporter.out(f"\n[agent6] detached: {layout.session_id} continues in the background.")
    reporter.out(f"          reattach:  agent6 attach {layout.session_id}")


def _escape_reason(exc: BaseException) -> _snapshot.SessionEndReason:
    """Return the end reason an escaping exception stands for."""
    return "interrupted" if isinstance(exc, KeyboardInterrupt) else "crashed"


def _journal_escape(
    events: agent6_events.EventSink, exc: BaseException, *, iterations: int
) -> _snapshot.SessionEndReason:
    """Journal the end an escape leaves.

    Without one every surface reads the dead run as running until the silence window expires;
    a dead journal must not mask the exit code.

    Args:
        events: The run's event sink.
        exc: The escaping exception.
        iterations: The turns reached.

    Returns:
        The reason journaled.
    """
    reason = _escape_reason(exc)
    with contextlib.suppress(agent6_events.EventWriteError):
        events.emit("session.end", reason=reason, iterations=iterations, all_passed=False)
    return reason


def run_execution(  # noqa: C901, PLR0911, PLR0912, PLR0915  # setup, run and teardown in reading order
    cfg: Config,
    layout: sessions_layout.SessionLayout,
    inputs: ExecutionInputs,
    *,
    frontend: app_frontend.SessionFrontend,
    reporter: app_reporter.Reporter,
    events: agent6_events.EventSink,
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
    mode, role = inputs.mode, inputs.role
    label = "resume" if inputs.resuming else "run"
    session: _session.SessionProviders | None = None
    prompt_reviser_provider: Provider | None = None
    try:
        # The interactive revision prompt reads the terminal, which the TUI may own.
        effective_revise_prompt = cfg.prompt.revise_prompt
        if effective_revise_prompt == "interactive" and (
            inputs.tui_enabled or frontend.select_revised_prompt is None
        ):
            owner = "the TUI owns it" if inputs.tui_enabled else "this surface has none"
            reporter.note(
                f"prompt.revise_prompt='interactive' needs the terminal; {owner}."
                " Skipping prompt revision for this execution."
            )
            effective_revise_prompt = "off"
        stream_text, console_stream = frontend.stream_modes(inputs.tui_enabled)
        if console_stream:
            frontend.attach_console_view(events)
        session = _session.build_session_providers(
            cfg,
            role=role,
            events=events,
            transcript_sink=transcript_sink,
            stream_text=stream_text,
            reporter=reporter,
        )
        budget = session.budget
        prompt_reviser_provider = providers.build_prompt_reviser_provider(
            cfg, transcript_sink=transcript_sink, budget=budget, events=events
        )
        cfg = inputs.gate(cfg, budget)

        steer_state = frontend.make_steer_state(
            events,
            layout.session_dir,
            _session.session_facts_provider(
                budget, session.rm_role.model, cfg.sandbox.run_commands, inputs.isolation
            ),
        )
    except (KeyboardInterrupt, Exception) as exc:
        # A parked run whose start crashes here never ran: it stays parked with no session.end.
        if manifest.parked_stamp(layout.session_dir) is not None:
            reporter.err(f"\n[agent6] {label} {_escape_reason(exc)}")
        else:
            reporter.err(f"\n[agent6] {label} {_journal_escape(events, exc, iterations=0)}")
        with contextlib.ExitStack() as cleanup:
            if prompt_reviser_provider is not None:
                cleanup.callback(providers.close_provider, prompt_reviser_provider)
            if session is not None:
                cleanup.callback(session.close)
        raise

    interrupted = False
    result: _snapshot.SessionResult | None = None
    wf: loop.Harness | None = None
    escape_handled = False
    undo_outcome: list[tuple[str, str]] = []
    dispatcher: dispatch.ToolDispatcher | None = None
    mcp_manager = None
    session_net: jail.SessionNetwork | None = None
    try:
        reporter.note(f"{'resume ' if inputs.resuming else ''}session id: {inputs.session_id}")

        # The session network before its first member; the MCP servers before the harness.
        if _setup.wants_session_network(cfg, inputs.isolation):
            session_net = jail.SessionNetwork.open()
            # Published so `agent6 exec` can join it: a namespace is named through a /proc entry.
            ipc.write_session_netns_pid(layout.session_dir, session_net.holder_pid)
        mcp_manager = _setup.start_mcp_manager_if_enabled(
            cfg, cwd, inputs.isolation, reporter=reporter, events=events, session_net=session_net
        )

        loop_log = frontend.loop_logger(mode)
        tools = _session.build_session_tools(
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
            rm_role=session.rm_role,
            worktree_git_dir=inputs.worktree_git_dir,
        )
        curator = tools.curator
        dispatcher = tools.dispatcher
        cfg = tools.cfg

        def _undo_forker() -> tuple[str, str] | None:
            """Return the `/undo` fork's outcome, kept for the end block."""
            got = inputs.undo_forker()
            if got is not None:
                undo_outcome.append(got)
            return got

        after_auto_commit: Callable[[int, str], kinds.AutoCommitDirective] = (
            frontend.build_repl_hook(cwd, budget, inputs.session_id, mcp_manager)
            if inputs.interactive and mode == "run"
            else (lambda _i, _s: "continue")
        )
        wf = loop.Harness(
            chain=_chain.RunChain(
                cwd,
                # `git.control = "model"` suspends the chain: the model's commits are the record.
                ref=git_ops.chain_ref_for(inputs.session_id)
                if mode == "run" and cfg.git.control != "model"
                else None,
                branch=inputs.chain_branch,
                fallback_parent=inputs.base_sha or None,
                untracked_at_start=inputs.untracked_at_start,
                identity=_chain.commit_identity(
                    cfg.git.commit,
                    commit_message.render_commit_trailer(
                        cfg.git.commit.trailer, models=(session.rm_role.model,)
                    ),
                ),
                per_step=cfg.git.commit_per_step,
                base_sha=inputs.base_sha,
            ),
            config=cfg,
            standing_goal=inputs.standing_goal,
            interactive=inputs.interactive and mode == "run",
            initial_pins=inputs.pins,
            max_iterations=cfg.harness.max_iterations,
            provider=session.provider,
            dispatcher=dispatcher,
            logger=loop_log,
            events=events,
            curator=curator,
            bridge=_operator.OperatorBridge(
                steer_requested=steer_state.requested,
                steer_clear=steer_state.clear,
                steer_prompt=steer_state.prompt,
                steer_reset=steer_state.reset_stage,
                compact_requested=lambda: ipc.read_compact_request(layout.session_dir),
                compact_clear=lambda: ipc.clear_compact_request(layout.session_dir),
                stop_requested=lambda: ipc.stop_request_pending(layout.session_dir),
                stop_clear=lambda: ipc.clear_stop_request(layout.session_dir),
                should_abort=steer_state.abort_pending,
                should_interrupt=steer_state.interrupt,
                take_requests=lambda: ipc.drain_requests(layout.session_dir),
                after_auto_commit=after_auto_commit,
                undo_forker=_undo_forker,
                # None in plan and ask, and inside a lane.
                lane_spawner=frontend.build_coordinator_spawner(
                    cfg,
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
                ),
            ),
            budget=budget,
            state_dir=state_dir,
            resume_state_path=inputs.resume_state_path,
            mode=mode,
            plan_output_path=(layout.session_dir / "plan.md" if mode == "plan" else None),
            review=_reviewer.ReviewSettings(
                trigger=cfg.review.trigger,
                period=cfg.review.period,
                seats=session.review_seats,
                decision=cfg.review.decision,
                quorum=cfg.review.quorum,
                max_total_rejections=cfg.review.max_total_rejections,
                budget_fraction=cfg.review.budget_fraction,
                concurrency=cfg.review.concurrency,
            ),
            revision=_prompt_revision.RevisionSettings(
                reviser=prompt_reviser_provider,
                mode=effective_revise_prompt,
                temperature=providers.role_temperature(cfg, "reviewer"),
                selector=(
                    frontend.select_revised_prompt
                    if effective_revise_prompt == "interactive"
                    else None
                ),
            ),
            call=_provider_call.CallSettings(temperature=providers.role_temperature(cfg, role)),
            compaction=_compaction.CompactionSettings(
                drop_at_chars=tools.compact_drop_at_chars,
                summarise_at_chars=tools.compact_summarise_at_chars,
                tool_result_cap_bytes=_session.tool_result_cap_bytes(cfg, role),
                keep_recent_chars=tools.keep_recent_chars,
                keep_thinking_turns=cfg.context.keep_thinking_turns,
                elision_gists=cfg.context.elision_gists,
                summary_max_tokens=cfg.context.summary_max_tokens,
                summariser=session.summariser_provider,
            ),
        )
        if inputs.task is not None:
            manifest.unpark(layout.session_dir, run_branch=inputs.chain_branch)
        try:
            with frontend.tui_session(layout.session_dir, inputs.tui_enabled):
                try:
                    if inputs.task is None:
                        result = wf.resume()
                    elif mode == "ask" and inputs.interactive:
                        result = frontend.run_ask_repl(wf, budget, layout, inputs.task)
                    else:
                        result = wf.run(inputs.task)
                    # Settled now, not at teardown: a viewer left open reads `/shells` meanwhile.
                    dispatcher.settle_background()
                except (KeyboardInterrupt, Exception) as exc:
                    # Journaled inside the TUI scope: the dashboard leaves only on an end it sees.
                    escape_handled = True
                    if result is None:
                        _journal_escape(events, exc, iterations=wf.iterations_reached)
                    raise
        except _snapshot.ResumeError as exc:
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
            and manifest.parked_stamp(layout.session_dir) is None
        ):
            _journal_escape(events, exc, iterations=wf.iterations_reached if wf else 0)
        reporter.err(f"\n[agent6] {label} {_escape_reason(exc)}")
        raise
    finally:

        def _close_mcp() -> None:
            """Close the MCP servers and journal any that survived."""
            if mcp_manager is not None and (survivors := mcp_manager.close()):
                with contextlib.suppress(
                    agent6_events.EventWriteError
                ):  # a dead journal must not skip cleanup
                    events.emit("jail.degraded", detail=jail.survivors_message(survivors))

        # ExitStack runs every close even when one raises, and re-raises after: no merge lands
        # on an execution whose teardown failed.
        try:
            with contextlib.ExitStack() as cleanup:
                if session_net is not None:
                    cleanup.callback(ipc.clear_session_netns_pid, layout.session_dir)
                    cleanup.callback(session_net.close)
                cleanup.callback(_close_mcp)
                if dispatcher is not None:
                    cleanup.callback(dispatcher.close)
                if prompt_reviser_provider is not None:
                    cleanup.callback(providers.close_provider, prompt_reviser_provider)
                cleanup.callback(session.close)
                cleanup.callback(steer_state.restore)
                # A stop never read at a boundary would stop the next execution at its first.
                cleanup.callback(ipc.clear_stop_request, layout.session_dir)
            if (
                not interrupted
                and result is not None
                and finalize.auto_merge_eligible(result)
                and cfg.git.auto_merge
            ):
                finalize.finalize_auto_merge(
                    cwd, layout=layout, cfg=cfg, reporter=reporter, budget=budget, events=events
                )
        finally:
            # Never leave root-owned run state in the user's repo (the sudo case).
            paths.chown_to_real_user(state_dir)

    if interrupted:
        finalize.print_interrupt_end(layout=layout, cwd=cwd, budget=budget, reporter=reporter)
        return ExecutionEnd(130)
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

    if mode == "ask":
        # stdout gets the answer alone; the REPL already printed and saved each turn.
        if inputs.ask_transcript_task is not None:
            reporter.out(result.summary)
            frontend.save_ask_transcript(layout, inputs.ask_transcript_task, result.summary)
            reporter.err(f"\n[agent6] answer saved to {layout.session_dir / 'transcript.md'}")
        reporter.err(budget.format_summary())
        return ExecutionEnd(finalize.session_exit_code(result))

    finalize.print_session_end(
        result,
        layout=layout,
        cwd=cwd,
        budget=budget,
        console_stream=console_stream,
        reporter=reporter,
    )
    finalize.fire_notify_hook(
        cfg.notify,
        session_id=layout.session_id,
        session_dir=layout.session_dir,
        ok=result.completed,
        reason=result.reason,
        verified=result.verified,
        reporter=reporter,
    )
    return ExecutionEnd(
        finalize.session_exit_code(result, stranded=finalize.stranded_edits(result, layout, cwd))
    )
