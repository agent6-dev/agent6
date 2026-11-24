# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Build the shared harness from the inputs of a fresh or resumed execution."""

from __future__ import annotations

import functools
import pathlib
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from agent6 import evidence, kinds
from agent6.app import frontend as app_frontend
from agent6.app import reporter as app_reporter
from agent6.app import (
    role_providers,
)
from agent6.app.parallel import children as app_children
from agent6.app.parallel import launch as child_launch
from agent6.app.parallel import resume as child_resume
from agent6.app.review import actions as review_actions
from agent6.app.review import directive as review_directive
from agent6.app.session import runtime as session_runtime
from agent6.config import Config
from agent6.harness import loop, operator_bridge, provider_call, run_chain
from agent6.harness.compaction import window as compaction_window
from agent6.harness.review import gate as review_gate
from agent6.sessions import event_log, ipc, journal
from agent6.sessions import layout as sessions_layout
from agent6.vcs import commit_message
from agent6.vcs.git import read as git_read
from agent6.viewmodel import execution_settings, policy, transcript_text

if TYPE_CHECKING:
    from agent6.app import execution


def headless_marker(event: dict[str, Any], *, log: Callable[[str], None]) -> None:
    """Print warnings and child activity through the run's text log.

    Args:
        event: One journal event.
        log: The run's text logger.
    """
    if event.get("type") in (
        "jail.degraded",
        "pack.quarantined",
        "parallel.child_activity",
        "parallel.child_continued",
        "parallel.child_merged",
        "parallel.returned",
    ):
        log(f"[agent6] {transcript_text.MARKER_BODIES[event['type']](event)}")


def start_header(
    event: dict[str, Any],
    *,
    session_dir: pathlib.Path,
    settings: tuple[journal.SettingObservation, ...],
    mode: kinds.LoopMode,
    log: Callable[[str], None],
) -> None:
    """Print a headless execution's settings, environment and gate when it starts.

    Args:
        event: One journal event.
        session_dir: The session directory.
        settings: The settings this execution runs under.
        mode: The session mode; an ask has no gate to name.
        log: Where the line goes.
    """
    if event.get("type") not in journal.SESSION_START_EVENTS:
        return
    facts = policy.session_policy(session_dir)
    parts = [execution_settings.summary(settings), *facts.environment[:1]]
    parts += [e for e in facts.environment if e == evidence.GIT_WRITABLE]
    if mode != "ask":
        parts.append(facts.gate())
    if line := " · ".join(p for p in parts if p):
        log(line)


def build_harness(
    cfg: Config,
    layout: sessions_layout.SessionLayout,
    inputs: execution.ExecutionInputs,
    *,
    session: session_runtime.SessionProviders,
    tools: session_runtime.SessionTools,
    steer_state: app_frontend.SteerHooks,
    spawner: app_children.GroupSpawner | None,
    after_auto_commit: Callable[[int, str], kinds.AutoCommitDirective],
    undo_outcome: list[tuple[str, str]],
    untracked_at_start: frozenset[str],
    hidden: frozenset[str],
    mirror: run_chain.MirrorTarget | None,
    loop_log: Callable[[str], None],
    console_stream: bool,
    events: event_log.EventSink,
    reporter: app_reporter.Reporter,
    cwd: pathlib.Path,
    state_dir: pathlib.Path,
) -> loop.Harness:
    """Wire the Harness an execution drives: the one place a fresh and a resumed one meet.

    Args:
        cfg: The config the tools resolved.
        layout: The session's directory layout.
        inputs: What the lifecycle hands the body.
        session: The run's providers.
        tools: The curator and dispatcher pair and the compaction thresholds.
        steer_state: The steer state the bridge polls.
        spawner: The dispatch spawner, or None.
        after_auto_commit: The hook a `run -i` calls after each commit.
        undo_outcome: Takes each `/undo` fork's outcome, for the end block.
        untracked_at_start: The operator's untracked files and the setup command's.
        hidden: The run's hidden paths inside the repository.
        mirror: The repository a session in a clone copies its commits into, or None.
        loop_log: The loop's logger.
        console_stream: Whether a console view prints the run's markers.
        events: The run's event sink.
        reporter: The output channels.
        cwd: The workspace.
        state_dir: The repo's state directory.

    Returns:
        The harness.
    """
    mode, role = inputs.mode, inputs.role

    def undo_forker() -> tuple[str, str] | None:
        """Return the `/undo` fork's outcome, kept for the end block.

        Returns:
            The fork's id and the text taken back, or None with the reason printed.
        """
        got = inputs.undo_forker()
        if got is not None:
            undo_outcome.append(got)
        return got

    def steer_requested() -> bool:
        review_actions.drain(events, execution_id=wf.execution_id, resume_steer=inputs.resume_steer)
        ipc.promote_queued_steer(layout.session_dir)
        return steer_state.requested()

    launch = child_launch.read()
    settings = execution_settings.observations(cfg, inputs.effective)
    if not inputs.tui_enabled:
        events.subscribe(
            functools.partial(
                start_header,
                session_dir=layout.session_dir,
                settings=settings,
                mode=mode,
                log=loop_log if console_stream else reporter.note,
            )
        )
    wf = loop.Harness(
        task_source=launch.task_source if launch is not None else "operator",
        sibling_notice=launch.sibling_notice if launch is not None else "",
        settings=settings,
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
            model_git=mode == "run" and cfg.git.control == "model",
            base_sha=inputs.base_sha,
            mirror_into=mirror,
        ),
        config=cfg,
        base_rules=inputs.base_rules,
        review_target=inputs.review_target,
        standing_goal=inputs.standing_goal,
        interactive=inputs.interactive and mode == "run",
        web_search=cfg.offers_web_search(mode),
        initial_pins=inputs.pins,
        max_turns=cfg.harness.max_turns,
        provider=session.provider,
        dispatcher=tools.dispatcher,
        prompts=inputs.prompts,
        logger=loop_log,
        events=events,
        curator=tools.curator,
        bridge=operator_bridge.OperatorBridge(
            steer_requested=steer_requested,
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
            undo_forker=undo_forker,
            review_preparer=functools.partial(
                review_directive.live, root=cwd, hide=cfg.sandbox.hide_paths
            ),
            child_authority=child_launch.dispatch_authority(
                cfg, mode=mode, session_id=inputs.session_id, root=cwd
            ),
            child_spawner=spawner,
            child_recoverable=functools.partial(
                child_resume.recoverable,
                cfg=cfg,
                origin=cwd,
                record_state=layout.state_dir,
                parent=inputs.session_id,
            ),
            child_recoverer=app_children.build_child_recoverer(
                cfg, cwd, layout.state_dir, mode=mode, reporter=reporter
            ),
            rank_recorder=app_children.build_rank_recorder(
                cwd,
                layout.state_dir,
                parent_dir=layout.session_dir,
                events=events,
                reporter=reporter,
            )
            if mode == "run"
            else None,
        ),
        budget=session.budget,
        state_dir=state_dir,
        resume_state_path=inputs.resume_state_path,
        operator_edits=inputs.operator_edits,
        merged_in=inputs.merged_in,
        mode=mode,
        contract=inputs.contract,
        opening=inputs.opening,
        plan_output_path=(layout.session_dir / "plan.md" if mode == "plan" else None),
        review=review_gate.ReviewSettings(
            when=cfg.review.when,
            interval_turns=cfg.review.interval_turns,
            reviewers=session.reviewers,
            repeat_provider=session.repeat_provider if cfg.review.merge_repeats else None,
            block_on=cfg.review.block_on,
            min_blocking_models=cfg.review.min_blocking_models,
            max_rejections=cfg.review.max_rejections,
            min_remaining_budget_fraction=cfg.review.min_remaining_budget_fraction,
            concurrency=cfg.review.concurrency,
            max_turns=cfg.review.max_turns,
        ),
        call=provider_call.CallSettings(temperature=role_providers.role_temperature(cfg, role)),
        compaction=compaction_window.CompactionSettings(
            cold_start=cfg.compaction.cold_start,
            drop_at_chars=tools.compact_drop_at_chars,
            summarize_at_chars=tools.compact_summarize_at_chars,
            tool_result_cap_bytes=session_runtime.tool_result_cap_bytes(cfg, role),
            keep_recent_chars=tools.keep_recent_chars,
            keep_operator_messages=cfg.compaction.keep_operator_messages,
            keep_recent_turns=cfg.compaction.keep_recent_turns,
            keep_thinking_turns=cfg.compaction.keep_thinking_turns,
            elision_gists=cfg.compaction.elision_gists,
            summary_max_tokens=cfg.compaction.summary_max_tokens,
            summarizer=cfg.compaction.summarizer,
            helper=session.summarizer_provider,
            helper_context_tokens=role_providers.helper_context_window(cfg),
            endpoint=role_providers.endpoint_compaction(cfg, session.rm_role),
            idle=role_providers.idle_compaction(cfg, session.rm_role),
        ),
    )
    return wf
