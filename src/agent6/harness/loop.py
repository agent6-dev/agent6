# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Run the agent loop: one system prompt, one model driving tool calls, a harness around it.

The harness bounds the loop (jail, budget, verify timeout, turn cap) and records it (events,
the task graph, per-turn commits of every tree a verify certified). The heuristics that nudge or
end a run are advisor functions (`guards`, `metric_rules`, `quiet_turns`) and finish gates
(`finish_gates`), run in a declared order; the loop applies their answers. The review panel
gates checkpoints and never steers.
"""

from __future__ import annotations

import dataclasses
import functools
import itertools
import json
import pathlib
import time
import uuid
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import pydantic

from agent6 import budget as agent6_budget
from agent6 import (
    child_records,
    directive,
    errors,
    kinds,
    media,
    review_records,
    task_text,
    terminal,
)
from agent6.config import Config
from agent6.context import memory as agent6_memory
from agent6.context import skills
from agent6.graph import curator as graph_curator
from agent6.graph import models, order
from agent6.harness import (
    advice,
    dag_focus,
    finish_gates,
    guards,
    loop_conversation,
    loop_resume,
    loop_snapshot,
    loop_state,
    loop_steer,
    metric_sampler,
    nudges,
    operator_bridge,
    operator_tasks,
    parallel_dispatch,
    prompt_blocks,
    provider_call,
    quiet_turns,
    repo_context,
    run_chain,
    run_checkpoint,
    standing_goal,
    tool_effects,
    toolset,
)
from agent6.harness.compaction import driver as compaction_driver
from agent6.harness.compaction import elision as compaction_elision
from agent6.harness.compaction import window as compaction_window
from agent6.harness.review import gate as review_gate
from agent6.harness.review import history as review_history
from agent6.harness.review import panel as review_panel
from agent6.harness.verify import gate as verify_gate
from agent6.harness.verify import verdict as verify_verdict
from agent6.providers import (
    Provider,
    ProviderAborted,
    ProviderError,
    ProviderInterrupted,
    ProviderResponse,
    ToolDefinition,
)
from agent6.sessions import budget_events, completion, journal, process
from agent6.tools import dispatch, operator_prompts, schema, tool_errors
from agent6.tools import results as tools_results
from agent6.vcs.git import ops as git_ops

# Bytes a repeated tool result must exceed before the dedupe stub replaces it.
_DEDUPE_MIN_CHARS = 500


if TYPE_CHECKING:
    from agent6.sessions import event_log


# Consecutive quiet turns after which a metric run drops the worker's output cap.
_STARVATION_BACKOFF_AFTER_QUIETS = 2


@dataclasses.dataclass
class Harness:
    """Drive one execution of the agent loop over a session.

    The model decides everything through tool calls in one loop: when to read, edit, verify,
    measure and stop. The harness keeps the loop bounded and observable.

    Attributes:
        chain: The run's commit chain: the repository root, the per-turn commits, the worktree.
        config: The effective config.
        provider: The model provider the worker calls go to.
        dispatcher: The tool dispatcher.
        logger: Where log lines go.
        events: The run's event sink; None emits nothing.
        curator: The task graph; None runs without task persistence (bench and one-off runs).
        budget: The budget tracker the provider shares; None degrades the metric plateau rule
            to fixed counts.
        state_dir: The per-repo state dir holding the memory store; None runs memory-less.
        max_turns: The cap on turns for this execution (-1 unbounded); a resumed
            execution re-arms it.
        finish_validator: A machine state's finish contract, returning the problems with a
            finish_session payload; None leaves finishes ungated.
        bridge: What the operator can do to the run, as the front-end injects it.
        call: The worker call's knobs: retries, temperature, output caps.
        compaction: The compaction tiers' thresholds, the tail kept, the summariser.
        review: The in-loop review panel's trigger, reviewers and decision rule.
        initial_pins: Pins seeded before the first turn (a child inherits the parent's);
            fresh runs only, a resume restores pins from the snapshot.
        resume_state_path: Where the provider-agnostic resume snapshot is written before
            every model call; None writes none.
        merged_in: A branch merged into the run's chain before this execution, with the
            paths its conflict markers wait in; the model hears it at its first turn.
        operator_edits: The paths the operator changed between executions and left in the
            tree; a resumed execution names them to the model before its first call.
        standing_goal: The operator's `run --standing` goal, seeded under the root; "" is none.
        review_target: A `/review` session's comparison and its diff; the panel sits on it
            once as the session starts, so the Findings card reads the reviewed branch.
        base_rules: A review session's base-tree AGENTS.md, which binds it in place of the
            checkout's; None outside a review.
        interactive: An operator is watching and can steer live, so a quiet turn parks for a
            steer instead of ending.
        mode: The mode; `plan` uses the planning prompt and tool list, never auto-commits,
            and writes `plan_markdown` to `plan_output_path` on finish_planning.
        plan_output_path: Where a plan is written; required when `mode="plan"`.
        contract: The bounded conversation a `reviewer` session drives in place of a
            worker's: its own system prompt and opening, a call ceiling, no gates, no
            advisors, no compaction; None for the worker's execution.
        opening: The dispatch a fresh run makes in place of its first model call, joined by
            rank: `run --parallel`'s spec and task. The run ends when it lands. None for a
            run that opens with the model.
        turns_reached: The turn being driven, 0 before the loop starts; the app's
            interrupt fallbacks read it for a truthful `session.stop`.
    """

    chain: run_chain.RunChain
    config: Config
    provider: Provider
    dispatcher: dispatch.ToolDispatcher
    logger: Callable[[str], None] = dataclasses.field(default=print)
    events: event_log.EventSink | None = None
    curator: graph_curator.GraphCurator | None = None
    budget: agent6_budget.BudgetTracker | None = None
    state_dir: pathlib.Path | None = None
    max_turns: int = 200
    finish_validator: Callable[[dict[str, Any] | None], list[str]] | None = None
    bridge: operator_bridge.OperatorBridge = dataclasses.field(
        default_factory=operator_bridge.OperatorBridge
    )
    # The one gate to the operator: its queued questions and the presence it reads.
    prompts: operator_prompts.OperatorPrompts = dataclasses.field(
        default_factory=operator_prompts.OperatorPrompts
    )
    call: provider_call.CallSettings = dataclasses.field(default_factory=provider_call.CallSettings)
    compaction: compaction_window.CompactionSettings = dataclasses.field(
        default_factory=compaction_window.CompactionSettings
    )
    review: review_gate.ReviewSettings = dataclasses.field(
        default_factory=review_gate.ReviewSettings
    )
    media_input: media.InputCapabilities = dataclasses.field(
        default_factory=media.InputCapabilities
    )
    media_rejections: provider_call.MediaRejections = dataclasses.field(default_factory=dict)
    initial_pins: Sequence[str] = ()
    resume_state_path: pathlib.Path | None = None
    operator_edits: Sequence[str] = ()
    merged_in: git_ops.ChainMerge | None = None
    standing_goal: str = ""
    base_rules: str | None = None
    review_target: tuple[review_records.ReviewTarget, str] | None = None
    # The gate a resumed execution carried from the last one, set before the state is built.
    adopted_on_resume: tuple[str, ...] = ()
    interactive: bool = False
    mode: kinds.LoopMode = "run"
    plan_output_path: pathlib.Path | None = None
    contract: kinds.ConversationContract | None = None
    opening: directive.Segment | None = None
    # A snapshot write fault warns once, not every turn; it disables resume, never the run.
    snapshot_write_failed: bool = dataclasses.field(default=False, init=False)
    turns_reached: int = dataclasses.field(default=0, init=False)
    # What a contract's ceiling cut, for its completion record.
    _contract_diagnostics: list[str] = dataclasses.field(default_factory=list, init=False)
    _contract_response: ProviderResponse | None = dataclasses.field(default=None, init=False)
    _execution_started: bool = dataclasses.field(default=False, init=False)
    # The turn whose tools are running, for the `parallel` tool's dispatch.
    _tool_turn: tuple[loop_conversation.Conversation, loop_state.LoopState, int] | None = (
        dataclasses.field(default=None, init=False)
    )

    def __post_init__(self) -> None:
        """Install the loop-owned tool handlers on the dispatcher."""
        self.dispatcher.parallel_handler = self._parallel_tool

    # ---- run / resume entry --------------------------------------------------

    def run(
        self, user_task: str, *, operator_task: str | None = None
    ) -> loop_snapshot.SessionResult:
        """Drive a fresh execution of the loop to its end.

        Args:
            user_task: The task, with any seed digest or skill block prepended.
            operator_task: The words before composition; None means `user_task` is uncomposed.

        Returns:
            How the execution ended.

        Raises:
            ValueError: Plan mode with no `plan_output_path`.
        """
        self.bridge.steer_reset()  # an execution starts with no armed Ctrl-C
        if self.mode == "plan" and self.plan_output_path is None:
            raise ValueError("Harness(mode='plan') requires plan_output_path to be set")
        if operator_task is None:
            operator_task = user_task
        self._emit_start(
            journal.SessionStart(
                session_id=self.session_id,
                user_task=operator_task,
                mode=self.mode,
                execution_id=self.execution_id,
            )
        )
        system, conversation, root_id = (
            self._open_task(user_task)
            if self.contract is None
            else self._open_contract(self.contract)
        )
        return self._drive_loop(
            system=system,
            conversation=conversation,
            tool_calls=0,
            start_turn=1,
            root_task_id=root_id,
            original_task=user_task,
            operator_task=operator_task,
        )

    def _open_task(self, user_task: str) -> tuple[str, loop_conversation.Conversation, str | None]:
        """Open a worker's execution: the assembled system prompt, the task block, the root task.

        Args:
            user_task: The task, with any seed digest or skill block prepended.

        Returns:
            The system prompt, the conversation holding the task block, and the root task's
            id, None without a curator.
        """
        self.log("LOOP: LOAD_CONTEXT")
        self._prepare_rulings_file()
        repo = repo_context.load_repo_summary(self.chain.root, self.config.sandbox.hide_paths)
        if self.base_rules is not None:
            # The checkout is the reviewed head: its rules are evidence, the base's bind.
            repo = dataclasses.replace(repo, agents_md=self.base_rules)
        system = prompt_blocks.build_system_prompt(
            config=self.config,
            repo=repo,
            mode=self.mode,
            memory_index=self._load_memory_index(),
            memory_dir_path=str(agent6_memory.memory_dir(self.state_dir))
            if self.state_dir is not None
            else "",
            decisions=self._load_decisions(),
            skills=self._load_skills(),
            isolation=self.dispatcher.isolation,
            commands_allowed=self.dispatcher.command_policy() != "deny",
            protected_paths=bool(self.dispatcher.extra_protect_paths),
            dag_available=self.dispatcher.dag_available,
            container_roots=self._container_roots(),
        )

        # The root task is what `add_task` with `parent_id=None` attaches under.
        root_id = self.operator_tasks.seed_root(user_task)
        if root_id is not None:
            self.dispatcher.set_run_root_node_id(root_id)
            self.log(f"LOOP: DAG root task seeded: {root_id}")
            self.operator_tasks.seed_standing(root_id, self.standing_goal)
            self._emit_graph_snapshot()  # show the root in the live task view

        self.log(f"LOOP: mode={self.mode} system={len(system)} chars, task={len(user_task)} chars")

        dag_hint = dag_focus.initial_dag_hint(
            root_id, self.mode, self.config.harness.prompt.decompose == "on"
        )
        instructions = prompt_blocks.initial_instructions(
            self.mode,
            self.config.sandbox.run_commands,
            has_gate=self.gate.present(
                verify_verdict.VerifyVerdict()
            ),  # the start: nothing adopted or denied
        )
        initial_user = f"TASK:\n{user_task}\n\n{instructions}{dag_hint}"
        conversation = loop_conversation.Conversation()
        conversation.notice(initial_user, origin="operator")
        return system, conversation, root_id

    def _open_contract(
        self, contract: kinds.ConversationContract
    ) -> tuple[str, loop_conversation.Conversation, None]:
        """Open a bounded conversation: the contract's system prompt and opening, no task block.

        Args:
            contract: The contract.

        Returns:
            The system prompt, the conversation holding the opening, and no root task.
        """
        self.dispatcher.allowed_tools = contract.allowed_tools
        self.log(
            f"LOOP: contract max_turns={contract.max_turns}"
            f" tools={','.join(contract.allowed_tools) or 'none'}"
        )
        conversation = loop_conversation.Conversation()
        conversation.notice(contract.opening, origin="operator")
        return contract.system, conversation, None

    def _container_roots(self) -> tuple[str, str] | None:
        """Return the command and host roots when the commands run in a container, else None."""
        target = self.dispatcher.execution_target
        if target is None or target.transport != "container":
            return None
        return str(target.command_root), str(target.host_root)

    def _prepare_rulings_file(self) -> None:
        """Move a store's legacy rulings file out of the model's reach and ensure the new one.

        Every execution does this before its first turn: a resumed pre-upgrade session would
        otherwise leave `memory/DECISIONS.md` writable, and the next start would read what the
        model wrote there as the operator's rulings.
        """
        if self.state_dir is None:
            return
        if note := agent6_memory.migrate_legacy_decisions(self.state_dir):
            self.log(f"LOOP: {note}")
        agent6_memory.ensure_decisions_file(self.state_dir)

    def resume(self, *, steer: str = "") -> loop_snapshot.SessionResult:
        """Resume a session from its snapshot and drive the new execution to its end.

        The snapshot is the one written before the last model call; the root task id is
        reattached to the dispatcher and the loop re-enters at the saved turn with the
        saved conversation. The budget tracker is fresh per execution.

        Args:
            steer: The operator's steer that started this resumed execution.

        Returns:
            How the execution ended.

        Raises:
            ResumeError: No snapshot path, or a snapshot that cannot be read.
        """
        self.bridge.steer_reset()  # an execution starts with no armed Ctrl-C
        if self.resume_state_path is None:
            raise loop_snapshot.ResumeError("resume() called but resume_state_path is None")
        self._prepare_rulings_file()
        try:
            snapshot = loop_snapshot.load_session_snapshot(self.resume_state_path)
            snapshot, interrupted = loop_resume.fold_interrupted_turn(self, snapshot)
            conversation = loop_conversation.Conversation.from_wire(snapshot.messages)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise loop_snapshot.ResumeError(
                f"failed to load resume snapshot from {self.resume_state_path}: {exc}"
            ) from exc

        # The execution's log opens with this event, so it identifies itself like session.start.
        self._emit_start(
            journal.SessionResume(
                session_id=self.session_id,
                user_task=snapshot.operator_task
                or task_text.operator_task_text(snapshot.original_task),
                steer=steer,
                mode=self.mode,
                turn=snapshot.next_turn,
                messages=len(snapshot.messages),
                execution_id=self.execution_id,
            )
        )
        self.log(
            f"LOOP: RESUME from {self.resume_state_path} "
            f"(turn={snapshot.next_turn}, messages={len(snapshot.messages)}, "
            f"tool_calls={snapshot.tool_calls})"
        )
        if interrupted:
            self.log(
                f"LOOP: the last execution died in turn {snapshot.next_turn - 1};"
                f" not re-run: {', '.join(interrupted)}"
            )
            self.emit(
                journal.SessionResumeInterrupted(
                    turn=snapshot.next_turn - 1,
                    tools=list(interrupted),
                )
            )

        if snapshot.root_task_id is not None:
            self.dispatcher.set_run_root_node_id(snapshot.root_task_id)
            self.log(f"LOOP: DAG root task restored: {snapshot.root_task_id}")

        # The system prompt is frozen, so a gate that changed between executions is announced.
        self.adopted_on_resume = loop_resume.carry_adopted_gate(self, snapshot)
        gate = self.gate.configured or self.adopted_on_resume
        withheld = (
            not gate
            and bool(snapshot.verify_command)
            and self.dispatcher.command_policy() == "deny"
        )
        if gate != snapshot.verify_command and not withheld:
            was = " ".join(snapshot.verify_command) or "none"
            now = " ".join(gate) or "none"
            conversation.notice(
                f"[harness] This run's verify gate changed between executions: it was `{was}`,"
                f" it is now `{now}`. The instructions above still name the old one."
            )
            self.log(f"LOOP: verify gate swapped on resume: {was} -> {now}")
            self.emit(journal.VerifySwapped(was=list(snapshot.verify_command), now=list(gate)))

        if snapshot.verify_withheld and gate and self.dispatcher.command_policy() != "deny":
            # Denial resets per execution; its notice can survive even in a compacted summary.
            conversation.notice(
                "[harness verify] Resumed: run_verify_command is available in this execution."
                " Any earlier notice withholding the gate"
                " applied only to that execution. This is not a verify verdict."
            )
        if self.merged_in is not None:
            conversation.notice(loop_steer.merged_in_notice(self.merged_in))
        if self.operator_edits:
            # The model's last read of these files is stale; said before the steer that
            # started this execution, so the steer stays the operator's last word.
            shown = [terminal.printable_name(p) for p in self.operator_edits[:8]]
            more = len(self.operator_edits) - len(shown)
            names = ", ".join(shown) + (f", +{more} more" if more else "")
            conversation.notice(
                f"[harness] The operator changed {names} since the last execution and left the"
                " edits uncommitted in the working tree. Read them before editing them."
            )
            self.log(f"LOOP: operator edits kept in the tree: {names}")

        return self._drive_loop(
            system=snapshot.system,
            conversation=conversation,
            tool_calls=snapshot.tool_calls,
            start_turn=snapshot.next_turn,
            root_task_id=snapshot.root_task_id,
            original_task=snapshot.original_task,
            operator_task=snapshot.operator_task,
            resume_from=snapshot,
        )

    # ---- the turn pipeline ---------------------------------------------------

    def _sit_review_target(
        self, state: loop_state.LoopState, conversation: loop_conversation.Conversation
    ) -> None:
        """Sit the panel on a `/review` session's range before the worker reads it."""
        if self.review_target is None or not self.review_gate.available():
            return
        critique = self.review_gate.review_target(state, *self.review_target)
        if critique is not None:
            conversation.notice(review_panel.review_notice(critique.text))

    def _drive_loop(  # noqa: PLR0911, PLR0912
        self,
        *,
        system: str,
        conversation: loop_conversation.Conversation,
        tool_calls: int,
        start_turn: int,
        root_task_id: str | None,
        original_task: str,
        operator_task: str = "",
        resume_from: loop_snapshot.SessionSnapshot | None = None,
    ) -> loop_snapshot.SessionResult:
        """Drive the turns of one execution until a phase ends it.

        One `TurnState` per turn, through the phases in order; any phase returning a
        `SessionResult` ends the execution.

        Args:
            system: The system prompt, frozen for the session.
            conversation: The conversation so far.
            tool_calls: Tool calls already made in earlier executions.
            start_turn: The first turn of this execution.
            root_task_id: The root task the graph's nodes attach under.
            original_task: The task verbatim; the review panel grounds on it.
            operator_task: The operator's words before composition; "" when the same.
            resume_from: The snapshot a resumed execution restores; None for a fresh run.

        Returns:
            How the execution ended.
        """
        state = loop_state.LoopState(
            original_task=original_task,
            execution_id=self.execution_id,
            review_panels=review_history.load_panels(self.events.path) if self.events else (),
            operator_task=operator_task,
            tool_calls=tool_calls,
            root_task_id=root_task_id,
            system=system,
        )
        loop_resume.seed_carryover(self, state, conversation, resume_from)
        if resume_from is None:
            aborted = self._start_fresh(state, conversation, start_turn=start_turn)
            if aborted is not None:
                return aborted
        # The allowance is this execution's: a resumed one re-arms it (-1 is unbounded).
        for turn in (
            range(start_turn, start_turn + self.max_turns)
            if self.max_turns >= 0
            else itertools.count(start_turn)
        ):
            self.turns_reached = turn
            if turn == start_turn and self.contract is None:
                opened = loop_steer.execution_opening(
                    self, conversation, turn, state, resuming=resume_from is not None
                )
                if opened is not None:
                    return opened
            # Rebuilt per turn: a gate adopted or a policy denied mid-run changes the tool list.
            tools = toolset.tool_definitions(self.dispatcher, mode=self.mode)
            ctx = self._turn_context(state, turn=turn, execution_start=start_turn)
            wire = self._turn_pre_call(
                conversation=conversation,
                state=state,
                ctx=ctx,
                prefix_chars=compaction_window.request_prefix_chars(system, tools),
            )
            if isinstance(wire, loop_snapshot.SessionResult):
                return wire
            got = self._turn_provider_call(
                system,
                conversation,
                tools,
                state,
                turn=turn,
            )
            if isinstance(got, loop_snapshot.SessionResult):
                return got
            if isinstance(got, loop_state.NextTurn):
                continue
            # The response's blocks enter the history verbatim, so tool_use ids round-trip.
            assistant = conversation.assistant(got.raw.get("content") or [])
            ended = self._contract_turn(state, got, assistant, turn=turn)
            if ended is not None:
                return ended
            if not assistant.tool_uses:
                result = quiet_turns.handle_no_tool_use(
                    self, got, assistant, conversation, state, ctx
                )
                if result is not None:
                    return result
                # A prose-only turn also reaches the snapshot and the operator boundary.
                loop_resume.save_resume_snapshot(
                    self, state, conversation.to_wire(), next_turn=turn + 1
                )
                outcome = loop_steer.operator_boundary(self, conversation, turn, state)
                if outcome is not None:
                    return outcome
                continue
            turn_state = loop_state.TurnState(number=turn, resp=got, assistant=assistant)
            # The marker outlives a death mid-dispatch, so a resume never runs these calls again.
            loop_resume.mark_turn(self, turn_state)
            result = self._turn_dispatch_tools(state, turn_state, ctx)
            if result is not None:
                return result
            # One graph snapshot per turn, however many mutations the turn made.
            if turn_state.dag_mutated:
                self._emit_graph_snapshot()
            result = self._turn_after_tools(state, turn_state, ctx)
            if result is not None:
                return result
            conversation.results(turn_state.tool_results)
            # Snapshot, then clear the marker: a stale marker is cleared, never missed.
            loop_resume.save_resume_snapshot(
                self, state, conversation.to_wire(), next_turn=turn + 1
            )
            if self.resume_state_path is not None:
                loop_snapshot.clear_turn_marker(
                    self.resume_state_path.parent / loop_snapshot.TURN_IN_FLIGHT_NAME
                )
            result = self._turn_stop_checks(state, turn_state, conversation)
            if result is not None:
                return result
            outcome = loop_steer.operator_boundary(self, conversation, turn, state)
            if outcome is not None:
                return outcome

        self.log(f"LOOP: max_turns={self.max_turns} reached")
        return self.finish(
            state,
            loop_snapshot.End(
                "max_turns",
                f"max_turns={self.max_turns} reached without finish_session",
            ),
            turn=self.turns_reached,
        )

    def _start_fresh(
        self,
        state: loop_state.LoopState,
        conversation: loop_conversation.Conversation,
        *,
        start_turn: int,
    ) -> loop_snapshot.SessionResult | None:
        """Open a fresh execution: the worker's baseline, the stagnation window, the review target.

        Args:
            state: The execution's state.
            conversation: The conversation the baseline's notices enter.
            start_turn: The first turn.

        Returns:
            The end when the operator's verify command cannot execute, else None.
        """
        # A run that opens with a ranked dispatch leaves the baseline to its children.
        if self.contract is None and self.opening is None:
            try:
                for notice in self.gate.start_baseline(state):
                    conversation.notice(notice.text)
            except tool_errors.OperatorCommandUnexecutableError as exc:
                return self.unexecutable_abort(exc, turn=start_turn, state=state)
        # The start gate is the harness's time: the stagnation window opens at the first turn.
        state.stagnation = guards.StagnationGuard(wait_at_start=self.dispatcher.operator_wait_s)
        self._sit_review_target(state, conversation)
        return None

    def _turn_after_tools(
        self, state: loop_state.LoopState, turn: loop_state.TurnState, ctx: advice.TurnContext
    ) -> loop_snapshot.SessionResult | None:
        """Run the worker's phases after the turn's tools: commit and metric, review, gates, advice.

        A contract keeps the commit step, which run mode alone takes, and none of the others.

        Args:
            state: The execution's state.
            turn: This turn's state.
            ctx: This turn's context.

        Returns:
            The end a phase declared, else None.
        """
        result = self._turn_auto_commit_and_metric(state, turn)
        if result is not None or self.contract is not None:
            return result
        self.review_gate.triggers(state, turn)
        self._turn_finish_gates(state, turn, ctx)
        return self._turn_advisors(state, turn, ctx)

    def _turn_pre_call(
        self,
        *,
        conversation: loop_conversation.Conversation,
        state: loop_state.LoopState,
        ctx: advice.TurnContext,
        prefix_chars: int = 0,
    ) -> list[dict[str, Any]] | loop_snapshot.SessionResult:
        """Prepare the context for this turn's model call.

        Budget heartbeat, compaction, the plan re-read, the before-call advisors, the cache
        marks, then the pre-call snapshot, so the snapshot and the call carry the same wire.

        Args:
            conversation: The conversation to prepare.
            state: The execution's state.
            ctx: This turn's context.
            prefix_chars: The system prompt's size, counted against the context budget.

        Returns:
            The wire the model call sends, or the parked end when plan.md cannot be read.
        """
        # The turn the `parallel` tool dispatches for, set before its calls run.
        self._tool_turn = (conversation, state, ctx.turn)
        self._emit_budget(ctx.turn)
        # A restart keeps the first message, which is a contract's opening.
        restarted = self.compactor.compact(conversation, state, prefix_chars=prefix_chars)
        if self.contract is None:
            if restarted:
                # A tier-2 restart wiped the focus banner and the plan block; both go back below.
                state.focus.surfaced_task_id = None
                state.plan_injected = ""
            parked = self._maybe_inject_plan(conversation, state, turn=ctx.turn)
            if parked is not None:
                return parked
            self._turn_before_call(conversation, state, ctx)
        for call_id, reason in conversation.limit_attachments():
            self.emit(journal.MediaRefused(provider_tool_call_id=call_id, reason=reason))
        conversation.roll_cache_marks()
        wire = conversation.to_wire()
        # The one numbered-checkpoint writer: the state this turn's model call consumes.
        loop_resume.save_resume_snapshot(
            self, state, wire, next_turn=ctx.turn, write_checkpoint=True
        )
        return wire

    def _maybe_inject_plan(
        self,
        conversation: loop_conversation.Conversation,
        state: loop_state.LoopState,
        *,
        turn: int,
    ) -> loop_snapshot.SessionResult | None:
        """Show the planner the plan.md on disk when it changed.

        The file is the plan and `agent6 plan edit` writes the operator's answers to it between
        executions; an unreadable file parks the execution rather than run on stale direction.

        Args:
            conversation: The conversation the plan enters.
            state: The execution's state; remembers the last plan shown.
            turn: The current turn.

        Returns:
            The parked end when the file is unreadable, else None.
        """
        if self.mode != "plan" or self.plan_output_path is None:
            return None
        try:
            text = self.plan_output_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None  # no plan yet; the first finish_planning creates it
        except (OSError, UnicodeDecodeError) as exc:
            session_id = self.session_id or "<session-id>"
            remedy = f"plan.md unreadable: {exc}; fix it and `agent6 resume {session_id}`"
            self.log(f"LOOP: {remedy}")
            self.emit(journal.PlanReadFailed(path=str(self.plan_output_path), error=str(exc)))
            return self.finish(
                state,
                loop_snapshot.End("plan_unreadable", remedy, checkpoint=False),
                turn=turn,
            )
        if text == state.plan_injected:
            return None
        state.plan_injected = text
        conversation.notice(f"{nudges.PLAN_ON_DISK_HEADER}\n\n{text}")
        self.log(f"  plan re-read from disk: {len(text)} chars")
        self.emit(
            journal.PlanReread(
                path=str(self.plan_output_path),
                bytes=len(text.encode("utf-8")),
            )
        )

    def _turn_before_call(
        self,
        conversation: loop_conversation.Conversation,
        state: loop_state.LoopState,
        ctx: advice.TurnContext,
    ) -> None:
        """Add the operator's queued tasks, the focus banner, then the before-call advice.

        The advice comes last so a finish directive is the most recent message.

        Args:
            conversation: The conversation the notices enter.
            state: The execution's state.
            ctx: This turn's context.
        """
        self.operator_tasks.take(state.root_task_id)
        self._maybe_surface_current_task(conversation, state)
        for advisor in guards.BEFORE_CALL:
            self.tell(conversation, advisor(state, ctx))

    def _maybe_surface_current_task(
        self, conversation: loop_conversation.Conversation, state: loop_state.LoopState
    ) -> None:
        """Keep the worker on one task: advance the cursor and post the focus banner.

        The current task is the cursor while it points at an open subtask, else the first
        open subtask whose dependencies passed; the banner posts when the focus changes.
        Run mode only; a curator write that fails logs and continues.

        Args:
            conversation: The conversation the banner enters.
            state: The execution's state; remembers the surfaced task.
        """
        if self.mode != "run" or self.curator is None:
            return
        cursor = self.curator.cursor()
        nodes = self.curator.nodes()
        current_id = dag_focus.current_task_id(nodes, cursor)
        if current_id is None:
            state.focus.clear()
            return  # nothing decomposed yet, or the frontier is empty
        if cursor != current_id:
            # A passed cursor task drops out of the frontier, so this moves forward.
            try:
                self.curator.set_cursor(models.SetCursorIntent(id=current_id))
            except (
                graph_curator.CuratorError,
                OSError,
                pydantic.ValidationError,
            ) as exc:  # advisory; never fatal
                self.log(f"LOOP: cursor advance skipped: {exc}")
        self.tell(conversation, guards.stuck_on_task(state, current_id, nodes[current_id], nodes))
        if current_id == state.focus.surfaced_task_id:
            return  # already surfaced; the banner survives tier-1 elision
        node = nodes[current_id]
        if node.status == "pending":
            # Best-effort: the graph shows the task as worked.
            try:
                self.curator.update_status(
                    models.UpdateStatusIntent(id=current_id, new_status="in_progress")
                )
            except (graph_curator.CuratorError, OSError, pydantic.ValidationError) as exc:
                self.log(f"LOOP: mark in_progress skipped: {exc}")
        banner = dag_focus.current_task_banner(
            current_id, node, decompose=self.config.harness.prompt.decompose == "on"
        )
        conversation.notice(banner)
        state.focus.surfaced_task_id = current_id
        self.log(f"LOOP: surfaced current task {current_id}")
        self.emit(journal.TaskSurfaced(task_id=current_id))
        # The cursor and status writes above bypass dispatch, which emits graph.update.
        self._emit_graph_snapshot()

    def _turn_provider_call(
        self,
        system: str,
        conversation: loop_conversation.Conversation,
        tools: list[ToolDefinition],
        state: loop_state.LoopState,
        *,
        turn: int,
    ) -> loop_snapshot.SessionResult | loop_state.NextTurn | ProviderResponse:
        """Make the turn's model call and classify a terminal error.

        Args:
            system: The system prompt.
            conversation: The canonical history, updated on a steer or a media refusal.
            tools: The tools offered this turn.
            state: The execution's state.
            turn: The current turn.

        Returns:
            The response, the end of the execution, or `NEXT_TURN` when a mid-stream
            steer discarded the turn.
        """

        def _logical_call(messages: list[dict[str, Any]]) -> ProviderResponse:
            # One identity per logical call: its attempts share it, a media-free retry is a
            # new call, and the turn's tool rows name the call that returned.
            binding = self.contract.review_binding if self.contract is not None else None
            state.call_context = kinds.CallContext(
                execution_id=self.execution_id,
                panel_id=binding.panel_id if binding else None,
                reviewer_index=binding.reviewer_index if binding else None,
            )
            return self.caller.call(
                system,
                messages,
                tools,
                self._worker_max_tokens(state),
                initiator=conversation.initiator,
                call_context=state.call_context,
            )

        try:
            if self.budget is not None:
                # A child's settled spend stops the parent before its next call, not after it.
                self.budget.check()
            return provider_call.call_with_media_recovery(
                _logical_call,
                conversation,
                provider_call.MediaRecovery(
                    self.media_input,
                    self.media_rejections.setdefault(self.media_input.route, set()),
                ),
                emit=self.emit,
                changed=lambda: loop_resume.save_resume_snapshot(
                    self,
                    state,
                    conversation.to_wire(),
                    next_turn=turn,
                ),
            )
        except agent6_budget.BudgetExceededError as exc:
            self.log(f"LOOP: budget exhausted at turn {turn}")
            return self.finish(
                state,
                loop_snapshot.End("budget_exhausted", f"{exc} (at turn {turn})"),
                turn=turn,
            )
        except ProviderAborted:
            self.bridge.steer_clear()  # consume the stop; don't leave it on disk to re-read
            self.log(f"LOOP: operator stop during turn {turn}; model call was cut short")
            return self.finish(
                state,
                loop_snapshot.End(
                    "steer_abort",
                    f"operator stopped the run during turn {turn}; model call was cut short"
                    f"{self.dirty_tree_note()}",
                    checkpoint=False,
                ),
                turn=turn,
            )
        except ProviderInterrupted:
            # The watchdog ended the turn for a steer; the partial turn is discarded.
            self.log(f"LOOP: steer requested mid-turn at turn {turn}")
            outcome = loop_steer.steer_outcome(
                self, self.steering.handle(conversation, turn, state), turn, state
            )
            if outcome is not None:
                return outcome
            return loop_state.NEXT_TURN  # "continue" or an injected instruction -> re-do the turn
        except ProviderError as exc:
            if self.contract is not None:
                self._contract_diagnostics.append(f"provider: {exc}")
            hint = provider_call.provider_error_hint(exc.status_code, exc.provider)
            attempts = f" after {exc.attempts} attempts" if exc.attempts > 1 else ""
            # The upstream body lands in this one log line; the summary below stays short.
            self.log(f"LOOP: provider error{attempts} at turn {turn}: {exc}{hint}")
            status = f" (HTTP {exc.status_code})" if exc.status_code else ""
            # A fatal error or a transport failure has no other reason to name.
            detail = f": {exc}" if exc.fatal or exc.status_code is None else ""
            return self.finish(
                state,
                loop_snapshot.End(
                    "provider_error",
                    f"provider error{attempts} at turn {turn}{status}{hint}{detail}",
                ),
                turn=turn,
            )

    def _contract_turn(
        self,
        state: loop_state.LoopState,
        resp: ProviderResponse,
        assistant: loop_conversation.AssistantMessage,
        *,
        turn: int,
    ) -> loop_snapshot.SessionResult | None:
        """End a bounded conversation on a response with no tool call, or at its turn limit.

        The response's text is the deliverable either way. A response with no tool call is an
        answer, on the last allowed turn too; the tool calls of a response at the limit are
        not executed, and the completion record names them.

        Args:
            state: The execution's state.
            resp: The model's response.
            assistant: The turn as it entered the conversation.
            turn: The current turn.

        Returns:
            The end, or None when the turn's tool calls run or no contract drives the loop.
        """
        if self.contract is None:
            return None
        self._contract_response = resp
        text = (resp.text or "").strip()
        if not assistant.tool_uses:
            self.log(f"LOOP: answered at turn {turn}")
            return self.finish(
                state,
                loop_snapshot.End(
                    "answered", text, completed=True, verdict="passed", checkpoint=False
                ),
                turn=turn,
            )
        if turn >= self.contract.max_turns:
            names = ", ".join(tu.name for tu in assistant.tool_uses)
            self._contract_diagnostics.append(
                f"not executed, the turn limit {self.contract.max_turns} was reached: {names}"
            )
            self.log(f"LOOP: max_turns={self.contract.max_turns} reached")
            return self.finish(
                state, loop_snapshot.End("max_turns", text, checkpoint=False), turn=turn
            )
        return None

    def _worker_max_tokens(self, state: loop_state.LoopState) -> int:
        """Return the output cap for the worker's call.

        A metric run lifts the cap to `metric_task_max_tokens` so one turn can rewrite a
        hot function whole; two consecutive quiet turns drop it back, since a model that
        spends the whole budget on reasoning otherwise repeats the binge every nudge.

        Args:
            state: The execution's state; counts the quiet turns.

        Returns:
            The cap in tokens.
        """
        if (
            self.metrics.active
            and state.quiet.went_quiet_nudges_used < _STARVATION_BACKOFF_AFTER_QUIETS
        ):
            return max(self.call.per_call_max_tokens, self.call.metric_task_max_tokens)
        return self.call.per_call_max_tokens

    def _turn_dispatch_tools(
        self, state: loop_state.LoopState, turn: loop_state.TurnState, ctx: advice.TurnContext
    ) -> loop_snapshot.SessionResult | None:
        """Dispatch the turn's tool calls, then the harness's own gate run when one is due.

        Each call appends one tool result and notes its effects on `turn`; a tool error is
        an error result the model recovers from.

        Args:
            state: The execution's state.
            turn: This turn's state.
            ctx: This turn's context.

        Returns:
            The end when the operator's command cannot execute, else None.
        """  # noqa: DOC501  # the refusal's ToolError is caught below
        # A turn with tool calls refills the went-quiet nudge budget.
        state.quiet.went_quiet_nudges_used = 0
        media_budget = media.AttachmentBudget()
        for tu in turn.assistant.tool_uses:
            name, tool_input = tu.name, tu.input
            if turn.finish is not None:
                # The calls after a finish are not executed, as the finish tools state.
                turn.tool_results.append(
                    loop_conversation.ToolResultItem(
                        tool_use_id=tu.id,
                        content=json.dumps(
                            {"error": f"{name} not executed: it follows {turn.finish.kind}"}
                        ),
                        for_call=tu,
                    )
                )
                loop_resume.mark_turn(self, turn)
                continue
            state.tool_calls += 1
            # The call's signature for the spiral guard; stable JSON so key order cannot differ.
            try:
                sig = f"{name}:{json.dumps(tool_input, sort_keys=True, ensure_ascii=False)}"
            except (TypeError, ValueError):
                sig = f"{name}:<unhashable>"
            state.spiral.note_call(sig, polling=name == schema.ReadBackgroundInput.TOOL_NAME)
            served = None
            attachment = None
            try:
                refusal = turn.resp.refused.get(tu.id)
                if refusal is not None:
                    # The provider's front-end refused the input; the same error is the result.
                    raise tool_errors.ToolError(refusal)
                tree_before = tool_effects.tree_before_command(self, name)
                result = self.dispatcher.dispatch(
                    name,
                    tool_input,
                    origin=(
                        kinds.ToolCallOrigin(state.call_context, tu.id, turn.number)
                        if state.call_context is not None
                        else None
                    ),
                )
                attachment = result.attachment
                if attachment is not None and (reason := media_budget.refusal(attachment)):
                    attachment = None
                    self.emit(journal.MediaRefused(provider_tool_call_id=tu.id, reason=reason))
                    raise tool_errors.ToolError(reason)
                wire_result = result.to_wire()
                if note := tool_effects.left_out_note(self, turn, name, result, tool_input):
                    wire_result["note"] = note
                content = json.dumps(wire_result, ensure_ascii=False)
                tool_effects.note(
                    self, state, turn, name, result, tool_input, tree_before=tree_before
                )
                # A repeated call with an unchanged result serves a stub, so a re-read stays small.
                if attachment is None and state.spiral.stub_repeat(
                    content, min_chars=_DEDUPE_MIN_CHARS
                ):
                    served = json.dumps(
                        {
                            "repeated": (
                                f"Identical to your previous {name} call --"
                                f" result unchanged ({len(content.encode())} bytes elided)."
                                " Do not re-issue the same call; if you need"
                                " different data, change the arguments, otherwise"
                                " act on what you already have."
                            )
                        }
                    )
                state.spiral.note_success(content)
                # A finish call is not work: counting it would reset the standing fruitless streak.
                if name not in ("finish_session", "finish_planning"):
                    state.ok_tool_calls += 1
                self._take(
                    state, turn, ctx, guards.unreachable_tool(state, name, tool_input, result)
                )
                # Only a dispatched finish counts; a refused one is an error result.
                tool_effects.capture_finish(self, state, turn, name, tool_input)
            except tool_errors.ToolError as exc:
                content = tool_effects.note_tool_error(self, state, name, tool_input, exc)
                self._take(state, turn, ctx, guards.tool_error_ladder(turn, state, ctx))
            except tool_errors.OperatorCommandUnexecutableError as exc:
                return self.unexecutable_abort(exc, turn=turn.number, state=state)
            turn.tool_results.append(
                loop_conversation.ToolResultItem(
                    tool_use_id=tu.id,
                    content=content
                    if attachment is not None
                    else compaction_elision.cap_tool_result(
                        served if served is not None else content,
                        tool_name=name,
                        cap=self.compaction.tool_result_cap_bytes,
                    ),
                    for_call=tu,
                    attachment=attachment,
                )
            )
            loop_resume.mark_turn(self, turn)
        # The gate run the harness adds to the turn, after the model's calls.
        try:
            self.gate.harness_verify(state, turn)
        except tool_errors.OperatorCommandUnexecutableError as exc:
            return self.unexecutable_abort(exc, turn=turn.number, state=state)
        return None

    def _turn_auto_commit_and_metric(
        self, state: loop_state.LoopState, turn: loop_state.TurnState
    ) -> loop_snapshot.SessionResult | None:
        """Commit the turn's work, then take the automatic metric sample.

        A step the gate judged green commits as verified; any other step that changed the
        tree commits as a checkpoint. Plan mode never commits; a failed commit logs and
        the execution goes on.

        Args:
            state: The execution's state.
            turn: This turn's state.

        Returns:
            The end when the operator stops at the commit hook or the metric command cannot
            execute, else None.
        """
        # A green verify with no edit after it makes a verified step; any other editing turn
        # makes a checkpoint, whatever the gate policy, so every turn stays a point to cut at.
        verified_commit = turn.verify_just_passed and not turn.edit_since_verify_pass
        unjudged_changed = not verified_commit and (turn.edited or self.chain.dirty())
        if self.mode != "run" or not (verified_commit or unjudged_changed):
            return None
        if unjudged_changed:
            # The idle-stop net needs this where no green verify fires per step.
            state.settled.gateless_ever_edited = True
        if not self.chain.per_turn:
            # `commit_per_turn` governs the commit only; the metric block is promised regardless.
            return self._sample_metric(state, turn, sha="")
        fallback = "checkpoint" if unjudged_changed else "verify passed"
        commit_subject = fallback
        sha = ""
        try:
            # The subject reads git too (the chain's delta): a fault there is a failed commit.
            commit_subject = self.checkpoints.subject(turn, fallback=fallback)
            sha = self.checkpoints.commit(
                commit_subject, turn=turn.number, task_id=state.focus.surfaced_task_id
            )
            turn.committed = bool(sha)
            # Adoption fills an absent command; a configured gate nobody may run stays as is.
            if sha and not self.gate.command(state.verify) and self.gate.may_run():
                self.gate.maybe_adopt(state, turn)
        except (errors.GitError, OSError) as exc:
            self.checkpoints.report_failure(exc, commit_subject, turn=turn.number)
        # The operator's after-commit hook; the default answers "continue".
        if sha:
            directive = self.bridge.after_auto_commit(turn.number, sha)
            if directive in ("undo", "exit"):
                ended = loop_steer.steer_outcome(self, directive, turn.number, state)
                if ended is not None:
                    return ended
            if directive == "stop":
                self.log(f"LOOP: interactive stop at turn {turn.number}")
                # An operator stop reads "stopped", never "passed".
                return self.finish(
                    state,
                    loop_snapshot.End(
                        "interactive_stop",
                        f"stopped interactively after turn {turn.number}{self.dirty_tree_note()}",
                        completed=True,
                        checkpoint=False,
                        roots=True,
                    ),
                    turn=turn.number,
                )
        return self._sample_metric(state, turn, sha=sha)

    def _sample_metric(
        self, state: loop_state.LoopState, turn: loop_state.TurnState, *, sha: str
    ) -> loop_snapshot.SessionResult | None:
        """Run the configured metric over the step and hand the model the reading.

        One reading per state of the tree, and none when the model ran the metric itself.

        Args:
            state: The execution's state.
            turn: This turn's state.
            sha: The step's commit, or "" when nothing committed.

        Returns:
            The end when the operator's metric command cannot execute, else None.
        """
        if turn.metric_sampled or state.metric.denied:
            return None
        tree = self.chain.tree_sha()
        if tree and tree == state.metric.tree:
            return None
        state.metric.tree = tree
        # The same abort as the per-tool handler's.
        try:
            turn.metric_feedback = self.metrics.auto_feedback(state, turn=turn.number, sha=sha)
        except tool_errors.OperatorCommandUnexecutableError as exc:
            return self.unexecutable_abort(exc, turn=turn.number, state=state)
        turn.metric_plateau_finish = self.metrics.plateau_finish(state.metric.history)
        return None

    def _turn_notices(self, state: loop_state.LoopState, turn: loop_state.TurnState) -> None:
        """Append the turn's review findings and metric feedback ahead of the advisors' notices.

        Args:
            state: The execution's state.
            turn: This turn's state.
        """
        if turn.review_text:
            turn.tool_results.append(
                loop_conversation.Notice(review_panel.review_notice(turn.review_text))
            )
            turn.review_text = None
        if turn.metric_feedback:
            turn.tool_results.append(loop_conversation.Notice(turn.metric_feedback))

    # ---- finish gates --------------------------------------------------------

    def _turn_finish_gates(
        self, state: loop_state.LoopState, turn: loop_state.TurnState, ctx: advice.TurnContext
    ) -> None:
        """Run a finish_session through the finish gates; the first refusal revokes it.

        Args:
            state: The execution's state.
            turn: This turn's state.
            ctx: This turn's context.
        """
        if turn.finish is None or turn.finish.kind != "finish_session":
            return
        turn.ending = "finish_session"
        for gate in finish_gates.FINISH_GATES:
            if self._refuse(state, turn, gate(turn, state, ctx)):
                return

    def _refuse(
        self,
        state: loop_state.LoopState,
        turn: loop_state.TurnState,
        refusal: advice.Refusal | None,
    ) -> bool:
        """Apply a gate's refusal of the turn's end.

        The finish is revoked, the model gets the refusal's text, and the settle streak
        starts over.

        Args:
            state: The execution's state.
            turn: This turn's state.
            refusal: The gate's answer; None when it let the end through.

        Returns:
            True when the end was refused.
        """
        if refusal is None:
            return False
        turn.finish = None
        turn.end_returned = True
        if refusal.text:
            turn.tool_results.append(loop_conversation.Notice(refusal.text))
        self._record(refusal)
        state.settled.restart()
        return True

    def end_gates(
        self,
        state: loop_state.LoopState,
        turn: loop_state.TurnState,
        ctx: advice.TurnContext,
        *,
        ending: str,
        gates: tuple[advice.Gate, ...],
    ) -> loop_snapshot.SessionResult | None:
        """Run an end declared without finish_session through the given gates.

        The harness gate runs first on the ending turn; the first refusal hands the end
        back with its reason.

        Args:
            state: The execution's state.
            turn: This turn's state.
            ctx: This turn's context.
            ending: The end declared, `settled` or `silent_finish`.
            gates: The gates the end must pass.

        Returns:
            The end when the operator's command cannot execute, else None.
        """
        try:
            self.gate.harness_verify(state, turn, ending=True)
        except tool_errors.OperatorCommandUnexecutableError as exc:
            return self.unexecutable_abort(exc, turn=turn.number, state=state)
        turn.ending = ending
        for gate in gates:
            if self._refuse(state, turn, gate(turn, state, ctx)):
                break
        if turn.review_text:
            # The turn's notices went out before these gates, so the panel's findings go here.
            turn.tool_results.append(
                loop_conversation.Notice(review_panel.review_notice(turn.review_text))
            )
            turn.review_text = None
        return None

    # ---- the advisors --------------------------------------------------------

    def _turn_context(
        self, state: loop_state.LoopState, *, turn: int, execution_start: int
    ) -> advice.TurnContext:
        """Build the facts the advisors read this turn.

        Args:
            state: The execution's state.
            turn: The current turn.
            execution_start: The first turn of this execution.

        Returns:
            This turn's context.
        """
        return advice.TurnContext(
            mode=self.mode,
            turn=turn,
            execution_start=execution_start,
            went_quiet_max_nudges=self.config.harness.went_quiet_max_nudges,
            loop_guard_kill_threshold=self.config.harness.loop_guard_kill_threshold,
            settled_stop_after=self.config.harness.settled_stop_after,
            stagnation_notice_after_s=self.config.harness.stagnation_notice_after_s,
            verify_when=self.gate.when,
            verify_retries=self.gate.retries,
            finish_validator=self.finish_validator,
            metric=self.metrics.goal is not None,
            memory_advisory=self.state_dir is not None and self.config.harness.memory_advisory,
            end_rejected=lambda turn, ending: self.review_gate.end_rejected(
                state, turn, ending=ending
            ),
            standing_absorb=lambda reason, turn: self.standing.absorb(
                state, reason=reason, turn=turn
            ),
            gate_present=lambda: self.gate.present(state.verify),
            verify_command=lambda: self.gate.command(state.verify),
            tree_sha=self.chain.tree_sha,
            head_sha=self.chain.head_sha,
            tree_green=lambda: self.gate.tree_green(state.verify),
            budget_remaining=self._budget_fraction_remaining,
            operator_wait_s=lambda: self.dispatcher.operator_wait_s,
            open_subtasks=self.open_subtasks,
        )

    def _budget_fraction_remaining(self) -> float | None:
        """Return the fraction of the token budget left, or None without a tracker."""
        if self.budget is None:
            return None
        return self.budget.fraction_remaining()

    def _turn_advisors(
        self, state: loop_state.LoopState, turn: loop_state.TurnState, ctx: advice.TurnContext
    ) -> loop_snapshot.SessionResult | None:
        """Post the turn's notices, then apply each after-tools advisor's answer.

        Args:
            state: The execution's state.
            turn: This turn's state.
            ctx: This turn's context.

        Returns:
            The end when a gate's verify could not run, else None.
        """
        self._turn_notices(state, turn)
        for advisor in guards.AFTER_TOOLS:
            aborted = self._take(state, turn, ctx, advisor(turn, state, ctx))
            if aborted is not None:
                return aborted
        return None

    def _take(
        self,
        state: loop_state.LoopState,
        turn: loop_state.TurnState,
        ctx: advice.TurnContext,
        outcome: advice.Nudge | advice.Stop | None,
    ) -> loop_snapshot.SessionResult | None:
        """Apply one after-tools advisor's answer.

        A nudge joins the turn's results; a stop joins the stop checks, after the end
        gates when it declares an ending.

        Args:
            state: The execution's state.
            turn: This turn's state.
            ctx: This turn's context.
            outcome: The advisor's answer; None when it had none.

        Returns:
            The end when the gates' verify could not run, else None.
        """
        if outcome is None:
            return None
        if isinstance(outcome, advice.Nudge):
            turn.tool_results.append(loop_conversation.Notice(outcome.text))
            self._record(outcome)
            return None
        if outcome.event is not None:
            self.emit(outcome.event)
        if outcome.declared and turn.finish is None:
            aborted = self.end_gates(
                state, turn, ctx, ending=outcome.declared, gates=finish_gates.END_GATES
            )
            if aborted is not None:
                return aborted
            if turn.end_returned:
                return None
        turn.stops.append(outcome)
        return None

    def tell(
        self, conversation: loop_conversation.Conversation, nudge: advice.Nudge | None
    ) -> None:
        """Apply a before-call advisor's answer: the notice joins the conversation.

        Args:
            conversation: The conversation the notice enters.
            nudge: The advisor's answer; None when it had none.
        """
        if nudge is None:
            return
        conversation.notice(nudge.text)
        self._record(nudge)

    def _record(self, answer: advice.Nudge) -> None:
        """Emit an answer's event and log its line, each when present.

        Args:
            answer: The advisor's or gate's answer.
        """
        if answer.event is not None:
            self.emit(answer.event)
        if answer.log:
            self.log(answer.log)

    # ---- stop checks, silent finish, went-quiet ------------------------------

    def _turn_stop_checks(
        self,
        state: loop_state.LoopState,
        turn: loop_state.TurnState,
        conversation: loop_conversation.Conversation,
    ) -> loop_snapshot.SessionResult | None:
        """End the execution on an advisor's stop, else on a finish that passed the gates.

        Runs after the turn's tool results are in the conversation and snapshotted.

        Args:
            state: The execution's state.
            turn: This turn's state.
            conversation: The conversation so far.

        Returns:
            The end, or None when the execution goes on.
        """
        self.standing.absorb_soft_stop(state, turn, conversation)
        for stop in turn.stops:
            if stop.log:
                self.log(stop.log)
            return self.finish(state, stop.end(), turn=turn.number)
        finish = turn.finish
        if finish is not None:
            if finish.kind == "finish_session" and loop_steer.operator_state(
                self, conversation, state, boundary=False
            ):
                turn.finish = None
                state.settled.restart()
                conversation.notice(
                    "[harness] finish_session deferred: an operator answer arrived."
                    " Read it before finishing."
                )
                return None
            self.log(f"LOOP: {finish.kind} called at turn {turn.number}")
            self.checkpoints.final(turn=turn.number, task_id=state.focus.surfaced_task_id)
            # A finish_session over a red or stale verify reads "finished", not "passed".
            reason = finish_gates.finish_reason(
                finish.kind,
                stale_gate=finish.stale_gate,
                tree_green=self.gate.tree_green(state.verify),
                verify=state.verify,
            )
            self._check_decisions_recorded(state)
            return self.finish(
                state,
                loop_snapshot.End(
                    reason,
                    advice.with_open_tasks(finish.summary, self.open_subtasks()),
                    completed=True,
                    verdict="grounded" if finish.kind == "finish_session" else "passed",
                    checkpoint=False,
                    finish_payload=finish.payload,
                ),
                turn=turn.number,
            )
        return None

    # ---- the end -------------------------------------------------------------

    def finish(
        self, state: loop_state.LoopState, end: loop_snapshot.End, *, turn: int
    ) -> loop_snapshot.SessionResult:
        """Record the end and return the execution's result.

        Every stop carries the gate's word for the final tree.

        Args:
            state: The execution's state.
            end: The end declared.
            turn: The turn it ended on.

        Returns:
            The execution's result.
        """
        if end.checkpoint:
            self.checkpoints.final(turn=turn, task_id=state.focus.surfaced_task_id)
        # An end the loop declares is no death: the next resume re-enters at the snapshot.
        if self.resume_state_path is not None:
            loop_snapshot.clear_turn_marker(
                self.resume_state_path.parent / loop_snapshot.TURN_IN_FLIGHT_NAME
            )
        roots = end.roots if end.roots is not None else end.verdict != "failed"
        if roots:
            self._pass_pending_root_tasks()
        verification = self.gate.verification(state.verify)
        if end.reason == "settled" and verification == "passed":
            # The base's green is not the run's pass: the settled guard says so in its reason.
            verification = "unverified"
        # The tree a run leaves: at its next start, what differs from it changed outside the run.
        tree = self.chain.tree_sha() if end.event and self.mode == "run" else ""
        left: dict[str, Any] = {"final_verify_id": state.verify.final_verify_id}
        if tree:
            left["tree"] = tree
        if end.event:
            self.emit(
                journal.SessionStop(
                    execution_id=self.execution_id,
                    reason=end.reason,
                    summary=end.summary,
                    turns=turn,
                    verification=verification,
                    scoped=state.verify.scoped if end.verdict == "grounded" else end.scoped,
                    **left,
                    **end.fields,
                )
            )
        self._record_memory_use(state)
        if self.contract is not None and self.resume_state_path is not None:
            completion.write_completion(
                self.resume_state_path.parent,
                completion.CompletionRecord(
                    text=end.summary,
                    stop_reason=end.reason,
                    output_tokens=self.caller.counter.output_tokens,
                    calls=self.caller.counter.attempts,
                    diagnostics=tuple(self._contract_diagnostics),
                    session_id=self.session_id,
                    review_binding=self.contract.review_binding,
                    response_stop_reason=self._contract_response.stop_reason
                    if self._contract_response
                    else "",
                    response_output_tokens=self._contract_response.output_tokens
                    if self._contract_response
                    else 0,
                    reasoning_chars=sum(
                        len(str(b.get("thinking") or ""))
                        for b in (self._contract_response.raw.get("content") or [])
                        if isinstance(b, dict) and b.get("type") == "thinking"
                    )
                    if self._contract_response
                    else 0,
                    asked_for_tools=bool(
                        self._contract_response and self._contract_response.tool_uses
                    ),
                ),
            )
        return loop_snapshot.SessionResult(
            completed=end.completed,
            verification=verification,
            reason=end.reason,
            summary=end.summary,
            turns=turn,
            tool_calls=state.tool_calls,
            finish_payload=end.finish_payload,
            stale_gate=state.gates.stale_gate,
            final_verify_id=state.verify.final_verify_id,
        )

    def _pass_pending_root_tasks(self) -> None:
        """Mark the open root tasks passed at a completed end.

        The worker finishes without touching the root it was seeded, so a completed
        execution would otherwise read `tasks 0/1`; open subtasks stay as they are.
        """
        if self.curator is None:
            return
        changed = False
        for nid, node in self.curator.nodes().items():
            if node.parent_id is None and node.status in order.OPEN_STATUSES:
                try:
                    self.curator.update_status(
                        models.UpdateStatusIntent(id=nid, new_status="passed")
                    )
                    changed = True
                except graph_curator.CuratorError as exc:  # this root refused; the next may not
                    self.log(f"LOOP: auto-pass root {nid} refused: {exc}")
                except (
                    OSError,
                    pydantic.ValidationError,
                ) as exc:  # a write fault must not break finish
                    self.log(f"LOOP: auto-pass root {nid} failed: {exc}")
                    break  # a curator write failure fails for every remaining node too
        if changed:
            self._emit_graph_snapshot()

    def _record_memory_use(self, state: loop_state.LoopState) -> None:
        """Persist the memory facts this execution wrote and read; a write fault logs.

        Args:
            state: The execution's state.
        """
        memory = state.memory
        if self.state_dir is None or not (memory.wrote or memory.read or memory.deleted):
            return
        try:
            agent6_memory.record_use(
                self.state_dir,
                session=self.session_id or "?",
                wrote=tuple(state.memory.wrote),
                created=tuple(state.memory.created),
                deleted=tuple(state.memory.deleted),
                read=dict(state.memory.read),
            )
        except OSError as exc:
            self.log(f"LOOP: memory use record failed: {exc}")

    def unexecutable_abort(
        self,
        exc: tool_errors.OperatorCommandUnexecutableError,
        *,
        turn: int,
        state: loop_state.LoopState,
    ) -> loop_snapshot.SessionResult:
        """End the execution when the operator's verify or metric command cannot run.

        The model cannot fix operator config, so the end is loud rather than a gate that
        never executes.

        Args:
            exc: The error naming the command.
            turn: The current turn.
            state: The execution's state.

        Returns:
            The execution's result.
        """
        self.log(f"LOOP: aborting -- {exc}")
        # Verify never went green, so the edits may exist only in the worktree.
        return self.finish(
            state, loop_snapshot.End("verify_command_unexecutable", str(exc)), turn=turn
        )

    def dirty_tree_note(self) -> str:
        """Return the summary suffix naming an uncommitted worktree, or "" outside run mode."""
        return self.chain.dirty_note() if self.mode == "run" else ""

    def verify_fact(self, state: loop_state.LoopState) -> str:
        """Return the verify fact the worker reads at a resume or a restart.

        Args:
            state: The execution's state.

        Returns:
            `VerifyVerdict.fact`, or "" when no gate can judge the run and none has.
        """
        verdict = state.verify
        if not (self.gate.present(verdict) or verdict.verify_tree or verdict.last_ok is not None):
            return ""
        return verdict.fact()

    # ---- the task graph ------------------------------------------------------

    def _emit_graph_snapshot(self) -> None:
        """Emit the task graph for the live viewers.

        Each node projects to the six fields the viewers render; a full dump would carry
        unbounded model text into the event log.
        """
        if self.curator is None:
            return
        cursor = self.curator.cursor()
        # A frozen wire shape, pinned by test_graph_update_snapshot_payload_is_wire_stable.
        nodes = {
            nid: {
                "title": n.title,
                "status": n.status,
                "parent_id": n.parent_id,
                "children": list(n.children),
                "created_by": n.created_by,
                "standing": n.standing,
            }
            for nid, n in self.curator.nodes().items()
        }
        self.emit(journal.GraphUpdate(nodes=nodes, cursor=cursor))

    def open_subtasks(self) -> list[tuple[str, str]]:
        """List the worker's open subtasks as (id, title) pairs, in run mode.

        The root stays open until the end, so counting it would deadlock every gate; a
        plan's tasks are its deliverable, open by design.

        Returns:
            The open subtasks; empty without a curator or outside run mode.
        """
        if self.curator is None or self.mode != "run":
            return []
        return advice.open_subtasks(self.curator.nodes())

    def checkpoint_graph_version(self) -> int:
        """Return the graph version for the per-turn checkpoint, 0 without a curator."""
        if self.curator is None:
            return 0
        return self.curator.graph_version

    # ---- the state dir: memory, decisions, skills ----------------------------

    def _load_memory_index(self) -> str:
        """Return the repo memory index for the system prompt.

        Returns:
            The index text; "" without a state dir, in read_only mode, or when unreadable.
        """
        if self.state_dir is None or self.mode == "read_only":
            return ""
        return agent6_memory.index_text(self.state_dir)

    def _load_decisions(self) -> str:
        """Return the operator's recorded rulings for the prompt, "" without a state dir."""
        return agent6_memory.decisions_text(self.state_dir) if self.state_dir is not None else ""

    def record_decision(self, state: loop_state.LoopState, question: str, answer: str) -> None:
        """Append an operator's answer to the repo's decisions.md as a ruling.

        Args:
            state: The execution's state; remembers the entry for the finish-time check.
            question: The question asked.
            answer: The operator's answer.
        """
        if self.state_dir is None or not answer.strip():
            return
        try:
            entry = agent6_memory.record_decision(
                self.state_dir, question=question, answer=answer, session=self.session_id
            )
        except OSError as exc:
            self.log(f"LOOP: decision not recorded: {exc}")
            self.emit(journal.DecisionUnrecorded(error=str(exc)))
            return
        state.decisions_recorded.append(entry)
        self.log(f"LOOP: decision recorded ({len(answer)} chars)")
        self.emit(journal.DecisionRecorded(question=question[:200], answer=answer[:200]))

    def _check_decisions_recorded(self, state: loop_state.LoopState) -> None:
        """Report a recorded ruling missing from decisions.md at the finish; never a block.

        Args:
            state: The execution's state.
        """
        if self.state_dir is None or not state.decisions_recorded:
            return
        try:
            text = agent6_memory.decisions_path(self.state_dir).read_text(encoding="utf-8")
        except OSError:
            text = ""
        missing = [e for e in state.decisions_recorded if e.strip() not in text]
        if missing:
            self.log(f"LOOP: {len(missing)} recorded decision(s) missing from decisions.md")
            self.emit(journal.DecisionUnrecorded(missing=len(missing)))

    def _load_skills(self) -> skills.ResolvedSkills | None:
        """Return the installed skills for the system prompt, in run mode.

        The dispatcher's resolution is reused, so the index and what use_skill serves
        cannot diverge.

        Returns:
            The resolved skills, or None when nothing renders.
        """
        if self.mode != "run":
            return None
        resolved = self.dispatcher.resolved_skills()
        for w in resolved.warnings:
            self.log(f"LOOP: skills: WARNING: {w}")
            self.emit(journal.SkillsWarning(warning=str(w)))
        if resolved.enabled or resolved.always:
            self.log(
                f"LOOP: skills: {len(resolved.enabled)} indexed, {len(resolved.always)} always-on"
            )
            return resolved
        return None

    # ---- the run's helpers ---------------------------------------------------

    @functools.cached_property
    def gate(self) -> verify_gate.VerifyGate:
        """Return the verify gate over the config's command."""
        verify = self.config.harness.verify
        return verify_gate.VerifyGate(
            configured=tuple(verify.command),
            when=verify.when,
            at_start=verify.at_start,
            retries=verify.retries,
            timeout_s=verify.timeout_s,
            infer=verify.infer,
            hide=self.config.sandbox.hide_paths,
            mode=self.mode,
            chain=self.chain,
            dispatcher=self.dispatcher,
            log=self.log,
            emit=self.emit,
        )

    @functools.cached_property
    def checkpoints(self) -> run_checkpoint.Checkpoints:
        """Return the per-turn commit writer."""
        return run_checkpoint.Checkpoints(
            chain=self.chain,
            style=self.config.git.commit.checkpoint_message,
            enabled=self.mode == "run",
            provider=self.provider,
            log=self.log,
            emit=self.emit,
        )

    @functools.cached_property
    def compactor(self) -> compaction_driver.Compactor:
        """Return the context compaction driver."""
        return compaction_driver.Compactor(
            settings=self.compaction,
            provider=self.provider,
            curator=self.curator,
            mode=self.mode,
            dag_available=self.dispatcher.dag_available,
            decisions=self._load_decisions,
            verify_fact=self.verify_fact,
            compact_requested=self.bridge.compact_requested,
            compact_clear=self.bridge.compact_clear,
            log=self.log,
            emit=self.emit,
            emit_graph_snapshot=self._emit_graph_snapshot,
        )

    @functools.cached_property
    def standing(self) -> standing_goal.Standing:
        """Return the standing goal, the re-entry a soft end converts into."""
        return standing_goal.Standing(
            curator=self.curator,
            patience=self.config.harness.standing_patience,
            budget_remaining=self._budget_fraction_remaining,
            log=self.log,
            emit=self.emit,
        )

    @functools.cached_property
    def metrics(self) -> metric_sampler.MetricSampler:
        """Return the metric sampler."""
        return metric_sampler.MetricSampler(
            settings=self.config.harness.metric,
            enabled=self.mode == "run",
            dispatcher=self.dispatcher,
            log=self.log,
            emit=self.emit,
        )

    @functools.cached_property
    def operator_tasks(self) -> operator_tasks.OperatorTasks:
        """Return the taker of the operator's queued tasks."""
        return operator_tasks.OperatorTasks(
            curator=self.curator,
            take_requests=self.bridge.take_requests,
            log=self.log,
            emit=self.emit,
            emit_graph_snapshot=self._emit_graph_snapshot,
        )

    def _parallel_tool(
        self, raw: dict[str, Any], origin: kinds.ToolCallOrigin | None
    ) -> tools_results.ToolResult:
        """Dispatch children for the model's `parallel` call through the steer's own path.

        The children run under the model-child ceilings and the run's reserved allowance; the
        dispatch is recorded with the model as its source and its result is the tool's
        answer, never a steer notice.

        Args:
            raw: The validated arguments.
            origin: The model call the tool call answers; None outside a turn.

        Returns:
            The dispatch's group, message and joins.

        Raises:
            ToolError: No turn is in progress.
        """
        del origin  # the record's identity is the group id the dispatcher allocates
        if self._tool_turn is None:
            raise tool_errors.ToolError("not available: no turn in progress")
        conversation, state, turn = self._tool_turn
        request = schema.ParallelInput.model_validate(raw)
        limits = child_records.ChildLimits(
            max_turns=parallel_dispatch.MODEL_CHILD_MAX_TURNS,
            deadline_at=time.time() + parallel_dispatch.MODEL_CHILD_DEADLINE_S,
        )
        result = self.parallel.dispatch(
            conversation,
            turn,
            state,
            [directive.Segment(task=request.task, spec=request.spec)],
            source="model",
            limits=limits,
        )
        self.parallel.record_delivery(result)
        return tools_results.RawResult(
            {
                "group": result.group,
                "dispatched": result.dispatched,
                "message": result.message,
                "children": [dataclasses.asdict(join) for join in result.children],
            }
        )

    @functools.cached_property
    def parallel(self) -> parallel_dispatch.ParallelDispatcher:
        """Return the `/parallel` child dispatcher."""
        return parallel_dispatch.ParallelDispatcher(
            chain=self.chain,
            curator=self.curator,
            max_children=self.config.harness.parallel.max_children,
            child_spawner=self.bridge.child_spawner,
            authority=self.bridge.child_authority,
            save_snapshot=lambda *args, **kwargs: loop_resume.save_resume_snapshot(
                self, *args, **kwargs
            ),
            log=self.log,
            emit=self.emit,
            emit_graph_snapshot=self._emit_graph_snapshot,
            session_id=self.session_id,
            execution_id=self.execution_id,
            budget=self.budget,
            session_dir=(
                self.resume_state_path.parent if self.resume_state_path is not None else None
            ),
            child_recoverer=self.bridge.child_recoverer,
            ranker=self.bridge.child_ranker,
            stop_pending=lambda: self.bridge.stop_requested() or self.bridge.should_abort(),
        )

    @functools.cached_property
    def steering(self) -> operator_bridge.Steering:
        """Return the reader of a steer's text."""
        return operator_bridge.Steering(
            bridge=self.bridge,
            parallel=lambda: self.parallel,
            dispatcher=self.dispatcher,
            record_decision=self.record_decision,
            log=self.log,
            emit=self.emit,
        )

    @functools.cached_property
    def review_gate(self) -> review_gate.ReviewGate:
        """Return the in-loop review panel."""
        return review_gate.ReviewGate(
            settings=self.review,
            session_id=self.session_id,
            chain=self.chain,
            hide=self.config.sandbox.hide_paths,
            executor=lambda: self.parallel,
            # A reviewer inherits the state of the world its parent has now, tools included.
            tools=lambda: tuple(
                d.name for d in toolset.tool_definitions(self.dispatcher, mode=self.mode)
            ),
            budget_remaining=self._budget_fraction_remaining,
            log=self.log,
            emit=self.emit,
        )

    @functools.cached_property
    def caller(self) -> provider_call.ProviderCaller:
        """Return the worker's provider caller under the retry knobs and the contract's bounds."""
        contract = self.contract
        return provider_call.ProviderCaller(
            provider=self.provider,
            retry_count=self.call.retry_count,
            retry_delay_s=self.call.retry_delay_s,
            retry_max_delay_s=self.call.retry_max_delay_s,
            temperature=(
                contract.temperature
                if contract is not None and contract.temperature is not None
                else self.call.temperature
            ),
            should_abort=self.bridge.should_abort,
            should_interrupt=self.bridge.should_interrupt,
            log=self.log,
            emit=self.emit,
        )

    # ---- the log and the events ----------------------------------------------

    @property
    def execution_id(self) -> str:
        """The identity shared with setup and command evidence, including before loop start."""
        return self.dispatcher.execution_id

    @property
    def session_id(self) -> str:
        """Return the session id, the session dir's name; "" without a log."""
        return self.events.path.parent.name if self.events is not None else ""

    def log(self, msg: str) -> None:
        """Write one line to the run log."""
        self.logger(f"[agent6] {msg}")

    def emit(self, event: journal.Event) -> None:
        """Emit one record when a log is wired."""
        if self.events is not None:
            self.events.emit(event)

    def _emit_start(self, event: journal.SessionStart | journal.SessionResume) -> None:
        """Emit a start-family record through the emitter that stamps the worker pid."""
        if self._execution_started:
            self.dispatcher.execution_id = uuid.uuid4().hex
        self._execution_started = True
        self.dispatcher.verify_tree = self.chain.tree_sha
        if self.events is not None:
            # The id is read after the rotation above, so a second start names its own execution.
            event = event.model_copy(update={"execution_id": self.execution_id})
            process.emit_session_start(self.events, self.events.path.parent, event)

    def _emit_budget(self, turn: int) -> None:
        """Emit the per-turn usage heartbeat that tells a long model call from a stall.

        Args:
            turn: The turn about to call.
        """
        if self.budget is not None:
            self.emit(budget_events.budget_turn(self.budget, turn))
