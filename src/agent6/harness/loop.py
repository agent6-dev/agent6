# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Run the agent loop: one system prompt, one model driving tool calls, a harness around it.

The harness bounds the loop (jail, budget, verify timeout, iteration cap) and records it (events,
the task graph, per-step commits of every tree a verify certified). The heuristics that nudge or
end a run are advisor functions (`_guards`, `_metric`, `_quiet_turns`) and finish gates
(`_finish_gates`), run in a declared order; the loop applies their answers. The review panel
gates checkpoints and never steers.
"""

from __future__ import annotations

import dataclasses
import functools
import itertools
import json
import pathlib
import time
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal

import pydantic

from agent6 import budget as agent6_budget
from agent6 import git_ops, paths, portable, skills, task_text
from agent6 import memory as agent6_memory
from agent6.config import Config
from agent6.graph import curator as graph_curator
from agent6.graph import models, order
from agent6.harness import (
    _advice,
    _chain,
    _checkpoint,
    _compaction,
    _compactor,
    _context,
    _conversation,
    _dag_focus,
    _finish_gates,
    _guards,
    _loop_state,
    _memory_touch,
    _metric,
    _metric_sampler,
    _nudges,
    _operator,
    _operator_tasks,
    _panel,
    _parallel_dispatch,
    _prompt_blocks,
    _prompt_revision,
    _provider_call,
    _quiet_turns,
    _reviewer,
    _snapshot,
    _standing,
    _toolset,
    _verify_gate,
    _verify_verdict,
)
from agent6.prompts import revision as prompts_revision
from agent6.providers import (
    Provider,
    ProviderAborted,
    ProviderError,
    ProviderInterrupted,
    ProviderResponse,
    ToolDefinition,
)
from agent6.sessions import ipc
from agent6.tools import dispatch, errors, mcp_client, results, schema

# Bytes a repeated tool result must exceed before the dedupe stub replaces it.
_DEDUPE_MIN_CHARS = 500


if TYPE_CHECKING:
    from agent6 import event_log


# Consecutive quiet turns after which a metric run drops the worker's output cap.
_STARVATION_BACKOFF_AFTER_QUIETS = 2


@dataclasses.dataclass
class Harness:
    """Drive one execution of the agent loop over a session.

    The model decides everything through tool calls in one loop: when to read, edit, verify,
    measure and stop. The harness keeps the loop bounded and observable.

    Attributes:
        chain: The run's commit chain: the repository root, the per-step commits, the worktree.
        config: The effective config.
        provider: The model provider the worker calls go to.
        dispatcher: The tool dispatcher.
        logger: Where log lines go.
        events: The run's event sink; None emits nothing.
        curator: The task graph; None runs without task persistence (bench and one-off runs).
        budget: The budget tracker the provider shares; None degrades the metric plateau rule
            to fixed counts.
        state_dir: The per-repo state dir holding the memory store; None runs memory-less.
        max_iterations: The cap on turns for this execution (-1 unbounded); a resumed
            execution re-arms it.
        finish_validator: A machine state's finish contract, returning the problems with a
            finish_session payload; None leaves finishes ungated.
        bridge: What the operator can do to the run, as the front-end injects it.
        call: The worker call's knobs: retries, temperature, output caps.
        compaction: The compaction tiers' thresholds, the tail kept, the summariser.
        review: The in-loop review panel's trigger, seats and decision rule.
        revision: The one-shot prompt revision before the first worker call.
        initial_pins: Pins seeded before the first turn (a lane inherits the coordinator's);
            fresh runs only, a resume restores pins from the snapshot.
        resume_state_path: Where the provider-agnostic resume snapshot is written before
            every model call; None writes none.
        standing_goal: The operator's `run --standing` goal, seeded under the root; "" is none.
        interactive: An operator is watching and can steer live, so a quiet turn parks for a
            steer instead of ending.
        mode: The mode; `plan` uses the planning prompt and tool list, never auto-commits,
            and writes `plan_markdown` to `plan_output_path` on finish_planning.
        plan_output_path: Where a plan is written; required when `mode="plan"`.
        iterations_reached: The iteration being driven, 0 before the loop starts; the app's
            interrupt fallbacks read it for a truthful `session.end`.
    """

    chain: _chain.RunChain
    config: Config
    provider: Provider
    dispatcher: dispatch.ToolDispatcher
    logger: Callable[[str], None] = dataclasses.field(default=print)
    events: event_log.EventSink | None = None
    curator: graph_curator.GraphCurator | None = None
    budget: agent6_budget.BudgetTracker | None = None
    state_dir: pathlib.Path | None = None
    max_iterations: int = 200
    finish_validator: Callable[[dict[str, Any] | None], list[str]] | None = None
    bridge: _operator.OperatorBridge = dataclasses.field(default_factory=_operator.OperatorBridge)
    call: _provider_call.CallSettings = dataclasses.field(
        default_factory=_provider_call.CallSettings
    )
    compaction: _compaction.CompactionSettings = dataclasses.field(
        default_factory=_compaction.CompactionSettings
    )
    review: _reviewer.ReviewSettings = dataclasses.field(default_factory=_reviewer.ReviewSettings)
    revision: _prompt_revision.RevisionSettings = dataclasses.field(
        default_factory=_prompt_revision.RevisionSettings
    )
    initial_pins: Sequence[str] = ()
    resume_state_path: pathlib.Path | None = None
    standing_goal: str = ""
    # The gate a resumed execution carried from the last one, set before the state is built.
    _adopted_on_resume: tuple[str, ...] = ()
    interactive: bool = False
    mode: Literal["run", "plan", "ask", "agent"] = "run"
    plan_output_path: pathlib.Path | None = None
    # A snapshot write fault warns once, not every turn; it disables resume, never the run.
    _snapshot_write_failed: bool = dataclasses.field(default=False, init=False)
    iterations_reached: int = dataclasses.field(default=0, init=False)

    # ---- run / resume entry --------------------------------------------------

    def run(self, user_task: str) -> _snapshot.SessionResult:
        """Drive a fresh execution of the loop to its end.

        Args:
            user_task: The task, with any seed digest or skill block prepended.

        Returns:
            How the execution ended.

        Raises:
            ValueError: Plan mode with no `plan_output_path`.
        """
        self.bridge.steer_reset()  # an execution starts with no armed Ctrl-C
        if self.mode == "plan" and self.plan_output_path is None:
            raise ValueError("Harness(mode='plan') requires plan_output_path to be set")
        # Every headline reads this field: the operator's own words, without a seed or skill block.
        self._emit_start(
            "session.start",
            session_id=self.session_id,
            user_task=task_text.operator_task_text(user_task)[:200],
            mode=self.mode,
        )
        self._log("LOOP: LOAD_CONTEXT")
        repo = _context.load_repo_summary(self.chain.root)
        system = _prompt_blocks.build_system_prompt(
            config=self.config,
            repo=repo,
            mode=self.mode,
            memory_index=self._load_memory_index(),
            memory_dir_path=str(agent6_memory.memory_dir(self.state_dir))
            if self.state_dir is not None
            else "",
            decisions=self._load_decisions(),
            decisions_path=str(agent6_memory.decisions_path(self.state_dir))
            if self.state_dir
            else "",
            skills=self._load_skills(),
            isolation=self.dispatcher.isolation,
            commands_allowed=self.dispatcher.command_policy() != "no",
            protected_paths=bool(self.dispatcher.extra_protect_paths),
            dag_available=self.dispatcher.dag_available,
        )

        try:
            effective_task = _prompt_revision.revise_prompt(
                self.revision, user_task, repo, log=self._log, emit=self._emit
            )
        except _prompt_revision.PromptRevisionError as exc:
            declined = isinstance(exc, _prompt_revision.PromptRevisionDeclined)
            end_reason: _snapshot.SessionEndReason = (
                "steer_abort" if declined else "prompt_revision_failed"
            )
            self._log(f"LOOP: prompt revision {'declined' if declined else 'failed'}: {exc}")
            self._emit("session.end", reason=end_reason, iterations=0, all_passed=False)
            return _snapshot.SessionResult(
                completed=False, reason=end_reason, summary=str(exc), iterations=0, tool_calls=0
            )

        # The root task is what `add_task` with `parent_id=None` attaches under.
        root_id = self.operator_tasks.seed_root(effective_task)
        if root_id is not None:
            self.dispatcher.set_run_root_node_id(root_id)
            self._log(f"LOOP: DAG root task seeded: {root_id}")
            self.operator_tasks.seed_standing(root_id, self.standing_goal)
            self._emit_graph_snapshot()  # show the root in the live task view

        self._log(
            f"LOOP: mode={self.mode} system={len(system)} chars, task={len(effective_task)} chars"
        )

        dag_hint = _dag_focus.initial_dag_hint(
            root_id, self.mode, self.config.prompt.decompose == "on"
        )
        instructions = _prompt_blocks.initial_instructions(
            self.mode,
            self.config.sandbox.run_commands,
            has_gate=self.gate.present(
                _verify_verdict.VerifyVerdict()
            ),  # the start: nothing adopted or denied
        )
        initial_user = f"TASK:\n{effective_task}\n\n{instructions}{dag_hint}"
        conversation = _conversation.Conversation()
        conversation.notice(initial_user)

        return self._drive_loop(
            system=system,
            conversation=conversation,
            tool_calls=0,
            start_iteration=1,
            root_task_id=root_id,
            original_task=effective_task,
        )

    def resume(self) -> _snapshot.SessionResult:
        """Resume a session from its snapshot and drive the new execution to its end.

        The snapshot is the one written before the last model call; the root task id is
        reattached to the dispatcher and the loop re-enters at the saved iteration with the
        saved conversation. The budget tracker is fresh per execution.

        Returns:
            How the execution ended.

        Raises:
            ResumeError: No snapshot path, or a snapshot that cannot be read.
        """
        self.bridge.steer_reset()  # an execution starts with no armed Ctrl-C
        if self.resume_state_path is None:
            raise _snapshot.ResumeError("resume() called but resume_state_path is None")
        try:
            snapshot = _snapshot.load_session_snapshot(self.resume_state_path)
            conversation = _conversation.Conversation.from_wire(snapshot.messages)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise _snapshot.ResumeError(
                f"failed to load resume snapshot from {self.resume_state_path}: {exc}"
            ) from exc

        # The execution's log opens with this event, so it identifies itself like session.start.
        self._emit_start(
            "loop.resume.start",
            session_id=self.session_id,
            mode=self.mode,
            iteration=snapshot.next_iteration,
            messages=len(snapshot.messages),
        )
        self._log(
            f"LOOP: RESUME from {self.resume_state_path} "
            f"(iter={snapshot.next_iteration}, messages={len(snapshot.messages)}, "
            f"tool_calls={snapshot.tool_calls})"
        )

        if snapshot.root_task_id is not None:
            self.dispatcher.set_run_root_node_id(snapshot.root_task_id)
            self._log(f"LOOP: DAG root task restored: {snapshot.root_task_id}")

        # The system prompt is frozen, so a gate that changed between executions is announced.
        self._adopted_on_resume = self._carry_adopted_gate(snapshot)
        gate = self.gate.configured or self._adopted_on_resume
        withheld = (
            not gate and bool(snapshot.verify_command) and self.dispatcher.command_policy() == "no"
        )
        if gate != snapshot.verify_command and not withheld:
            was = " ".join(snapshot.verify_command) or "none"
            now = " ".join(gate) or "none"
            conversation.notice(
                f"[harness] This run's verify gate changed between executions: it was `{was}`,"
                f" it is now `{now}`. The instructions above still name the old one."
            )
            self._log(f"LOOP: verify gate swapped on resume: {was} -> {now}")
            self._emit("loop.verify_swapped", was=list(snapshot.verify_command), now=list(gate))

        return self._drive_loop(
            system=snapshot.system,
            conversation=conversation,
            tool_calls=snapshot.tool_calls,
            start_iteration=snapshot.next_iteration,
            root_task_id=snapshot.root_task_id,
            original_task=snapshot.original_task,
            resume_from=snapshot,
        )

    # ---- snapshots and carryover ---------------------------------------------

    def _seed_carryover(
        self,
        state: _loop_state.LoopState,
        conversation: _conversation.Conversation,
        resume_from: _snapshot.SessionSnapshot | None,
    ) -> None:
        """Seed the execution's carried state and announce it for the read model.

        A resumed or forked execution re-announces its restored pins and elision counters,
        even when empty: the fold replaces on these events, so the surfaces never list a pin no
        restart will re-inject. A fresh run seeded with pins takes the same path.

        Args:
            state: The execution's state.
            conversation: The conversation the pins are announced into.
            resume_from: The snapshot a resumed execution restores; None for a fresh run.
        """
        if resume_from is not None:
            _loop_state.restore_completion_state(state, resume_from)
            state.verify.adopted = self._adopted_on_resume
            self._carry_verify_verdict(state, resume_from)
            self._emit("loop.pin.restored", pins=list(state.pins), count=len(state.pins))
            elided, gists = _compaction.count_elisions(conversation)
            self._emit("loop.compact.restored", elided=elided, gists=gists)
        elif self.initial_pins:
            # The pin owner applies the cap and the non-empty check; a refused pin is logged.
            for pin in self.initial_pins:
                if not _operator.try_pin(state.pins, pin):
                    self._log(
                        f"  --pin refused (empty or over the {_operator.PINS_MAX_CHARS}-char cap)"
                    )
                    self._emit("loop.pin.refused", chars=len(pin), limit=_operator.PINS_MAX_CHARS)
            self._emit("loop.pin.restored", pins=list(state.pins), count=len(state.pins))
            if state.pins:
                conversation.notice(prompts_revision.pinned_block(state.pins))

    def _carry_adopted_gate(self, snapshot: _snapshot.SessionSnapshot) -> tuple[str, ...]:
        """Carry the gate a gateless run adopted in an earlier execution.

        Args:
            snapshot: The snapshot the execution resumes from.

        Returns:
            The snapshot's command when the config names none and the jail can run it; `()`
            otherwise, and the execution-start notice reads the gate as swapped.
        """
        argv = tuple(snapshot.verify_command)
        if self.gate.configured or not argv or not self.dispatcher.adopt_verify_command(argv):
            return ()
        self._log(f"LOOP: verify gate carried from the last execution: {' '.join(argv)}")
        self._emit(
            "loop.verify_inferred",
            command=list(argv),
            source="resumed",
            adopted_at=snapshot.next_iteration,
        )
        return argv

    def _carry_verify_verdict(
        self, state: _loop_state.LoopState, snap: _snapshot.SessionSnapshot
    ) -> None:
        """Carry the prior execution's verify observation when it still describes this tree.

        The chain tip must be the snapshot's and the worktree clean; an operator commit or edit
        in between fails closed, so the execution starts unobserved rather than wrongly green
        or red. `baseline_ok` is about the base commit, which resume never moves.

        Args:
            state: The execution's state.
            snap: The snapshot the execution resumes from.
        """
        state.verify.baseline_ok = snap.baseline_ok
        state.standing.tools_mark = snap.standing_tools_mark
        state.standing.fruitless = snap.standing_fruitless
        state.ok_tool_calls = snap.ok_tool_calls
        if snap.last_verify_ok is None or not snap.head_sha:
            return
        try:
            if snap.head_sha != self.chain.checkpoint_head_sha():
                return
            dirty = self.chain.is_dirty()
        except (git_ops.GitError, OSError):
            return
        if not dirty:
            state.verify.last_ok = snap.last_verify_ok
            state.verify.edited_since = snap.edited_since_verify

    def _save_resume_snapshot(
        self,
        state: _loop_state.LoopState,
        messages: list[dict[str, Any]],
        *,
        next_iteration: int,
        write_checkpoint: bool = False,
    ) -> None:
        """Write the resume snapshot, and the turn's checkpoint when asked.

        Written before each model call and again after the turn's tool results land, so a
        crash after a non-idempotent tool resumes from after it. Every write advances the
        latest pointer; only the pre-call write owns `checkpoints/<next_iteration>.json`, so
        `fork --at-turn N` has one meaning. Atomic, so a crash mid-write keeps the prior
        snapshot. A None `resume_state_path` writes nothing.

        Args:
            state: The execution's state.
            messages: The conversation in wire form.
            next_iteration: The iteration the snapshot resumes at.
            write_checkpoint: Also write the turn's checkpoint file.
        """
        if self.resume_state_path is None:
            return
        goal = self.metrics.goal
        best = (
            _metric.best_metric_sample(state.metric.history, goal=goal)
            if goal is not None
            else None
        )
        snapshot = _snapshot.SessionSnapshot(
            system=state.system,
            messages=messages,
            tool_calls=state.tool_calls,
            next_iteration=next_iteration,
            root_task_id=state.root_task_id,
            original_task=state.original_task,
            verify_command=self.gate.command(state.verify),
            review_rejections_total=state.gates.review_total,
            verify_ever_passed=state.verify.ever_passed,
            verify_ever_failed=state.verify.ever_failed,
            gateless_ever_edited=state.settled.gateless_ever_edited,
            parallel_groups_dispatched=state.parallel_groups_dispatched,
            pins=tuple(state.pins),
            metric_best_score=best.score if best is not None else None,
            metric_at_ceiling=state.metric.at_ceiling(),
            last_verify_ok=state.verify.last_ok,
            edited_since_verify=state.verify.edited_since,
            baseline_ok=state.verify.baseline_ok,
            verify_scoped=state.verify.scoped,
            memory_written=state.memory.written,
            memory_flip_nudged=state.memory.flip_nudged,
            memory_finish_nudged=state.memory.finish_nudged,
            standing_tools_mark=state.standing.tools_mark,
            standing_fruitless=state.standing.fruitless,
            ok_tool_calls=state.ok_tool_calls,
            head_sha=self.chain.checkpoint_head_sha(),
            graph_version=self._checkpoint_graph_version(),
        )
        blob = snapshot.model_dump_json()
        # Recovery state, not run output: an unwritable state dir warns once and never aborts.
        try:
            # The checkpoint first, then the latest pointer, so a fork keeps a durable target.
            if write_checkpoint:
                cp_dir = self.resume_state_path.parent / "checkpoints"
                portable.atomic_write(cp_dir / f"{next_iteration:04d}.json", blob)
            portable.atomic_write(self.resume_state_path, blob)
        except OSError as exc:
            if not self._snapshot_write_failed:
                self._snapshot_write_failed = True
                self._log(
                    f"LOOP: WARNING could not persist resume snapshot ({exc}); "
                    "resume/fork are unavailable for this run, continuing anyway"
                )

    # ---- the turn pipeline ---------------------------------------------------

    def _drive_loop(  # noqa: PLR0911, PLR0912
        self,
        *,
        system: str,
        conversation: _conversation.Conversation,
        tool_calls: int,
        start_iteration: int,
        root_task_id: str | None,
        original_task: str,
        resume_from: _snapshot.SessionSnapshot | None = None,
    ) -> _snapshot.SessionResult:
        """Drive the turns of one execution until a phase ends it.

        One `TurnState` per turn, through the phases in order; any phase returning a
        `SessionResult` ends the execution.

        Args:
            system: The system prompt, frozen for the session.
            conversation: The conversation so far.
            tool_calls: Tool calls already made in earlier executions.
            start_iteration: The first iteration of this execution.
            root_task_id: The root task the graph's nodes attach under.
            original_task: The task verbatim; the review panel grounds on it.
            resume_from: The snapshot a resumed execution restores; None for a fresh run.

        Returns:
            How the execution ended.
        """
        state = _loop_state.LoopState(
            original_task=original_task,
            tool_calls=tool_calls,
            root_task_id=root_task_id,
            system=system,
        )
        self._seed_carryover(state, conversation, resume_from)
        # The allowance is this execution's: a resumed one re-arms it (-1 is unbounded).
        for iteration in (
            range(start_iteration, start_iteration + self.max_iterations)
            if self.max_iterations >= 0
            else itertools.count(start_iteration)
        ):
            self.iterations_reached = iteration
            if resume_from is not None and iteration == start_iteration:
                seeded = self._seeded_steer(conversation, iteration, state)
                if seeded is not None:
                    return seeded
            # Rebuilt per turn: a gate adopted or a policy denied mid-run changes the tool list.
            tools = _toolset.tool_definitions(self.dispatcher, mode=self.mode)
            ctx = self._turn_context(state, iteration=iteration, execution_start=start_iteration)
            wire = self._turn_pre_call(
                conversation=conversation,
                state=state,
                ctx=ctx,
                prefix_chars=_compaction.request_prefix_chars(system, tools),
            )
            if isinstance(wire, _snapshot.SessionResult):
                return wire
            got = self._turn_provider_call(
                system,
                conversation,
                wire,
                tools,
                state,
                iteration=iteration,
            )
            if isinstance(got, _snapshot.SessionResult):
                return got
            if isinstance(got, _loop_state.NextTurn):
                continue
            # The response's blocks enter the history verbatim, so tool_use ids round-trip.
            assistant = conversation.assistant(got.raw.get("content") or [])
            if not assistant.tool_uses:
                result = self._handle_no_tool_use(got, assistant, conversation, state, ctx)
                if result is not None:
                    return result
                # A prose turn is a completed iteration: snapshotted, then the operator boundary.
                self._save_resume_snapshot(
                    state, conversation.to_wire(), next_iteration=iteration + 1
                )
                outcome = self._operator_boundary(conversation, iteration, state)
                if outcome is not None:
                    return outcome
                continue
            turn = _loop_state.TurnState(iteration=iteration, resp=got, assistant=assistant)
            # The marker outlives a crash mid-dispatch, so resume asks before replaying a tool.
            if self.resume_state_path is not None:
                _snapshot.write_turn_marker(
                    self.resume_state_path.parent / _snapshot.TURN_IN_FLIGHT_NAME,
                    iteration,
                    tuple(tu.name for tu in assistant.tool_uses),
                )
            result = self._turn_dispatch_tools(state, turn, ctx)
            if result is not None:
                return result
            # One graph snapshot per turn, however many mutations the turn made.
            if turn.dag_mutated:
                self._emit_graph_snapshot()
            result = self._turn_auto_commit_and_metric(state, turn)
            if result is not None:
                return result
            self.reviewer.triggers(state, turn)
            self._turn_finish_gates(state, turn, ctx)
            result = self._turn_advisors(state, turn, ctx)
            if result is not None:
                return result
            conversation.results(turn.tool_results)
            # Snapshot, then clear the marker: a stale marker is cleared, never missed.
            self._save_resume_snapshot(state, conversation.to_wire(), next_iteration=iteration + 1)
            if self.resume_state_path is not None:
                _snapshot.clear_turn_marker(
                    self.resume_state_path.parent / _snapshot.TURN_IN_FLIGHT_NAME
                )
            result = self._turn_stop_checks(state, turn, conversation)
            if result is not None:
                return result
            outcome = self._operator_boundary(conversation, iteration, state)
            if outcome is not None:
                return outcome

        self._log(f"LOOP: max_iterations={self.max_iterations} reached")
        return self._finish(
            state,
            _snapshot.End(
                "max_iterations",
                f"max_iterations={self.max_iterations} reached without finish_session",
            ),
            iteration=self.iterations_reached,
        )

    def _seeded_steer(
        self, conversation: _conversation.Conversation, iteration: int, state: _loop_state.LoopState
    ) -> _snapshot.SessionResult | None:
        """Consume the follow-up a `resume --steer` queued before the loop started.

        It enters the conversation ahead of the first model call, since a resumed conversation
        that was already finished ends on iteration 1 before the end-of-iteration poll runs.

        Args:
            conversation: The conversation the steer enters.
            iteration: The first iteration of the execution.
            state: The execution's state.

        Returns:
            The end the steer asked for, or None when the execution goes on.
        """
        return self._steer_outcome(
            self.steering.handle(conversation, iteration, state), iteration, state
        )

    def _turn_pre_call(
        self,
        *,
        conversation: _conversation.Conversation,
        state: _loop_state.LoopState,
        ctx: _advice.TurnContext,
        prefix_chars: int = 0,
    ) -> list[dict[str, Any]] | _snapshot.SessionResult:
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
        self._emit_budget(ctx.iteration)
        if self.compactor.compact(conversation, state, prefix_chars=prefix_chars):
            # A tier-2 restart wiped the focus banner and the plan block; both go back below.
            state.focus.surfaced_task_id = None
            state.plan_injected = ""
        parked = self._maybe_inject_plan(conversation, state, iteration=ctx.iteration)
        if parked is not None:
            return parked
        self._turn_before_call(conversation, state, ctx)
        conversation.roll_cache_marks()
        wire = conversation.to_wire()
        # The one numbered-checkpoint writer: the state this turn's model call consumes.
        self._save_resume_snapshot(state, wire, next_iteration=ctx.iteration, write_checkpoint=True)
        return wire

    def _maybe_inject_plan(
        self,
        conversation: _conversation.Conversation,
        state: _loop_state.LoopState,
        *,
        iteration: int,
    ) -> _snapshot.SessionResult | None:
        """Show the planner the plan.md on disk when it changed.

        The file is the plan and `agent6 plan edit` writes the operator's answers to it between
        executions; an unreadable file parks the execution rather than run on stale direction.

        Args:
            conversation: The conversation the plan enters.
            state: The execution's state; remembers the last plan shown.
            iteration: The current iteration.

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
            self._log(f"LOOP: {remedy}")
            self._emit("loop.plan_read.failed", path=str(self.plan_output_path), error=str(exc))
            return self._finish(
                state,
                _snapshot.End("plan_unreadable", remedy, checkpoint=False),
                iteration=iteration,
            )
        if text == state.plan_injected:
            return None
        state.plan_injected = text
        conversation.notice(f"{_nudges.PLAN_ON_DISK_HEADER}\n\n{text}")
        self._log(f"  plan re-read from disk: {len(text)} chars")
        self._emit(
            "loop.plan_reread",
            path=str(self.plan_output_path),
            bytes=len(text.encode("utf-8")),
        )

    def _turn_before_call(
        self,
        conversation: _conversation.Conversation,
        state: _loop_state.LoopState,
        ctx: _advice.TurnContext,
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
        for advisor in _guards.BEFORE_CALL:
            self._tell(conversation, advisor(state, ctx))

    def _maybe_surface_current_task(
        self, conversation: _conversation.Conversation, state: _loop_state.LoopState
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
        current_id = _dag_focus.current_task_id(nodes, cursor)
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
                self._log(f"LOOP: cursor advance skipped: {exc}")
        self._tell(conversation, _guards.stuck_on_task(state, current_id, nodes[current_id], nodes))
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
                self._log(f"LOOP: mark in_progress skipped: {exc}")
        banner = _dag_focus.current_task_banner(
            current_id, node, decompose=self.config.prompt.decompose == "on"
        )
        conversation.notice(banner)
        state.focus.surfaced_task_id = current_id
        self._log(f"LOOP: surfaced current task {current_id}")
        self._emit("loop.task.surfaced", task_id=current_id)
        # The cursor and status writes above bypass dispatch, which emits graph.update.
        self._emit_graph_snapshot()

    def _turn_provider_call(
        self,
        system: str,
        conversation: _conversation.Conversation,
        wire: list[dict[str, Any]],
        tools: list[ToolDefinition],
        state: _loop_state.LoopState,
        *,
        iteration: int,
    ) -> _snapshot.SessionResult | _loop_state.NextTurn | ProviderResponse:
        """Make the turn's model call and classify a terminal error.

        Args:
            system: The system prompt.
            conversation: The conversation; touched only on the steer path.
            wire: The pre-call serialization, already snapshotted.
            tools: The tools offered this turn.
            state: The execution's state.
            iteration: The current iteration.

        Returns:
            The response, the end of the execution, or `NEXT_TURN` when a mid-stream
            steer discarded the turn.
        """
        try:
            return self.caller.call(system, wire, tools, self._worker_max_tokens(state))
        except agent6_budget.BudgetExceededError as exc:
            self._log(f"LOOP: budget exhausted at iter {iteration} ({exc})")
            return self._finish(
                state,
                _snapshot.End("budget_exhausted", f"budget exhausted at iter {iteration}: {exc}"),
                iteration=iteration,
            )
        except ProviderAborted:
            self.bridge.steer_clear()  # consume the stop; don't leave it on disk to re-read
            self._log(f"LOOP: operator stopped the run mid-turn at iter {iteration}")
            return self._finish(
                state,
                _snapshot.End(
                    "steer_abort",
                    f"operator stopped the run at iter {iteration}{self._dirty_tree_note()}",
                    checkpoint=False,
                ),
                iteration=iteration,
            )
        except ProviderInterrupted:
            # The watchdog ended the turn for a steer; the partial turn is discarded.
            self._log(f"LOOP: steer requested mid-turn at iter {iteration}")
            outcome = self._steer_outcome(
                self.steering.handle(conversation, iteration, state), iteration, state
            )
            if outcome is not None:
                return outcome
            return _loop_state.NEXT_TURN  # "continue" or an injected instruction -> re-do the turn
        except ProviderError as exc:
            hint = _provider_call.provider_error_hint(exc.status_code, exc.provider)
            attempts = f" after {exc.attempts} attempts" if exc.attempts > 1 else ""
            # The upstream body lands in this one log line; the summary below stays short.
            self._log(f"LOOP: provider error{attempts} at iter {iteration}: {exc}{hint}")
            status = f" (HTTP {exc.status_code})" if exc.status_code else ""
            # A fatal error or a transport failure has no other reason to name.
            detail = f": {exc}" if exc.fatal or exc.status_code is None else ""
            return self._finish(
                state,
                _snapshot.End(
                    "provider_error",
                    f"provider error{attempts} at iter {iteration}{status}{hint}{detail}",
                ),
                iteration=iteration,
            )

    def _worker_max_tokens(self, state: _loop_state.LoopState) -> int:
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
        self, state: _loop_state.LoopState, turn: _loop_state.TurnState, ctx: _advice.TurnContext
    ) -> _snapshot.SessionResult | None:
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
        for tu in turn.assistant.tool_uses:
            name = tu.name
            tool_input = tu.input
            if turn.finish is not None:
                # The calls after a finish are not executed, as the finish tools state.
                turn.tool_results.append(
                    _conversation.ToolResultItem(
                        tool_use_id=tu.id,
                        content=json.dumps(
                            {"error": f"{name} not executed: it follows {turn.finish.kind}"}
                        ),
                        for_call=tu,
                    )
                )
                continue
            state.tool_calls += 1
            # The call's signature for the spiral guard; stable JSON so key order cannot differ.
            try:
                sig = f"{name}:{json.dumps(tool_input, sort_keys=True, ensure_ascii=False)}"
            except (TypeError, ValueError):
                sig = f"{name}:<unhashable>"
            state.spiral.note_call(sig, polling=name == schema.ReadBackgroundInput.TOOL_NAME)
            self._emit("loop.tool.call", name=name, iteration=turn.iteration)
            served = None
            try:
                refusal = turn.resp.refused.get(tu.id)
                if refusal is not None:
                    # The provider's front-end refused the input; the same error is the result.
                    raise errors.ToolError(refusal)
                tree_before = self._tree_before_command(name)
                result = self.dispatcher.dispatch(name, tool_input)
                content = json.dumps(result.to_wire(), ensure_ascii=False)
                self._note_tool_effects(
                    state, turn, name, result, tool_input, tree_before=tree_before
                )
                # A repeated call with an unchanged result serves a stub, so a re-read stays small.
                if state.spiral.stub_repeat(content, min_chars=_DEDUPE_MIN_CHARS):
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
                    state, turn, ctx, _guards.unreachable_tool(state, name, tool_input, result)
                )
                # Only a dispatched finish counts; a refused one is an error result.
                self._capture_finish(turn, name, tool_input)
            except errors.ToolError as exc:
                content = self._note_tool_error(state, name, tool_input, exc)
                self._take(state, turn, ctx, _guards.tool_error_ladder(turn, state, ctx))
            except errors.OperatorCommandUnexecutableError as exc:
                return self._unexecutable_abort(exc, iteration=turn.iteration, state=state)
            turn.tool_results.append(
                _conversation.ToolResultItem(
                    tool_use_id=tu.id,
                    content=_compaction.cap_tool_result(
                        served if served is not None else content,
                        tool_name=name,
                        cap=self.compaction.tool_result_cap_bytes,
                    ),
                    for_call=tu,
                )
            )
        # The gate run the harness adds to the turn, after the model's calls.
        try:
            self.gate.harness_verify(state, turn)
        except errors.OperatorCommandUnexecutableError as exc:
            return self._unexecutable_abort(exc, iteration=turn.iteration, state=state)
        return None

    def _tree_before_command(self, name: str) -> str:
        """Return the tree's content sha before a child-process tool's call.

        The operator's own gates are excluded: the caches they drop must not invalidate
        the pass they just produced.

        Args:
            name: The tool about to run.

        Returns:
            The sha, or "" for a tool that cannot touch the tree.
        """
        if name != "run_command" and not name.startswith(mcp_client.MCP_TOOL_PREFIX):
            return ""
        return self.chain.tree_sha()

    def _left_the_tree_dirty(self, tree_before: str) -> bool:
        """Report whether a child-process tool changed the tree.

        Git decides, so a read-only probe costs its pass nothing and gitignored build
        output never counts.

        Args:
            tree_before: The sha from `_tree_before_command`; "" reads as unchanged.

        Returns:
            True when the tree's content sha differs from `tree_before`.
        """
        if not tree_before:
            return False
        after = self.chain.tree_sha()
        return bool(after) and after != tree_before

    def _note_tool_effects(
        self,
        state: _loop_state.LoopState,
        turn: _loop_state.TurnState,
        name: str,
        result: results.ToolResult,
        tool_input: Any,
        tree_before: str = "",
    ) -> None:
        """Note a dispatched tool's effects on the turn.

        Verify results, metric samples, tree edits and graph mutations feed the commit,
        the review panel and the finish gates.

        Args:
            state: The execution's state.
            turn: This turn's state.
            name: The tool that ran.
            result: Its result.
            tool_input: Its input.
            tree_before: The sha from `_tree_before_command` for a child-process tool.
        """
        if name == "ask_user" and isinstance(result, results.AnswersResult):
            # The result carries the questions the dispatcher accepted; the input is not reparsed.
            for question, answer in zip(result.asked, result.answers, strict=False):
                self._record_decision(state, question, answer)
        if name == "run_verify_command" and isinstance(result, results.ExecResult):
            # A timed-out model gate gets the same scoped follow-up as the harness gate.
            if not (
                self.mode == "run"
                and self.gate.when != "never"
                and result.returncode == _verify_gate.EXIT_TIMEOUT
                and not state.verify.scoped
                and self.gate.scoped_followup(state, turn) is not None
            ):
                if result.returncode == 0:
                    # The model's call runs the full argv, so its green is a full pass.
                    state.verify.scoped = False
                self.gate.note_result(state, turn, result)
        elif name == "run_metric_command" and isinstance(result, results.MetricResult):
            turn.metric_sampled = True
            # The tree this reading covers, so the auto path does not sample it again.
            state.metric.tree = self.chain.tree_sha()
            turn.metric_feedback = self.metrics.record(
                state.metric.history,
                result,
                iteration=turn.iteration,
                label=f"manual iter {turn.iteration}",
                sha="",
            )
            if turn.verify_just_passed:
                turn.metric_plateau_finish = self.metrics.plateau_finish(state.metric.history)
        if name in ("apply_edit", "apply_patch") and isinstance(result, results.PreviewResult):
            return  # a dry run writes nothing: no memory write, no tree edit
        if self._note_memory_touch(state, name, result, tool_input):
            # A memory write is not workspace work: the tree bookkeeping below does not apply.
            return
        if name in ("apply_edit", "apply_patch"):
            turn.edited = True
            state.ever_edited = True
            # A same-turn verify pass no longer covers this tree.
            turn.edit_since_verify_pass = True
            state.verify.note_edit()
        elif self._left_the_tree_dirty(tree_before):
            # A command or an MCP tool changed the tree, so a green verify no longer covers it.
            turn.edit_since_verify_pass = True
            state.verify.note_edit()
        if name in _dag_focus.DAG_MUTATING_TOOLS:
            turn.dag_mutated = True  # snapshot once after the turn

    def _note_memory_touch(
        self, state: _loop_state.LoopState, name: str, result: results.ToolResult, tool_input: Any
    ) -> bool:
        """Count a read of a memory fact, or record an edit tool's write to the store.

        Args:
            state: The execution's state; holds the memory bookkeeping.
            name: The tool that ran.
            result: Its result.
            tool_input: Its input.

        Returns:
            True for a write to the store, which is not workspace work.
        """
        facts = _memory_touch.memory_store_facts(self.state_dir, name, result, tool_input)
        if facts is None:
            return False
        if name == "read_file":
            for fact in facts:
                state.memory.read[fact] = state.memory.read.get(fact, 0) + 1
            return False
        if name in ("apply_edit", "apply_patch"):
            state.memory.written = True
            for fact, op in facts.items():
                state.memory.note_write(fact, op)
            return True
        return False

    def _capture_finish(self, turn: _loop_state.TurnState, name: str, tool_input: Any) -> None:
        """Record a dispatched finish on the turn; a finish_planning also writes its plan.

        The finish gates may still revoke it.

        Args:
            turn: This turn's state.
            name: The tool that ran.
            tool_input: Its input.
        """
        finish = _finish_gates.FinishCall.parse(name, tool_input)
        if finish is None:
            return
        turn.finish = finish
        if finish.kind != "finish_planning":
            return
        if finish.plan_salvaged:
            self._log("  plan salvaged: folded summary into a title-only plan_markdown")
        if self.plan_output_path is None or not finish.plan_markdown:
            return
        try:
            paths.mkdir_for_real_user(self.plan_output_path.parent)
            self.plan_output_path.write_text(finish.plan_markdown, encoding="utf-8")
            self._log(
                f"  plan written: {self.plan_output_path} ({len(finish.plan_markdown)} chars)"
            )
            self._emit(
                "loop.plan_written",
                path=str(self.plan_output_path),
                bytes=len(finish.plan_markdown.encode("utf-8")),
            )
        except OSError as exc:
            self._log(f"  plan write failed: {exc}")
            self._emit("loop.plan_write.failed", path=str(self.plan_output_path), error=str(exc))

    def _note_tool_error(
        self,
        state: _loop_state.LoopState,
        name: str,
        tool_input: dict[str, Any],
        exc: errors.ToolError,
    ) -> str:
        """Note one failed dispatch for the spiral guard and the reachability note.

        Args:
            state: The execution's state.
            name: The tool that failed.
            tool_input: Its input.
            exc: The error.

        Returns:
            The error content served as the tool result.
        """
        content = json.dumps({"error": str(exc)})
        self._log(f"  tool_error: {name}: {exc}")
        state.spiral.note_error(
            _nudges.tool_error_signature(name, str(exc)),
            denial=isinstance(exc, errors.ToolDeniedError),
            content=content,
        )
        return content

    def _turn_auto_commit_and_metric(
        self, state: _loop_state.LoopState, turn: _loop_state.TurnState
    ) -> _snapshot.SessionResult | None:
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
        gateless = not self.gate.present(state.verify)
        # Unjudged: no gate may run, or under `verify_when = "finish"` the model ran none.
        unjudged = gateless or (
            self.gate.when == "finish" and not (turn.verify_just_passed or turn.verify_just_failed)
        )
        unjudged_changed = unjudged and (turn.edited or self.chain.dirty())
        verified_commit = turn.verify_just_passed and not turn.edit_since_verify_pass
        if self.mode != "run" or not (verified_commit or unjudged_changed):
            return None
        if unjudged_changed:
            # The idle-stop net needs this where no green verify fires per step.
            state.settled.gateless_ever_edited = True
        if not self.chain.per_step:
            # `commit_per_step` governs the commit only; the metric block is promised regardless.
            return self._sample_metric(state, turn, sha="")
        commit_subject = self.checkpoints.subject(
            turn, fallback="checkpoint" if unjudged_changed else "verify passed"
        )
        sha = ""
        try:
            sha = self.checkpoints.commit(commit_subject, iteration=turn.iteration)
            turn.committed = bool(sha)
            # Adoption fills an absent command; a configured gate nobody may run stays as is.
            if (
                sha
                and not self.gate.command(state.verify)
                and self.gate.may_run(denied=state.verify.denied)
            ):
                self.gate.maybe_adopt(state, turn)
        except (git_ops.GitError, OSError) as exc:
            self.checkpoints.report_failure(exc, commit_subject, iteration=turn.iteration)
        # The operator's after-commit hook; the default answers "continue".
        if sha:
            directive = self.bridge.after_auto_commit(turn.iteration, sha)
            if directive in ("undo", "exit"):
                ended = self._steer_outcome(directive, turn.iteration, state)
                if ended is not None:
                    return ended
            if directive == "stop":
                self._log(f"LOOP: interactive stop at iter {turn.iteration}")
                # An operator stop reads "stopped", never "passed".
                return self._finish(
                    state,
                    _snapshot.End(
                        "interactive_stop",
                        f"stopped interactively after iter {turn.iteration}"
                        f"{self._dirty_tree_note()}",
                        completed=True,
                        checkpoint=False,
                        roots=True,
                    ),
                    iteration=turn.iteration,
                )
        return self._sample_metric(state, turn, sha=sha)

    def _sample_metric(
        self, state: _loop_state.LoopState, turn: _loop_state.TurnState, *, sha: str
    ) -> _snapshot.SessionResult | None:
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
            turn.metric_feedback = self.metrics.auto_feedback(
                state, iteration=turn.iteration, sha=sha
            )
        except errors.OperatorCommandUnexecutableError as exc:
            return self._unexecutable_abort(exc, iteration=turn.iteration, state=state)
        turn.metric_plateau_finish = self.metrics.plateau_finish(state.metric.history)
        return None

    def _turn_notices(self, state: _loop_state.LoopState, turn: _loop_state.TurnState) -> None:
        """Append the turn's review findings and metric feedback ahead of the advisors' notices.

        Args:
            state: The execution's state.
            turn: This turn's state.
        """
        if turn.review_text:
            turn.tool_results.append(_conversation.Notice(_panel.review_notice(turn.review_text)))
            turn.review_text = None
        if turn.metric_feedback:
            turn.tool_results.append(_conversation.Notice(turn.metric_feedback))

    # ---- finish gates --------------------------------------------------------

    def _turn_finish_gates(
        self, state: _loop_state.LoopState, turn: _loop_state.TurnState, ctx: _advice.TurnContext
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
        for gate in _finish_gates.FINISH_GATES:
            if self._refuse(state, turn, gate(turn, state, ctx)):
                return

    def _refuse(
        self,
        state: _loop_state.LoopState,
        turn: _loop_state.TurnState,
        refusal: _advice.Refusal | None,
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
            turn.tool_results.append(_conversation.Notice(refusal.text))
        self._record(refusal)
        state.settled.restart()
        return True

    def _end_gates(
        self,
        state: _loop_state.LoopState,
        turn: _loop_state.TurnState,
        ctx: _advice.TurnContext,
        *,
        ending: str,
        gates: tuple[_advice.Gate, ...],
    ) -> _snapshot.SessionResult | None:
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
        except errors.OperatorCommandUnexecutableError as exc:
            return self._unexecutable_abort(exc, iteration=turn.iteration, state=state)
        turn.ending = ending
        for gate in gates:
            if self._refuse(state, turn, gate(turn, state, ctx)):
                break
        if turn.review_text:
            # The turn's notices went out before these gates, so the panel's findings go here.
            turn.tool_results.append(_conversation.Notice(_panel.review_notice(turn.review_text)))
            turn.review_text = None
        return None

    # ---- the advisors --------------------------------------------------------

    def _turn_context(
        self, state: _loop_state.LoopState, *, iteration: int, execution_start: int
    ) -> _advice.TurnContext:
        """Build the facts the advisors read this turn.

        Args:
            state: The execution's state.
            iteration: The current iteration.
            execution_start: The first iteration of this execution.

        Returns:
            This turn's context.
        """
        return _advice.TurnContext(
            mode=self.mode,
            iteration=iteration,
            execution_start=execution_start,
            went_quiet_max_nudges=self.config.harness.went_quiet_max_nudges,
            loop_guard_kill_threshold=self.config.harness.loop_guard_kill_threshold,
            stagnation_notice_after_s=self.config.harness.stagnation_notice_after_s,
            verify_when=self.gate.when,
            verify_retries=self.gate.retries,
            finish_validator=self.finish_validator,
            metric=self.metrics.goal is not None,
            memory_wired=self.state_dir is not None,
            end_rejected=lambda turn, ending: self.reviewer.end_rejected(
                state, turn, ending=ending
            ),
            standing_absorb=lambda reason, iteration: self.standing.absorb(
                state, reason=reason, iteration=iteration
            ),
            gate_present=lambda: self.gate.present(state.verify),
            verify_command=lambda: self.gate.command(state.verify),
            tree_sha=self.chain.tree_sha,
            tree_green=lambda: self.gate.tree_green(state.verify),
            budget_remaining=self._budget_fraction_remaining,
            operator_wait_s=lambda: self.dispatcher.operator_wait_s,
            open_subtasks=self._open_subtasks,
        )

    def _budget_fraction_remaining(self) -> float | None:
        """Return the fraction of the token budget left, or None without a tracker."""
        if self.budget is None:
            return None
        return self.budget.fraction_remaining()

    def _turn_advisors(
        self, state: _loop_state.LoopState, turn: _loop_state.TurnState, ctx: _advice.TurnContext
    ) -> _snapshot.SessionResult | None:
        """Post the turn's notices, then apply each after-tools advisor's answer.

        Args:
            state: The execution's state.
            turn: This turn's state.
            ctx: This turn's context.

        Returns:
            The end when a gate's verify could not run, else None.
        """
        self._turn_notices(state, turn)
        for advisor in _guards.AFTER_TOOLS:
            aborted = self._take(state, turn, ctx, advisor(turn, state, ctx))
            if aborted is not None:
                return aborted
        return None

    def _take(
        self,
        state: _loop_state.LoopState,
        turn: _loop_state.TurnState,
        ctx: _advice.TurnContext,
        outcome: _advice.Nudge | _advice.Stop | None,
    ) -> _snapshot.SessionResult | None:
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
        if isinstance(outcome, _advice.Nudge):
            turn.tool_results.append(_conversation.Notice(outcome.text))
            self._record(outcome)
            return None
        if outcome.event:
            self._emit(outcome.event, **outcome.fields)
        if outcome.declared and turn.finish is None:
            aborted = self._end_gates(
                state, turn, ctx, ending=outcome.declared, gates=_finish_gates.END_GATES
            )
            if aborted is not None:
                return aborted
            if turn.end_returned:
                return None
        turn.stops.append(outcome)
        return None

    def _tell(self, conversation: _conversation.Conversation, nudge: _advice.Nudge | None) -> None:
        """Apply a before-call advisor's answer: the notice joins the conversation.

        Args:
            conversation: The conversation the notice enters.
            nudge: The advisor's answer; None when it had none.
        """
        if nudge is None:
            return
        conversation.notice(nudge.text)
        self._record(nudge)

    def _record(self, answer: _advice.Nudge) -> None:
        """Emit an answer's event and log its line, each when present.

        Args:
            answer: The advisor's or gate's answer.
        """
        if answer.event:
            self._emit(answer.event, **answer.fields)
        if answer.log:
            self._log(answer.log)

    # ---- stop checks, silent finish, went-quiet ------------------------------

    def _turn_stop_checks(
        self,
        state: _loop_state.LoopState,
        turn: _loop_state.TurnState,
        conversation: _conversation.Conversation,
    ) -> _snapshot.SessionResult | None:
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
                self._log(stop.log)
            return self._finish(state, stop.end(), iteration=turn.iteration)
        finish = turn.finish
        if finish is not None:
            self._log(f"LOOP: {finish.kind} called at iter {turn.iteration}")
            self.checkpoints.final(iteration=turn.iteration)
            # A finish_session over a red or stale verify reads "finished", not "passed".
            reason = _finish_gates.finish_reason(
                finish.kind,
                stale_gate=finish.stale_gate,
                tree_green=self.gate.tree_green(state.verify),
                verify=state.verify,
            )
            self._check_decisions_recorded(state)
            return self._finish(
                state,
                _snapshot.End(
                    reason,
                    _advice.with_open_tasks(finish.summary, self._open_subtasks()),
                    completed=True,
                    verdict="grounded" if finish.kind == "finish_session" else "passed",
                    checkpoint=False,
                    finish_payload=finish.payload,
                    stale_gate=finish.stale_gate,
                ),
                iteration=turn.iteration,
            )
        return None

    def _handle_no_tool_use(
        self,
        resp: ProviderResponse,
        assistant: _conversation.AssistantTurn,
        conversation: _conversation.Conversation,
        state: _loop_state.LoopState,
        ctx: _advice.TurnContext,
    ) -> _snapshot.SessionResult | None:
        """Handle a turn with no tool call: a silent finish with text, went-quiet without.

        A silent finish passes the gates a finish_session would; an empty turn is nudged
        up to a cap and never reads as success.

        Args:
            resp: The model's response.
            assistant: The turn as it entered the conversation.
            conversation: The conversation so far.
            state: The execution's state.
            ctx: This turn's context.

        Returns:
            The end, or None to go on after a nudge.
        """
        text = resp.text.strip() if resp.text else ""
        if text:
            # A prose turn is non-empty, so the went-quiet nudge budget refills.
            state.quiet.went_quiet_nudges_used = 0
            turn = _loop_state.TurnState(iteration=ctx.iteration, resp=resp, assistant=assistant)
            return self._handle_silent_finish(text, conversation, state, turn, ctx)
        return self._handle_went_quiet(resp, conversation, state, ctx)

    def _handle_silent_finish(
        self,
        text: str,
        conversation: _conversation.Conversation,
        state: _loop_state.LoopState,
        turn: _loop_state.TurnState,
        ctx: _advice.TurnContext,
    ) -> _snapshot.SessionResult | None:
        """Run a prose turn through the end gates as an implicit finish.

        Args:
            text: The turn's prose.
            conversation: The conversation the gates' notices enter.
            state: The execution's state.
            turn: This turn's state.
            ctx: This turn's context.

        Returns:
            The silent-finish end, or None when a gate sent the worker back to work.
        """
        iteration = turn.iteration
        if (stall := _quiet_turns.silent_no_work(state, ctx)) is not None:
            self._tell(conversation, stall)
            return None
        aborted = self._end_gates(
            state, turn, ctx, ending="silent_finish", gates=_finish_gates.SILENT_END_GATES
        )
        # A prose turn has no tool results, so the gates' notices go to the conversation.
        for item in turn.tool_results:
            if isinstance(item, _conversation.Notice):
                conversation.notice(item.text)
        if aborted is not None or turn.end_returned:
            return aborted
        if (asked := _quiet_turns.question_in_prose(state, ctx, text)) is not None:
            self._tell(conversation, asked)
            return None
        # A standing goal re-enters, else an interactive run parks for a steer; ask mode ends.
        cont = self._quiet_continuation(
            conversation, state, iteration=iteration, reason="silent_finish"
        )
        if cont is not None:
            return None if isinstance(cont, _loop_state.NextTurn) else cont
        # In ask mode the prose is the answer, so the end reads "answered".
        reason: _snapshot.SessionEndReason = "answered" if self.mode == "ask" else "silent_finish"
        if self.mode == "ask":
            self._log(f"  ask answered at iter {iteration}")
        else:
            self._log(
                f"LOOP: silent_finish at iter {iteration} - agent emitted text but no tool_use"
            )
        # Run and plan ground on the verify state as finish_session does; ask keeps the answer.
        return self._finish(
            state,
            _snapshot.End(
                reason,
                text
                if self.mode == "ask"
                else _advice.with_open_tasks(text[:1000], self._open_subtasks()),
                completed=True,
                verdict="grounded" if reason == "silent_finish" else "passed",
            ),
            iteration=iteration,
        )

    def _handle_went_quiet(
        self,
        resp: ProviderResponse,
        conversation: _conversation.Conversation,
        state: _loop_state.LoopState,
        ctx: _advice.TurnContext,
    ) -> _snapshot.SessionResult | None:
        """Nudge an empty turn up to the per-streak cap, then end the execution as went_quiet.

        The empty assistant turn leaves the conversation first: providers reject an
        assistant message with no content.

        Args:
            resp: The model's response.
            conversation: The conversation so far.
            state: The execution's state.
            ctx: This turn's context.

        Returns:
            The end, or None to go on after a nudge.
        """
        iteration = ctx.iteration
        reasoning_chars = _provider_call.reasoning_starvation(resp)
        starved = reasoning_chars > 0
        if starved:
            self._log(
                f"LOOP: reasoning_starvation at iter {iteration}"
                f" - stop_reason=length, reasoning_chars={reasoning_chars},"
                f" output_tokens={resp.output_tokens}; the model spent"
                f" its entire output budget on reasoning_content."
                f" Add this model to _REASONING_MODEL_HINTS in"
                f" providers/openai.py if it isn't already."
            )
            self._emit(
                "loop.reasoning_starvation",
                iteration=iteration,
                reasoning_chars=reasoning_chars,
                output_tokens=resp.output_tokens,
                stop_reason=resp.stop_reason,
            )
        # A subscription plan is metered in points, so it says "spent", not "billed".
        plan_metered = self.budget is not None and self.budget.snapshot().plan_latest is not None
        spent_word = "spent" if plan_metered else "billed"
        billed = (
            f" ({resp.output_tokens} output tokens {spent_word} on it: reasoning that never"
            " surfaced, or a tool call the provider dropped)"
            if resp.output_tokens > 0 and not starved
            else ""
        )
        self._log(
            f"LOOP: went_quiet at iter {iteration} - agent emitted no text and no tool_use{billed}"
        )
        # Every path below calls again or snapshots, and a provider rejects the empty turn.
        conversation.pop_quiet_assistant()
        if (nudge := _quiet_turns.went_quiet(state, ctx, resp)) is not None:
            self._tell(conversation, nudge)
            return None
        cont = self._quiet_continuation(
            conversation, state, iteration=iteration, reason="went_quiet"
        )
        if cont is not None:
            return None if isinstance(cont, _loop_state.NextTurn) else cont
        return self._finish(
            state,
            _snapshot.End("went_quiet", "(agent emitted no text and no tool_use)"),
            iteration=iteration,
        )

    # ---- the end -------------------------------------------------------------

    def _finish(
        self, state: _loop_state.LoopState, end: _snapshot.End, *, iteration: int
    ) -> _snapshot.SessionResult:
        """Record the end and return the execution's result.

        A grounded end reads `all_passed` off the final tree: True when it is observed
        green, False when red or stale, None when nothing gated it.

        Args:
            state: The execution's state.
            end: The end declared.
            iteration: The iteration it ended on.

        Returns:
            The execution's result.
        """
        if end.checkpoint:
            self.checkpoints.final(iteration=iteration)
        roots = end.roots if end.roots is not None else end.verdict != "failed"
        if roots:
            self._pass_pending_root_tasks()
        if end.event and end.verdict == "failed":
            self._emit(
                "session.end",
                reason=end.reason,
                iterations=iteration,
                all_passed=False,
                **end.fields,
            )
        elif end.event:
            grounded = end.verdict == "grounded"
            self._emit(
                "session.end",
                reason=end.reason,
                iterations=iteration,
                all_passed=self.gate.tree_green(state.verify) if grounded else True,
                scoped=state.verify.scoped if grounded else end.scoped,
            )
        self._record_memory_use(state)
        return _snapshot.SessionResult(
            completed=end.completed,
            verified=self.gate.verification(state.verify),
            reason=end.reason,
            summary=end.summary,
            iterations=iteration,
            tool_calls=state.tool_calls,
            finish_payload=end.finish_payload,
            stale_gate=end.stale_gate,
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
                    self._log(f"LOOP: auto-pass root {nid} refused: {exc}")
                except (
                    OSError,
                    pydantic.ValidationError,
                ) as exc:  # a write fault must not break finish
                    self._log(f"LOOP: auto-pass root {nid} failed: {exc}")
                    break  # a curator write failure fails for every remaining node too
        if changed:
            self._emit_graph_snapshot()

    def _record_memory_use(self, state: _loop_state.LoopState) -> None:
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
            self._log(f"LOOP: memory use record failed: {exc}")

    def _unexecutable_abort(
        self,
        exc: errors.OperatorCommandUnexecutableError,
        *,
        iteration: int,
        state: _loop_state.LoopState,
    ) -> _snapshot.SessionResult:
        """End the execution when the operator's verify or metric command cannot run.

        The model cannot fix operator config, so the end is loud rather than a gate that
        never executes.

        Args:
            exc: The error naming the command.
            iteration: The current iteration.
            state: The execution's state.

        Returns:
            The execution's result.
        """
        self._log(f"LOOP: aborting -- {exc}")
        # Verify never went green, so the edits may exist only in the worktree.
        return self._finish(
            state, _snapshot.End("verify_command_unexecutable", str(exc)), iteration=iteration
        )

    def _dirty_tree_note(self) -> str:
        """Return the summary suffix naming an uncommitted worktree, or "" outside run mode."""
        return self.chain.dirty_note() if self.mode == "run" else ""

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
        self._emit("graph.update", nodes=nodes, cursor=cursor)

    def _open_subtasks(self) -> list[tuple[str, str]]:
        """List the worker's open subtasks as (id, title) pairs, in run mode.

        The root stays open until the end, so counting it would deadlock every gate; a
        plan's tasks are its deliverable, open by design.

        Returns:
            The open subtasks; empty without a curator or outside run mode.
        """
        if self.curator is None or self.mode != "run":
            return []
        return _advice.open_subtasks(self.curator.nodes())

    def _checkpoint_graph_version(self) -> int:
        """Return the graph version for the per-turn checkpoint, 0 without a curator."""
        if self.curator is None:
            return 0
        return self.curator.graph_version

    # ---- the state dir: memory, decisions, skills ----------------------------

    def _load_memory_index(self) -> str:
        """Return the repo memory index for the system prompt.

        Returns:
            The index text; "" without a state dir, in agent mode, or when unreadable.
        """
        if self.state_dir is None or self.mode == "agent":
            return ""
        return agent6_memory.index_text(self.state_dir)

    def _load_decisions(self) -> str:
        """Return the operator's recorded rulings for the prompt, "" without a state dir."""
        return agent6_memory.decisions_text(self.state_dir) if self.state_dir is not None else ""

    def _record_decision(self, state: _loop_state.LoopState, question: str, answer: str) -> None:
        """Append an operator's answer to the repo's DECISIONS.md as a ruling.

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
            self._log(f"LOOP: decision not recorded: {exc}")
            self._emit("loop.decision.unrecorded", error=str(exc))
            return
        state.decisions_recorded.append(entry)
        self._log(f"LOOP: decision recorded ({len(answer)} chars)")
        self._emit("loop.decision.recorded", question=question[:200], answer=answer[:200])

    def _check_decisions_recorded(self, state: _loop_state.LoopState) -> None:
        """Report a recorded ruling missing from DECISIONS.md at the finish; never a block.

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
            self._log(f"LOOP: {len(missing)} recorded decision(s) missing from DECISIONS.md")
            self._emit("loop.decision.unrecorded", missing=len(missing))

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
            self._log(f"LOOP: skills: WARNING: {w}")
            self._emit("loop.skills.warning", warning=str(w))
        if resolved.enabled or resolved.always:
            self._log(
                f"LOOP: skills: {len(resolved.enabled)} indexed, {len(resolved.always)} always-on"
            )
            return resolved
        return None

    # ---- the run's helpers ---------------------------------------------------

    @functools.cached_property
    def gate(self) -> _verify_gate.VerifyGate:
        """Return the verify gate over the config's command."""
        wf = self.config.harness
        return _verify_gate.VerifyGate(
            configured=tuple(wf.verify_command),
            when=wf.verify_when,
            retries=wf.verify_retries,
            timeout_s=wf.verify_timeout_s,
            infer=wf.verify_infer,
            mode=self.mode,
            chain=self.chain,
            dispatcher=self.dispatcher,
            log=self._log,
            emit=self._emit,
        )

    @functools.cached_property
    def checkpoints(self) -> _checkpoint.Checkpoints:
        """Return the per-step commit writer."""
        return _checkpoint.Checkpoints(
            chain=self.chain,
            style=self.config.git.commit.checkpoint.message,
            enabled=self.mode == "run",
            provider=self.provider,
            log=self._log,
            emit=self._emit,
        )

    @functools.cached_property
    def compactor(self) -> _compactor.Compactor:
        """Return the context compaction driver."""
        return _compactor.Compactor(
            settings=self.compaction,
            provider=self.provider,
            curator=self.curator,
            mode=self.mode,
            dag_available=self.dispatcher.dag_available,
            decisions=self._load_decisions,
            compact_requested=self.bridge.compact_requested,
            compact_clear=self.bridge.compact_clear,
            log=self._log,
            emit=self._emit,
            emit_graph_snapshot=self._emit_graph_snapshot,
        )

    @functools.cached_property
    def standing(self) -> _standing.Standing:
        """Return the standing goal, the re-entry a soft end converts into."""
        return _standing.Standing(
            curator=self.curator,
            patience=self.config.harness.standing_patience,
            budget_remaining=self._budget_fraction_remaining,
            log=self._log,
            emit=self._emit,
        )

    @functools.cached_property
    def metrics(self) -> _metric_sampler.MetricSampler:
        """Return the metric sampler."""
        return _metric_sampler.MetricSampler(
            settings=self.config.harness.metric,
            enabled=self.mode == "run",
            dispatcher=self.dispatcher,
            log=self._log,
            emit=self._emit,
        )

    @functools.cached_property
    def operator_tasks(self) -> _operator_tasks.OperatorTasks:
        """Return the taker of the operator's queued tasks."""
        return _operator_tasks.OperatorTasks(
            curator=self.curator,
            take_requests=self.bridge.take_requests,
            revision=self.revision,
            root=self.chain.root,
            log=self._log,
            emit=self._emit,
            emit_graph_snapshot=self._emit_graph_snapshot,
        )

    @functools.cached_property
    def parallel(self) -> _parallel_dispatch.ParallelDispatcher:
        """Return the `/parallel` lane dispatcher."""
        return _parallel_dispatch.ParallelDispatcher(
            chain=self.chain,
            curator=self.curator,
            max_lanes=self.config.parallel.max_lanes,
            lane_spawner=self.bridge.lane_spawner,
            save_snapshot=self._save_resume_snapshot,
            log=self._log,
            emit=self._emit,
            emit_graph_snapshot=self._emit_graph_snapshot,
        )

    @functools.cached_property
    def steering(self) -> _operator.Steering:
        """Return the reader of a steer's text."""
        return _operator.Steering(
            bridge=self.bridge,
            parallel=lambda: self.parallel,
            dispatcher=self.dispatcher,
            record_decision=self._record_decision,
            log=self._log,
            emit=self._emit,
        )

    @functools.cached_property
    def reviewer(self) -> _reviewer.Reviewer:
        """Return the in-loop review panel."""
        return _reviewer.Reviewer(
            settings=self.review,
            chain=self.chain,
            review_tools=lambda: _toolset.build_readonly_review_tools(self.dispatcher),
            budget_remaining=self._budget_fraction_remaining,
            log=self._log,
            emit=self._emit,
        )

    @functools.cached_property
    def caller(self) -> _provider_call.ProviderCaller:
        """Return the worker's provider caller under the retry knobs."""
        return _provider_call.ProviderCaller(
            provider=self.provider,
            retry_count=self.call.retry_count,
            retry_delay_s=self.call.retry_delay_s,
            retry_max_delay_s=self.call.retry_max_delay_s,
            temperature=self.call.temperature,
            should_abort=self.bridge.should_abort,
            should_interrupt=self.bridge.should_interrupt,
            log=self._log,
            emit=self._emit,
        )

    # ---- steering and operator boundaries ------------------------------------

    def _operator_boundary(
        self, conversation: _conversation.Conversation, iteration: int, state: _loop_state.LoopState
    ) -> _snapshot.SessionResult | None:
        """Honour a pending stop, then poll the steer flag, after every completed iteration.

        A stop or an injected instruction never splits a tool call from its result.

        Args:
            conversation: The conversation a steer enters.
            iteration: The iteration just completed.
            state: The execution's state.

        Returns:
            The end the operator asked for, or None when the execution goes on.
        """
        # A background command's ending reaches disk only when observed; `/shells` reads it.
        self.dispatcher.settle_background()
        if self.bridge.stop_requested():
            self.bridge.stop_clear()
            self._log(f"LOOP: operator stop at the step boundary (iter {iteration})")
            return self._finish(
                state,
                _snapshot.End(
                    "steer_abort",
                    f"operator stopped the run after step {iteration}{self._dirty_tree_note()}",
                    checkpoint=False,
                ),
                iteration=iteration,
            )
        return self._steer_outcome(
            self.steering.handle(conversation, iteration, state), iteration, state
        )

    def _quiet_continuation(
        self,
        conversation: _conversation.Conversation,
        state: _loop_state.LoopState,
        *,
        iteration: int,
        reason: str,
    ) -> _snapshot.SessionResult | _loop_state.NextTurn | None:
        """Continue a quiet turn in run mode: a standing goal re-enters, else a park.

        Args:
            conversation: The conversation the re-entry enters.
            state: The execution's state.
            iteration: The current iteration.
            reason: The quiet end being continued.

        Returns:
            `NEXT_TURN` to go on, the end a park chose, or None when neither applies.
        """
        if self.mode != "run":
            return None
        nudge = self.standing.absorb(state, reason=reason, iteration=iteration)
        if nudge is not None:
            conversation.notice(nudge)
            return _loop_state.NEXT_TURN
        if self.interactive:
            parked = self._park_for_steer(conversation, state, iteration=iteration, reason=reason)
            return _loop_state.NEXT_TURN if parked is None else parked
        return None

    def _park_for_steer(
        self,
        conversation: _conversation.Conversation,
        state: _loop_state.LoopState,
        *,
        iteration: int,
        reason: str,
    ) -> _snapshot.SessionResult | None:
        """Park a quiet interactive execution until the operator steers it.

        The conversation stays in memory with its snapshot on disk; there is no timeout.

        Args:
            conversation: The conversation a steer enters.
            state: The execution's state.
            iteration: The current iteration.
            reason: The quiet end being parked.

        Returns:
            The end a steer verb chose, or None when the execution continues.
        """
        self._log(
            f"LOOP: parked at iter {iteration} ({reason}) - waiting for your steer"
            " (any composer or the pause menu; abort ends the run)"
        )
        self._emit("loop.parked", iteration=iteration, reason=reason)
        while True:
            if self.bridge.stop_requested():
                self.bridge.stop_clear()
                return self._finish(
                    state,
                    _snapshot.End(
                        "steer_abort",
                        f"operator stopped the parked run{self._dirty_tree_note()}",
                        checkpoint=False,
                    ),
                    iteration=iteration,
                )
            if self.bridge.should_abort():
                return self._steer_outcome("abort", iteration, state)
            if self.bridge.steer_requested():
                verb = self.steering.handle(conversation, iteration, state)
                if verb is not None:
                    return self._steer_outcome(verb, iteration, state)
                # Injected (or a bare poke): the run continues where it parked.
                self._emit("loop.parked.resumed", iteration=iteration)
                return None
            if self.operator_tasks.take(state.root_task_id):
                # A queued task, goal or retirement is work; the next focus banner names it.
                self._emit("loop.parked.resumed", iteration=iteration)
                return None
            time.sleep(0.5)

    def _steer_outcome(
        self, steer_result: str | None, iteration: int, state: _loop_state.LoopState
    ) -> _snapshot.SessionResult | None:
        """Map a steer verb to the execution's end.

        Args:
            steer_result: The verb `Steering.handle` returned; None for an injected instruction.
            iteration: The current iteration.
            state: The execution's state.

        Returns:
            The end, or None when the execution goes on.
        """
        if steer_result in ("abort", "exit"):
            # "exit" is the same stop; its reason tells the CLI to skip the follow-up prompt.
            reason: _snapshot.SessionEndReason = (
                "steer_exit" if steer_result == "exit" else "steer_abort"
            )
            return self._finish(
                state,
                _snapshot.End(
                    reason,
                    f"operator {'exited' if steer_result == 'exit' else 'aborted'}"
                    f" at iter {iteration} via steering prompt{self._dirty_tree_note()}",
                    checkpoint=False,
                ),
                iteration=iteration,
            )
        if steer_result == "undo":
            forked = self.bridge.undo_forker() if self.bridge.undo_forker is not None else None
            if forked is None:
                # The forker printed why, or none is wired.
                self._log("  /undo: nothing to undo; continuing")
                return None
            new_id, undone_text = forked
            self._emit("session.undone", new_session_id=new_id, undone_text=undone_text)
            # An undo is the operator's own end; without a session.end the run reads stale.
            return self._finish(
                state,
                _snapshot.End(
                    "undone",
                    f"operator undid the last message at iter {iteration}; forked to {new_id}",
                    checkpoint=False,
                ),
                iteration=iteration,
            )
        if steer_result == "detach":
            # No session.end: the caller respawns a detached `resume` that appends to this log.
            return self._finish(
                state,
                _snapshot.End(
                    "detached",
                    f"operator detached at iter {iteration}; resuming in the background",
                    checkpoint=False,
                    event=False,
                ),
                iteration=iteration,
            )
        return None

    # ---- the log and the events ----------------------------------------------

    @property
    def session_id(self) -> str:
        """Return the session id, the run dir's name; "" without a log."""
        return self.events.path.parent.name if self.events is not None else ""

    def _log(self, msg: str) -> None:
        """Write one line to the run log."""
        self.logger(f"[agent6] {msg}")

    def _emit(self, event_type: str, **fields: Any) -> None:
        """Emit one event when a log is wired."""
        if self.events is not None:
            self.events.emit(event_type, **fields)

    def _emit_start(self, event_type: str, **fields: Any) -> None:
        """Emit a start-family event through the emitter that stamps the worker pid."""
        if self.events is not None:
            ipc.emit_session_start(self.events, self.events.path.parent, event_type, **fields)

    def _emit_budget(self, iteration: int) -> None:
        """Emit the per-iteration usage heartbeat that tells a long model call from a stall.

        Args:
            iteration: The iteration about to call.
        """
        if self.budget is None:
            return
        snap = self.budget.snapshot()
        cost, _ = self.budget.estimate_usd()
        self._emit(
            "loop.budget",
            iteration=iteration,
            input_tokens=snap.input_total,
            output_tokens=snap.output_total,
            cache_read_tokens=snap.cache_read_total,
            cost_usd=round(cost, 6),
        )
