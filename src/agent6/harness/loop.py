# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The agent loop: one system prompt, one model driving tool calls, and a
deterministic harness around it (jail, budget, verify timeout, DAG curator
for persistence and resume). One driver: the review panel gates checkpoints
and never steers. Green verifies auto-commit, so the chain records each tree
a verify certified. The heuristics that nudge or end a run are advisor
functions (`_guards`, `_metric`, `_quiet_turns`) and finish gates
(`_finish_gates`), run in a declared order; the loop applies their answers.
"""

from __future__ import annotations

import itertools
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import ValidationError

from agent6.budget import BudgetExceeded, BudgetTracker
from agent6.config import Config
from agent6.git_ops import (
    GitError,
)
from agent6.graph.curator import CuratorError, GraphCurator
from agent6.graph.models import (
    SetCursorIntent,
    UpdateStatusIntent,
)
from agent6.graph.order import OPEN_STATUSES
from agent6.harness._advice import (
    Gate,
    Nudge,
    Refusal,
    Stop,
    TurnContext,
    open_subtasks,
    with_open_tasks,
)
from agent6.harness._chain import RunChain
from agent6.harness._checkpoint import Checkpoints
from agent6.harness._compaction import (
    CompactionSettings,
    cap_tool_result,
    count_elisions,
    request_prefix_chars,
)
from agent6.harness._compactor import Compactor
from agent6.harness._context import load_repo_summary
from agent6.harness._conversation import (
    AssistantTurn,
    Conversation,
    Notice,
    ToolResultItem,
)
from agent6.harness._dag_focus import (
    DAG_MUTATING_TOOLS,
    current_task_banner,
    current_task_id,
    initial_dag_hint,
)
from agent6.harness._finish_gates import (
    END_GATES,
    FINISH_GATES,
    SILENT_END_GATES,
    FinishCall,
    finish_reason,
)
from agent6.harness._guards import (
    AFTER_TOOLS,
    BEFORE_CALL,
    stuck_on_task,
    tool_error_ladder,
    unreachable_tool,
)
from agent6.harness._loop_state import (
    NEXT_TURN,
    LoopState,
    NextTurn,
    TurnState,
    restore_completion_state,
)
from agent6.harness._memory_touch import memory_store_facts
from agent6.harness._metric import (
    best_metric_sample,
)
from agent6.harness._metric_sampler import MetricSampler
from agent6.harness._nudges import (
    PLAN_ON_DISK_HEADER,
    tool_error_signature,
)
from agent6.harness._operator import PINS_MAX_CHARS, OperatorBridge, Steering, try_pin
from agent6.harness._operator_tasks import OperatorTasks
from agent6.harness._panel import (
    review_notice,
)
from agent6.harness._parallel_dispatch import (
    ParallelDispatcher,
)
from agent6.harness._prompt_blocks import build_system_prompt, initial_instructions
from agent6.harness._prompt_revision import (
    PromptRevisionDeclined,
    PromptRevisionError,
    RevisionSettings,
    revise_prompt,
)
from agent6.harness._provider_call import (
    CallSettings,
    ProviderCaller,
    provider_error_hint,
    reasoning_starvation,
)
from agent6.harness._quiet_turns import question_in_prose, silent_no_work, went_quiet
from agent6.harness._review import Reviewer, ReviewSettings
from agent6.harness._snapshot import (
    TURN_IN_FLIGHT_NAME,
    End,
    ResumeError,
    SessionEndReason,
    SessionResult,
    SessionSnapshot,
    clear_turn_marker,
    load_session_snapshot,
    write_turn_marker,
)
from agent6.harness._standing import Standing
from agent6.harness._toolset import (
    build_readonly_review_tools,
    tool_definitions,
)
from agent6.harness._verify_gate import EXIT_TIMEOUT, VerifyGate
from agent6.harness._verify_verdict import VerifyVerdict
from agent6.memory import (
    decisions_path,
    decisions_text,
    memory_dir,
    record_decision,
    record_use,
)
from agent6.memory import index_text as memory_index_text
from agent6.paths import mkdir_for_real_user
from agent6.portable import atomic_write
from agent6.prompts.revision import (
    pinned_block,
)
from agent6.providers import (
    Provider,
    ProviderAborted,
    ProviderError,
    ProviderInterrupted,
    ProviderResponse,
    ToolDefinition,
)
from agent6.sessions.ipc import (
    emit_session_start,
)
from agent6.skills import ResolvedSkills
from agent6.task_text import operator_task_text
from agent6.tools.dispatch import (
    OperatorCommandUnexecutable,
    ToolDenied,
    ToolDispatcher,
    ToolError,
)
from agent6.tools.mcp_client import MCP_TOOL_PREFIX
from agent6.tools.results import (
    AnswersResult,
    ExecResult,
    MetricResult,
    PreviewResult,
    ToolResult,
)
from agent6.tools.schema import (
    ReadBackgroundInput,
)

# A re-served tool result must exceed this many bytes before the back-to-back
# dedupe elides it; below it the stub would not save enough to matter and the
# small results (finish/dag echoes) should pass through verbatim.
_DEDUPE_MIN_CHARS = 500


if TYPE_CHECKING:
    from agent6.events import EventSink


# Consecutive went-quiet turns after which a metric run drops the worker's
# per-call output cap from metric_task_max_tokens back to per_call_max_tokens
# (see Harness._worker_max_tokens). 2 spares a one-off starvation its full
# recovery room while breaking a reasoning-binge spiral.
_STARVATION_BACKOFF_AFTER_QUIETS = 2


@dataclass
class Harness:
    """Single-loop agent harness.

    The agent decides everything via tool calls in one large loop:
    when to read, when to plan (implicitly via subsequent tool calls),
    when to edit, when to verify, when to measure the metric, when to
    pivot, when to stop. The harness keeps the loop bounded
    (max_iterations, budget caps, verify_timeout) and observable
    (events).
    """

    # The run's commit chain: the repository root, where per-step commits go
    # and what the worktree holds beyond them.
    chain: RunChain
    config: Config
    provider: Provider
    dispatcher: ToolDispatcher
    logger: Callable[[str], None] = field(default=print)
    events: EventSink | None = None
    # In-process GraphCurator. When None,
    # DAG-as-tool handlers raise ToolError and the loop runs without DAG
    # persistence (still usable for bench / one-off tasks). When wired,
    # Harness.run() seeds a root task and the agent can add subtasks
    # and update statuses; survives crashes via <run-dir>/graph.jsonl.
    curator: GraphCurator | None = None
    # Per-invocation token budget tracker (the same instance wired into
    # the provider). When present the loop can read how much budget
    # remains and use it to decide whether a metric plateau is worth
    # quitting on. None in test / MCP paths; the loop degrades to fixed
    # count-based heuristics when it is unset.
    budget: BudgetTracker | None = None
    # Per-repo state dir holding the cross-run memory store
    # (<state_dir>/memory/). When set, the memory index is injected into
    # the system prompt at run start; the CLI wires the same path into the
    # dispatcher so memory-dir edits persist across runs.
    # None (bench / tests / one-off embedders) runs memory-less.
    state_dir: Path | None = None
    # Cap on assistant turns for THIS execution (config [harness].max_iterations;
    # -1 unlimited). Each turn = one provider.call. A resumed execution re-arms the
    # allowance: the cap is relative to its start_iteration, so a standing
    # run is bounded per execution, never by the sum of its history.
    max_iterations: int = 200
    # A machine agent state's finish contract: called on each finish_session
    # payload, returning the problems (empty = conforms). Injected by the
    # machine execution builder from the state's output_schema; None (every plain
    # run) leaves finishes ungated. The engine's own validation of the
    # recorded fact stays the authority.
    finish_validator: Callable[[dict[str, Any] | None], list[str]] | None = None
    # What the operator can do to the run, as the front-end injects it.
    bridge: OperatorBridge = field(default_factory=OperatorBridge)
    # The worker call's knobs: retries, temperature, output caps.
    call: CallSettings = field(default_factory=CallSettings)
    # Context compaction: the tiers' thresholds, the tail kept, the summariser.
    compaction: CompactionSettings = field(default_factory=CompactionSettings)
    # The in-loop review panel: its trigger, seats and decision rule.
    review: ReviewSettings = field(default_factory=ReviewSettings)
    # The one-shot prompt revision before the first worker call.
    revision: RevisionSettings = field(default_factory=RevisionSettings)
    # Pins seeded before the first turn (a /parallel lane inherits the
    # coordinator's standing instructions via the spawner's --pin channel,
    # out-of-band of user_task). Fresh runs only; resume/fork restore pins
    # from the snapshot instead.
    initial_pins: Sequence[str] = ()
    # When set, Harness writes a JSON snapshot of (system, messages,
    # tool_calls, next_iteration, root_task_id) before every LLM call. The
    # snapshot is provider-agnostic (it holds the anthropic-shaped message
    # list the loop maintains internally, not the on-the-wire OpenAI-shaped body
    # the openai provider sends) so `agent6 resume` works regardless of which
    # provider the prior run used. Atomic write (tmp + rename) so a crash
    # mid-write leaves the prior snapshot intact.
    resume_state_path: Path | None = None
    # The operator's standing goal (`run --standing`): seeded as a standing
    # task under the root at run start. "" = none.
    standing_goal: str = ""
    # The gate a resumed execution carried from the last one (`_carry_adopted_gate`),
    # set at the execution's start for the state the execution then builds.
    _adopted_on_resume: tuple[str, ...] = ()
    # An operator is watching and can steer live (a foreground CLI/TUI run or
    # an interactive resume). A quiet turn then PARKS for a steer instead of
    # ending: interactively, going quiet is the most normal thing an agent
    # does, not a failure.
    interactive: bool = False
    # Plan mode. When `mode="plan"`, the harness uses the
    # planning system prompt + plan-mode tool list (no apply_edit /
    # apply_patch; finish_planning replaces finish_session), skips auto-
    # commit-on-verify-pass, and on finish_planning writes the
    # `plan_markdown` argument to `plan_output_path` before exiting.
    # `plan_output_path` is required when `mode="plan"`.
    mode: Literal["run", "plan", "ask", "agent"] = "run"
    plan_output_path: Path | None = None
    # The guards' knobs: the quiet-turn cap, the loop-guard kill, the stagnation notice.
    # One-shot guard so a persistently unwritable state dir (full disk, quota,
    # read-only mount) warns once instead of every turn. Snapshot persistence is
    # recovery state; a failure disables resume/fork but must not abort the run.
    _snapshot_write_failed: bool = field(default=False, init=False)
    # The loop iteration currently being driven (0 before the loop starts). The
    # app-level KeyboardInterrupt fallbacks in run/resume read it so their
    # emergency session.end carries a truthful iteration count, matching the shape
    # the loop's own session.end emitters use.
    iterations_reached: int = field(default=0, init=False)

    # ---- run / resume entry --------------------------------------------------

    def run(self, user_task: str) -> SessionResult:
        """Drive the single-loop agent to completion."""
        self.bridge.steer_reset()  # an execution starts with no armed Ctrl-C
        if self.mode == "plan" and self.plan_output_path is None:
            raise ValueError("Harness(mode='plan') requires plan_output_path to be set")
        # The event carries the operator's own words (a seed digest or skill
        # block prepended by `run --from`/`--skill` is context, not the task),
        # clipped: every headline reads this field.
        self._emit_start(
            "session.start",
            session_id=self.session_id,
            user_task=operator_task_text(user_task)[:200],
            mode=self.mode,
        )
        self._log("LOOP: LOAD_CONTEXT")
        repo = load_repo_summary(self.chain.root)
        system = build_system_prompt(
            config=self.config,
            repo=repo,
            mode=self.mode,
            memory_index=self._load_memory_index(),
            memory_dir_path=str(memory_dir(self.state_dir)) if self.state_dir is not None else "",
            decisions=self._load_decisions(),
            decisions_path=str(decisions_path(self.state_dir)) if self.state_dir else "",
            skills=self._load_skills(),
            isolation=self.dispatcher.isolation,
            commands_allowed=self.dispatcher.command_policy() != "no",
            protected_paths=bool(self.dispatcher.extra_protect_paths),
            dag_available=self.dispatcher.dag_available,
        )

        try:
            effective_task = revise_prompt(
                self.revision, user_task, repo, log=self._log, emit=self._emit
            )
        except PromptRevisionError as exc:
            declined = isinstance(exc, PromptRevisionDeclined)
            end_reason: SessionEndReason = "steer_abort" if declined else "prompt_revision_failed"
            self._log(f"LOOP: prompt revision {'declined' if declined else 'failed'}: {exc}")
            self._emit("session.end", reason=end_reason, iterations=0, all_passed=False)
            return SessionResult(
                completed=False, reason=end_reason, summary=str(exc), iterations=0, tool_calls=0
            )

        # Seed the run's root task and wire its id into the
        # dispatcher so add_task with parent_id=None has a parent. Skipped
        # gracefully if no curator is configured (DAG tools then
        # raise ToolError if called).
        root_id = self.operator_tasks.seed_root(effective_task)
        if root_id is not None:
            self.dispatcher.set_run_root_node_id(root_id)
            self._log(f"LOOP: DAG root task seeded: {root_id}")
            self.operator_tasks.seed_standing(root_id, self.standing_goal)
            self._emit_graph_snapshot()  # show the root in the live task view

        self._log(
            f"LOOP: mode={self.mode} system={len(system)} chars, task={len(effective_task)} chars"
        )

        # Initial user turn - the task + a brief operational header.
        # Cache breakpoints are rolled by the conversation each iteration,
        # so the growing history stays cached across turns.
        dag_hint = initial_dag_hint(root_id, self.mode, self.config.prompt.decompose == "on")
        instructions = initial_instructions(
            self.mode,
            self.config.sandbox.run_commands,
            has_gate=self.gate.present(VerifyVerdict()),  # the start: nothing adopted or denied
        )
        initial_user = f"TASK:\n{effective_task}\n\n{instructions}{dag_hint}"
        conversation = Conversation()
        conversation.notice(initial_user)

        return self._drive_loop(
            system=system,
            conversation=conversation,
            tool_calls=0,
            start_iteration=1,
            root_task_id=root_id,
            original_task=effective_task,
        )

    def resume(self) -> SessionResult:
        """Resume a paused/crashed run from its snapshot.

        Reads `self.resume_state_path` (the snapshot written by the
        loop before each LLM call), reattaches the DAG root task id to
        the dispatcher, and re-enters the loop at the saved iteration
        with the saved conversation. The budget tracker is fresh per
        invocation (by design - see `agent6.budget` docstring); the
        DAG state on disk is restored by spawning a curator against the
        same run layout in the CLI.
        """
        self.bridge.steer_reset()  # an execution starts with no armed Ctrl-C
        if self.resume_state_path is None:
            raise ResumeError("resume() called but resume_state_path is None")
        try:
            snapshot = load_session_snapshot(self.resume_state_path)
            conversation = Conversation.from_wire(snapshot.messages)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise ResumeError(
                f"failed to load resume snapshot from {self.resume_state_path}: {exc}"
            ) from exc

        # The execution's log opens with this event: stamp session_id + mode like
        # session.start so the log identifies itself (the manifest owns the task).
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

        # The system prompt is the run's, frozen: config that gained (or lost) a
        # verify command between executions swaps what judges the work while the
        # instructions still name the old gate. Say so rather than let the
        # worker run a command nothing checks. A gate the execution dropped because
        # commands are withheld is no swap: no command can run, that one
        # included.
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
        self, state: LoopState, conversation: Conversation, resume_from: SessionSnapshot | None
    ) -> None:
        """Seed the execution's carried state and announce it for the read model.

        A resumed/forked execution re-announces its restored pins and elision
        counters: a fork's fresh logs.jsonl has no pin.added or compact
        events to fold (the fold REPLACES on these events, so a plain resume
        never double-counts). Announced even when empty: a pin whose
        pin.added reached the log but whose snapshot never did is still
        folded from the same log, so only a replace with the real (empty)
        list stops the surfaces listing a pin no restart will re-inject.

        A FRESH run seeded with pins (--pin; the /parallel lane channel) uses
        the same state, the same replace-fold event, and the same block a
        restart re-shows, so the wording never depends on the delivery path.
        """
        if resume_from is not None:
            restore_completion_state(state, resume_from)
            state.verify.adopted = self._adopted_on_resume
            self._carry_verify_verdict(state, resume_from)
            self._emit("loop.pin.restored", pins=list(state.pins), count=len(state.pins))
            elided, gists = count_elisions(conversation)
            self._emit("loop.compact.restored", elided=elided, gists=gists)
        elif self.initial_pins:
            # Seed via the pin owner so --pin honors the cap + non-empty check;
            # a --pin that doesn't fit is refused loudly.
            for pin in self.initial_pins:
                if not try_pin(state.pins, pin):
                    self._log(f"  --pin refused (empty or over the {PINS_MAX_CHARS}-char cap)")
                    self._emit("loop.pin.refused", chars=len(pin), limit=PINS_MAX_CHARS)
            self._emit("loop.pin.restored", pins=list(state.pins), count=len(state.pins))
            if state.pins:
                conversation.notice(pinned_block(state.pins))

    def _carry_adopted_gate(self, snapshot: SessionSnapshot) -> tuple[str, ...]:
        """The gate a gateless run adopted in an earlier execution, carried into this
        one: the snapshot's command when the config names none and the jail
        can run it (the dispatcher takes it again, as the adoption did). `()`
        otherwise, and the execution-start notice reads the gate as swapped."""
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

    def _carry_verify_verdict(self, state: LoopState, snap: SessionSnapshot) -> None:
        """Carry the prior execution's verify observation when it still describes THIS
        tree: the chain tip is the snapshot's (`RunChain.checkpoint_head_sha` wrote
        it; a chain commit moves neither HEAD nor the checkout) and the
        worktree holds nothing the chain does not. An operator commit or edit
        between executions invalidates it -- fails closed, like the baseline probe,
        so the execution starts unobserved rather than wrongly green or red.
        `baseline_ok` is about the BASE commit, which resume never moves: it
        carries unconditionally."""
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
        except (GitError, OSError):
            return
        if not dirty:
            state.verify.last_ok = snap.last_verify_ok
            state.verify.edited_since = snap.edited_since_verify

    def _save_resume_snapshot(
        self,
        state: LoopState,
        messages: list[dict[str, Any]],
        *,
        next_iteration: int,
        write_checkpoint: bool = False,
    ) -> None:
        """Write loop state to disk for resume.

        Called before each LLM call and again at the end of each iteration
        (after the executed tool_results are appended) so a crash after a
        non-idempotent tool dispatch resumes from AFTER the executed tools
        rather than replaying them. Every call advances `loop_state.json`
        (the latest pointer resume follows); only the pre-call save passes
        `write_checkpoint` and owns `checkpoints/<next_iteration>.json` -- the
        state that turn's provider call consumes, written once, so
        `fork --at-turn N` has one meaning. Atomic via tmp-file + replace so a
        crash mid-write leaves the prior snapshot intact. No-op if
        `resume_state_path` is None (e.g. unit tests).
        """
        if self.resume_state_path is None:
            return
        goal = self.metrics.goal
        best = best_metric_sample(state.metric.history, goal=goal) if goal is not None else None
        snapshot = SessionSnapshot(
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
        # The snapshot is recovery state, not run output: an unwritable state dir
        # (full disk, quota, read-only mount) disables resume/fork but must not
        # abort an otherwise-healthy run whose edits + commits are already on disk
        # independently. Warn once, then continue.
        try:
            # Write the append-only checkpoint first, then advance loop_state.json
            # as the latest pointer. If the second write fails, default fork still
            # follows loop_state.json, while explicit --at-turn can use the durable
            # checkpoint.
            if write_checkpoint:
                cp_dir = self.resume_state_path.parent / "checkpoints"
                atomic_write(cp_dir / f"{next_iteration:04d}.json", blob)
            atomic_write(self.resume_state_path, blob)
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
        conversation: Conversation,
        tool_calls: int,
        start_iteration: int,
        root_task_id: str | None,
        original_task: str,
        resume_from: SessionSnapshot | None = None,
    ) -> SessionResult:
        """Shared loop body for both fresh `run()` and `resume()`: one
        `TurnState` per tool-use iteration, driven through the turn phases
        in order. Any phase returning a SessionResult ends the run.

        `original_task` is the exact task string (in-loop review calls ground
        on it): run() threads it straight through, resume() reads it verbatim
        from the snapshot -- never re-derived from the message history.

        Before each provider call, writes a snapshot of the harness's
        in-memory state to `self.resume_state_path` (if set) so a
        crash mid-call can be resumed from the same point.
        """
        state = LoopState(
            original_task=original_task,
            tool_calls=tool_calls,
            # steer-boundary phases parent DAG nodes here, and snapshot with
            # the system prompt (see ParallelDispatcher.dispatch).
            root_task_id=root_task_id,
            system=system,
        )
        self._seed_carryover(state, conversation, resume_from)
        # This EXECUTION's allowance: start..start-1+max (-1 = unbounded); a resumed
        # execution re-arms rather than inheriting a spent absolute counter.
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
            # Rebuilt per turn, not per execution: a gate adopted mid-run, or a
            # policy the operator denies mid-run, changes what the worker has.
            # A frozen list offers a tool that is gone, or keeps offering one
            # that only raises. Built BEFORE the context prep, which measures
            # the request the tools ride in.
            tools = tool_definitions(self.dispatcher, mode=self.mode)
            ctx = self._turn_context(state, iteration=iteration, execution_start=start_iteration)
            wire = self._turn_pre_call(
                conversation=conversation,
                state=state,
                ctx=ctx,
                prefix_chars=request_prefix_chars(system, tools),
            )
            if isinstance(wire, SessionResult):
                return wire
            got = self._turn_provider_call(
                system,
                conversation,
                wire,
                tools,
                state,
                iteration=iteration,
            )
            if isinstance(got, SessionResult):
                return got
            if isinstance(got, NextTurn):
                continue
            # The response's blocks enter the history verbatim, so tool_use
            # IDs (and thinking blocks) round-trip cleanly.
            assistant = conversation.assistant(got.raw.get("content") or [])
            if not assistant.tool_uses:
                result = self._handle_no_tool_use(got, assistant, conversation, state, ctx)
                if result is not None:
                    return result
                # A completed prose turn is snapshotted like a tool turn, so an
                # operator stop at the boundary below resumes from AFTER the
                # prose + nudge instead of re-paying the provider call.
                self._save_resume_snapshot(
                    state, conversation.to_wire(), next_iteration=iteration + 1
                )
                # A prose turn is a completed iteration too: without this
                # boundary a model answering in prose could never be stopped
                # or steered.
                outcome = self._operator_boundary(conversation, iteration, state)
                if outcome is not None:
                    return outcome
                continue
            turn = TurnState(iteration=iteration, resp=got, assistant=assistant)
            # BEFORE dispatch: a crash between a tool's side effect and the
            # after-tools snapshot below leaves this marker at the iteration
            # resume would re-run, so resume can ask instead of silently
            # replaying a non-idempotent effect.
            if self.resume_state_path is not None:
                write_turn_marker(
                    self.resume_state_path.parent / TURN_IN_FLIGHT_NAME,
                    iteration,
                    tuple(tu.name for tu in assistant.tool_uses),
                )
            result = self._turn_dispatch_tools(state, turn, ctx)
            if result is not None:
                return result
            # One task-DAG snapshot per turn (not per mutation), so several
            # add_task/update_task calls in a turn collapse to a single event.
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
            # Snapshot AFTER the executed tools (assistant turn + tool_results
            # are in the conversation) so a crash before iteration N+1's
            # pre-call snapshot resumes from AFTER the dispatched tools instead
            # of replaying them. The dispatch->snapshot window itself stays
            # open (the side effect and this write are not atomic); the
            # in-flight marker above covers it, so resume detects the one case
            # where replay may repeat a non-idempotent effect and asks. Marker
            # deletion comes AFTER this write: a crash mid-snapshot then leaves
            # a stale marker resume clears silently, never a missed one.
            self._save_resume_snapshot(state, conversation.to_wire(), next_iteration=iteration + 1)
            if self.resume_state_path is not None:
                clear_turn_marker(self.resume_state_path.parent / TURN_IN_FLIGHT_NAME)
            result = self._turn_stop_checks(state, turn, conversation)
            if result is not None:
                return result
            outcome = self._operator_boundary(conversation, iteration, state)
            if outcome is not None:
                return outcome

        self._log(f"LOOP: max_iterations={self.max_iterations} reached")
        return self._finish(
            state,
            End(
                "max_iterations",
                f"max_iterations={self.max_iterations} reached without finish_session",
            ),
            iteration=self.iterations_reached,
        )

    def _seeded_steer(
        self, conversation: Conversation, iteration: int, state: LoopState
    ) -> SessionResult | None:
        """Consume the follow-up a `resume --steer` queued before the loop
        started (`resume.py` write_steer_answer).

        Up front, so it enters the conversation ahead of the first provider
        call and drives this turn: a resumed already-finished conversation
        silent-finishes on iteration 1 and returns before the end-of-iteration
        poll ever runs, dropping the follow-up. Only the first resumed
        iteration -- mid-run Ctrl-C steering stays on the completed-iteration
        poll, and a Ctrl-C cannot precede this point."""
        return self._steer_outcome(
            self.steering.handle(conversation, iteration, state), iteration, state
        )

    def _turn_pre_call(
        self,
        *,
        conversation: Conversation,
        state: LoopState,
        ctx: TurnContext,
        prefix_chars: int = 0,
    ) -> list[dict[str, Any]] | SessionResult:
        """Prepare the context for this turn's provider call: budget heartbeat,
        tiered compaction, the plan re-read, pre-call nudges, rolling cache
        breakpoints, then the pre-call resume snapshot. Returns the serialized
        wire, so the snapshot on disk and the provider call carry the same list
        by construction -- or the parked SessionResult when the plan file
        cannot be read.

        The cache breakpoints advance AFTER compaction + nudges (the tail must
        be final) and BEFORE the snapshot (markers persist across resume).
        After the snapshot write, a crash anywhere up to the next iteration's
        snapshot can be resumed by re-running this same call."""
        self._emit_budget(ctx.iteration)
        if self.compactor.compact(conversation, state, prefix_chars=prefix_chars):
            # A tier-2 restart wiped the surfaced focus banner and the plan
            # block; let the passes below put both back into the fresh context.
            state.focus.surfaced_task_id = None
            state.plan_injected = ""
        parked = self._maybe_inject_plan(conversation, state, iteration=ctx.iteration)
        if parked is not None:
            return parked
        self._turn_before_call(conversation, state, ctx)
        conversation.roll_cache_marks()
        wire = conversation.to_wire()
        # The one numbered-checkpoint writer: this state is what turn
        # `iteration`'s provider call consumes.
        self._save_resume_snapshot(state, wire, next_iteration=ctx.iteration, write_checkpoint=True)
        return wire

    def _maybe_inject_plan(
        self, conversation: Conversation, state: LoopState, *, iteration: int
    ) -> SessionResult | None:
        """Put the CURRENT plan.md in front of the planner, every turn.

        plan.md on disk is the plan; the conversation only ever holds a copy, and
        `agent6 plan edit` writes the operator's answers to the file between executions.
        So the file is re-read here rather than resynced at one chosen moment, and
        injected only when it differs from what the planner was last shown -- an
        untouched plan costs nothing. finish_planning stays the only writer.

        An UNREADABLE plan parks the execution (the returned SessionResult): the file
        may carry operator answers the planner's own copy supersedes, and
        continuing without them spends budget on stale direction. A missing
        file is normal (the first finish_planning creates it).
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
                state, End("plan_unreadable", remedy, checkpoint=False), iteration=iteration
            )
        if text == state.plan_injected:
            return None
        state.plan_injected = text
        conversation.notice(f"{PLAN_ON_DISK_HEADER}\n\n{text}")
        self._log(f"  plan re-read from disk: {len(text)} chars")
        self._emit(
            "loop.plan_reread",
            path=str(self.plan_output_path),
            bytes=len(text.encode("utf-8")),
        )

    def _turn_before_call(
        self, conversation: Conversation, state: LoopState, ctx: TurnContext
    ) -> None:
        """Before the provider call: anything the operator queued joins the
        graph, then the focus banner, then the before-call advisors, so a
        finish directive a low budget draws is the most recent message, not the
        banner."""
        self.operator_tasks.take(state.root_task_id)
        self._maybe_surface_current_task(conversation, state)
        for advisor in BEFORE_CALL:
            self._tell(conversation, advisor(state, ctx))

    def _maybe_surface_current_task(self, conversation: Conversation, state: LoopState) -> None:
        """Surface-current-task: keep the worker on ONE task at a time.

        Compute the current task (the cursor if it still points at an open
        subtask, else the first dependency-satisfied open subtask), advance the
        cursor to it, and inject a focus banner when the focus first appears,
        changes, or was wiped by a tier-2 restart (`surfaced_task_id` reset to
        None there). Advancing the cursor each turn means that once the worker
        marks the current task passed, the next turn's frontier recompute moves
        focus to the next ready task -- the cursor walks the frontier on its own.

        Also runs the anti-grind counter (`stuck_on_task`).

        Run mode only; no curator or no open subtask is a no-op (the finish-gate
        covers the empty-frontier finish). A curator mutation that fails logs
        and continues.
        """
        if self.mode != "run" or self.curator is None:
            return
        cursor = self.curator.cursor()
        nodes = self.curator.nodes()
        current_id = current_task_id(nodes, cursor)
        if current_id is None:
            state.focus.clear()
            return  # nothing decomposed yet, or the frontier is empty
        if cursor != current_id:
            # Advance the cursor onto the frontier task (auto-advance: a passed
            # cursor task drops out of the frontier, so this moves forward).
            try:
                self.curator.set_cursor(SetCursorIntent(id=current_id))
            except (CuratorError, OSError, ValidationError) as exc:  # advisory; never fatal
                self._log(f"LOOP: cursor advance skipped: {exc}")
        self._tell(conversation, stuck_on_task(state, current_id, nodes[current_id], nodes))
        if current_id == state.focus.surfaced_task_id:
            return  # already surfaced; the banner survives tier-1 elision
        node = nodes[current_id]
        if node.status == "pending":
            # Reflect that this task is now being worked, keeping the DAG honest
            # for the TUI and the check-off / finish-gate "open" set. Best-effort.
            try:
                self.curator.update_status(
                    UpdateStatusIntent(id=current_id, new_status="in_progress")
                )
            except (CuratorError, OSError, ValidationError) as exc:
                self._log(f"LOOP: mark in_progress skipped: {exc}")
        banner = current_task_banner(
            current_id, node, decompose=self.config.prompt.decompose == "on"
        )
        conversation.notice(banner)
        state.focus.surfaced_task_id = current_id
        self._log(f"LOOP: surfaced current task {current_id}")
        self._emit("loop.task.surfaced", task_id=current_id)
        # The harness-driven cursor/status writes bypass the tool-dispatch path
        # that emits graph.update, so refresh the live view here.
        self._emit_graph_snapshot()

    def _turn_provider_call(
        self,
        system: str,
        conversation: Conversation,
        wire: list[dict[str, Any]],
        tools: list[ToolDefinition],
        state: LoopState,
        *,
        iteration: int,
    ) -> SessionResult | NextTurn | ProviderResponse:
        """One worker call with terminal-error classification. Returns the
        provider response on success, a SessionResult to end the run, or
        `NEXT_TURN` when a mid-stream steer discarded the turn (the menu
        chose continue, or injected an instruction, so the turn is re-done).
        `wire` is the pre-call serialization (already snapshotted); the
        conversation is only touched on the steer path."""
        try:
            return self.caller.call(system, wire, tools, self._worker_max_tokens(state))
        except BudgetExceeded as exc:
            self._log(f"LOOP: budget exhausted at iter {iteration} ({exc})")
            return self._finish(
                state,
                End("budget_exhausted", f"budget exhausted at iter {iteration}: {exc}"),
                iteration=iteration,
            )
        except ProviderAborted:
            self.bridge.steer_clear()  # consume the stop; don't leave it on disk to re-read
            self._log(f"LOOP: operator stopped the run mid-turn at iter {iteration}")
            return self._finish(
                state,
                End(
                    "steer_abort",
                    f"operator stopped the run at iter {iteration}{self._dirty_tree_note()}",
                    checkpoint=False,
                ),
                iteration=iteration,
            )
        except ProviderInterrupted:
            # A steer was requested mid-stream; the watchdog ended the (thinking)
            # turn so the loop handles it now rather than waiting it out. The partial turn
            # is discarded; the menu decides continue / steer / stop / detach.
            self._log(f"LOOP: steer requested mid-turn at iter {iteration}")
            outcome = self._steer_outcome(
                self.steering.handle(conversation, iteration, state), iteration, state
            )
            if outcome is not None:
                return outcome
            return NEXT_TURN  # "continue" or an injected instruction -> re-do the turn
        except ProviderError as exc:
            hint = provider_error_hint(exc.status_code, exc.provider)
            attempts = f" after {exc.attempts} attempts" if exc.attempts > 1 else ""
            # The full upstream body (which can carry a noisy account user_id)
            # goes in this one diagnostic log line; the end-block summary below
            # stays concise so the raw blob is not echoed to the operator twice.
            self._log(f"LOOP: provider error{attempts} at iter {iteration}: {exc}{hint}")
            status = f" (HTTP {exc.status_code})" if exc.status_code else ""
            # A fatal error's text and a statusless transport failure are the
            # only available reason; an HTTP response's raw body stays in the log.
            detail = f": {exc}" if exc.fatal or exc.status_code is None else ""
            return self._finish(
                state,
                End(
                    "provider_error",
                    f"provider error{attempts} at iter {iteration}{status}{hint}{detail}",
                ),
                iteration=iteration,
            )

    def _worker_max_tokens(self, state: LoopState) -> int:
        """Per-call output cap for the worker turn.

        Metric-optimization runs (mode "run" with a configured continuous
        metric) lift the ceiling to `metric_task_max_tokens` so a single turn
        can rewrite a hot function wholesale without truncating mid-apply_patch.
        Every other run keeps `per_call_max_tokens`.

        Starvation backoff: once the worker has gone quiet (no text + no
        tool_use -- typically a reasoning model that spent its whole output
        budget on reasoning_content) on >= 2 CONSECUTIVE turns, drop back to
        `per_call_max_tokens` even on a metric run. A spiraling over-reasoner
        (observed: GLM 5.2) otherwise burns a fresh ~65k-token reasoning binge
        every nudged turn until it exhausts `went_quiet_max_nudges` and the run
        dies with zero progress. A tight cap plus the forceful "emit a tool_use
        now" nudge pressures it to ACT; `went_quiet_nudges_used` resets to 0 on
        the first productive turn, so the very next turn gets the full ceiling
        back for the real edit (the recovery edit itself is never truncated).
        The 2-quiet threshold spares the model the high ceiling was raised FOR
        (Kimi K2.x finishes its reasoning within 65k and rarely goes quiet, let
        alone twice in a row).
        """
        if (
            self.metrics.active
            and state.quiet.went_quiet_nudges_used < _STARVATION_BACKOFF_AFTER_QUIETS
        ):
            return max(self.call.per_call_max_tokens, self.call.metric_task_max_tokens)
        return self.call.per_call_max_tokens

    def _turn_dispatch_tools(
        self, state: LoopState, turn: TurnState, ctx: TurnContext
    ) -> SessionResult | None:
        """Dispatch each tool_use in the turn, appending one tool_result per
        call and noting effects (verify / metric / edits / DAG / finish) on
        `turn`, then the harness's own gate run when one is due. Returns a
        SessionResult only for the unexecutable-operator-command abort; tool
        errors become error tool_results instead."""
        # This iteration produced tool_uses, so the went_quiet
        # nudge budget refills (failures are per-streak, not per-run).
        state.quiet.went_quiet_nudges_used = 0
        for tu in turn.assistant.tool_uses:
            name = tu.name
            tool_input = tu.input
            if turn.finish is not None:
                # A finish ends the turn's work: the calls after it are not
                # executed, as the finish tools' descriptions state.
                turn.tool_results.append(
                    ToolResultItem(
                        tool_use_id=tu.id,
                        content=json.dumps(
                            {"error": f"{name} not executed: it follows {turn.finish.kind}"}
                        ),
                        for_call=tu,
                    )
                )
                continue
            state.tool_calls += 1
            # degenerate-loop signature tracking. Stable
            # JSON so dict key order does not break equality. Same
            # (name, args) back-to-back across iterations increments
            # `state.spiral.call_streak`; anything else resets it.
            try:
                sig = f"{name}:{json.dumps(tool_input, sort_keys=True, ensure_ascii=False)}"
            except (TypeError, ValueError):
                sig = f"{name}:<unhashable>"
            state.spiral.note_call(sig, polling=name == ReadBackgroundInput.TOOL_NAME)
            self._emit("loop.tool.call", name=name, iteration=turn.iteration)
            served = None
            try:
                refusal = turn.resp.refused.get(tu.id)
                if refusal is not None:
                    # The provider's front-end checked the input and answered
                    # the model itself; the same error is the result here.
                    raise ToolError(refusal)
                tree_before = self._tree_before_command(name)
                result = self.dispatcher.dispatch(name, tool_input)
                content = json.dumps(result.to_wire(), ensure_ascii=False)
                self._note_tool_effects(
                    state, turn, name, result, tool_input, tree_before=tree_before
                )
                # Dedupe a back-to-back identical (name, args) call whose result
                # bytes are unchanged: serve a short stub instead of re-sending
                # the full payload, so a re-read spiral cannot grow the context.
                # The call still dispatched (a CHANGED result serves in full);
                # only the redundant re-serve is elided.
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
                # Control verbs are not WORK: a revoked finish_session that
                # counted here would reset the standing fruitless streak
                # every round, and standing_patience could never engage.
                if name not in ("finish_session", "finish_planning"):
                    state.ok_tool_calls += 1
                self._take(state, turn, ctx, unreachable_tool(state, name, tool_input, result))
                # Only a DISPATCHED finish counts: a refused finish tool (mode
                # backstop, schema error) is an error result the model recovers
                # from, not an end to the run.
                self._capture_finish(turn, name, tool_input)
            except ToolError as exc:
                content = self._note_tool_error(state, name, tool_input, exc)
                self._take(state, turn, ctx, tool_error_ladder(turn, state, ctx))
            except OperatorCommandUnexecutable as exc:
                return self._unexecutable_abort(exc, iteration=turn.iteration, state=state)
            turn.tool_results.append(
                ToolResultItem(
                    tool_use_id=tu.id,
                    content=cap_tool_result(
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
        except OperatorCommandUnexecutable as exc:
            return self._unexecutable_abort(exc, iteration=turn.iteration, state=state)
        return None

    def _tree_before_command(self, name: str) -> str:
        """The worktree's content sha ahead of a child-process tool's call,
        for `_left_the_tree_dirty`; "" for every other tool. `run_verify_command`
        and `run_metric_command` are the operator's own gates, and the caches
        they drop must not invalidate the pass they just produced."""
        if name != "run_command" and not name.startswith(MCP_TOOL_PREFIX):
            return ""
        return self.chain.tree_sha()

    def _left_the_tree_dirty(self, tree_before: str) -> bool:
        """True when a child-process tool changed the tree: its content sha
        after the call differs from *tree_before*. Git decides, so a read-only
        probe (`ls`, `grep`) costs its pass nothing, over uncommitted work too,
        and gitignored build artifacts never count as a change. "" (no sha, or
        a tool that cannot touch the tree) reads as unchanged."""
        if not tree_before:
            return False
        after = self.chain.tree_sha()
        return bool(after) and after != tree_before

    def _note_tool_effects(
        self,
        state: LoopState,
        turn: TurnState,
        name: str,
        result: ToolResult,
        tool_input: Any,
        tree_before: str = "",
    ) -> None:
        """Record a dispatched tool's side effects on the turn: verify results
        (they feed auto-commit-on-verify-pass and ground the review panel:
        verify-pass presumes correctness, verify-red is the hard signal),
        manual metric samples, tree edits, and DAG mutations. *tree_before* is
        `_tree_before_command`'s sha for a child-process tool."""
        if name == "ask_user" and isinstance(result, AnswersResult):
            # The result carries the questions the dispatcher accepted (one
            # flat, a stringified list); the raw input is never parsed twice.
            for question, answer in zip(result.asked, result.answers, strict=False):
                self._record_decision(state, question, answer)
        if name == "run_verify_command" and isinstance(result, ExecResult):
            # The model's own gate overran its budget: the same scoped
            # follow-up the harness gate gets, whose verdict is the turn's
            # (the 124 is not noted beside it); under `never` the harness
            # runs nothing and the timeout is the verdict.
            if not (
                self.mode == "run"
                and self.gate.when != "never"
                and result.returncode == EXIT_TIMEOUT
                and not state.verify.scoped
                and self.gate.scoped_followup(state, turn) is not None
            ):
                if result.returncode == 0:
                    # The model's call runs the full argv: a green there is a
                    # full pass, so later harness gates run full again.
                    state.verify.scoped = False
                self.gate.note_result(state, turn, result)
        elif name == "run_metric_command" and isinstance(result, MetricResult):
            turn.metric_sampled = True
            # The tree this reading covers: without the stamp the auto path
            # samples it again on every turn that reads the tree as changed
            # (all of them, with nothing committing between steps).
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
        if name in ("apply_edit", "apply_patch") and isinstance(result, PreviewResult):
            return  # a dry run writes nothing: no memory write, no tree edit
        if self._note_memory_touch(state, name, result, tool_input):
            # An edit under the memory dir is a memory write, not workspace
            # work: both memory nudges stay quiet for the rest of the run and
            # none of the tree bookkeeping below applies (the gate's tree is
            # untouched).
            return
        if name in ("apply_edit", "apply_patch"):
            turn.edited = True
            state.ever_edited = True
            # Invalidate a same-turn earlier verify pass: the commit
            # gate must not label this edited tree "verify passed".
            turn.edit_since_verify_pass = True
            state.verify.note_edit()
        elif self._left_the_tree_dirty(tree_before):
            # A command (or an MCP tool) can change the tree just as an edit
            # tool can, and a green verify must not survive it: the tree the
            # gate approved is no longer the tree we have. Asked of git rather
            # than assumed from the tool name, so a read-only `ls` or `grep`
            # through run_command keeps the pass it had.
            turn.edit_since_verify_pass = True
            state.verify.note_edit()
        if name in DAG_MUTATING_TOOLS:
            turn.dag_mutated = True  # snapshot once after the turn

    def _note_memory_touch(
        self, state: LoopState, name: str, result: ToolResult, tool_input: Any
    ) -> bool:
        """Count a `read_file` of a fact in the memory store, or record an
        edit tool's write there (the facts it created, edited or deleted, and
        `written` for the nudges). True for a write: the store sits outside
        the workspace, so it is not workspace work."""
        facts = memory_store_facts(self.state_dir, name, result, tool_input)
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

    def _capture_finish(self, turn: TurnState, name: str, tool_input: Any) -> None:
        """A dispatched finish ends the turn's work; the finish gates may still
        revoke it. A finish_planning also writes its plan to `plan_output_path`."""
        finish = FinishCall.parse(name, tool_input)
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
            mkdir_for_real_user(self.plan_output_path.parent)
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
        self, state: LoopState, name: str, tool_input: dict[str, Any], exc: ToolError
    ) -> str:
        """Bookkeeping for one failed dispatch: the served error content, the
        denial/binary records the reachability note reads, and the
        same-signature streak the nudge ladder climbs."""
        content = json.dumps({"error": str(exc)})
        self._log(f"  tool_error: {name}: {exc}")
        state.spiral.note_error(
            tool_error_signature(name, str(exc)),
            denial=isinstance(exc, ToolDenied),
            content=content,
        )
        return content

    def _turn_auto_commit_and_metric(
        self, state: LoopState, turn: TurnState
    ) -> SessionResult | None:
        """Auto-commit the turn's work, then take the automatic metric sample.

        A step the gate judged green commits as a verified step; every other
        editing step (a gateless run, and under `verify_when = "finish"` a step
        the model did not verify itself) commits as an un-gated checkpoint, so
        resume and the audit trail still work.
        `turn.edited` (apply_edit/apply_patch) is the cheap fast-path; the
        worktree-dirty fallback catches run_command-authored edits (else they'd
        never be committed gateless). Plan mode is read-only and never commits.
        Best-effort: commit failures (e.g. nothing to commit) are logged but
        don't abort the run; the catch includes OSError so a transient FS
        hiccup doesn't kill an otherwise-fine run.

        Returns a SessionResult for the REPL hook's "stop" directive or an
        unexecutable operator metric command; None otherwise."""
        gateless = not self.gate.present(state.verify)
        # A step no gate judged commits as a checkpoint: every gateless step
        # (no command, or one nobody may run), and under `verify_when =
        # "finish"` every step the model did not verify itself (the gate
        # certifies the tree the run ends on).
        unjudged = gateless or (
            self.gate.when == "finish" and not (turn.verify_just_passed or turn.verify_just_failed)
        )
        unjudged_changed = unjudged and (turn.edited or self.chain.dirty())
        verified_commit = turn.verify_just_passed and not turn.edit_since_verify_pass
        if self.mode != "run" or not (verified_commit or unjudged_changed):
            return None
        if unjudged_changed:
            # Seed the idle-stop net for runs where no green verify fires per
            # step (see the verify-settled bookkeeping), commits or not.
            state.settled.gateless_ever_edited = True
        if not self.chain.per_step:
            # `commit_per_step` governs the COMMIT. The metric is measurement:
            # the prompt promises a [harness metric] block after every verified
            # edit, so the model sees the number it is asked to move.
            return self._sample_metric(state, turn, sha="")
        commit_subject = self.checkpoints.subject(
            turn, fallback="checkpoint" if unjudged_changed else "verify passed"
        )
        sha = ""
        try:
            sha = self.checkpoints.commit(commit_subject, iteration=turn.iteration)
            turn.committed = bool(sha)
            # Adoption fills an ABSENT command, for a worker who may run one:
            # a configured gate nobody may run stays the operator's.
            if (
                sha
                and not self.gate.command(state.verify)
                and self.gate.may_run(denied=state.verify.denied)
            ):
                self.gate.maybe_adopt(state, turn)
        except (GitError, OSError) as exc:
            self.checkpoints.report_failure(exc, commit_subject, iteration=turn.iteration)
        # REPL hook. Default no-op returns "continue".
        if sha:
            directive = self.bridge.after_auto_commit(turn.iteration, sha)
            if directive in ("undo", "exit"):
                ended = self._steer_outcome(directive, turn.iteration, state)
                if ended is not None:
                    return ended
            if directive == "stop":
                self._log(f"LOOP: interactive stop at iter {turn.iteration}")
                # An operator stop is deliberate, not verified success: the
                # same truth rule as steer_abort ("stopped", never "passed").
                return self._finish(
                    state,
                    End(
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
        self, state: LoopState, turn: TurnState, *, sha: str
    ) -> SessionResult | None:
        """Run the configured metric over the step just taken and hand the
        model the reading, unless it ran the metric itself this turn.

        One reading per state of the tree: with nothing committing between
        steps (`commit_per_step = false`) the tree stays dirty for the rest of
        the run, and sampling on dirt alone would re-run the operator's
        benchmark on every turn, read-only ones included."""
        if turn.metric_sampled or state.metric.denied:
            return None
        tree = self.chain.tree_sha()
        if tree and tree == state.metric.tree:
            return None
        state.metric.tree = tree
        # The auto path raises OperatorCommandUnexecutable just like a manual
        # run_metric_command would: the same abort as the per-tool handler's.
        try:
            turn.metric_feedback = self.metrics.auto_feedback(
                state, iteration=turn.iteration, sha=sha
            )
        except OperatorCommandUnexecutable as exc:
            return self._unexecutable_abort(exc, iteration=turn.iteration, state=state)
        turn.metric_plateau_finish = self.metrics.plateau_finish(state.metric.history)
        return None

    def _turn_notices(self, state: LoopState, turn: TurnState) -> None:
        """Append the turn's review findings and metric feedback to the
        tool_results block, ahead of the advisors' notices."""
        if turn.review_text:
            turn.tool_results.append(Notice(review_notice(turn.review_text)))
            turn.review_text = None
        if turn.metric_feedback:
            turn.tool_results.append(Notice(turn.metric_feedback))

    # ---- finish gates --------------------------------------------------------

    def _turn_finish_gates(self, state: LoopState, turn: TurnState, ctx: TurnContext) -> None:
        """The gates a finish_session must pass, in precedence order: the
        finish contract, the before-finish panel, the metric early-finish
        rule, the open subtasks, the verify certification, the memory backstop,
        the standing goal. The first refusal revokes the finish (`_refuse`)."""
        if turn.finish is None or turn.finish.kind != "finish_session":
            return
        turn.ending = "finish_session"
        for gate in FINISH_GATES:
            if self._refuse(state, turn, gate(turn, state, ctx)):
                return

    def _refuse(self, state: LoopState, turn: TurnState, refusal: Refusal | None) -> bool:
        """Apply a gate's refusal of the turn's end: a finish is revoked (its
        tool_result still goes back, so the call is not half-applied) and a
        declared end handed back (`turn.end_returned`), the model gets the
        refusal's text, the event and the line are recorded, and the settle
        streak starts over (the work a refusal asks for is idle to it). False
        when the gate let the end through."""
        if refusal is None:
            return False
        turn.finish = None
        turn.end_returned = True
        if refusal.text:
            turn.tool_results.append(Notice(refusal.text))
        self._record(refusal)
        state.settled.restart()
        return True

    def _end_gates(
        self,
        state: LoopState,
        turn: TurnState,
        ctx: TurnContext,
        *,
        ending: str,
        gates: tuple[Gate, ...],
    ) -> SessionResult | None:
        """An end declared without finish_session (`settled`: the harness's
        idle stop; `silent_finish`: a prose turn with no tool call) passes
        *gates*, the rules a finish_session would, through the one applier
        (`_refuse`): the first refusal hands the end back (`turn.end_returned`)
        with its reason. The harness gate runs first on the ending turn (the
        STANDING verdict decides the red: it is skipped over a tree a red
        already covers); the unexecutable-command abort ends the run as it
        does on the tool path. The panel's findings, when it sat, follow the
        refusal as a notice."""
        try:
            self.gate.harness_verify(state, turn, ending=True)
        except OperatorCommandUnexecutable as exc:
            return self._unexecutable_abort(exc, iteration=turn.iteration, state=state)
        turn.ending = ending
        for gate in gates:
            if self._refuse(state, turn, gate(turn, state, ctx)):
                break
        if turn.review_text:
            # The turn's notices went out before the settled and plateau
            # checks, so the panel's findings are delivered here.
            turn.tool_results.append(Notice(review_notice(turn.review_text)))
            turn.review_text = None
        return None

    # ---- the advisors --------------------------------------------------------

    def _turn_context(
        self, state: LoopState, *, iteration: int, execution_start: int
    ) -> TurnContext:
        """The facts the advisors read this turn (`TurnContext`)."""
        return TurnContext(
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
        """Fraction of the token budget still available, or None when no
        BudgetTracker is wired in (tests / MCP path)."""
        if self.budget is None:
            return None
        return self.budget.fraction_remaining()

    def _turn_advisors(
        self, state: LoopState, turn: TurnState, ctx: TurnContext
    ) -> SessionResult | None:
        """The turn's notices (review findings, metric feedback), then the
        after-tools advisors in order, each answer applied."""
        self._turn_notices(state, turn)
        for advisor in AFTER_TOOLS:
            aborted = self._take(state, turn, ctx, advisor(turn, state, ctx))
            if aborted is not None:
                return aborted
        return None

    def _take(
        self, state: LoopState, turn: TurnState, ctx: TurnContext, outcome: Nudge | Stop | None
    ) -> SessionResult | None:
        """Apply one advisor's answer. A nudge joins the turn's results, its
        event emitted and its line logged. A stop joins `turn.stops` for the
        stop checks; an ending the harness declares (`Stop.declared`) is
        judged by the end gates first, unless a finish call this turn already
        ran them, and dropped when they hand it back. Returns the abort when
        the gates' verify could not run."""
        if outcome is None:
            return None
        if isinstance(outcome, Nudge):
            turn.tool_results.append(Notice(outcome.text))
            self._record(outcome)
            return None
        if outcome.event:
            self._emit(outcome.event, **outcome.fields)
        if outcome.declared and turn.finish is None:
            aborted = self._end_gates(state, turn, ctx, ending=outcome.declared, gates=END_GATES)
            if aborted is not None:
                return aborted
            if turn.end_returned:
                return None
        turn.stops.append(outcome)
        return None

    def _tell(self, conversation: Conversation, nudge: Nudge | None) -> None:
        """Apply a before-call advisor's answer: the notice joins the
        conversation, its event is emitted, its line logged."""
        if nudge is None:
            return
        conversation.notice(nudge.text)
        self._record(nudge)

    def _record(self, answer: Nudge) -> None:
        """Record an advisor's or a gate's answer: its event emitted, its
        line logged (each skipped when empty)."""
        if answer.event:
            self._emit(answer.event, **answer.fields)
        if answer.log:
            self._log(answer.log)

    # ---- stop checks, silent finish, went-quiet ------------------------------

    def _turn_stop_checks(
        self, state: LoopState, turn: TurnState, conversation: Conversation
    ) -> SessionResult | None:
        """Terminal checks, run after the turn's tool_results are in
        `messages` and the post-tools snapshot is written, in precedence
        order: the advisors' stops as decided, then honouring a finish call
        that survived the gates."""
        self.standing.absorb_soft_stop(state, turn, conversation)
        for stop in turn.stops:
            if stop.log:
                self._log(stop.log)
            return self._finish(state, stop.end(), iteration=turn.iteration)
        finish = turn.finish
        if finish is not None:
            self._log(f"LOOP: {finish.kind} called at iter {turn.iteration}")
            self.checkpoints.final(iteration=turn.iteration)
            # Honest finish: finish_planning is always a clean finish, but a
            # finish_session over a red/stale verify is "finished", not "passed"
            # -- all_passed reflects the actual verify state, never just "the
            # model called finish_session".
            reason = finish_reason(
                finish.kind,
                stale_gate=finish.stale_gate,
                tree_green=self.gate.tree_green(state.verify),
                verify=state.verify,
            )
            self._check_decisions_recorded(state)
            return self._finish(
                state,
                End(
                    reason,
                    with_open_tasks(finish.summary, self._open_subtasks()),
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
        assistant: AssistantTurn,
        conversation: Conversation,
        state: LoopState,
        ctx: TurnContext,
    ) -> SessionResult | None:
        """Handle a turn with no tool_use. Either a silent finish (the agent
        emitted text; gated like an explicit finish_session) or went-quiet (an
        empty turn; nudged up to a cap). Returns a terminal SessionResult, or None
        to continue the loop after appending a nudge.

        Distinguishing the two matters: "agent talked then stopped" is likely
        an implicit finish (the user gets the text as summary), while "agent
        emitted nothing" is a went-quiet failure (an empty provider response,
        or a confused agent) that bench scoring must NOT treat as success."""
        text = resp.text.strip() if resp.text else ""
        if text:
            # A prose turn is NON-EMPTY: the went_quiet nudge budget refills
            # here exactly as on a tool_use turn (the documented per-streak
            # contract, "reset on any non-empty turn"). Without it, quiet
            # streaks interleaved with bounced prose turns (silent-finish
            # gates, question nudges) drain one shared budget and end the run
            # as went_quiet with no streak at the cap, and the starvation
            # output-cap backoff stays reduced.
            state.quiet.went_quiet_nudges_used = 0
            turn = TurnState(iteration=ctx.iteration, resp=resp, assistant=assistant)
            return self._handle_silent_finish(text, conversation, state, turn, ctx)
        return self._handle_went_quiet(resp, conversation, state, ctx)

    def _handle_silent_finish(
        self,
        text: str,
        conversation: Conversation,
        state: LoopState,
        turn: TurnState,
        ctx: TurnContext,
    ) -> SessionResult | None:
        """A no-tool_use turn WITH text: treat it as an implicit finish and run
        it through the same gates as an explicit finish_session. Returns None (with
        a nudge appended to the conversation) when a gate sends the worker back to
        work; the silent_finish SessionResult once every gate lets it through."""
        iteration = turn.iteration
        if (stall := silent_no_work(state, ctx)) is not None:
            self._tell(conversation, stall)
            return None
        aborted = self._end_gates(state, turn, ctx, ending="silent_finish", gates=SILENT_END_GATES)
        # A prose turn has no tool results, so the gates' notices go to the
        # conversation directly.
        for item in turn.tool_results:
            if isinstance(item, Notice):
                conversation.notice(item.text)
        if aborted is not None or turn.end_returned:
            return aborted
        if (asked := question_in_prose(state, ctx, text)) is not None:
            self._tell(conversation, asked)
            return None
        # A quiet run does not have to end: a standing goal re-enters, else an
        # interactive run parks for a steer (never in ask mode, where the
        # prose IS the answer).
        cont = self._quiet_continuation(
            conversation, state, iteration=iteration, reason="silent_finish"
        )
        if cont is not None:
            return None if isinstance(cont, NextTurn) else cont
        # In ask mode a prose answer with no tool call is the NORMAL success (the
        # answer IS the text), so end as "answered", not "silent_finish": the
        # latter reads as a failure diagnostic on a good answer. run/plan keep
        # silent_finish: there, stopping without finish_session is mildly anomalous.
        reason: SessionEndReason = "answered" if self.mode == "ask" else "silent_finish"
        if self.mode == "ask":
            self._log(f"  ask answered at iter {iteration}")
        else:
            self._log(
                f"LOOP: silent_finish at iter {iteration} - agent emitted text but no tool_use"
            )
        # Honest finish: run/plan ground exactly like the explicit
        # finish_session path (observed green -> "passed", red or stale ->
        # "failed", ungated -> "finished"). Ask mode's prose answer is the
        # success (it never runs verify), so it always ends passed, and the
        # final prose IS the answer the caller prints, so it is kept whole;
        # run/plan only need a short summary line.
        return self._finish(
            state,
            End(
                reason,
                text if self.mode == "ask" else with_open_tasks(text[:1000], self._open_subtasks()),
                completed=True,
                verdict="grounded" if reason == "silent_finish" else "passed",
            ),
            iteration=iteration,
        )

    def _handle_went_quiet(
        self,
        resp: ProviderResponse,
        conversation: Conversation,
        state: LoopState,
        ctx: TurnContext,
    ) -> SessionResult | None:
        """A fully-empty turn (no text, no tool_use): surface reasoning
        starvation explicitly, then nudge-and-retry up to the per-streak cap
        (`went_quiet`) before ending the run as went_quiet.

        The nudge is cheap (~50 input tokens vs aborting the entire run) and
        almost always gets a weak open-weights model back on track. The empty
        assistant turn is dropped from the conversation first: Anthropic rejects an
        assistant message with empty content, a THINKING-ONLY turn (reasoning
        starvation: blocks but no text/tool_use) translates to one with no
        content and no tool_calls that strict OpenAI-compatible backends reject
        with a non-retryable 400, and either way it is dead context."""
        iteration = ctx.iteration
        reasoning_chars = reasoning_starvation(resp)
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
        # An empty turn the provider still billed output tokens for is not a
        # model that chose silence: the tokens went to reasoning that never
        # surfaced, or to a tool call the upstream failed to parse and dropped
        # (seen on OpenRouter-routed qwen at temperature 0, deterministically
        # per prompt). Say so, in the log and on the event, so the transcript
        # file is not the only place the difference shows.
        # "billed" is a dollar word: on a subscription plan those tokens cost
        # $0, so the plan-metered run says "spent" instead of claiming a bill.
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
        # Drop the dead turn before any exit: a provider rejects an assistant
        # message with empty content, and every path below either calls again
        # (nudge, standing goal, park) or snapshots the conversation for resume.
        conversation.pop_quiet_assistant()
        if (nudge := went_quiet(state, ctx, resp)) is not None:
            self._tell(conversation, nudge)
            return None
        cont = self._quiet_continuation(
            conversation, state, iteration=iteration, reason="went_quiet"
        )
        if cont is not None:
            return None if isinstance(cont, NextTurn) else cont
        return self._finish(
            state, End("went_quiet", "(agent emitted no text and no tool_use)"), iteration=iteration
        )

    # ---- the end -------------------------------------------------------------

    def _finish(self, state: LoopState, end: End, *, iteration: int) -> SessionResult:
        """Record *end* (its checkpoint, the pending roots it passes, its
        `session.end`) and return the run's result.

        A clean end grounds `all_passed` on the FINAL tree: True only when it
        is OBSERVED verify-green, False when it is red or stale, None when
        nothing gated it, so "passed" never means "ended over a red or stale
        verify", and an ungated end reads "finished", never "failed";
        `_verification` gives the same state its not_applicable verdict. The
        roots pass either way: the DAG tracks work items and the run-level
        word carries the verify truth, so a red-verify finish would otherwise
        read `tasks 0/1` forever. `scoped` says the gate ran scoped to the
        tests nearest the diff, so a scoped green reads "passed · scoped
        gate" on every surface."""
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
        return SessionResult(
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
        """On successful completion, mark still-pending root task(s) as passed.

        The loop seeds one root task per `run()` (each ask REPL follow-up seeds
        another), but the worker finishes via `finish_session` without ever
        touching it -- so a completed ask/run otherwise reads `tasks 0/1`. Pass
        any root (`parent_id is None`) still pending/in-progress so the DAG --
        and every viewer + resume -- agrees the run completed. Subtasks the
        worker deliberately left unfinished are untouched (kept honest).
        Best-effort: a curator hiccup must never break completion."""
        if self.curator is None:
            return
        changed = False
        for nid, node in self.curator.nodes().items():
            if node.parent_id is None and node.status in OPEN_STATUSES:
                try:
                    self.curator.update_status(UpdateStatusIntent(id=nid, new_status="passed"))
                    changed = True
                except CuratorError as exc:  # this root refused; the next may not
                    self._log(f"LOOP: auto-pass root {nid} refused: {exc}")
                except (OSError, ValidationError) as exc:  # a write fault must not break finish
                    self._log(f"LOOP: auto-pass root {nid} failed: {exc}")
                    break  # a curator write failure fails for every remaining node too
        if changed:
            self._emit_graph_snapshot()

    def _record_memory_use(self, state: LoopState) -> None:
        """Persist the facts this execution wrote and read (`memory list` shows them);
        a write fault must not break the end."""
        memory = state.memory
        if self.state_dir is None or not (memory.wrote or memory.read or memory.deleted):
            return
        try:
            record_use(
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
        self, exc: OperatorCommandUnexecutable, *, iteration: int, state: LoopState
    ) -> SessionResult:
        """Graceful abort when an operator verify/metric command cannot run in
        the jail (e.g. its binary is not on the jail PATH). The model cannot fix
        operator config, so stop loudly rather than flail against a gate that
        never executes or silently report success. Shared by the manual per-tool
        path and the auto-metric-after-verify path so the same misconfiguration
        ends the same way regardless of who triggered the command."""
        self._log(f"LOOP: aborting -- {exc}")
        # The worst checkpoint case of all the harness ends: verify can never
        # go green here, so the per-turn auto-commit never fired and ALL of
        # the run's edits may exist only in the worktree.
        return self._finish(
            state, End("verify_command_unexecutable", str(exc)), iteration=iteration
        )

    def _dirty_tree_note(self) -> str:
        """Summary suffix naming an uncommitted worktree (`RunChain.dirty_note`),
        for a run; "" in the modes that never commit."""
        return self.chain.dirty_note() if self.mode == "run" else ""

    # ---- the task graph ------------------------------------------------------

    def _emit_graph_snapshot(self) -> None:
        """Emit the current task DAG so a live viewer (the TUI) can render it.
        The worker's add_task/update_task tree lives in the curator, not the
        event log, so we snapshot it (once per turn, see the call site).

        Project to ONLY the fields the viewer renders, a full node dump carries
        unbounded model-authored text (rationale/acceptance/notes/paths) that
        bloats the fsync'd event log for no benefit."""
        if self.curator is None:
            return
        cursor = self.curator.cursor()
        # FROZEN wire surface: project each node to exactly these six fields,
        # children as a JSON list -- the graph.update shape the viewmodel fold,
        # web and TUI hold. `created_by` and `standing` tell the operator's own
        # tasks from the model's (graph.models.owner_note); a run dir written
        # before a field existed reads it as the model's.
        # Pinned by test_graph_update_snapshot_payload_is_wire_stable.
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
        """The worker's own subtasks still open: `(id, title)` pairs. Only
        SUBTASKS (parent_id is not None) count -- the auto-root is pending until
        the run ends, so counting it would deadlock every gate. Run mode only:
        a plan's tasks are its deliverable, open by design. Best-effort: no
        curator -> nothing open."""
        if self.curator is None or self.mode != "run":
            return []
        return open_subtasks(self.curator.nodes())

    def _checkpoint_graph_version(self) -> int:
        """Curator DAG version for the per-turn checkpoint; 0 if no curator."""
        if self.curator is None:
            return 0
        return self.curator.graph_version

    # ---- the state dir: memory, decisions, skills ----------------------------

    def _load_memory_index(self) -> str:
        """The repo memory index for the system prompt.

        "" when no state_dir is wired, and for machine/agent modes (whose
        prompt assembly drops repo context). An unreadable index degrades to
        "" inside the store: memory is context, not correctness.
        """
        if self.state_dir is None or self.mode == "agent":
            return ""
        return memory_index_text(self.state_dir)

    def _load_decisions(self) -> str:
        """The operator's recorded rulings for the prompt ("" without a state
        dir); every mode sees them, a ruling binds a planner as much as a
        worker."""
        return decisions_text(self.state_dir) if self.state_dir is not None else ""

    def _record_decision(self, state: LoopState, question: str, answer: str) -> None:
        """An operator answer becomes a durable ruling the moment it arrives:
        appended to the repo's DECISIONS.md by the harness (never by the
        model), remembered for the finish-time check."""
        if self.state_dir is None or not answer.strip():
            return
        try:
            entry = record_decision(
                self.state_dir, question=question, answer=answer, session=self.session_id
            )
        except OSError as exc:
            self._log(f"LOOP: decision not recorded: {exc}")
            self._emit("loop.decision.unrecorded", error=str(exc))
            return
        state.decisions_recorded.append(entry)
        self._log(f"LOOP: decision recorded ({len(answer)} chars)")
        self._emit("loop.decision.recorded", question=question[:200], answer=answer[:200])

    def _check_decisions_recorded(self, state: LoopState) -> None:
        """The finish-time check: every ruling this execution recorded is in the
        file. A miss is reported (log + event), never a block."""
        if self.state_dir is None or not state.decisions_recorded:
            return
        try:
            text = decisions_path(self.state_dir).read_text(encoding="utf-8")
        except OSError:
            text = ""
        missing = [e for e in state.decisions_recorded if e.strip() not in text]
        if missing:
            self._log(f"LOOP: {len(missing)} recorded decision(s) missing from DECISIONS.md")
            self._emit("loop.decision.unrecorded", missing=len(missing))

    def _load_skills(self) -> ResolvedSkills | None:
        """Operator-installed skills for the system prompt, run mode only.

        Reuses the dispatcher's one-shot resolution so the <skills> index and
        what use_skill actually serves can never diverge. None (nothing
        installed, subsystem off, or non-run mode) renders no block.
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

    @cached_property
    def gate(self) -> VerifyGate:
        """The run's verify gate over the config's command; `gate.command`
        reads the one in force, the config's or the adopted one."""
        wf = self.config.harness
        return VerifyGate(
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

    @cached_property
    def checkpoints(self) -> Checkpoints:
        return Checkpoints(
            chain=self.chain,
            style=self.config.git.commit.checkpoint.message,
            enabled=self.mode == "run",
            provider=self.provider,
            log=self._log,
            emit=self._emit,
        )

    @cached_property
    def compactor(self) -> Compactor:
        """The run's context compaction driver."""
        return Compactor(
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

    @cached_property
    def standing(self) -> Standing:
        """The run's standing goal: the re-entry a soft end converts into."""
        return Standing(
            curator=self.curator,
            patience=self.config.harness.standing_patience,
            budget_remaining=self._budget_fraction_remaining,
            log=self._log,
            emit=self._emit,
        )

    @cached_property
    def metrics(self) -> MetricSampler:
        return MetricSampler(
            settings=self.config.harness.metric,
            enabled=self.mode == "run",
            dispatcher=self.dispatcher,
            log=self._log,
            emit=self._emit,
        )

    @cached_property
    def operator_tasks(self) -> OperatorTasks:
        return OperatorTasks(
            curator=self.curator,
            take_requests=self.bridge.take_requests,
            revision=self.revision,
            root=self.chain.root,
            log=self._log,
            emit=self._emit,
            emit_graph_snapshot=self._emit_graph_snapshot,
        )

    @cached_property
    def parallel(self) -> ParallelDispatcher:
        """The run's `/parallel` lane dispatch."""
        return ParallelDispatcher(
            chain=self.chain,
            curator=self.curator,
            max_lanes=self.config.parallel.max_lanes,
            lane_spawner=self.bridge.lane_spawner,
            save_snapshot=self._save_resume_snapshot,
            log=self._log,
            emit=self._emit,
            emit_graph_snapshot=self._emit_graph_snapshot,
        )

    @cached_property
    def steering(self) -> Steering:
        """What a steer's text means for the run (`Steering.handle`)."""
        return Steering(
            bridge=self.bridge,
            parallel=lambda: self.parallel,
            dispatcher=self.dispatcher,
            record_decision=self._record_decision,
            log=self._log,
            emit=self._emit,
        )

    @cached_property
    def reviewer(self) -> Reviewer:
        """The run's in-loop review panel."""
        return Reviewer(
            settings=self.review,
            chain=self.chain,
            review_tools=lambda: build_readonly_review_tools(self.dispatcher),
            budget_remaining=self._budget_fraction_remaining,
            log=self._log,
            emit=self._emit,
        )

    @cached_property
    def caller(self) -> ProviderCaller:
        """The worker's provider under the run's retry knobs and steer callables."""
        return ProviderCaller(
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
        self, conversation: Conversation, iteration: int, state: LoopState
    ) -> SessionResult | None:
        """The end-of-iteration operator-control boundary, run after EVERY
        completed iteration (tool turns and prose turns alike): honor a
        pending "stop after this step" marker, then poll the steering flag.
        The safe point is AFTER a complete iteration, so a stop or an
        injected instruction never splits a tool_use / tool_result pair; the
        per-iteration snapshot is the resume point."""
        # Before the menu below can print them: a background command's ending
        # only reaches disk when someone observes it, and `/shells` reads from
        # there.
        self.dispatcher.settle_background()
        if self.bridge.stop_requested():
            self.bridge.stop_clear()
            self._log(f"LOOP: operator stop at the step boundary (iter {iteration})")
            return self._finish(
                state,
                End(
                    "steer_abort",
                    f"operator stopped the run after step {iteration}{self._dirty_tree_note()}",
                    checkpoint=False,
                ),
                iteration=iteration,
            )
        # The operator can press Ctrl-C once to drop a steering instruction
        # into the conversation; a second Ctrl-C within 2s raises
        # KeyboardInterrupt and aborts.
        return self._steer_outcome(
            self.steering.handle(conversation, iteration, state), iteration, state
        )

    def _quiet_continuation(
        self, conversation: Conversation, state: LoopState, *, iteration: int, reason: str
    ) -> SessionResult | NextTurn | None:
        """The run-mode continuations for a quiet turn, in priority order: a
        standing goal re-enters (autonomy first), else an interactive run
        parks for a steer. Returns NEXT_TURN to continue the loop, a park's
        terminal steer verb, or None when neither applies (the caller ends
        the run)."""
        if self.mode != "run":
            return None
        nudge = self.standing.absorb(state, reason=reason, iteration=iteration)
        if nudge is not None:
            conversation.notice(nudge)
            return NEXT_TURN
        if self.interactive:
            parked = self._park_for_steer(conversation, state, iteration=iteration, reason=reason)
            return NEXT_TURN if parked is None else parked
        return None

    def _park_for_steer(
        self, conversation: Conversation, state: LoopState, *, iteration: int, reason: str
    ) -> SessionResult | None:
        """The interactive turn boundary: the model went quiet, so the run
        parks -- the SAME in-memory conversation, its snapshot already on
        disk -- until the operator steers it from any composer or the pause
        menu. Returns None when a steer (or a bare poke) continued the run,
        or the steer verb's terminal result. No timeout: parked-until-steered
        is the point; stop/abort are the exits."""
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
                    End(
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
                # A queued task, goal or retirement is work: the run continues
                # and the next turn's focus banner names it.
                self._emit("loop.parked.resumed", iteration=iteration)
                return None
            time.sleep(0.5)

    def _steer_outcome(
        self, steer_result: str | None, iteration: int, state: LoopState
    ) -> SessionResult | None:
        """Map a `Steering.handle` result to a terminal SessionResult, or None to keep
        going (empty steer, or an instruction injected into messages)."""
        if steer_result in ("abort", "exit"):
            # "exit" is /exit at the pause menu: the same stop, but the end
            # reason tells the CLI to skip the follow-up prompt and leave.
            reason: SessionEndReason = "steer_exit" if steer_result == "exit" else "steer_abort"
            return self._finish(
                state,
                End(
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
                # The forker printed why (or no forker is wired); keep running.
                self._log("  /undo: nothing to undo; continuing")
                return None
            new_id, undone_text = forked
            self._emit("session.undone", new_session_id=new_id, undone_text=undone_text)
            # An undo is the operator's own end, like an abort: without a
            # session.end the run reads "stale" (a dead worker and no end).
            return self._finish(
                state,
                End(
                    "undone",
                    f"operator undid the last message at iter {iteration}; forked to {new_id}",
                    checkpoint=False,
                ),
                iteration=iteration,
            )
        if steer_result == "detach":
            # Not an end: the caller respawns a detached `resume` that appends to this
            # same log, so a persistent viewer follows straight through (no session.end).
            # The per-iteration snapshot is the resume point.
            return self._finish(
                state,
                End(
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
        """The run dir's name, the authoritative run id (stamped into the
        start events so every fold reads it from there); empty without a log."""
        return self.events.path.parent.name if self.events is not None else ""

    def _log(self, msg: str) -> None:
        self.logger(f"[agent6] {msg}")

    def _emit(self, event_type: str, **fields: Any) -> None:
        if self.events is not None:
            self.events.emit(event_type, **fields)

    def _emit_start(self, event_type: str, **fields: Any) -> None:
        """A start-family event goes through the one emitter that stamps the
        worker pid first (see :func:`agent6.sessions.ipc.emit_session_start`)."""
        if self.events is not None:
            emit_session_start(self.events, self.events.path.parent, event_type, **fields)

    def _emit_budget(self, iteration: int) -> None:
        """Per-iteration usage heartbeat: running token + cost totals. The fold
        keeps only its timestamp (the idle anchor) and reads totals from
        `budget.update`; the event at the start of each iteration keeps a long
        provider call distinguishable from a stall."""
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
