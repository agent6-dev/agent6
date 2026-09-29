# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The operator's side of a run: the callables a front-end injects for
steering, stopping and compacting the loop, the steer verbs, and the pin
invariant."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from agent6.directive import DirectiveError, parse_directive, parse_pin
from agent6.harness._conversation import last_assistant_prose
from agent6.harness._nudges import ending_question
from agent6.harness.subrun import GroupLaneSpawner
from agent6.kinds import AutoCommitDirective
from agent6.sessions.ipc import OperatorRequest
from agent6.skills import skill_command, skill_steer_payload

if TYPE_CHECKING:
    from agent6.harness._conversation import Conversation
    from agent6.harness._loop_state import LoopState
    from agent6.harness._parallel_dispatch import ParallelDispatcher
    from agent6.tools.dispatch import ToolDispatcher

# `/pin` instructions are re-injected verbatim after every tier-2 restart, so
# their total is capped. Over the cap a pin lands as an ordinary steer (the
# instruction still reaches the model once); only its survives-compaction
# durability is refused.
PINS_MAX_CHARS = 4_000

# The steer texts that end or hand over the run, as the operator types them:
# the verb the loop acts on, the event that records it, the log line.
STEER_VERBS: dict[str, tuple[str, str, str]] = {
    "abort": ("abort", "loop.steer.aborted", "abort - halting the run"),
    "exit": ("exit", "loop.steer.exited", "exit - halting the run and leaving the terminal"),
    "/undo": ("undo", "loop.steer.undo", "/undo - forking back before the last message"),
    "detach": ("detach", "loop.steer.detached", "detach - stopping to resume in the background"),
}


@dataclass(frozen=True, slots=True)
class OperatorBridge:
    """What the operator can do to a running loop, as the front-end injects
    it. The defaults do nothing: a loop with no operator runs unattended."""

    # Polled between iterations; on a positive the loop asks `steer_prompt`
    # for the instruction (or "abort") and `steer_clear` consumes the request.
    steer_requested: Callable[[], bool] = field(default=lambda: False)
    steer_clear: Callable[[], None] = field(default=lambda: None)
    steer_prompt: Callable[[], str | None] = field(default=lambda: None)
    # Called at each execution entry (run/resume): disarms a SIGINT stage the prior
    # execution never consumed, without touching the steer marker files.
    steer_reset: Callable[[], None] = field(default=lambda: None)
    # "Compact now" from a front-end: polled at the same pre-call boundary as
    # the tiered thresholds; a positive forces the tier-2 summarise-and-restart.
    # The marker travels the same file bridge as steer.
    compact_requested: Callable[[], str | None] = field(default=lambda: None)
    compact_clear: Callable[[], None] = field(default=lambda: None)
    # "Stop after this step": polled at each completed-iteration boundary
    # (tool results and auto-commit landed), ending the run cleanly there. The
    # mid-turn immediate stop is the steer "abort" answer.
    stop_requested: Callable[[], bool] = field(default=lambda: False)
    stop_clear: Callable[[], None] = field(default=lambda: None)
    # Polled DURING a streaming model call: True once the operator asked to
    # stop, so a long reasoning turn aborts promptly.
    should_abort: Callable[[], bool] = field(default=lambda: False)
    # Polled DURING a streaming call: True once the operator asked to steer
    # (Ctrl-C, the TUI's `s`), so the watchdog ends the turn and the loop
    # reaches its steer boundary at once instead of waiting the turn out.
    should_interrupt: Callable[[], bool] = field(default=lambda: False)
    # What the operator queued for the graph (`/task`, `/standing`, `/retire`),
    # taken at each pre-call boundary and while the run is parked; each call
    # returns what arrived since the last, oldest first.
    take_requests: Callable[[], list[OperatorRequest]] = field(default=list)
    # Called once per landed auto-commit. "stop" ends the loop cleanly as
    # interactive_stop; "undo" takes the steer's /undo path; "continue" (the
    # default) runs the next iteration. `agent6 run -i` installs its REPL
    # prompt here.
    after_auto_commit: Callable[[int, str], AutoCommitDirective] = field(
        default=lambda _i, _sha: "continue"
    )
    # `/undo`: commits the tree as it stands onto the session's ref, forks the
    # session at the state before its last operator message and puts the
    # checkout back to that tree (app.undo.undo_fork, injected: harness never
    # import app); returns (new_session_id, undone_text), or None with the
    # reason printed.
    undo_forker: Callable[[], tuple[str, str] | None] | None = None
    # `/parallel` steer dispatch: the ui-side group spawner that runs a sibling
    # group of subordinate lanes to completion and imports their branches into
    # this run's repo. None (the default, every headless path, and inside a
    # lane: depth 1) makes a `/parallel` directive answer with feedback and
    # continue.
    lane_spawner: GroupLaneSpawner | None = None


def try_pin(pins: list[str], instruction: str) -> bool:
    """Append *instruction* to *pins* when it is non-empty and fits the
    PINS_MAX_CHARS total; whether it was pinned. The one owner of the pin
    invariants: `/pin` and the pre-run --pin seeding both go through it."""
    instruction = instruction.strip()
    if not instruction:
        return False
    if sum(len(p) for p in pins) + len(instruction) > PINS_MAX_CHARS:
        return False
    pins.append(instruction)
    return True


@dataclass(frozen=True, slots=True)
class Steering:
    """What a steer's text means for the run, taken at an operator boundary
    or a park: a verb (`STEER_VERBS`: the name the loop maps to an end), a
    `/parallel` directive dispatched at once, a `/pin` that survives
    compaction, a `/<skill>` expanded to its text, or an instruction
    injected into the conversation (paired with the question it answers,
    when the model just asked one)."""

    bridge: OperatorBridge
    # Built on the first `/parallel` (the dispatcher reads the config then).
    parallel: Callable[[], ParallelDispatcher]
    dispatcher: ToolDispatcher
    record_decision: Callable[[LoopState, str, str], None]
    log: Callable[[str], None]
    emit: Callable[..., None]

    def handle(
        self,
        conversation: Conversation,
        iteration: int,
        state: LoopState,
    ) -> str | None:
        """Operator steering between iterations.

        Returns `"abort"` if the operator typed "abort" at the prompt;
        the loop should then return a steer_abort result. Returns `None`
        in all other cases (no request, empty steer, `/parallel` dispatch,
        or instruction injected into the conversation).

        Polls steer_requested() and, on a positive, calls steer_prompt()
        to capture operator text. Empty / None / KeyboardInterrupt aborts;
        boundary is between completed iters so a tool_use / tool_result pair
        is never split. A message starting with the exact `/parallel` token
        is a dispatch directive (see `ParallelDispatcher.dispatch`), not an injected
        instruction.
        """
        if not self.bridge.steer_requested():
            return None
        self.emit("loop.steer.requested", iteration=iteration)
        self.log(f"STEER: operator steering at iter {iteration}")
        try:
            text = self.bridge.steer_prompt()
        finally:
            self.bridge.steer_clear()
        if text is None or not text.strip():
            self.log("  (empty - continuing)")
            return None
        steer_text = text.strip()
        if verb := STEER_VERBS.get(steer_text.lower()):
            name, event, line = verb
            self.emit(event)
            self.log(f"  {line}")
            return name
        if (
            self.directive(conversation, iteration, state, steer_text)
            or self.pin(conversation, state, steer_text)
            or self.skill(conversation, steer_text)
        ):
            return None
        self.log(f"  injecting steering instruction ({len(steer_text)} chars)")
        self.emit("loop.steer.injected", chars=len(steer_text), text=steer_text)
        asked = last_assistant_prose(conversation)
        if question := ending_question(asked):
            self.record_decision(state, question, steer_text)
        conversation.notice(
            f"OPERATOR STEERING (a mid-run instruction from the operator):\n{steer_text}"
        )
        return None

    def skill(self, conversation: Conversation, steer_text: str) -> bool:
        """Handle a `/<skill> [args]` steer from any composer: the skill's
        full text is injected as the instruction (the same payload on every
        surface). Returns True when handled; False when *steer_text* names no
        enabled skill."""
        if not steer_text.startswith("/"):
            return False
        found = skill_command(steer_text, self.dispatcher.resolved_skills())
        if found is None:
            return False
        skill, args = found
        self.log(f"  skill steer: {skill.name}")
        self.emit("loop.steer.skill", name=skill.name, args=args)
        conversation.notice(
            "OPERATOR STEERING (a mid-run instruction from the operator):\n"
            + skill_steer_payload(skill.name, skill.text, args)
        )
        return True

    def pin(self, conversation: Conversation, state: LoopState, steer_text: str) -> bool:
        """Handle a steer that is a `/pin` directive. A recorded pin is injected
        as a marked instruction AND re-injected verbatim after every tier-2
        restart. Over the total cap, the instruction is still delivered as an
        ordinary steer -- only the durability is refused, loudly. Returns True
        when handled; False when *steer_text* is not a pin directive."""
        try:
            instruction = parse_pin(steer_text)
        except DirectiveError as exc:
            conversation.notice(f"OPERATOR STEERING: nothing pinned: {exc}")
            self.log(f"  /pin refused: {exc}")
            return True
        if instruction is None:
            return False
        if not try_pin(state.pins, instruction):
            # parse_pin already rejects an empty directive, so a refusal here is
            # always the cap: deliver the instruction as an ordinary steer.
            self.log(f"  /pin over cap (> {PINS_MAX_CHARS}); delivered as an ordinary steer")
            self.emit("loop.pin.refused", chars=len(instruction), limit=PINS_MAX_CHARS)
            conversation.notice(
                f"OPERATOR STEERING (not pinned: the {PINS_MAX_CHARS}-char pin cap is"
                " full, so this instruction does not survive context compaction):\n"
                f"{instruction}"
            )
            return True
        self.log(f"  pinned instruction ({len(instruction)} chars, {len(state.pins)} pins)")
        self.emit("loop.pin.added", text=instruction, chars=len(instruction), count=len(state.pins))
        conversation.notice(
            "OPERATOR STEERING (pinned: this instruction survives context compaction"
            " and binds for the rest of the run):\n"
            f"{instruction}"
        )
        return True

    def directive(
        self,
        conversation: Conversation,
        iteration: int,
        state: LoopState,
        steer_text: str,
    ) -> bool:
        """Handle a steer that is a `/parallel` directive: dispatch a valid one,
        or answer a malformed one (a bare `/parallel`, a spec with no task) and
        continue. Returns True when handled; False when *steer_text* is ordinary
        steering to inject as an instruction."""
        try:
            segments = parse_directive(steer_text)
        except DirectiveError as exc:
            self.parallel().feedback(conversation, f"nothing dispatched: {exc}")
            return True
        if segments is None:
            return False
        self.parallel().dispatch(conversation, iteration, state, segments)
        return True
