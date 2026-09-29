# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Hold the operator's side of a run.

The callables a front-end injects for steering, stopping and compacting the loop, the steer
verbs, and the pin invariant.
"""

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

# `/pin` instructions are re-injected after every tier-2 restart, so their total is capped.
# Over the cap a pin lands as an ordinary steer; only its survival across compaction is refused.
PINS_MAX_CHARS = 4_000

# The steer texts that end or hand over the run: the verb, its event, its log line.
STEER_VERBS: dict[str, tuple[str, str, str]] = {
    "abort": ("abort", "loop.steer.aborted", "abort - halting the run"),
    "exit": ("exit", "loop.steer.exited", "exit - halting the run and leaving the terminal"),
    "/undo": ("undo", "loop.steer.undo", "/undo - forking back before the last message"),
    "detach": ("detach", "loop.steer.detached", "detach - stopping to resume in the background"),
}


@dataclass(frozen=True, slots=True)
class OperatorBridge:
    """What the operator can do to a running loop, as the front-end injects it.

    The defaults do nothing: a loop with no operator runs unattended.

    Attributes:
        steer_requested: Polled between iterations; a positive asks `steer_prompt`.
        steer_clear: Consumes the steer request.
        steer_prompt: The operator's instruction, or None.
        steer_reset: Called at each execution entry; disarms a SIGINT stage the prior execution
            never consumed, leaving the steer marker files alone.
        compact_requested: Polled at the pre-call boundary; a positive forces the tier-2
            summarise-and-restart. The marker travels the same file bridge as steer.
        compact_clear: Consumes the compact request.
        stop_requested: Polled at each completed-iteration boundary; a positive ends the run
            cleanly there. The mid-turn immediate stop is the steer "abort" answer.
        stop_clear: Consumes the stop request.
        should_abort: Polled during a streaming call; True once the operator asked to stop.
        should_interrupt: Polled during a streaming call; True once the operator asked to steer,
            so the watchdog ends the turn and the loop reaches its steer boundary at once.
        take_requests: What the operator queued for the graph (`/task`, `/standing`, `/retire`)
            since the last call, oldest first.
        after_auto_commit: Called with the iteration and sha of each landed auto-commit; "stop"
            ends the loop as interactive_stop, "undo" takes the `/undo` path, "continue" runs on.
        undo_forker: `/undo`: commits the tree onto the session's ref, forks the session before
            its last operator message and restores that tree; returns the new session id and
            the undone text, or None with the reason printed. Injected: harness never imports app.
        lane_spawner: The ui-side group spawner `/parallel` dispatches through; None (headless,
            or inside a lane) makes the directive answer with feedback and continue.
    """

    steer_requested: Callable[[], bool] = field(default=lambda: False)
    steer_clear: Callable[[], None] = field(default=lambda: None)
    steer_prompt: Callable[[], str | None] = field(default=lambda: None)
    steer_reset: Callable[[], None] = field(default=lambda: None)
    compact_requested: Callable[[], str | None] = field(default=lambda: None)
    compact_clear: Callable[[], None] = field(default=lambda: None)
    stop_requested: Callable[[], bool] = field(default=lambda: False)
    stop_clear: Callable[[], None] = field(default=lambda: None)
    should_abort: Callable[[], bool] = field(default=lambda: False)
    should_interrupt: Callable[[], bool] = field(default=lambda: False)
    take_requests: Callable[[], list[OperatorRequest]] = field(default=list)
    after_auto_commit: Callable[[int, str], AutoCommitDirective] = field(
        default=lambda _i, _sha: "continue"
    )
    undo_forker: Callable[[], tuple[str, str] | None] | None = None
    lane_spawner: GroupLaneSpawner | None = None


def try_pin(pins: list[str], instruction: str) -> bool:
    """Append an instruction to the pins when it is non-empty and fits `PINS_MAX_CHARS`.

    The one owner of the pin invariants: `/pin` and the pre-run `--pin` seeding both go through
    it.

    Args:
        pins: The run's pins, appended in place.
        instruction: The instruction text.

    Returns:
        Whether it was pinned.
    """
    instruction = instruction.strip()
    if not instruction:
        return False
    if sum(len(p) for p in pins) + len(instruction) > PINS_MAX_CHARS:
        return False
    pins.append(instruction)
    return True


@dataclass(frozen=True, slots=True)
class Steering:
    """What a steer's text means for the run, taken at an operator boundary or a park.

    A verb (`STEER_VERBS`), a `/parallel` directive dispatched at once, a `/pin` that survives
    compaction, a `/<skill>` expanded to its text, or an instruction injected into the
    conversation, paired with the question it answers when the model just asked one.

    Attributes:
        bridge: The operator's callables.
        parallel: Builds the dispatcher on the first `/parallel`.
        dispatcher: The run's tool dispatcher, for the resolved skills.
        record_decision: Records an answered question as a ruling.
        log: The run's text logger.
        emit: The run's event sink.
    """

    bridge: OperatorBridge
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
        """Take the operator's steer between iterations.

        The boundary is between completed iterations, so a tool_use and tool_result pair is
        never split.

        Args:
            conversation: The run's conversation.
            iteration: The iteration just completed.
            state: The loop state.

        Returns:
            The verb's name ("abort", "exit", "undo", "detach") when the steer was one; None
            otherwise (no request, an empty steer, a directive, a pin, a skill, or an injected
            instruction).
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
        """Handle a `/<skill> [args]` steer: the skill's full text is injected as the instruction.

        Args:
            conversation: The run's conversation.
            steer_text: The steer as typed.

        Returns:
            True when handled; False when the text names no enabled skill.
        """
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
        """Handle a `/pin` steer.

        A recorded pin is injected as a marked instruction and re-injected after every tier-2
        restart. Over the total cap the instruction is delivered as an ordinary steer, and only
        the durability is refused, loudly.

        Args:
            conversation: The run's conversation.
            state: The loop state holding the pins.
            steer_text: The steer as typed.

        Returns:
            True when handled; False when the text is not a pin directive.
        """
        try:
            instruction = parse_pin(steer_text)
        except DirectiveError as exc:
            conversation.notice(f"OPERATOR STEERING: nothing pinned: {exc}")
            self.log(f"  /pin refused: {exc}")
            return True
        if instruction is None:
            return False
        if not try_pin(state.pins, instruction):
            # parse_pin rejects an empty directive, so a refusal here is always the cap.
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
        """Handle a `/parallel` steer: dispatch a valid one, answer a malformed one and continue.

        Args:
            conversation: The run's conversation.
            iteration: The iteration just completed.
            state: The loop state.
            steer_text: The steer as typed.

        Returns:
            True when handled; False when the text is ordinary steering to inject.
        """
        try:
            segments = parse_directive(steer_text)
        except DirectiveError as exc:
            self.parallel().feedback(conversation, f"nothing dispatched: {exc}")
            return True
        if segments is None:
            return False
        self.parallel().dispatch(conversation, iteration, state, segments)
        return True
