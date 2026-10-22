# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Apply steering at execution boundaries."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from agent6.harness import _conversation, _loop_state, _snapshot

if TYPE_CHECKING:
    from agent6.harness import loop


def seeded_steer(
    harness: loop.Harness,
    conversation: _conversation.Conversation,
    turn: int,
    state: _loop_state.LoopState,
) -> _snapshot.SessionResult | None:
    """Consume the follow-up a `resume --steer` queued before the loop started.

    It enters the conversation ahead of the first model call, since a resumed conversation
    that was already finished ends on turn 1 before the end-of-turn poll runs.

    Args:
        harness: The execution's harness.
        conversation: The conversation the steer enters.
        turn: The first turn of the execution.
        state: The execution's state.

    Returns:
        The end the steer asked for, or None when the execution goes on.
    """
    # This resume path handles both finished and interrupted runs.
    return steer_outcome(
        harness,
        harness.steering.handle(
            conversation,
            turn,
            state,
            heading=(
                "OPERATOR STEERING: This execution continues the run even if it previously"
                " finished, and the steer below is its task."
            ),
        ),
        turn,
        state,
    )


def operator_boundary(
    harness: loop.Harness,
    conversation: _conversation.Conversation,
    turn: int,
    state: _loop_state.LoopState,
) -> _snapshot.SessionResult | None:
    """Honour a pending stop, then poll the steer flag, after every completed turn.

    A stop or an injected instruction never splits a tool call from its result.

    Args:
        harness: The execution's harness.
        conversation: The conversation a steer enters.
        turn: The turn just completed.
        state: The execution's state.

    Returns:
        The end the operator asked for, or None when the execution goes on.
    """
    # A background command's ending reaches disk only when observed; `/shells` reads it.
    harness.dispatcher.settle_background()
    if harness.bridge.stop_requested() or harness.bridge.should_abort():
        immediate = harness.bridge.stop_is_now() or harness.bridge.should_abort()
        harness.bridge.stop_clear()
        when = f"during step {turn}" if immediate else f"after step {turn}"
        cut = "; running command was cut short" if immediate and state.command_cut else ""
        if immediate:
            harness.bridge.steer_clear()
            harness.log(f"LOOP: operator stop {when}{cut}")
        else:
            harness.log(f"LOOP: operator stop at the step boundary (turn {turn})")
        return harness.finish(
            state,
            _snapshot.End(
                "steer_abort",
                f"operator stopped the run {when}{cut}{harness.dirty_tree_note()}",
                checkpoint=False,
            ),
            turn=turn,
        )
    return steer_outcome(harness, harness.steering.handle(conversation, turn, state), turn, state)


def quiet_continuation(
    harness: loop.Harness,
    conversation: _conversation.Conversation,
    state: _loop_state.LoopState,
    *,
    turn: int,
    reason: str,
) -> _snapshot.SessionResult | _loop_state.NextTurn | None:
    """Continue a quiet turn in run mode: a standing goal re-enters, else a park.

    Args:
        harness: The execution's harness.
        conversation: The conversation the re-entry enters.
        state: The execution's state.
        turn: The current turn.
        reason: The quiet end being continued.

    Returns:
        `NEXT_TURN` to go on, the end a park chose, or None when neither applies.
    """
    if harness.mode != "run":
        return None
    nudge = harness.standing.absorb(state, reason=reason, turn=turn)
    if nudge is not None:
        conversation.notice(nudge)
        return _loop_state.NEXT_TURN
    if harness.interactive:
        parked = park_for_steer(harness, conversation, state, turn=turn, reason=reason)
        return _loop_state.NEXT_TURN if parked is None else parked
    return None


def park_for_steer(
    harness: loop.Harness,
    conversation: _conversation.Conversation,
    state: _loop_state.LoopState,
    *,
    turn: int,
    reason: str,
) -> _snapshot.SessionResult | None:
    """Park a quiet interactive execution until the operator steers it.

    The conversation stays in memory with its snapshot on disk; there is no timeout.

    Args:
        harness: The execution's harness.
        conversation: The conversation a steer enters.
        state: The execution's state.
        turn: The current turn.
        reason: The quiet end being parked.

    Returns:
        The end a steer verb chose, or None when the execution continues.
    """
    harness.log(
        f"LOOP: parked at turn {turn} ({reason}) - waiting for your steer"
        " (any composer or the pause menu; abort ends the run)"
    )
    harness.emit("loop.parked", turn=turn, reason=reason)
    while True:
        if harness.bridge.stop_requested():
            harness.bridge.stop_clear()
            return harness.finish(
                state,
                _snapshot.End(
                    "steer_abort",
                    f"operator stopped the parked run{harness.dirty_tree_note()}",
                    checkpoint=False,
                ),
                turn=turn,
            )
        if harness.bridge.should_abort():
            return steer_outcome(harness, "abort", turn, state)
        if harness.bridge.steer_requested():
            verb = harness.steering.handle(conversation, turn, state)
            if verb is not None:
                return steer_outcome(harness, verb, turn, state)
            # Injected (or a bare poke): the run continues where it parked.
            harness.emit("loop.parked.resumed", turn=turn)
            return None
        if harness.operator_tasks.take(state.root_task_id):
            # A queued task, goal or retirement is work; the next focus banner names it.
            harness.emit("loop.parked.resumed", turn=turn)
            return None
        time.sleep(0.5)


def steer_outcome(
    harness: loop.Harness, steer_result: str | None, turn: int, state: _loop_state.LoopState
) -> _snapshot.SessionResult | None:
    """Map a steer verb to the execution's end.

    Args:
        harness: The execution's harness.
        steer_result: The verb `Steering.handle` returned; None for an injected instruction.
        turn: The current turn.
        state: The execution's state.

    Returns:
        The end, or None when the execution goes on.
    """
    if steer_result in ("abort", "exit"):
        # "exit" is the same stop; its reason tells the CLI to skip the follow-up prompt.
        reason: _snapshot.SessionEndReason = (
            "steer_exit" if steer_result == "exit" else "steer_abort"
        )
        return harness.finish(
            state,
            _snapshot.End(
                reason,
                f"operator {'exited' if steer_result == 'exit' else 'aborted'}"
                f" at turn {turn} via steering prompt{harness.dirty_tree_note()}",
                checkpoint=False,
            ),
            turn=turn,
        )
    if steer_result == "undo":
        forked = harness.bridge.undo_forker() if harness.bridge.undo_forker is not None else None
        if forked is None:
            # The forker printed why, or none is wired.
            harness.log("  /undo: nothing to undo; continuing")
            return None
        new_id, undone_text = forked
        harness.emit("session.undone", new_session_id=new_id, undone_text=undone_text)
        # An undo is the operator's own end; without a session.end the run reads stale.
        return harness.finish(
            state,
            _snapshot.End(
                "undone",
                f"operator undid the last message at turn {turn}; forked to {new_id}",
                checkpoint=False,
            ),
            turn=turn,
        )
    if steer_result == "detach":
        # No session.end: the caller respawns a detached `resume` that appends to this log.
        return harness.finish(
            state,
            _snapshot.End(
                "detached",
                f"operator detached at turn {turn}; resuming in the background",
                checkpoint=False,
                event=False,
            ),
            turn=turn,
        )
    return None
