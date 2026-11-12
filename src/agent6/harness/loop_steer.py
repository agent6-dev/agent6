# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Apply steering at turn and execution boundaries."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import TYPE_CHECKING

from agent6 import terminal
from agent6.harness import loop_conversation, loop_snapshot, loop_state, parallel_join
from agent6.sessions import journal, lane_dispatch
from agent6.tools import operator_prompts, tool_errors
from agent6.vcs.git import ops as git_ops

if TYPE_CHECKING:
    from agent6.harness import loop


def seeded_steer(
    harness: loop.Harness,
    conversation: loop_conversation.Conversation,
    turn: int,
    state: loop_state.LoopState,
) -> loop_snapshot.SessionResult | None:
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
    conversation: loop_conversation.Conversation,
    turn: int,
    state: loop_state.LoopState,
) -> loop_snapshot.SessionResult | None:
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
    stopped = operator_stop(harness, turn, state)
    if stopped is not None:
        return stopped
    operator_state(harness, conversation, state)
    return steer_outcome(harness, harness.steering.handle(conversation, turn, state), turn, state)


def operator_stop(
    harness: loop.Harness, turn: int, state: loop_state.LoopState
) -> loop_snapshot.SessionResult | None:
    """End the execution on a pending stop request, consuming it.

    Args:
        harness: The execution's harness.
        turn: The turn the stop arrived in or after.
        state: The execution's state.

    Returns:
        The end, or None when no stop is pending.
    """
    if not (harness.bridge.stop_requested() or harness.bridge.should_abort()):
        return None
    immediate = harness.bridge.stop_is_now() or harness.bridge.should_abort()
    harness.bridge.stop_clear()
    when = f"during turn {turn}" if immediate else f"after turn {turn}"
    cut = "; running command was cut short" if immediate and state.command_cut else ""
    if immediate:
        harness.bridge.steer_clear()
        harness.log(f"LOOP: operator stop {when}{cut}")
    else:
        harness.log(f"LOOP: operator stop at the turn boundary (turn {turn})")
    return harness.finish(
        state,
        loop_snapshot.End(
            "steer_abort",
            f"operator stopped the run {when}{cut}{harness.dirty_tree_note()}",
            checkpoint=False,
        ),
        turn=turn,
    )


def ranked_opening(
    harness: loop.Harness,
    conversation: loop_conversation.Conversation,
    turn: int,
    state: loop_state.LoopState,
) -> loop_snapshot.SessionResult | None:
    """Run the dispatch a run opens with, joined by rank, and end the run on what landed.

    The ranking's winner joins the run's chain; the run's own gate then judges the joined
    tree. No model call is made unless the operator steered while the lanes ran: the run
    then goes on with that steer, as a resume with a steer does later.

    Args:
        harness: The execution's harness, built with an opening.
        conversation: The run's conversation, snapshotted before the lanes run.
        turn: The run's first turn.
        state: The execution's state.

    Returns:
        The end: `lanes_ranked` when a lane joined (or, with no chain to join onto, was
        placed), `no_lane_passed` when the ranking crowned none, `budget_exhausted` when the
        budget ended every lane, `no_lane_result` otherwise, or the operator's stop; None
        when a steer gives the run more work.
    """
    assert harness.opening is not None
    result = harness.parallel.dispatch(
        conversation, turn, state, [harness.opening], source="operator", join="rank"
    )
    stopped = operator_stop(harness, turn, state)
    if stopped is not None:
        return stopped
    if harness.bridge.steer_requested():
        # A steer sent while the lanes ran is the run's next work, from what landed.
        harness.parallel.deliver(conversation, result)
        size = len(conversation)
        asked = steer_outcome(
            harness, harness.steering.handle(conversation, turn, state), turn, state
        )
        if asked is not None or len(conversation) > size:
            return asked
    else:
        harness.parallel.record_delivery(result)
    # A refusal before any lane ran has no lanes to word: its own message is the summary.
    summary = parallel_join.rank_summary(result.lanes) if result.lanes else result.message
    statuses = {join.status for join in result.lanes}
    # Without a chain (`git.control = "model"`) a placed lane is the whole outcome.
    landed = statuses & {"joined", "no_changes"} or ("ranked" in statuses and not harness.chain.ref)
    if not landed:
        spent = harness.budget is not None and harness.budget.is_exhausted()
        reason: loop_snapshot.SessionEndReason = (
            "no_lane_passed"
            if "ranked" in statuses
            else "budget_exhausted"
            if spent
            else "no_lane_result"
        )
        return harness.finish(state, loop_snapshot.End(reason, summary, roots=False), turn=turn)
    if "joined" in statuses:
        try:
            harness.gate.after_rank_join(state)
        except tool_errors.OperatorCommandUnexecutableError as exc:
            return harness.unexecutable_abort(exc, turn=turn, state=state)
    return harness.finish(
        state,
        loop_snapshot.End("lanes_ranked", summary, completed=True, verdict="grounded"),
        turn=turn,
    )


def operator_state(
    harness: loop.Harness,
    conversation: loop_conversation.Conversation,
    state: loop_state.LoopState,
    *,
    boundary: bool = True,
) -> bool:
    """Record operator presence and deliver queued answers once.

    Presence changes are journaled on every poll but reach the model only at a turn boundary.

    Args:
        harness: The execution's harness.
        conversation: The conversation the notices enter.
        state: The execution's state.
        boundary: Whether to deliver the current presence rather than only journal it.

    Returns:
        Whether an answer was delivered.
    """
    prompts = harness.prompts
    presence = prompts.presence()
    if presence and presence != state.operator_presence:
        state.operator_presence = presence
        harness.emit(journal.OperatorPresence(presence=presence))
    if boundary and presence and presence != state.operator_presence_notified:
        state.operator_presence_notified = presence
        conversation.notice(f"OPERATOR PRESENCE: {presence}.")
    late = prompts.collect_answers()
    if not late:
        return False
    for item in late:
        for question, answer in zip(item.questions, item.answers, strict=False):
            harness.steering.record_decision(state, question.question, answer)
        state.delivered_questions.append(item.id)
    conversation.notice(answers_notice(late), origin="operator")
    harness.emit(journal.QuestionDelivered(ids=[item.id for item in late]))
    return True


def answers_notice(late: Sequence[operator_prompts.LateAnswer]) -> str:
    """Return the notice carrying queued questions' answers: each id, its questions, the answers."""
    lines = ["OPERATOR ANSWERS (to questions asked earlier):"]
    for item in late:
        lines.append(item.id)
        for question, answer in zip(item.questions, item.answers, strict=False):
            lines.append(f"  Q: {question.question}")
            lines.append(f"  A: {answer}")
    return "\n".join(lines)


def pending_notice(pending: Sequence[operator_prompts.QuestionRequest]) -> str:
    """Return the notice listing the questions still open, one line per question."""
    lines = ["QUESTIONS PENDING (asked earlier, no answer yet):"]
    for request in pending:
        lines.extend(f"{request.id}: {question.question}" for question in request.questions)
    return "\n".join(lines)


def resume_opening(
    harness: loop.Harness, conversation: loop_conversation.Conversation, state: loop_state.LoopState
) -> None:
    """Open a resumed execution with the operator's side: presence, missed answers, open questions.

    Args:
        harness: The execution's harness.
        conversation: The conversation the notices enter.
        state: The execution's state.
    """
    operator_state(harness, conversation, state)
    pending = harness.prompts.pending()
    if pending:
        conversation.notice(pending_notice(pending))


def merged_in_notice(merged: git_ops.ChainMerge) -> str:
    """Return the opening notice for a branch merged into the run's work before this execution.

    Args:
        merged: What the merge left behind.

    Returns:
        The branch and commit, then the paths carrying conflict markers and what to do with
        them; or that the work already held the branch.
    """
    if not merged.merged:
        return f"The run's work already contains {merged.branch}; nothing was merged in."
    head = (
        f"{merged.branch} was merged into this run's work before this execution"
        f" (commit {merged.sha[:12]})."
    )
    if not merged.conflicts:
        return f"{head} No conflicts. Check the merged result against the task before finishing."
    shown = ", ".join(terminal.printable_name(p) for p in merged.conflicts)
    notes = "".join(f"\n  {terminal.printable_name(n)}" for n in merged.notes)
    return (
        f"{head} Conflicts were left in: {shown}. A text conflict carries both sides between"
        f" markers; a binary or a modify/delete keeps one side, the other at {merged.branch}."
        f" Git's notes:{notes}\nResolve each one keeping both intents, remove every marker, and"
        " keep the verify command green before finishing."
    )


def execution_opening(
    harness: loop.Harness,
    conversation: loop_conversation.Conversation,
    turn: int,
    state: loop_state.LoopState,
    *,
    resuming: bool,
) -> loop_snapshot.SessionResult | None:
    """Open an execution's first turn with the operator's side, before the first model call.

    A fresh run hears the operator's presence; a resume hears it with the missed answers and
    the open questions, then takes the steer that started it.

    Args:
        harness: The execution's harness.
        conversation: The conversation the notices enter.
        turn: The first turn of the execution.
        state: The execution's state.
        resuming: Whether the execution resumes a snapshot.

    Returns:
        The end the seeded steer asked for, or None when the execution goes on.
    """
    if not resuming:
        operator_state(harness, conversation, state)
        if harness.opening is not None:
            return ranked_opening(harness, conversation, turn, state)
        return None
    resume_opening(harness, conversation, state)
    # An interrupted dispatch is finished or diagnosed before the first call.
    if harness.resume_state_path is not None and (
        lane_dispatch.records_dir(harness.resume_state_path.parent).is_dir()
    ):
        harness.parallel.reconcile(state, conversation)
    return seeded_steer(harness, conversation, turn, state)


def quiet_continuation(
    harness: loop.Harness,
    conversation: loop_conversation.Conversation,
    state: loop_state.LoopState,
    *,
    turn: int,
    reason: str,
) -> loop_snapshot.SessionResult | loop_state.NextTurn | None:
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
        return loop_state.NEXT_TURN
    if harness.interactive:
        parked = park_for_steer(harness, conversation, state, turn=turn, reason=reason)
        return loop_state.NEXT_TURN if parked is None else parked
    return None


def park_for_steer(
    harness: loop.Harness,
    conversation: loop_conversation.Conversation,
    state: loop_state.LoopState,
    *,
    turn: int,
    reason: str,
) -> loop_snapshot.SessionResult | None:
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
    harness.emit(journal.LoopParked(turn=turn, reason=reason))
    while True:
        if harness.bridge.stop_requested():
            harness.bridge.stop_clear()
            return harness.finish(
                state,
                loop_snapshot.End(
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
            harness.emit(journal.LoopParkedResumed(turn=turn))
            return None
        if harness.operator_tasks.take(state.root_task_id):
            # A queued task, goal or retirement is work; the next focus banner names it.
            harness.emit(journal.LoopParkedResumed(turn=turn))
            return None
        if operator_state(harness, conversation, state, boundary=False):
            # A queued question's answer is work too; the model reads it on its next turn.
            harness.emit(journal.LoopParkedResumed(turn=turn))
            return None
        time.sleep(0.5)


def steer_outcome(
    harness: loop.Harness, steer_result: str | None, turn: int, state: loop_state.LoopState
) -> loop_snapshot.SessionResult | None:
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
        reason: loop_snapshot.SessionEndReason = (
            "steer_exit" if steer_result == "exit" else "steer_abort"
        )
        return harness.finish(
            state,
            loop_snapshot.End(
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
        harness.emit(journal.SessionUndone(new_session_id=new_id, undone_text=undone_text))
        # An undo is the operator's own end; without a session.stop the run reads stale.
        return harness.finish(
            state,
            loop_snapshot.End(
                "undone",
                f"operator undid the last message at turn {turn}; forked to {new_id}",
                checkpoint=False,
            ),
            turn=turn,
        )
    if steer_result == "detach":
        # No session.stop: the caller respawns a detached `resume` that appends to this log.
        return harness.finish(
            state,
            loop_snapshot.End(
                "detached",
                f"operator detached at turn {turn}; resuming in the background",
                checkpoint=False,
                event=False,
            ),
            turn=turn,
        )
    return None
