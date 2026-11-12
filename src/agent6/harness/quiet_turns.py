# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Drive a turn with no tool call: a nudge, a silent finish or a went-quiet end.

`QuietGuard` holds the counters; each rule answers one kind of quiet turn with the `Nudge`
the loop puts in the conversation, or None when the turn stands; the `handle_*` functions
run the turn through the rules, the end gates and the steer continuation.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from agent6.harness import (
    advice,
    finish_gates,
    loop_conversation,
    loop_snapshot,
    loop_state,
    loop_steer,
    nudges,
    provider_call,
)
from agent6.providers import ProviderResponse
from agent6.sessions import journal

if TYPE_CHECKING:
    from agent6.harness import loop

# An early prose turn on an untouched tree is a stall for this many turns, then a finish.
SILENT_NO_WORK_UNTIL = 3


@dataclasses.dataclass(slots=True)
class QuietGuard:
    """Count the nudges the quiet turns of one execution drew.

    Attributes:
        went_quiet_nudges_used: Nudges the streak of empty turns drew; any other turn resets it.
        silent_no_work_nudges_used: Nudges early prose turns on an untouched tree drew.
        question_nudged: Whether the once-per-run nudge for a prose turn ending on a question fired.
    """

    went_quiet_nudges_used: int = 0
    silent_no_work_nudges_used: int = 0
    question_nudged: bool = False


def silent_no_work(state: loop_state.LoopState, ctx: advice.TurnContext) -> advice.Nudge | None:
    """Steer an early prose turn on an untouched tree back to the tools.

    A prose turn within the first `SILENT_NO_WORK_UNTIL` turns of a run that edited nothing
    is a stall, nudged up to `SILENT_NO_WORK_PATIENCE` times; a later prose turn is a finish.

    Args:
        state: The execution's state.
        ctx: The turn's context.

    Returns:
        The nudge, or None when the turn stands as a finish.
    """
    quiet = state.quiet
    if not (
        ctx.mode == "run"
        and ctx.turn <= SILENT_NO_WORK_UNTIL
        and not state.ever_edited
        and not state.verify.ever_passed
        and quiet.silent_no_work_nudges_used < nudges.SILENT_NO_WORK_PATIENCE
    ):
        return None
    quiet.silent_no_work_nudges_used += 1
    return advice.Nudge(
        nudges.SILENT_NO_WORK_NUDGE,
        event=journal.NudgeSilentNoWork(
            turn=ctx.turn, nudges_used=quiet.silent_no_work_nudges_used
        ),
        log=(
            f"  silent finish rejected: no work yet (nudge"
            f" #{quiet.silent_no_work_nudges_used}) at turn {ctx.turn}"
        ),
    )


def question_in_prose(
    state: loop_state.LoopState, ctx: advice.TurnContext, text: str
) -> advice.Nudge | None:
    """Tell the model once per run that a question in prose reaches nobody.

    A second question is accepted as the finish, so a stubborn model cannot loop the run.

    Args:
        state: The execution's state.
        ctx: The turn's context.
        text: The turn's prose.

    Returns:
        The nudge to call ask_user or finish_session, or None.
    """
    if ctx.mode != "run" or state.quiet.question_nudged or not nudges.ends_with_question(text):
        return None
    state.quiet.question_nudged = True
    return advice.Nudge(
        nudges.QUESTION_NUDGE,
        event=journal.NudgeQuestion(turn=ctx.turn),
        log=f"  silent_finish nudged: ended on a question at turn {ctx.turn}",
    )


def went_quiet(
    state: loop_state.LoopState, ctx: advice.TurnContext, resp: ProviderResponse
) -> advice.Nudge | None:
    """Nudge an empty turn, up to the cap per streak.

    A turn that spent its whole output budget on reasoning gets the starved wording.

    Args:
        state: The execution's state.
        ctx: The turn's context, which carries the cap.
        resp: The empty response.

    Returns:
        The nudge, or None once the cap is spent and the run ends as went_quiet.
    """
    cap = ctx.went_quiet_max_nudges
    quiet = state.quiet
    if quiet.went_quiet_nudges_used >= cap:
        return None
    quiet.went_quiet_nudges_used += 1
    starved = provider_call.reasoning_starvation(resp) > 0
    return advice.Nudge(
        nudges.reasoning_starved_nudge(resp.output_tokens) if starved else nudges.WENT_QUIET_NUDGE,
        event=journal.NudgeWentQuiet(
            turn=ctx.turn,
            nudges_used=quiet.went_quiet_nudges_used,
            nudges_max=cap,
            output_tokens=resp.output_tokens,
        ),
    )


def handle_no_tool_use(
    wf: loop.Harness,
    resp: ProviderResponse,
    assistant: loop_conversation.AssistantMessage,
    conversation: loop_conversation.Conversation,
    state: loop_state.LoopState,
    ctx: advice.TurnContext,
) -> loop_snapshot.SessionResult | None:
    """Handle a turn with no tool call: a silent finish with text, went-quiet without.

    A silent finish passes the gates a finish_session would; an empty turn is nudged
    up to a cap and never reads as success.

    Args:
        wf: The harness driving the execution.
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
        turn = loop_state.TurnState(number=ctx.turn, resp=resp, assistant=assistant)
        return handle_silent_finish(wf, text, conversation, state, turn, ctx)
    return handle_went_quiet(wf, resp, conversation, state, ctx)


def handle_silent_finish(
    wf: loop.Harness,
    text: str,
    conversation: loop_conversation.Conversation,
    state: loop_state.LoopState,
    turn: loop_state.TurnState,
    ctx: advice.TurnContext,
) -> loop_snapshot.SessionResult | None:
    """Run a prose turn through the end gates as an implicit finish.

    Args:
        wf: The harness driving the execution.
        text: The turn's prose.
        conversation: The conversation the gates' notices enter.
        state: The execution's state.
        turn: This turn's state.
        ctx: This turn's context.

    Returns:
        The silent-finish end, or None when a gate sent the worker back to work.
    """
    if (stall := silent_no_work(state, ctx)) is not None:
        wf.tell(conversation, stall)
        return None
    aborted = wf.end_gates(
        state, turn, ctx, ending="silent_finish", gates=finish_gates.SILENT_END_GATES
    )
    # A prose turn has no tool results, so the gates' notices go to the conversation.
    for item in turn.tool_results:
        if isinstance(item, loop_conversation.Notice):
            conversation.notice(item.text)
    if aborted is not None or turn.end_returned:
        return aborted
    if (asked := question_in_prose(state, ctx, text)) is not None:
        wf.tell(conversation, asked)
        return None
    # A standing goal re-enters, else an interactive run parks for a steer; ask mode ends.
    cont = loop_steer.quiet_continuation(
        wf, conversation, state, turn=turn.number, reason="silent_finish"
    )
    if cont is not None:
        return None if isinstance(cont, loop_state.NextTurn) else cont
    # In ask mode the prose is the answer, so the end reads "answered".
    reason: loop_snapshot.SessionEndReason = "answered" if wf.mode == "ask" else "silent_finish"
    if wf.mode == "ask":
        wf.log(f"  ask answered at turn {turn.number}")
    else:
        wf.log(f"LOOP: silent_finish at turn {turn.number} - agent emitted text but no tool_use")
    # Run and plan ground on the verify state as finish_session does; ask keeps the answer.
    return wf.finish(
        state,
        loop_snapshot.End(
            reason,
            text if wf.mode == "ask" else advice.with_open_tasks(text[:1000], wf.open_subtasks()),
            completed=True,
            verdict="grounded" if reason == "silent_finish" else "passed",
        ),
        turn=turn.number,
    )


def handle_went_quiet(
    wf: loop.Harness,
    resp: ProviderResponse,
    conversation: loop_conversation.Conversation,
    state: loop_state.LoopState,
    ctx: advice.TurnContext,
) -> loop_snapshot.SessionResult | None:
    """Nudge an empty turn up to the per-streak cap, then end the execution as went_quiet.

    The empty assistant message leaves the conversation first: providers reject an
    assistant message with no content.

    Args:
        wf: The harness driving the execution.
        resp: The model's response.
        conversation: The conversation so far.
        state: The execution's state.
        ctx: This turn's context.

    Returns:
        The end, or None to go on after a nudge.
    """
    turn = ctx.turn
    reasoning_chars = provider_call.reasoning_starvation(resp)
    starved = reasoning_chars > 0
    if starved:
        wf.log(
            f"LOOP: reasoning_starvation at turn {turn}"
            f" - stop_reason=length, reasoning_chars={reasoning_chars},"
            f" output_tokens={resp.output_tokens}; the model spent"
            f" its entire output budget on reasoning_content."
            f" Add this model to _REASONING_MODEL_HINTS in"
            f" providers/openai.py if it isn't already."
        )
        wf.emit(
            journal.ModelReasoningStarvation(
                turn=turn,
                reasoning_chars=reasoning_chars,
                output_tokens=resp.output_tokens,
                stop_reason=resp.stop_reason,
            )
        )
    # A subscription plan is metered in points, so it says "spent", not "billed".
    plan_metered = wf.budget is not None and wf.budget.snapshot().plan_latest is not None
    spent_word = "spent" if plan_metered else "billed"
    billed = (
        f" ({resp.output_tokens} output tokens {spent_word} on it: reasoning that never"
        " surfaced, or a tool call the provider dropped)"
        if resp.output_tokens > 0 and not starved
        else ""
    )
    wf.log(f"LOOP: went_quiet at turn {turn} - agent emitted no text and no tool_use{billed}")
    # Every path below calls again or snapshots, and a provider rejects the empty turn.
    conversation.pop_quiet_assistant()
    if (nudge := went_quiet(state, ctx, resp)) is not None:
        wf.tell(conversation, nudge)
        return None
    cont = loop_steer.quiet_continuation(wf, conversation, state, turn=turn, reason="went_quiet")
    if cont is not None:
        return None if isinstance(cont, loop_state.NextTurn) else cont
    return wf.finish(
        state,
        loop_snapshot.End("went_quiet", "(agent emitted no text and no tool_use)"),
        turn=turn,
    )
