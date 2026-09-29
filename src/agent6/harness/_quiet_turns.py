# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Answer the turns that say nothing with a nudge.

`QuietGuard` holds the counters; each function answers one kind of quiet turn with the
`Nudge` the loop puts in the conversation, or None when the turn stands.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from agent6.harness._advice import Nudge, TurnContext
from agent6.harness._nudges import (
    QUESTION_NUDGE,
    SILENT_NO_WORK_NUDGE,
    SILENT_NO_WORK_PATIENCE,
    WENT_QUIET_NUDGE,
    ends_with_question,
    reasoning_starved_nudge,
)
from agent6.harness._provider_call import reasoning_starvation
from agent6.providers import ProviderResponse

if TYPE_CHECKING:
    from agent6.harness._loop_state import LoopState

# An early prose turn on an untouched tree is a stall for this many iterations, then a finish.
SILENT_NO_WORK_UNTIL = 3


@dataclass(slots=True)
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


def silent_no_work(state: LoopState, ctx: TurnContext) -> Nudge | None:
    """Steer an early prose turn on an untouched tree back to the tools.

    A prose turn within the first `SILENT_NO_WORK_UNTIL` iterations of a run that edited nothing
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
        and ctx.iteration <= SILENT_NO_WORK_UNTIL
        and not state.ever_edited
        and not state.verify.ever_passed
        and quiet.silent_no_work_nudges_used < SILENT_NO_WORK_PATIENCE
    ):
        return None
    quiet.silent_no_work_nudges_used += 1
    return Nudge(
        SILENT_NO_WORK_NUDGE,
        event="loop.silent_no_work.nudge",
        fields={"iteration": ctx.iteration, "nudges_used": quiet.silent_no_work_nudges_used},
        log=(
            f"  silent finish rejected: no work yet (nudge"
            f" #{quiet.silent_no_work_nudges_used}) at iter {ctx.iteration}"
        ),
    )


def question_in_prose(state: LoopState, ctx: TurnContext, text: str) -> Nudge | None:
    """Tell the model once per run that a question in prose reaches nobody.

    A second question is accepted as the finish, so a stubborn model cannot loop the run.

    Args:
        state: The execution's state.
        ctx: The turn's context.
        text: The turn's prose.

    Returns:
        The nudge to call ask_user or finish_session, or None.
    """
    if ctx.mode != "run" or state.quiet.question_nudged or not ends_with_question(text):
        return None
    state.quiet.question_nudged = True
    return Nudge(
        QUESTION_NUDGE,
        event="loop.question_nudge",
        fields={"iteration": ctx.iteration},
        log=f"  silent_finish nudged: ended on a question at iter {ctx.iteration}",
    )


def went_quiet(state: LoopState, ctx: TurnContext, resp: ProviderResponse) -> Nudge | None:
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
    starved = reasoning_starvation(resp) > 0
    return Nudge(
        reasoning_starved_nudge(resp.output_tokens) if starved else WENT_QUIET_NUDGE,
        event="loop.went_quiet.nudge",
        fields={
            "iteration": ctx.iteration,
            "nudges_used": quiet.went_quiet_nudges_used,
            "nudges_max": cap,
            "output_tokens": resp.output_tokens,
        },
    )
