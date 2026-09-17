# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The turns that say nothing: the nudges a prose turn with no tool call and
an empty turn draw (`QuietGuard` holds their counters), each one function
answering with a `Nudge` the loop puts in the conversation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

from agent6.providers import ProviderResponse
from agent6.workflows._advice import Nudge, TurnContext
from agent6.workflows._nudges import (
    QUESTION_NUDGE,
    SILENT_NO_WORK_NUDGE,
    SILENT_NO_WORK_PATIENCE,
    WENT_QUIET_NUDGE,
    ends_with_question,
    reasoning_starved_nudge,
)
from agent6.workflows._provider_call import reasoning_starvation

if TYPE_CHECKING:
    from agent6.workflows._loop_state import LoopState

# An early prose turn on an untouched tree is a stall, not a finish, for
# this many iterations; a later prose finish is honoured.
SILENT_NO_WORK_UNTIL = 3


@dataclass(slots=True)
class QuietGuard:
    """The turns that say nothing: an empty turn draws a nudge up to the cap
    per streak (`went_quiet_nudges_used`, reset by any non-empty turn); an
    early prose turn on an untouched tree draws `SILENT_NO_WORK_PATIENCE`
    nudges; a prose turn ending on a question draws one nudge to call
    ask_user."""

    went_quiet_nudges_used: int = 0
    silent_no_work_nudges_used: int = 0
    question_nudged: bool = False


def silent_no_work(state: LoopState, ctx: TurnContext) -> Nudge | None:
    """A prose turn with no tool call on an untouched tree within the first
    `SILENT_NO_WORK_UNTIL` iterations of a run is a stall (a chat-tuned model
    answering the problem statement in prose), steered back to the tools up
    to `SILENT_NO_WORK_PATIENCE` times; a run that read its fill and answers
    in prose is a legitimate implicit finish."""
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
    """A run's prose turn that ends on a question reaches nobody: once per
    run, the model is told to call ask_user or finish_session; asking again
    is accepted as the finish, so a stubborn model cannot loop the run."""
    if ctx.mode != "run" or state.quiet.question_nudged or not ends_with_question(text):
        return None
    state.quiet.question_nudged = True
    return Nudge(
        QUESTION_NUDGE,
        event="loop.question_nudge",
        fields={"iteration": ctx.iteration},
        log=f"  silent_finish nudged: ended on a question at iter {ctx.iteration}",
    )


def went_quiet_cap(ctx: TurnContext) -> int:
    """The empty-turn nudge cap: `AGENT6_WENT_QUIET_MAX_NUDGES` when set,
    else `went_quiet_max_nudges`."""
    env_max = os.environ.get("AGENT6_WENT_QUIET_MAX_NUDGES", "").strip()
    return int(env_max) if env_max.isdigit() else ctx.guards.went_quiet_max_nudges


def went_quiet(state: LoopState, ctx: TurnContext, resp: ProviderResponse) -> Nudge | None:
    """An empty turn (no text, no tool call) is answered with a nudge up to
    the cap per streak (any non-empty turn refills it); a turn that spent
    its whole output budget on reasoning gets the starved wording. None once
    the cap is spent: the run ends as went_quiet unless a standing goal or a
    watching operator continues it."""
    cap = went_quiet_cap(ctx)
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
