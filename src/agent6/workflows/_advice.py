# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""What the loop's advisors and finish gates answer with, and what they
read: a `Nudge` (a notice for the model, recorded), a `Stop` (an end of the
run), a `Refusal` (a finish handed back), the frozen `TurnContext` of run
facts, and the operator's `GuardSettings`. The advisors live in `_guards`
and `_metric`, the gates in `_finish_gates`; the loop applies their
answers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from agent6.graph.models import TaskNode
from agent6.graph.order import OPEN_STATUSES
from agent6.workflows._session_state import End

if TYPE_CHECKING:
    from agent6.workflows._loop_state import LoopState, TurnState


@dataclass(frozen=True, slots=True)
class GuardSettings:
    """The guards' operator-facing knobs. `went_quiet_max_nudges`: an empty
    turn (no text, no tool call) is answered with a harness notice and
    re-asked up to this many times per streak, reasoning-starvation bursts
    included; 0 ends the run on the first. `loop_guard_kill_threshold`: the
    same (tool, args) call this many times in a row ends the run as
    loop_guard_killed (the notice fires from three, every other turn); 0
    leaves the notice alone. `stagnation_notice_after_s`: one notice when
    this much wall clock passes with no edit and no verify (a recall spiral
    makes few calls with long reasoning between them, below every
    call-count guard's horizon); 0 disables."""

    went_quiet_max_nudges: int = 4
    loop_guard_kill_threshold: int = 10
    stagnation_notice_after_s: float = 300.0


@dataclass(frozen=True, slots=True)
class Nudge:
    """What an advisor says to the model this turn, and how the harness
    records it: the notice text, the event (with its fields) and the log
    line; "" skips the event or the line."""

    text: str
    event: str = ""
    fields: Mapping[str, object] = field(default_factory=dict)
    log: str = ""


@dataclass(frozen=True, slots=True)
class Stop:
    """An advisor's decision to end the run, honoured once the turn's results
    and snapshot are on disk (`Workflow._turn_stop_checks`): `end` composes
    what `_finish` records, called then so it reads the run as the end gates
    left it; `log` is the line written then. `soft` names the reason a
    standing task's re-entry nudge carries when it absorbs the stop ("" = a
    hard stop nothing converts); `declared` names the ending the end gates
    judge as soon as the stop is decided ("" = a fault no gate judges). The
    event is emitted when the stop is decided."""

    end: Callable[[], End]
    soft: str = ""
    declared: str = ""
    event: str = ""
    fields: Mapping[str, object] = field(default_factory=dict)
    log: str = ""


@dataclass(frozen=True, slots=True)
class TurnContext:
    """The run facts an advisor reads, built once per turn. A fact that can
    move within the turn (a harness verify can deny or un-adopt the gate,
    an edit moves the tree) is a zero-arg callable read where it is needed."""

    mode: Literal["run", "plan", "ask", "agent"]
    iteration: int
    # The leg's first iteration: a turn allowance counts from it.
    leg_start: int
    guards: GuardSettings
    # `[workflow].verify_when` and `verify_retries`: what the red-gate return
    # rule reads (`verify_command` moves mid-run and is a callable below).
    verify_when: Literal["finish", "step", "never"]
    verify_retries: int
    # A machine agent state's finish contract: the problems with a finish
    # payload (empty = conforms); None leaves finishes ungated.
    finish_validator: Callable[[dict[str, Any] | None], list[str]] | None
    # A metric goal is configured: the plateau and ceiling rules own the
    # run's end, and the plain-run guards stand down.
    metric: bool
    # A memory store is wired, so the memory nudges apply.
    memory_wired: bool
    gate_present: Callable[[], bool]
    verify_command: Callable[[], tuple[str, ...]]
    tree_sha: Callable[[], str]
    tree_green: Callable[[], bool | None]
    budget_remaining: Callable[[], float | None]
    operator_wait_s: Callable[[], float]
    open_subtasks: Callable[[], list[tuple[str, str]]]
    # The before-finish panel's verdict over an end declared on the turn
    # (named by the ending it judges): True when it rejected the end. Sits
    # once per turn; it records its own verdict.
    end_reviewed: Callable[[TurnState, str], bool]
    # The standing goal's re-entry nudge for a soft end (the reason and the
    # iteration), or None when the run may end; it records the re-entry.
    standing_absorb: Callable[[str, int], str | None]


# An advisor: one heuristic over the turn, the run state and the context,
# answering with what to say, a decision to end the run, or nothing.
Advisor = Callable[["TurnState", "LoopState", TurnContext], Nudge | Stop | None]
# A before-call advisor speaks before the turn exists: its nudge goes to the
# conversation ahead of the provider call.
BeforeCallAdvisor = Callable[["LoopState", TurnContext], Nudge | None]


@dataclass(frozen=True, slots=True)
class Refusal:
    """A finish gate's answer: the finish call is revoked and the model gets
    `text` (or nothing, when the findings reach it another way), with the
    event and the log line recorded."""

    text: str = ""
    event: str = ""
    fields: Mapping[str, object] = field(default_factory=dict)
    log: str = ""


# A finish gate: one rule a finish_session must satisfy, judged over a turn
# that called it; a Refusal hands the finish back.
Gate = Callable[["TurnState", "LoopState", TurnContext], Refusal | None]


def open_subtasks(nodes: Mapping[str, TaskNode]) -> list[tuple[str, str]]:
    """The worker's own subtasks still open: `(id, title)` pairs. Only
    SUBTASKS (parent_id is not None) count: the auto-root is pending until
    the run ends, so counting it would deadlock every gate. A standing task
    is not unfinished work: it gates the finish via its own re-entry, never
    via the capped nudge.

    A task the operator queued carries that in its title: both consumers are
    prose the operator or the model reads (the end receipt, the finish
    deferral), and an end over one of those is worth naming as theirs."""
    return [
        (nid, node.title[:120] + (" (queued by you)" if node.created_by == "user" else ""))
        for nid, node in nodes.items()
        if node.parent_id is not None and node.status in OPEN_STATUSES and not node.standing
    ]


def with_open_tasks(summary: str, open_tasks: Sequence[tuple[str, str]]) -> str:
    """*summary* with the open subtasks named, when an end went through over
    them (the gate's cap): the receipt says what was left."""
    if not open_tasks:
        return summary
    titles = ", ".join(title for _tid, title in open_tasks)
    return f"{summary} ({len(open_tasks)} open task(s): {titles})"
