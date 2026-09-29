# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""What an advisor or a finish gate answers with, and the turn facts it reads.

A `Nudge` is a notice for the model, a `Stop` an end of the run, a `Refusal` a
finish handed back; `TurnContext` is the frozen set of run facts and guard
knobs. The advisors live in `_guards` and `_metric`, the gates in
`_finish_gates`; the loop applies their answers.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

from agent6.graph import models, order
from agent6.harness import _snapshot

if TYPE_CHECKING:
    from agent6.harness import _loop_state


@dataclasses.dataclass(frozen=True, slots=True)
class Nudge:
    """What an advisor says to the model this turn, and how the harness records it.

    Attributes:
        text: The notice the model reads.
        event: The event recorded with it; "" records none.
        fields: The event's fields.
        log: The log line; "" writes none.
    """

    text: str
    event: str = ""
    fields: Mapping[str, object] = dataclasses.field(default_factory=dict)
    log: str = ""


@dataclasses.dataclass(frozen=True, slots=True)
class Stop:
    """An advisor's decision to end the run.

    The loop honours it once the turn's results and snapshot are on disk
    (`Harness._turn_stop_checks`); the event is emitted when the stop is decided.

    Attributes:
        end: Composes what `_finish` records, called at the stop so it reads the run
            as the end gates left it.
        soft: The reason a standing task's re-entry nudge carries when it absorbs the
            stop; "" is a hard stop nothing converts.
        declared: The ending the end gates judge as soon as the stop is decided; "" is
            a fault no gate judges.
        event: The event recorded when the stop is decided; "" records none.
        fields: The event's fields.
        log: The log line written at the stop; "" writes none.
    """

    end: Callable[[], _snapshot.End]
    soft: str = ""
    declared: str = ""
    event: str = ""
    fields: Mapping[str, object] = dataclasses.field(default_factory=dict)
    log: str = ""


@dataclasses.dataclass(frozen=True, slots=True)
class TurnContext:
    """The run facts an advisor reads, built once per turn.

    A fact that can move within the turn (a harness verify can deny or un-adopt
    the gate, an edit moves the tree) is a zero-argument callable read where it
    is needed.
    """

    mode: Literal["run", "plan", "ask", "agent"]
    iteration: int
    # The execution's first iteration: a turn allowance counts from it.
    execution_start: int
    # `[harness]`'s guard knobs: the empty-turn nudge cap, the kill threshold, the notice delay.
    went_quiet_max_nudges: int
    loop_guard_kill_threshold: int
    stagnation_notice_after_s: float
    # `[harness].verify_when` and `verify_retries`, read by the red-gate return rule.
    verify_when: Literal["finish", "step", "never"]
    verify_retries: int
    # A machine agent state's finish contract: the problems with a payload; None gates nothing.
    finish_validator: Callable[[dict[str, Any] | None], list[str]] | None
    # A metric goal is configured: the plateau and ceiling rules own the run's end.
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
    # The before-finish panel over the turn's declared end: True when it rejected the end.
    end_rejected: Callable[[_loop_state.TurnState, str], bool]
    # The standing goal's re-entry nudge for a soft end, or None when the run may end.
    standing_absorb: Callable[[str, int], str | None]


# An advisor: one heuristic over the turn, answering with a nudge, a stop or nothing.
Advisor = Callable[
    ["_loop_state.TurnState", "_loop_state.LoopState", TurnContext], Nudge | Stop | None
]
# A before-call advisor's nudge goes to the conversation ahead of the provider call.
BeforeCallAdvisor = Callable[["_loop_state.LoopState", TurnContext], Nudge | None]


@dataclasses.dataclass(frozen=True, slots=True)
class Refusal(Nudge):
    """A finish gate's answer: the loop revokes the end and the model gets the text.

    An empty `text` means the findings reach the model another way.
    """


# A finish gate: one rule a finish must satisfy; a Refusal hands the finish back.
Gate = Callable[["_loop_state.TurnState", "_loop_state.LoopState", TurnContext], Refusal | None]


def open_subtasks(nodes: Mapping[str, models.TaskNode]) -> list[tuple[str, str]]:
    """Return the worker's own subtasks still open, as (id, title) pairs.

    Only subtasks count: the root is pending until the run ends, so counting it
    would deadlock every gate. A standing task gates the finish through its own
    re-entry, never through the capped nudge. A task the operator queued says so
    in its title, since the receipt and the deferral are read as prose.

    Args:
        nodes: The task graph's nodes by id.

    Returns:
        The open subtasks in graph order, titles clipped to 120 characters.
    """
    out: list[tuple[str, str]] = []
    for nid, node in nodes.items():
        if node.parent_id is None or node.status not in order.OPEN_STATUSES or node.standing:
            continue
        note = models.owner_note(
            created_by=node.created_by, parent_id=node.parent_id, standing=False
        )
        out.append((nid, node.title[:120] + (f" ({note})" if note else "")))
    return out


def with_open_tasks(summary: str, open_tasks: Sequence[tuple[str, str]]) -> str:
    """Return the summary with the open subtasks named, so the receipt says what was left.

    Args:
        summary: The end's summary.
        open_tasks: The (id, title) pairs `open_subtasks` returned.

    Returns:
        The summary, with a count and the titles appended when any task is open.
    """
    if not open_tasks:
        return summary
    titles = ", ".join(title for _tid, title in open_tasks)
    return f"{summary} ({len(open_tasks)} open task(s): {titles})"
