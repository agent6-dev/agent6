# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Re-enter the run's standing goal when its ordinary work runs out.

A standing goal is the never-passing task set by `run --standing` or `/standing`
(`OperatorTasks.set_standing`). `Standing` decides whether a soft end (a finish_session,
the settled family, a quiet turn) converts into re-entry and applies that to a turn's stops.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import TYPE_CHECKING

from agent6.graph import curator as graph_curator
from agent6.graph import order
from agent6.harness import _nudges

if TYPE_CHECKING:
    from agent6.harness import _conversation, _loop_state


@dataclasses.dataclass(frozen=True, slots=True)
class Standing:
    """Read the run's standing goal from the graph and convert soft ends into re-entry.

    Attributes:
        curator: The task graph, or None when the run has none.
        patience: `[harness].standing_patience`, the fruitless re-entries allowed; -1 is unbounded.
        budget_remaining: The fraction of the budget left, or None without a tracker.
        log: The run's text logger.
        emit: The run's event emitter.
    """

    curator: graph_curator.GraphCurator | None
    patience: int
    budget_remaining: Callable[[], float | None]
    log: Callable[[str], None]
    emit: Callable[..., None]

    def task(self) -> tuple[str, str] | None:
        """Return the ready standing task's (id, title), or None when the run has none."""
        if self.curator is None:
            return None
        nodes = self.curator.nodes()
        for nid, node in nodes.items():
            if node.standing and order.ready_subtask(nodes, node):
                return nid, node.title[:120]
        return None

    def absorb(self, state: _loop_state.LoopState, *, reason: str, iteration: int) -> str | None:
        """Convert a soft end into re-entry of the standing task.

        A re-entry with no executed tool call since the last one is fruitless; past `patience`
        of them the end is honoured. At the default (-1) the nudge escalates instead and the run
        ends on its budget, its iteration cap or an operator stop.

        Args:
            state: The execution's state.
            reason: The soft end being converted.
            iteration: The turn the conversion happens at.

        Returns:
            The nudge that re-enters the task, or None when the run ends: no ready standing
            task, a spent budget, or the patience used up.
        """
        st = self.task()
        if st is None:
            return None
        remaining = self.budget_remaining()
        if remaining is not None and remaining <= 0.0:
            return None
        nid, title = st
        if state.ok_tool_calls == state.standing.tools_mark:
            state.standing.fruitless += 1
            if 0 <= self.patience < state.standing.fruitless:
                self.log(
                    f"  standing: {state.standing.fruitless} fruitless re-entries >"
                    f" standing_patience {self.patience}; honouring {reason}"
                )
                return None
            nudge = _nudges.standing_fruitless_nudge(reason, nid, title, state.standing.fruitless)
        else:
            state.standing.fruitless = 0
            nudge = _nudges.standing_resume_nudge(reason, nid, title)
        state.standing.tools_mark = state.ok_tool_calls
        self.log(f"  standing re-entry ({reason}) -> {nid} at iter {iteration}")
        self.emit("loop.standing.resumed", reason=reason, task_id=nid, iteration=iteration)
        return nudge

    def absorb_soft_stop(
        self,
        state: _loop_state.LoopState,
        turn: _loop_state.TurnState,
        conversation: _conversation.Conversation,
    ) -> None:
        """Clear a turn's soft stop and put the re-entry nudge in the conversation.

        A hard stop in the same turn (a fault, the loop guard, a bound) ends the run regardless.

        Args:
            state: The execution's state.
            turn: The turn whose stops are judged.
            conversation: The conversation the nudge joins.
        """
        soft = next((stop.soft for stop in turn.stops if stop.soft), None)
        if soft is None or any(not stop.soft for stop in turn.stops):
            return
        nudge = self.absorb(state, reason=soft, iteration=turn.iteration)
        if nudge is None:
            return
        turn.stops = [stop for stop in turn.stops if not stop.soft]
        state.settled.restart()
        state.verify.fail_streak = 0
        state.no_progress.rearm()
        state.metric.rearm()
        conversation.notice(nudge)
