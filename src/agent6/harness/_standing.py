# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The run's standing goal: the never-passing task a run returns to when
its ordinary work runs out. `Standing` answers whether a soft end (a
finish_session, the settled family, a quiet turn) converts into re-entry,
with the nudge that re-enters, and applies that conversion to a turn's
stops. The goal itself is set by `run --standing` or `/standing`
(`OperatorTasks.set_standing`)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from agent6.graph.curator import GraphCurator
from agent6.harness._dag_focus import ready_subtask
from agent6.harness._nudges import standing_fruitless_nudge, standing_resume_nudge

if TYPE_CHECKING:
    from agent6.harness._conversation import Conversation
    from agent6.harness._loop_state import LoopState, TurnState


@dataclass(frozen=True, slots=True)
class Standing:
    """The run's standing goal, read from the graph; `patience` is
    `[harness].standing_patience`."""

    curator: GraphCurator | None
    patience: int
    budget_remaining: Callable[[], float | None]
    log: Callable[[str], None]
    emit: Callable[..., None]

    def task(self) -> tuple[str, str] | None:
        """The ready standing task's (id, title), if this run has one."""
        if self.curator is None:
            return None
        nodes = self.curator.nodes()
        for nid, node in nodes.items():
            if node.standing and ready_subtask(nodes, node):
                return nid, node.title[:120]
        return None

    def absorb(self, state: LoopState, *, reason: str, iteration: int) -> str | None:
        """The standing-goal conversion for a soft end: the nudge text to
        inject when the run should re-enter the standing task instead of
        ending, else None. None when there is no ready standing task, when
        the budget is spent (the hard bounds always win), or once
        `[harness].standing_patience` fruitless re-entries (no executed
        tool call since the last one) are used up. At the default (-1) a
        fruitless round never ends the run by itself: the nudge escalates
        to "dig deeper or try a different approach" instead, and the run
        ends on its budget, iteration cap, or an operator stop."""
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
            nudge = standing_fruitless_nudge(reason, nid, title, state.standing.fruitless)
        else:
            state.standing.fruitless = 0
            nudge = standing_resume_nudge(reason, nid, title)
        state.standing.tools_mark = state.ok_tool_calls
        self.log(f"  standing re-entry ({reason}) -> {nid} at iter {iteration}")
        self.emit("loop.standing.resumed", reason=reason, task_id=nid, iteration=iteration)
        return nudge

    def absorb_soft_stop(
        self, state: LoopState, turn: TurnState, conversation: Conversation
    ) -> None:
        """A standing task converts the soft out-of-work endings into
        re-entry: the pending stop flag is cleared and the standing nudge
        joins the conversation. Faults (tool_error), the loop guard, and
        every hard bound still end the run; the absorb itself refuses on
        spent budget or a spin."""
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
