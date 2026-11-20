# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The loop's side of `tools.child_tools.Children`: a run's children as its tools reach them."""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from agent6 import child_records, directive
from agent6.harness import parallel_dispatch
from agent6.tools import child_tools, results, tool_errors

if TYPE_CHECKING:
    from agent6.harness import loop


@dataclasses.dataclass(frozen=True, slots=True)
class ChildWork:
    """Start, list, await and stop a run's children for its tools.

    Attributes:
        harness: The run's loop, whose dispatcher holds the children.
    """

    harness: loop.Harness

    def start(self, task: str, spec: str) -> results.StartedResult:
        """Start one task's children for the model's `parallel` call, through the steer's path.

        The dispatch is recorded with the model as its source, and the call returns at once.

        Args:
            task: The task.
            spec: The child spec: "" for one child, a count or a selector list for attempts.

        Returns:
            The children's session ids and the line that says what started, or the refusal.

        Raises:
            ToolError: No turn is in progress.
        """
        if self.harness.tool_turn is None:
            raise tool_errors.ToolError("not available: no turn in progress")
        conversation, state, turn = self.harness.tool_turn
        started = self.harness.parallel.dispatch(
            conversation,
            turn,
            state,
            [directive.Segment(task=task, spec=spec)],
            source="model",
            limits=child_records.ChildLimits(
                max_turns=parallel_dispatch.MODEL_CHILD_MAX_TURNS,
                deadline_at=time.time() + parallel_dispatch.MODEL_CHILD_DEADLINE_S,
            ),
        )
        return results.StartedResult(started.children, started.message)

    def roster(self) -> tuple[child_tools.ChildLine, ...]:
        """Return every child the run's tasks started, in the order started."""
        dispatcher = self.harness.children
        return dispatcher.roster() if dispatcher is not None else ()

    def wait(self, child: str, wait_s: float, interrupted: Callable[[], bool]) -> None:
        """Wait up to `wait_s` for the child to end; with "", for the next one to end."""
        if (dispatcher := self.harness.children) is not None:
            dispatcher.wait_for(child, wait_s, interrupted)

    def stop(self, child: str) -> None:
        """Ask a running child to stop now; an ended one is left as it is."""
        if (dispatcher := self.harness.children) is not None:
            dispatcher.stop(child)
