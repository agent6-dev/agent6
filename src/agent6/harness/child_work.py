# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A run's children as its tools and its finish gate reach them.

`ChildWork` is the loop's side of `tools.child_tools.Children`; the functions read the
dispatcher's open dispatches: where each child stands, a wait for one to end, a stop.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from agent6 import directive
from agent6.harness import parallel_join
from agent6.sessions import child_dispatch
from agent6.tools import child_tools, results, tool_errors

if TYPE_CHECKING:
    from agent6.harness import loop, parallel_dispatch


def unlanded(dispatcher: parallel_dispatch.ParallelDispatcher) -> tuple[tuple[str, str], ...]:
    """Return each child whose task has not landed, with where it stands, oldest first.

    Args:
        dispatcher: The run's dispatcher.

    Returns:
        Each child's session id with `running`, `stopping` or `ended, not landed`.
    """
    return tuple(
        (
            started.session_id,
            "ended, not landed"
            if started.result.done()
            else "stopping"
            if started.session_id in dispatcher.stopping
            else "running",
        )
        for task in dispatcher.open.values()
        for started in task.children
    )


def roster(dispatcher: parallel_dispatch.ParallelDispatcher) -> tuple[child_tools.ChildLine, ...]:
    """Return every child the run's tasks started and where it stands, oldest first.

    Args:
        dispatcher: The run's dispatcher.

    Returns:
        One line per child of a work dispatch: the open ones as `unlanded` reads them, the
        landed ones with how their work joined.
    """
    if dispatcher.session_dir is None:
        return ()
    try:
        records = child_dispatch.read_records(dispatcher.session_dir)
    except child_dispatch.DispatchRecordError:
        return ()
    open_now = dict(unlanded(dispatcher))
    lines: list[child_tools.ChildLine] = []
    for record in records:
        for child in record.children if record.source != "panel" else ():
            sid = child.session_id
            state = open_now.get(sid, "ended, not landed")
            if child.join is not None:
                state = parallel_join.join_words(parallel_join.join_of(child))
            title = next((ln.strip() for ln in child.task.splitlines() if ln.strip()), "")
            running = state in ("running", "stopping")
            lines.append(child_tools.ChildLine(sid, state, title[:80], running))
    return tuple(lines)


def wait_for(
    dispatcher: parallel_dispatch.ParallelDispatcher,
    child: str,
    wait_s: float,
    interrupted: Callable[[], bool],
) -> None:
    """Wait for one running child to end, or with "" for the next of them all.

    Args:
        dispatcher: The run's dispatcher.
        child: The child's session id; "" for whichever ends next.
        wait_s: The most seconds to wait.
        interrupted: Cuts the wait short: the operator's stop reaches no turn boundary
            while a tool call waits.
    """
    pending = [
        started.result
        for task in dispatcher.open.values()
        for started in task.children
        if not started.result.done() and child in ("", started.session_id)
    ]
    deadline = time.monotonic() + wait_s
    while pending and not interrupted() and (left := deadline - time.monotonic()) > 0:
        done, _ = concurrent.futures.wait(
            pending, timeout=min(left, 0.5), return_when=concurrent.futures.FIRST_COMPLETED
        )
        if done:
            return


def stop(dispatcher: parallel_dispatch.ParallelDispatcher, child: str) -> None:
    """Ask a running child to stop now; its end is told and its commits land as any child's.

    Args:
        dispatcher: The run's dispatcher.
        child: The child's session id; one that has ended, or is no child, is left alone.
    """
    for task in dispatcher.open.values():
        for started in task.children:
            if started.session_id == child and not started.result.done():
                started.stop()
                dispatcher.stopping.add(child)


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
            conversation, turn, state, [directive.Segment(task=task, spec=spec)], source="model"
        )
        return results.StartedResult(started.children, started.message)

    def roster(self) -> tuple[child_tools.ChildLine, ...]:
        """Return every child the run's tasks started, in the order started."""
        dispatcher = self.harness.children
        return roster(dispatcher) if dispatcher is not None else ()

    def wait(self, child: str, wait_s: float, interrupted: Callable[[], bool]) -> None:
        """Wait up to `wait_s` for the child to end; with "", for the next one to end."""
        if (dispatcher := self.harness.children) is not None:
            wait_for(dispatcher, child, wait_s, interrupted)

    def stop(self, child: str) -> None:
        """Ask a running child to stop now; an ended one is left as it is."""
        if (dispatcher := self.harness.children) is not None:
            stop(dispatcher, child)
