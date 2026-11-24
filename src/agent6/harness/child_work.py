# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A run's children as its tools and its finish gate reach them.

`ChildWork` is the loop's side of `tools.child_tools.Children`; the functions read the
dispatcher's open dispatches: where each child stands, a wait for one to end, a steer, a stop.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from agent6 import child_records, directive, kinds
from agent6.harness import child_continue, child_merge, operator_bridge, parallel_merge
from agent6.sessions import child_dispatch, ipc
from agent6.sessions import layout as sessions_layout
from agent6.tools import child_tools, results, tool_errors
from agent6.viewmodel import children as vm_children
from agent6.viewmodel.session import log as session_log

if TYPE_CHECKING:
    from agent6.harness import loop, parallel_dispatch


def standing(
    dispatcher: parallel_dispatch.ParallelDispatcher,
    task: parallel_dispatch.OpenDispatch,
    started: child_records.StartedChild,
) -> str:
    """Return where a child of an open dispatch stands.

    Args:
        dispatcher: The run's dispatcher.
        task: The child's dispatch.
        started: The child.

    Returns:
        Its live state, `ended, task pending`, `being ranked`, or a stopped child's words.
    """
    if not started.result.done():
        if started.session_id in dispatcher.stopping:
            return "stopping"
        if dispatcher.session_dir is not None and sessions_layout.is_safe_session_id(
            started.session_id
        ):
            directory = dispatcher.session_dir.with_name(started.session_id)
            scan = session_log.scan_session_log(directory / sessions_layout.LOGS_NAME)
            if not vm_children.reported_start(task.record.child(started.session_id), scan):
                return "starting"
            if not scan.finished and scan.review.in_review(scan.execution_id):
                return "in review"
        return "running"
    res = dispatcher.result_of(task.record, started)
    if res.ok and res.stopped_by:
        return parallel_merge.merge_words(parallel_merge.stopped_merge(res))
    return "being ranked" if task.judging is not None and res.ok else "ended, task pending"


def unmerged(dispatcher: parallel_dispatch.ParallelDispatcher) -> tuple[tuple[str, str], ...]:
    """Return each child whose work may still merge, with where it stands, oldest first.

    A stopped child is not among them: it never merges, so it holds no end.

    Args:
        dispatcher: The run's dispatcher.

    Returns:
        Each child's session id and the state holding the parent's finish.
    """
    requested = tuple(
        (child.session_id, "merge requested")
        for record in (
            child_dispatch.read_records(dispatcher.session_dir) if dispatcher.session_dir else ()
        )
        for child in record.children
        if child_merge.pending(child)
    )
    return requested + tuple(
        (started.session_id, state)
        for task in dispatcher.open.values()
        for started in task.children
        if (state := standing(dispatcher, task, started))
        in ("starting", "running", "in review", "ended, task pending", "being ranked")
    )


def roster(dispatcher: parallel_dispatch.ParallelDispatcher) -> tuple[child_tools.ChildLine, ...]:
    """Return every child the run's tasks started and where it stands, oldest first.

    Args:
        dispatcher: The run's dispatcher.

    Returns:
        One line per child of a work dispatch: the open ones as `standing` reads them, the
        merged ones with how their work merged.
    """
    if dispatcher.session_dir is None:
        return ()
    try:
        records = child_dispatch.read_records(dispatcher.session_dir)
    except child_dispatch.DispatchRecordError:
        return ()
    open_now = {
        started.session_id: standing(dispatcher, task, started)
        for task in dispatcher.open.values()
        for started in task.children
    }
    lines: list[child_tools.ChildLine] = []
    for record in records:
        for child in record.children if record.source != "panel" else ():
            sid = child.session_id
            state = open_now.get(sid, "ended, task pending")
            if child.merge is not None:
                state = parallel_merge.merge_words(parallel_merge.merge_of(child))
            title = next((ln.strip() for ln in child.task.splitlines() if ln.strip()), "")
            # A wait has something to wait for: the child itself, or the judges ranking it.
            running = state in ("starting", "running", "in review", "stopping", "being ranked")
            mergeable = child_merge.available(child, model_git=not dispatcher.chain.ref)
            if child_merge.pending(child):
                state = "merge requested"
            commits = (
                child_dispatch.read_commit_count(
                    dispatcher.session_dir.with_name(sid), child.merge, version=record.version
                )
                if sessions_layout.is_safe_session_id(sid)
                else None
            )
            lines.append(
                child_tools.ChildLine(
                    sid, child.number, state, title[:80], running, mergeable, commits
                )
            )
    return tuple(lines)


def wait_for(
    dispatcher: parallel_dispatch.ParallelDispatcher,
    child: str,
    wait_s: float,
    interrupted: Callable[[], bool],
) -> None:
    """Wait for one running child to end, or with "" for the next child or judge of them all.

    Args:
        dispatcher: The run's dispatcher.
        child: The child's session id; "" for whichever ends next, a task's judge included.
        wait_s: The most seconds to wait.
        interrupted: Cuts the wait short: the operator's stop reaches no turn boundary
            while a tool call waits.
    """
    pending = [
        started.result
        for task in dispatcher.open.values()
        for started in (*task.children, *(task.judging.children if task.judging else ()))
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
    """Ask a running child to stop now; its end is told and its work never merges by itself.

    Args:
        dispatcher: The run's dispatcher.
        child: The child's session id; one that has ended, or is no child, is left alone.
    """
    for task in dispatcher.open.values():
        for started in task.children:
            if started.session_id == child and not started.result.done():
                started.stop()
                dispatcher.stopping.add(child)


def steer(dispatcher: parallel_dispatch.ParallelDispatcher, child: str, text: str) -> str:
    """Steer a running child or continue an unmerged child under its parent.

    A child taking another turn reads it under `operator_bridge.PARENT_STEERING`:
    text for its model, never a steer verb or a directive.

    Args:
        dispatcher: The run's dispatcher.
        child: The child's session id.
        text: The instruction.

    Returns:
        The confirmation of submission, not delivery.

    Raises:
        ToolError: No child in this run can take that instruction.
    """
    state = next(
        (
            standing(dispatcher, task, started)
            for task in dispatcher.open.values()
            for started in task.children
            if started.session_id == child and not started.result.done()
        ),
        None,
    )
    if state is None:
        ended = next((line for line in roster(dispatcher) if line.id == child), None)
        if ended is None:
            raise tool_errors.ToolError(
                f"no child {child} in this run; read_background with no id lists this run's"
                " children."
            )
        return child_continue.start(dispatcher, child, text)
    if state == "stopping":
        raise tool_errors.ToolError(f"child {child} is stopping; nothing was sent.")
    assert dispatcher.session_dir is not None
    state_dir = sessions_layout.layout_of(dispatcher.session_dir).state_dir
    child_dir = sessions_layout.bucket_dir(state_dir, kinds.session_bucket("run")) / child
    if (
        state == "starting"
        or not child_dir.is_dir()
        or not ipc.submit_steer(child_dir, f"{operator_bridge.PARENT_STEERING}\n{text}")
    ):
        raise tool_errors.ToolError(
            f"child {child} has not started yet; nothing was sent. Send it again once"
            " read_background shows its transcript."
        )
    review = " (in review)" if state == "in review" else ""
    return f"[parallel] child {child}{review}: steer queued for another turn, if any."


@dataclasses.dataclass(frozen=True, slots=True)
class ChildWork:
    """Start, list, await, steer and stop a run's children for its tools.

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

    def steer(self, child: str, text: str) -> str:
        """Steer a running child or continue an unmerged child with the instruction.

        Args:
            child: The child's session id.
            text: The instruction.

        Returns:
            The line that says when the child reads it.

        Raises:
            ToolError: The run started no child, or `steer` refused the id.
        """
        if (dispatcher := self.harness.children) is None:
            raise tool_errors.ToolError(
                f"no child {child} in this run; read_background with no id lists this run's"
                " children."
            )
        return steer(dispatcher, child, text)

    def merge(self, child: str) -> str:
        """Request a child's merge at the next boundary.

        Args:
            child: The child's session id.

        Returns:
            The line saying when the merge runs.

        Raises:
            ToolError: No dispatcher owns the child, or the request is refused.
        """
        if (dispatcher := self.harness.children) is None:
            raise tool_errors.ToolError(f"no child {child} in this run")
        return child_merge.request(dispatcher, child)
