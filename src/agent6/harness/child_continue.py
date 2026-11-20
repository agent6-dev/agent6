"""Parent-owned continuations of stopped children, recorded before they start."""

from __future__ import annotations

from agent6 import budget, child_records
from agent6.harness import child_merge, parallel_dispatch
from agent6.sessions import child_dispatch, journal
from agent6.tools import tool_errors


def start(  # noqa: PLR0912  # each refusal precedes the durable state change
    dispatcher: parallel_dispatch.ParallelDispatcher, session_id: str, instruction: str
) -> str:
    """Continue a stopped child in its own clone under this run's watch.

    Args:
        dispatcher: The parent, whose dispatch record owns the child.
        session_id: The child's id.
        instruction: Its next instruction, marked as from its parent.

    Returns:
        The notice that the continuation started.

    Raises:
        ToolError: The child cannot continue; no execution was started.
    """
    if not dispatcher.authority.work or dispatcher.child_spawner is None:
        raise tool_errors.ToolError("child continuation is unavailable in this execution")
    if dispatcher.session_dir is None:
        raise tool_errors.ToolError("child continuation needs the parent's session record")
    found = next(
        (
            (record, child)
            for record in child_dispatch.read_records(dispatcher.session_dir)
            for child in record.children
            if child.session_id == session_id
        ),
        None,
    )
    if found is None:
        raise tool_errors.ToolError(f"no child {session_id} in this run")
    record, child = found
    if child.join is None or child.join.status != "stopped":
        raise tool_errors.ToolError(f"child {session_id} is not stopped and unmerged")
    if record.state != "complete":
        raise tool_errors.ToolError("the task's other attempts have not ended; nothing continued")
    if any(
        c.join and c.join.status in ("joined", "no_changes", "retained") for c in record.children
    ):
        raise tool_errors.ToolError("an attempt already completed this task; nothing continued")
    if any(child_merge.pending(c) for c in record.children):
        raise tool_errors.ToolError("the task has a pending merge; nothing continued")
    if dispatcher.running() >= dispatcher.max_children:
        raise tool_errors.ToolError("the run is at parallel.max_children; nothing continued")
    if not child.settled or child.usage is None:
        raise tool_errors.ToolError("the stopped child's usage is unknown; nothing continued")
    if dispatcher.budget is not None and dispatcher.budget.is_exhausted():
        raise tool_errors.ToolError("the run's budget is exhausted; nothing continued")
    continuation = child_dispatch.ContinuationRecord(
        sequence=child.continuation.sequence + 1 if child.continuation is not None else 1,
        instruction=instruction,
        usage_before=child.usage,
    )
    resumed = child.model_copy(
        update={
            "continuation": continuation,
            "join": None,
            "merge_request": None,
            "reported": False,
            "report": "",
            "settled": False,
        }
    )
    running = record.with_child(resumed).model_copy(
        update={"state": "running", "ranking": None, "message": "", "delivered": False}
    )
    dispatcher.write(running)
    if dispatcher.budget is not None:
        dispatcher.budget.child_execution(
            session_id,
            resumed.account_id,
            budget.UsageSettlement(
                usd=child.usage.usd,
                usd_partial=child.usage.usd_partial,
                tokens=child.usage.tokens,
                model_tokens=child.usage.model_tokens,
            ),
        )
    task = child_records.ChildTask(
        task=child.task,
        model=child.model,
        limits=child_records.ChildLimits(),
        allowance=dispatcher.budget.available() if dispatcher.budget is not None else None,
        continuation=child_records.ChildContinuation(child.number, instruction),
    )
    started: list[child_records.StartedChild] = []
    try:
        started = dispatcher.child_spawner(
            [task], record.identity.removeprefix(f"{dispatcher.session_id}-")
        )
        if [c.session_id for c in started] != [session_id]:
            raise tool_errors.ToolError("the continuation did not start the requested child")
    except Exception as exc:
        # The injected spawner's failures leave the stopped record available for retry.
        for child in started:
            child.stop()
        dispatcher.write(record)
        raise tool_errors.ToolError(f"child {session_id} could not continue: {exc}") from exc
    dispatcher.open[record.identity] = parallel_dispatch.OpenDispatch(running, started)
    dispatcher.stopping.discard(session_id)
    dispatcher.emit(
        journal.ParallelChildContinued(
            group=record.identity, child=session_id, sequence=continuation.sequence
        )
    )
    return f"[parallel] child {session_id} continues in its own clone; read_background follows it."
