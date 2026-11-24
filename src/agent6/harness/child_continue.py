"""Parent-owned continuations of unmerged children, recorded before they start."""

from __future__ import annotations

from agent6 import budget, child_records
from agent6.harness import child_merge, child_tasks, parallel_dispatch
from agent6.sessions import child_dispatch, journal
from agent6.tools import tool_errors
from agent6.vcs.git import read as git_read


def start(  # noqa: PLR0912  # each refusal precedes the durable state change
    dispatcher: parallel_dispatch.ParallelDispatcher, session_id: str, instruction: str
) -> str:
    """Continue an unmerged child under this run's watch.

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
    if record.state != "complete":
        raise tool_errors.ToolError("the task's other attempts have not ended; nothing continued")
    merge = child.merge
    if merge is None:
        raise tool_errors.ToolError(f"child {session_id} has no recorded result")
    if already_merged(dispatcher, record, child):
        raise tool_errors.ToolError(
            f"child {session_id} already merged; its work is in the parent's tree."
            " parallel starts a new child from there."
        )
    if any(child_merge.pending(c) for c in record.children):
        raise tool_errors.ToolError("the task has a pending merge; nothing continued")
    if dispatcher.running() >= dispatcher.max_children:
        raise tool_errors.ToolError("the run is at parallel.max_children; nothing continued")
    if not child.settled or child.usage is None:
        raise tool_errors.ToolError("the child's usage is unknown; nothing continued")
    if dispatcher.budget is not None and dispatcher.budget.is_exhausted():
        raise tool_errors.ToolError("the run's budget is exhausted; nothing continued")
    continuation = child_dispatch.ContinuationRecord(
        sequence=child.continuation.sequence + 1 if child.continuation is not None else 1,
        owner=dispatcher.execution_id,
        instruction=instruction,
        usage_before=child.usage,
    )
    resumed = child_tasks.followup(dispatcher, record, child).model_copy(
        update={
            "continuation": continuation,
            "merge": None,
            "merge_request": None,
            "reported": False,
            "report": "",
            "settled": False,
        }
    )
    running = record.with_child(resumed).model_copy(
        update={"state": "running", "message": "", "delivered": False}
    )
    dispatcher.write(running)
    if dispatcher.budget is not None:
        dispatcher.budget.child_execution(
            session_id,
            resumed.account_id,
            budget.UsageSettlement(
                usd=child.usage.usd,
                usd_observed_credits=child.usage.usd_observed_credits,
                credit_readings=child.usage.credit_readings,
                usd_partial=child.usage.usd_partial,
                tokens=child.usage.tokens,
                model_tokens=child.usage.model_tokens,
            ),
        )
    task = child_records.ChildTask(
        task=child.task,
        task_source=record.source,
        model=child.model,
        limits=child_records.ChildLimits(),
        allowance=dispatcher.budget.available() if dispatcher.budget is not None else None,
        continuation=child_records.ChildContinuation(
            child.number, instruction, fresh=merge.status != "stopped", commits=merge.commits
        ),
    )
    started: list[child_records.StartedChild] = []
    try:
        started = dispatcher.child_spawner(
            [task], record.identity.removeprefix(f"{dispatcher.session_id}-")
        )
        if [c.session_id for c in started] != [session_id]:
            raise tool_errors.ToolError("the continuation did not start the requested child")
    except Exception as exc:
        # The injected spawner's failures leave the completed record available for retry.
        for child in started:
            child.stop()
        dispatcher.write(record)
        raise tool_errors.ToolError(f"child {session_id} could not continue: {exc}") from exc
    dispatcher.open[record.identity] = parallel_dispatch.OpenDispatch(running, started)
    dispatcher.emit_graph_snapshot()
    dispatcher.stopping.discard(session_id)
    dispatcher.emit(
        journal.ParallelChildContinued(
            group=record.identity, child=session_id, sequence=continuation.sequence
        )
    )
    workspace = started[0].workspace or "its kept clone"
    return f"[parallel] child {session_id} continues in {workspace}; read_background follows it."


def already_merged(
    dispatcher: parallel_dispatch.ParallelDispatcher,
    record: child_dispatch.DispatchRecord,
    child: child_dispatch.ChildRecord,
) -> bool:
    """Return whether agent6 recorded the merge or the parent chain contains the child's work."""
    merge, chain = child.merge, dispatcher.chain
    if merge is None or not chain.ref:
        return False
    if merge.status == "merged":
        return True
    tip = git_read.chain_tip(chain.root, merge.branch) if merge.branch else None
    return bool(tip and tip != record.base and git_read.is_ancestor(chain.root, tip, chain.ref))
