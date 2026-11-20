"""Queue and complete explicit merges of a parent's stopped children."""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

from agent6 import child_records
from agent6.harness import parallel_merge
from agent6.sessions import child_dispatch, journal
from agent6.tools import tool_errors
from agent6.vcs.git import read as git_read

if TYPE_CHECKING:
    from agent6.harness import loop_conversation, parallel_dispatch


def pending(child: child_dispatch.ChildRecord) -> bool:
    """Return whether a requested merge still needs its result delivered."""
    return child.merge_request is not None and not child.merge_request.delivered


def request(dispatcher: parallel_dispatch.ParallelDispatcher, child_id: str) -> str:
    """Record a stopped child's requested merge before answering its parent.

    Args:
        dispatcher: The parent that owns the stopped child.
        child_id: The child's session id, never a model-supplied git ref.

    Returns:
        The notice saying when the merge runs.

    Raises:
        ToolError: The child is unavailable, another attempt merged, or git is model-owned.
    """
    if not dispatcher.chain.ref or dispatcher.session_dir is None:
        raise tool_errors.ToolError("merge_child requires a parent whose git.control is agent6")
    for record in child_dispatch.read_records(dispatcher.session_dir):
        for child in record.children if record.source != "panel" else ():
            if child.session_id != child_id:
                continue
            if record.state != "complete" or child.join is None or child.join.status != "stopped":
                raise tool_errors.ToolError(
                    f"child {child_id} is not stopped with a completed task and unmerged work"
                )
            if any(c.join and c.join.status in ("joined", "no_changes") for c in record.children):
                raise tool_errors.ToolError(
                    "another attempt already completed this task; nothing merged"
                )
            if any(c.session_id != child_id and pending(c) for c in record.children):
                raise tool_errors.ToolError("another attempt already has a merge requested")
            if dispatcher.chain.dirty() and not dispatcher.chain.per_turn:
                raise tool_errors.ToolError("commit the working tree before requesting a merge")
            if not pending(child):
                dispatcher.write(
                    record.with_child(
                        child.model_copy(update={"merge_request": child_dispatch.MergeRequest()})
                    )
                )
            return f"[parallel] child {child_id} will be merged at the next turn boundary."
    raise tool_errors.ToolError(f"no child {child_id} in this run")


def settle(
    dispatcher: parallel_dispatch.ParallelDispatcher,
    conversation: loop_conversation.Conversation,
    turn: int,
) -> bool:
    """Complete pending merges and deliver their outcomes at a turn boundary.

    Args:
        dispatcher: The parent's dispatcher.
        conversation: Where the merge notices are delivered.
        turn: The completed turn, used for a checkpoint.

    Returns:
        Whether the conversation gained a notice.
    """
    if dispatcher.session_dir is None:
        return False
    told = False
    for found in child_dispatch.read_records(dispatcher.session_dir):
        record = found
        for entry in record.children:
            child = entry
            if not pending(child):
                continue
            asked = child.merge_request
            assert asked is not None
            outcome = asked.result
            if outcome is None:
                before = dispatcher.chain.tip() or ""
                join = merge(dispatcher, record, child, turn)
                outcome = child_dispatch.JoinRecord(
                    branch=join.branch,
                    status=join.status,
                    sha=join.sha,
                    detail=join.detail,
                    commits=join.commits,
                    target_before=before,
                    target_after=dispatcher.chain.tip() or "",
                )
                asked = asked.model_copy(update={"result": outcome})
                child = child.model_copy(update={"merge_request": asked})
                if join.status in ("joined", "no_changes"):
                    child = child.model_copy(update={"join": outcome})
                record = record.with_child(child)
                dispatcher.write(record)
            parallel_merge.stamp_segment_node(
                dispatcher.curator,
                record.node_id,
                [parallel_merge.join_of(c) for c in record.children],
                log=dispatcher.log,
            )
            dispatcher.emit_graph_snapshot()
            dispatcher.emit(
                journal.ParallelChildMerged(
                    group=record.identity,
                    child=child.session_id,
                    status=outcome.status,
                    ref=outcome.branch,
                    sha=outcome.sha,
                    detail=outcome.detail,
                )
            )
            result = parallel_merge.join_of(child.model_copy(update={"join": outcome}))
            conversation.notice(parallel_merge.summary_text(record.identity, [result]))
            record = record.with_child(
                child.model_copy(
                    update={"merge_request": asked.model_copy(update={"delivered": True})}
                )
            )
            dispatcher.write(record)
            told = True
    return told


def merge(
    dispatcher: parallel_dispatch.ParallelDispatcher,
    record: child_dispatch.DispatchRecord,
    child: child_dispatch.ChildRecord,
    turn: int,
) -> child_records.ChildJoin:
    """Merge the stopped ref through the same chain operation as a finished child.

    Args:
        dispatcher: The parent whose chain receives the work.
        record: The child's task record.
        child: The stopped child with a pending request.
        turn: The turn the merge follows.

    Returns:
        The merge result; a refusal changes neither the chain nor the worktree.
    """
    assert child.join is not None
    ref = child.join.branch
    refusal = ""
    if not dispatcher.chain.ref:
        refusal = "git.control is model; merge the ref with git"
    elif not dispatcher.ensure_clean(turn, "merging a stopped child"):
        refusal = "the working tree holds uncommitted work; commit it before retrying"
    if refusal:
        return child_records.ChildJoin(child.session_id, ref, "failed", "", refusal)
    tip = git_read.chain_tip(dispatcher.chain.root, ref) if child.join.commits else ""
    if child.join.commits and not tip:
        return child_records.ChildJoin(
            child.session_id, ref, "failed", "", "the work ref is missing"
        )
    res = child_records.ChildResult(
        spec=child_records.ChildSpec(child.number, child.session_id, pathlib.Path(), None),
        session_dir=pathlib.Path(),
        branch=ref,
        ok=True,
        error="",
        imported_ref=ref,
        imported_tip=tip or "",
        has_commits=bool(tip and tip != record.base),
        commits=child.join.commits,
    )
    return dispatcher.join_child(child, res)
