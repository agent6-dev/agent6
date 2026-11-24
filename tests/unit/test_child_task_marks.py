"""A child's work can return without meeting its task's acceptance."""

from __future__ import annotations

import pathlib

import pytest

from agent6.graph import models, order
from agent6.harness import loop_snapshot
from agent6.harness.parallel import work as child_work
from agent6.tools import tool_errors
from tests.unit import test_children_background as background


def test_a_failure_report_without_changes_leaves_the_task_unfinished(
    tmp_path: pathlib.Path,
) -> None:
    wf, children, graph = background._listed_task(tmp_path)
    dependent = graph.add_subtask(
        models.AddSubtaskIntent(
            parent_id="0001",
            draft=models.TaskNodeDraft(
                title="Use the code", created_by="worker", depends_on=("0002",)
            ),
        )
    )
    assert wf.tool_turn
    conversation, _, turn = wf.tool_turn
    (sid,) = child_work.ChildWork(wf).start("", "", task_id="0002").children
    children.end(sid, report="Could not read the missing file.")
    wf.parallel.settle(conversation, turn)
    node = graph.nodes()["0002"]
    assert node.status == "pending" and not node.held_by
    assert not order.dependencies_done(graph.nodes(), dependent)
    assert "no changes" in node.notes and sid in node.notes
    assert "Task 0002 awaits your mark" in str(conversation.to_wire())
    assert "Could not read the missing file" in str(conversation.to_wire())
    listed = wf.dispatcher.dispatch("list_tasks", {}).to_wire()["tasks"]
    assert listed[1]["awaiting_mark"] and listed[1]["notes"] == node.notes


@pytest.mark.parametrize("status", ["failed", "passed"])
def test_parent_marks_the_returned_task(tmp_path: pathlib.Path, status: str) -> None:
    wf, children, graph = background._listed_task(tmp_path)
    assert wf.tool_turn
    (sid,) = child_work.ChildWork(wf).start("", "", task_id="0002").children
    children.end(sid, path="code" if status == "passed" else "", report=status)
    wf.parallel.settle(wf.tool_turn[0], 2)
    wf.dispatcher.dispatch("update_task", {"id": "0002", "status": status})
    node = graph.nodes()["0002"]
    assert node.status == status and not node.awaiting_mark
    if status == "failed":
        work = child_work.ChildWork(wf)
        with pytest.raises(tool_errors.ToolError, match="not an open subtask"):
            work.steer(sid, "Try reading again")
        wf.dispatcher.dispatch("update_task", {"id": "0002", "status": "pending"})
        work.steer(sid, "Try reading again")
        assert graph.nodes()["0002"].held_by == (sid,)
    else:
        assert not any(nid == "0002" for nid, _ in wf.open_subtasks())
        assert node.commit_sha == wf.parallel.chain.tip() and node.commit_sha[:12] in node.notes


def test_receipt_names_the_unmarked_task_at_a_hard_stop(tmp_path: pathlib.Path) -> None:
    wf, children, _ = background._listed_task(tmp_path)
    assert wf.tool_turn
    conversation, state, turn = wf.tool_turn
    (sid,) = child_work.ChildWork(wf).start("", "", task_id="0002").children
    children.end(sid, report="No such file.")
    wf.parallel.settle(conversation, turn)
    end = wf.finish(state, loop_snapshot.End("max_turns", "turn limit"), turn=turn)
    assert end.summary == "turn limit (1 open task(s): Listed task)"
