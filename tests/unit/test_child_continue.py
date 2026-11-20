from __future__ import annotations

import concurrent.futures
import dataclasses
import pathlib

import pytest

from agent6 import child_records, directive
from agent6.harness import child_work, loop_conversation, loop_state
from agent6.sessions import child_dispatch
from agent6.tools import tool_errors
from tests.unit import test_children_background as background


@pytest.mark.parametrize("control", ["agent6", "model"])
def test_a_parent_continues_its_stopped_child(tmp_path: pathlib.Path, control: str) -> None:
    """The same child runs again, holds finish, and settles only its new usage."""
    repo = tmp_path / "repo"
    background._init_repo(repo)
    children = background._Children(repo, "parent")
    harness = background._harness(repo, tmp_path / "parent", None, children, max_turns=10)
    dispatcher = harness.parallel
    if control == "model":
        dispatcher.chain = dataclasses.replace(dispatcher.chain, ref=None)
    conversation = loop_conversation.Conversation()
    state = loop_state.LoopState(original_task="work", tool_calls=0)
    (sid,) = dispatcher.dispatch(
        conversation, 1, state, [directive.Segment(task="work", spec="")]
    ).children
    children.stop(sid, path="child.txt")
    dispatcher.settle(conversation, 2)
    previous = children.results[sid].result()
    continued: list[child_records.ChildTask] = []
    future: concurrent.futures.Future[child_records.ChildResult] = concurrent.futures.Future()

    def start(
        children: list[child_records.ChildTask], group: str, *, at: str | None = None
    ) -> list[child_records.StartedChild]:
        continued.extend(children)
        assert group == "p1"
        assert at is None
        return [child_records.StartedChild(sid, future)]

    dispatcher.child_spawner = start
    answer = child_work.steer(dispatcher, sid, "finish the test")
    assert "continues" in answer
    assert continued[0].task == "work"
    assert child_work.unmerged(dispatcher) == ((sid, "starting"),)
    record = child_dispatch.read_records(tmp_path / "parent")[0]
    continuation = record.child(sid).continuation
    assert continuation is not None
    assert continuation.instruction == "finish the test"
    assert continuation.sequence == 1
    child_task = continued[0].continuation
    assert child_task is not None
    assert child_task.number == 1
    assert child_task.instruction == "finish the test"
    future.set_result(dataclasses.replace(previous, stopped_by="", report="test complete"))
    dispatcher.settle(conversation, 3)
    assert not child_work.unmerged(dispatcher)
    assert (repo / "child.txt").exists() is (control == "agent6")
    assert "test complete" in str(conversation.to_wire())
    assert not dispatcher.settle(conversation, 4)


@pytest.mark.parametrize("control", ["agent6", "model"])
@pytest.mark.parametrize("blocked", ["pending merge", "winner", "cap", "usage"])
def test_continuation_refuses_before_start(
    tmp_path: pathlib.Path, control: str, blocked: str
) -> None:
    """A continuation cannot race a merge, replace a winner or bypass resource limits."""
    repo = tmp_path / "repo"
    background._init_repo(repo)
    children = background._Children(repo, "parent")
    harness = background._harness(repo, tmp_path / "parent", None, children, max_turns=10)
    dispatcher = harness.parallel
    if control == "model":
        dispatcher.chain = dataclasses.replace(dispatcher.chain, ref=None)
    conversation = loop_conversation.Conversation()
    state = loop_state.LoopState(original_task="work", tool_calls=0)
    (sid,) = dispatcher.dispatch(
        conversation, 1, state, [directive.Segment(task="work", spec="")]
    ).children
    children.stop(sid)
    dispatcher.settle(conversation, 2)
    record = child_dispatch.read_records(tmp_path / "parent")[0]
    child = record.child(sid)
    match = ""
    if blocked == "pending merge":
        child = child.model_copy(update={"merge_request": child_dispatch.MergeRequest()})
        match = "pending merge"
    elif blocked == "winner":
        winner = child.model_copy(
            update={
                "session_id": "parent-p1-l2",
                "merge": child_dispatch.MergeRecord(status="merged"),
            }
        )
        record = record.model_copy(update={"children": (*record.children, winner)})
        match = "already completed"
    elif blocked == "cap":
        dispatcher.max_children = 0
        match = "max_children"
    elif blocked == "usage":
        child = child.model_copy(update={"usage": None})
        match = "usage is unknown"
    dispatcher.write(record.with_child(child))
    with pytest.raises(tool_errors.ToolError, match=match):
        child_work.steer(dispatcher, sid, "continue")
    assert children.results[sid].done()
    assert not child_work.unmerged(dispatcher) or blocked == "pending merge"


@pytest.mark.parametrize("control", ["agent6", "model"])
def test_a_second_attempt_keeps_its_number_on_continuation(
    tmp_path: pathlib.Path, control: str
) -> None:
    """Continuing attempt two does not start attempt one or forget its stopped sibling."""
    repo = tmp_path / "repo"
    background._init_repo(repo)
    children = background._Children(repo, "parent")
    harness = background._harness(repo, tmp_path / "parent", None, children, max_turns=10)
    dispatcher = harness.parallel
    if control == "model":
        dispatcher.chain = dataclasses.replace(dispatcher.chain, ref=None)
    conversation = loop_conversation.Conversation()
    state = loop_state.LoopState(original_task="work", tool_calls=0)
    first, second = dispatcher.dispatch(
        conversation, 1, state, [directive.Segment(task="work", spec="2")]
    ).children
    children.stop(first)
    children.stop(second)
    dispatcher.settle(conversation, 2)
    before = child_dispatch.read_records(tmp_path / "parent")[0].child(first)
    future: concurrent.futures.Future[child_records.ChildResult] = concurrent.futures.Future()

    def start(
        children: list[child_records.ChildTask], group: str, *, at: str | None = None
    ) -> list[child_records.StartedChild]:
        assert children[0].continuation == child_records.ChildContinuation(2, "continue")
        return [child_records.StartedChild(second, future)]

    dispatcher.child_spawner = start
    child_work.steer(dispatcher, second, "continue")
    assert child_work.unmerged(dispatcher) == ((second, "starting"),)
    record = child_dispatch.read_records(tmp_path / "parent")[0]
    assert record.child(first) == before
    future.set_result(dataclasses.replace(children.results[second].result(), stopped_by=""))
    dispatcher.settle(conversation, 3)
    assert not child_work.unmerged(dispatcher)
    assert (repo / f"{second}.txt").exists() is (control == "agent6")
    assert not (repo / f"{first}.txt").exists()


def test_a_refused_restart_preserves_the_stopped_record(tmp_path: pathlib.Path) -> None:
    """A missing clone or other synchronous launch refusal keeps the stopped work available."""
    repo = tmp_path / "repo"
    background._init_repo(repo)
    children = background._Children(repo, "parent")
    harness = background._harness(repo, tmp_path / "parent", None, children, max_turns=10)
    dispatcher = harness.parallel
    conversation = loop_conversation.Conversation()
    state = loop_state.LoopState(original_task="work", tool_calls=0)
    (sid,) = dispatcher.dispatch(
        conversation, 1, state, [directive.Segment(task="work", spec="")]
    ).children
    children.stop(sid)
    dispatcher.settle(conversation, 2)
    before = child_dispatch.read_records(tmp_path / "parent")[0]

    def start(
        children: list[child_records.ChildTask], group: str, *, at: str | None = None
    ) -> list[child_records.StartedChild]:
        raise RuntimeError("child clone is missing")

    dispatcher.child_spawner = start
    with pytest.raises(tool_errors.ToolError, match="clone is missing"):
        child_work.steer(dispatcher, sid, "continue")
    after = child_dispatch.read_records(tmp_path / "parent")[0]
    assert after.children == before.children
    assert after.state == "complete"
    assert child_work.roster(dispatcher)[0].mergeable
