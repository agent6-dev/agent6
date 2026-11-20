from __future__ import annotations

import dataclasses
import pathlib

import pytest

from agent6 import directive
from agent6.graph import curator as graph_curator
from agent6.harness import child_work, loop_conversation, loop_state, parallel_recovery, toolset
from agent6.sessions import child_dispatch
from agent6.sessions import layout as sessions_layout
from agent6.tools import tool_errors
from tests.unit import test_children_background as background


def test_a_stopped_child_merges_only_after_the_requested_boundary(tmp_path: pathlib.Path) -> None:
    """A requested merge holds finish until the boundary and is delivered once."""
    repo = tmp_path / "repo"
    background._init_repo(repo)
    children = background._Children(repo, "parent")
    graph = graph_curator.GraphCurator(
        sessions_layout.SessionLayout(state_dir=tmp_path / "state", session_id="parent")
    )
    harness = background._harness(
        repo, tmp_path / "parent", None, children, max_turns=10, curator=graph
    )
    dispatcher = harness.parallel
    conversation = loop_conversation.Conversation()
    state = loop_state.LoopState(original_task="work", tool_calls=0)
    (sid,) = dispatcher.dispatch(
        conversation, 1, state, [directive.Segment(task="work", spec="")]
    ).children
    children.stop(sid, path="child.txt")
    dispatcher.settle(conversation, 2)
    assert {node.status for node in graph.nodes().values()} == {"failed"}
    before = dispatcher.chain.tip()

    def names() -> list[str]:
        return [t.name for t in toolset.tool_definitions(harness.dispatcher)]

    assert "merge_child" in names()
    answer = harness.dispatcher.dispatch("merge_child", {"id": sid})
    assert sid in str(answer.to_wire())
    assert dispatcher.chain.tip() == before
    assert child_work.unmerged(dispatcher) == ((sid, "merge requested"),)
    dispatcher.settle(conversation, 3)
    assert (repo / "child.txt").is_file()
    assert {node.status for node in graph.nodes().values()} == {"passed"}
    assert "merge_child" not in names()
    assert not child_work.unmerged(dispatcher)
    assert not dispatcher.settle(conversation, 4)
    assert len(background._events(tmp_path / "parent", "parallel.child_merged")) == 1


@pytest.mark.parametrize("conflict", [False, True])
def test_a_requested_merge_survives_resume(tmp_path: pathlib.Path, conflict: bool) -> None:
    """Recovery merges a pending request, or reports conflicting files without changing the tree."""
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
    children.end(sid, path="README.md", text="child\n", stopped_by="stopped")
    dispatcher.settle(conversation, 2)
    harness.dispatcher.dispatch("merge_child", {"id": sid})
    if conflict:
        (repo / "README.md").write_text("parent\n")
        dispatcher.chain.commit("parent edit")
    before = dispatcher.chain.tip()
    resumed = background._harness(repo, tmp_path / "parent", None, children, max_turns=10)
    parallel_recovery.reconcile(resumed.parallel, state, conversation, turn=3)
    record = child_dispatch.read_records(tmp_path / "parent")[0]
    merge = record.child(sid).merge
    assert merge is not None
    if conflict:
        assert resumed.parallel.chain.tip() == before
        assert (repo / "README.md").read_text() == "parent\n"
        assert "README.md" in str(conversation.to_wire())
        assert f"agent6/{sid}" in str(conversation.to_wire())
        assert merge.status == "stopped"
    else:
        assert (repo / "README.md").read_text() == "child\n"
        assert merge.status == "merged"


def test_merge_child_is_refused_without_stopped_work(tmp_path: pathlib.Path) -> None:
    """A hidden merge tool cannot merge an arbitrary ref or another run's child."""
    repo = tmp_path / "repo"
    background._init_repo(repo)
    children = background._Children(repo, "parent")
    harness = background._harness(repo, tmp_path / "parent", None, children, max_turns=10)
    assert "merge_child" not in [t.name for t in toolset.tool_definitions(harness.dispatcher)]
    with pytest.raises(tool_errors.ToolError):
        harness.dispatcher.dispatch("merge_child", {"id": "other-p1-l1"})


@pytest.mark.parametrize("control", ["agent6", "model"])
def test_merge_child_respects_git_ownership(tmp_path: pathlib.Path, control: str) -> None:
    """Model-owned git never exposes or accepts merge_child, even with stopped work."""
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
    cfg = harness.dispatcher.config.model_copy(
        update={"git": harness.dispatcher.config.git.model_copy(update={"control": control})}
    )
    harness.dispatcher._config = cfg
    if control == "model":
        dispatcher.chain = dataclasses.replace(dispatcher.chain, ref=None)
        assert "merge_child" not in [t.name for t in toolset.tool_definitions(harness.dispatcher)]
        with pytest.raises(tool_errors.ToolError, match=r"git\.control"):
            harness.dispatcher.dispatch("merge_child", {"id": sid})
    else:
        assert "merge_child" in [t.name for t in toolset.tool_definitions(harness.dispatcher)]


@pytest.mark.parametrize("reason", ["running", "winner", "dirty", "pending"])
def test_an_explicit_merge_preserves_one_merge_per_task(
    tmp_path: pathlib.Path, reason: str
) -> None:
    """An explicit merge cannot race another attempt or overwrite uncommitted work."""
    repo = tmp_path / "repo"
    background._init_repo(repo)
    children = background._Children(repo, "parent")
    harness = background._harness(repo, tmp_path / "parent", None, children, max_turns=10)
    dispatcher = harness.parallel
    conversation = loop_conversation.Conversation()
    state = loop_state.LoopState(original_task="work", tool_calls=0)
    one, two = dispatcher.dispatch(
        conversation, 1, state, [directive.Segment(task="work", spec="2")]
    ).children
    children.stop(one)
    if reason == "winner":
        children.end(two, path="winner.txt")
    elif reason != "running":
        children.stop(two)
    dispatcher.settle(conversation, 2)
    if reason == "dirty":
        dispatcher.chain = dataclasses.replace(dispatcher.chain, per_turn=False)
        (repo / "README.md").write_text("uncommitted\n")
    elif reason == "pending":
        harness.dispatcher.dispatch("merge_child", {"id": two})
    with pytest.raises(tool_errors.ToolError):
        harness.dispatcher.dispatch("merge_child", {"id": one})
    assert not (repo / f"{one}.txt").exists()
