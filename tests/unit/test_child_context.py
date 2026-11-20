"""Stopped children retain measured context and report its current age."""

from __future__ import annotations

import concurrent.futures
import dataclasses
import json
import pathlib

import pytest

from agent6 import child_records, directive
from agent6.harness import (
    child_work,
    loop_conversation,
    loop_state,
    parallel_merge,
    parallel_recovery,
)
from agent6.sessions import child_dispatch
from tests.unit import test_children_background as background


@pytest.mark.parametrize("control", ["agent6", "model"])
def test_stopped_context_survives_completion_and_resume(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, control: str
) -> None:
    """A delivered stop still enters a resumed parent's opening with its refreshed age."""
    repo = tmp_path / "repo"
    background._init_repo(repo)
    children = background._Children(repo, "parent")
    harness = background._harness(repo, tmp_path / "parent", None, children, max_turns=10)
    dispatcher = harness.parallel
    if control == "model":
        dispatcher.chain = dataclasses.replace(dispatcher.chain, ref=None)
    conversation = loop_conversation.Conversation()
    state = loop_state.LoopState(original_task="work", tool_calls=0)
    (sid,) = dispatcher.dispatch(conversation, 1, state, [directive.Segment("work", "")]).children
    children.stop(sid)
    result = dataclasses.replace(
        children.results[sid].result(),
        context=child_records.ChildContext(tokens=1234, last_run_at=1000),
    )
    future: concurrent.futures.Future[child_records.ChildResult] = concurrent.futures.Future()
    future.set_result(result)
    dispatcher.open[next(iter(dispatcher.open))].children[:] = [
        child_records.StartedChild(sid, future)
    ]
    monkeypatch.setattr(parallel_merge.time, "time", lambda: 1120)
    dispatcher.settle(conversation, 2)
    expected = "conversation at last call: 1234 tokens; last ran 2m ago"
    assert expected in json.dumps(conversation.to_wire())
    assert expected in child_work.roster(dispatcher)[0].state
    (record,) = child_dispatch.read_records(tmp_path / "parent")
    join = record.child(sid).join
    assert join is not None
    assert join.context == result.context
    assert not (repo / f"{sid}.txt").exists()
    monkeypatch.setattr(parallel_merge.time, "time", lambda: 1600)
    resumed = loop_conversation.Conversation()
    parallel_recovery.reconcile(dispatcher, state, resumed, turn=3)
    text = json.dumps(resumed.to_wire())
    assert "stopped children" in text
    assert "last ran 10m ago" in text
    assert "1234 tokens" in text and f"agent6/{sid} holds 1 commit(s)" in text
    assert "cache" not in text


def test_missing_context_is_not_reported_as_zero() -> None:
    """Old or incomplete child records retain their ref and name unknown measurements."""
    words = parallel_merge.join_words(
        child_records.ChildJoin("child", "agent6/child", "stopped", "", "stopped")
    )
    assert "agent6/child holds 0 commit(s)" in words
    assert "conversation at last call: unknown; last ran: unknown" in words
