# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A queued task joins the graph, and only the graph.

The point of queueing rather than steering is that the turn in flight never
sees it: no message, no notice, nothing the model reads until the frontier
reaches the task and its focus banner names it.
"""

from __future__ import annotations

import datetime
import json
import pathlib
import types as types_
from typing import Any
from unittest import mock

from agent6 import events
from agent6.graph import curator as graph_curator
from agent6.graph import models
from agent6.harness import _chain, _dag_focus, _loop_state, _operator, _prompt_revision, loop
from agent6.providers import types as providers_types
from agent6.sessions import ipc
from agent6.sessions import layout as sessions_layout

_NOW = datetime.datetime(2026, 9, 16, tzinfo=datetime.UTC)
SPEC = "Add a --json flag\n\nSame fields as the table, keyed by name."


def _workflow(curator: graph_curator.GraphCurator, sink: events.EventSink) -> loop.Harness:
    """A loop with only what the drain reads wired: the graph and the journal."""
    return loop.Harness(
        chain=_chain.RunChain(
            pathlib.Path("/tmp"), ref=None, branch=None, fallback_parent=None, per_step=False
        ),
        config=mock.MagicMock(),
        provider=mock.MagicMock(),
        dispatcher=mock.MagicMock(),
        logger=lambda _msg: None,
        curator=curator,
        events=sink,
        bridge=_operator.OperatorBridge(take_requests=lambda: ipc.drain_requests(sink.path.parent)),
    )


def _state(root: str) -> _loop_state.LoopState:
    return _loop_state.LoopState(original_task="t", tool_calls=0, root_task_id=root, system="")


def _run_dir(tmp_path: pathlib.Path) -> tuple[graph_curator.GraphCurator, events.EventSink, str]:
    layout = sessions_layout.SessionLayout(state_dir=tmp_path / ".agent6", session_id="run1")
    curator = graph_curator.GraphCurator(layout)
    root = curator.add_subtask(
        models.AddSubtaskIntent(
            parent_id=None, draft=models.TaskNodeDraft(title="the run", created_by="user")
        )
    )
    return curator, events.EventSink(layout.session_dir / "logs.jsonl"), root.id


def _events(sink: events.EventSink) -> list[dict[str, Any]]:
    if not sink.path.exists():
        return []
    return [json.loads(line) for line in sink.path.read_text(encoding="utf-8").splitlines()]


def test_a_queued_task_lands_under_the_root_whole(tmp_path: pathlib.Path) -> None:
    curator, sink, root = _run_dir(tmp_path)
    ipc.queue_request(sink.path.parent, "task", SPEC)
    ipc.queue_request(sink.path.parent, "task", "second thing")
    wf = _workflow(curator, sink)

    wf.operator_tasks.take(root)

    children = [curator.nodes()[cid] for cid in curator.get(root).children]
    assert [c.title for c in children] == ["Add a --json flag", "second thing"]
    assert [c.created_by for c in children] == ["user", "user"]
    # The title is the first line; the operator's whole text is kept beside it.
    assert children[0].rationale == SPEC
    assert children[1].rationale == ""


def test_the_drain_takes_each_task_once(tmp_path: pathlib.Path) -> None:
    curator, sink, root = _run_dir(tmp_path)
    ipc.queue_request(sink.path.parent, "task", "only once")
    wf = _workflow(curator, sink)
    state = _state(root)

    wf.operator_tasks.take(state.root_task_id)
    wf.operator_tasks.take(state.root_task_id)

    assert len(curator.get(root).children) == 1


def test_a_queued_task_is_journalled_for_the_surfaces(tmp_path: pathlib.Path) -> None:
    """A queued task arrives as a transcript line and a graph snapshot before the model reads it."""
    curator, sink, root = _run_dir(tmp_path)
    ipc.queue_request(sink.path.parent, "task", "note it")
    wf = _workflow(curator, sink)

    wf.operator_tasks.take(root)

    kinds = [e["type"] for e in _events(sink)]
    assert "loop.task.queued" in kinds
    assert "graph.update" in kinds
    queued = next(e for e in _events(sink) if e["type"] == "loop.task.queued")
    assert queued["title"] == "note it"


def test_an_empty_queue_journals_nothing(tmp_path: pathlib.Path) -> None:
    curator, sink, root = _run_dir(tmp_path)
    wf = _workflow(curator, sink)

    wf.operator_tasks.take(root)

    assert _events(sink) == []


def _node(**kw: Any) -> models.TaskNode:
    base: dict[str, Any] = {
        "id": "01M2M10ABAH7YRMW5YXKY4MDEX",
        "parent_id": "01M2M10AB8RER75QYT5QYHQ2JK",
        "title": "Add a --json flag",
        "created_by": "worker",
        "created_at": _NOW,
        "updated_at": _NOW,
    }
    return models.TaskNode(**(base | kw))


def test_the_banner_gives_an_operator_task_its_full_text(tmp_path: pathlib.Path) -> None:
    del tmp_path
    banner = _dag_focus.current_task_banner(
        "01M2M10ABAH7YRMW5YXKY4MDEX", _node(created_by="user", rationale=SPEC)
    )

    assert "Same fields as the table, keyed by name." in banner
    assert "The operator queued this task" in banner


def test_the_banner_leaves_the_models_own_tasks_alone(tmp_path: pathlib.Path) -> None:
    del tmp_path
    banner = _dag_focus.current_task_banner(
        "01M2M10ABAH7YRMW5YXKY4MDEX", _node(rationale="my own note")
    )

    assert "The operator queued this task" not in banner
    assert "my own note" not in banner


def test_a_queued_task_inherits_prompt_revision(tmp_path: pathlib.Path) -> None:
    """`[prompt].revise_prompt` covers a queued task, with the operator's words authoritative."""
    curator, sink, root = _run_dir(tmp_path)
    ipc.queue_request(sink.path.parent, "task", SPEC)
    reviser = mock.MagicMock()
    reviser.call.return_value = types_.SimpleNamespace(
        text="<revised_task>Add --json to `stats`, same fields as the table.</revised_task>"
    )
    wf = _workflow(curator, sink)
    wf.revision = _prompt_revision.RevisionSettings(reviser=reviser, mode="interactive")

    wf.operator_tasks.take(root)

    node = curator.nodes()[curator.get(root).children[0]]
    assert node.title == "Add a --json flag"  # the tree keeps the operator's words
    assert "Add --json to `stats`" in node.rationale
    assert "Original user task (authoritative if anything conflicts)" in node.rationale
    assert SPEC in node.rationale


def test_a_failed_revision_keeps_the_task_as_written(tmp_path: pathlib.Path) -> None:
    """A queued task is never lost to a reviser that could not answer."""
    curator, sink, root = _run_dir(tmp_path)
    ipc.queue_request(sink.path.parent, "task", SPEC)
    reviser = mock.MagicMock()
    reviser.call.side_effect = providers_types.ProviderError("reviser down")
    wf = _workflow(curator, sink)
    wf.revision = _prompt_revision.RevisionSettings(reviser=reviser, mode="auto")

    wf.operator_tasks.take(root)

    node = curator.nodes()[curator.get(root).children[0]]
    assert node.rationale == SPEC


def test_a_parked_run_takes_a_queued_task_and_continues(tmp_path: pathlib.Path) -> None:
    """A `/task` typed into a parked interactive run wakes it instead of waiting for a steer."""
    curator, sink, root = _run_dir(tmp_path)
    wf = _workflow(curator, sink)
    taken: list[int] = []

    def _take() -> list[ipc.OperatorRequest]:
        taken.append(1)
        return [ipc.OperatorRequest("task", "add a --json flag")] if len(taken) == 1 else []

    wf.bridge = _operator.OperatorBridge(take_requests=_take)

    result = wf._park_for_steer(  # pyright: ignore[reportPrivateUsage]
        mock.MagicMock(), _state(root), iteration=2, reason="quiet"
    )

    assert result is None
    assert [n.title for n in curator.nodes().values() if n.created_by == "user"] == [
        "the run",
        "add a --json flag",
    ]
    types = [e["type"] for e in _events(sink)]
    assert types[-3:] == ["loop.task.queued", "graph.update", "loop.parked.resumed"]
