# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""A queued task joins the graph, and only the graph.

The point of queueing rather than steering is that the turn in flight never
sees it: no message, no notice, nothing the model reads until the frontier
reaches the task and its focus banner names it.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from agent6.events import EventSink
from agent6.graph.curator import GraphCurator
from agent6.graph.models import AddSubtaskIntent, TaskNode, TaskNodeDraft
from agent6.sessions.ipc import queue_task
from agent6.sessions.layout import SessionLayout
from agent6.workflows._chain import RunChain
from agent6.workflows._dag_focus import current_task_banner
from agent6.workflows.loop import LoopState, Workflow

_NOW = datetime(2026, 9, 16, tzinfo=UTC)
SPEC = "Add a --json flag\n\nSame fields as the table, keyed by name."


def _workflow(curator: GraphCurator, sink: EventSink) -> Workflow:
    """A loop with only what the drain reads wired: the graph and the journal."""
    return Workflow(
        chain=RunChain(Path("/tmp"), ref=None, branch=None, fallback_parent=None, per_step=False),
        config=MagicMock(),
        provider=MagicMock(),
        dispatcher=MagicMock(),
        logger=lambda _msg: None,
        curator=curator,
        events=sink,
    )


def _state(root: str) -> LoopState:
    return LoopState(original_task="t", tool_calls=0, root_task_id=root, system="")


def _run_dir(tmp_path: Path) -> tuple[GraphCurator, EventSink, str]:
    layout = SessionLayout(state_dir=tmp_path / ".agent6", session_id="run1")
    curator = GraphCurator(layout)
    root = curator.add_subtask(
        AddSubtaskIntent(parent_id=None, draft=TaskNodeDraft(title="the run", created_by="user"))
    )
    return curator, EventSink(layout.session_dir / "logs.jsonl"), root.id


def _events(sink: EventSink) -> list[dict[str, Any]]:
    if not sink.path.exists():
        return []
    return [json.loads(line) for line in sink.path.read_text(encoding="utf-8").splitlines()]


def test_a_queued_task_lands_under_the_root_whole(tmp_path: Path) -> None:
    curator, sink, root = _run_dir(tmp_path)
    queue_task(sink.path.parent, SPEC)
    queue_task(sink.path.parent, "second thing")
    wf = _workflow(curator, sink)

    wf._drain_queued_tasks(_state(root))  # pyright: ignore[reportPrivateUsage]

    children = [curator.nodes()[cid] for cid in curator.get(root).children]
    assert [c.title for c in children] == ["Add a --json flag", "second thing"]
    assert [c.created_by for c in children] == ["user", "user"]
    # The title is the first line; the operator's whole text is kept beside it.
    assert children[0].rationale == SPEC
    assert children[1].rationale == ""


def test_the_drain_takes_each_task_once(tmp_path: Path) -> None:
    curator, sink, root = _run_dir(tmp_path)
    queue_task(sink.path.parent, "only once")
    wf = _workflow(curator, sink)
    state = _state(root)

    wf._drain_queued_tasks(state)  # pyright: ignore[reportPrivateUsage]
    wf._drain_queued_tasks(state)  # pyright: ignore[reportPrivateUsage]

    assert len(curator.get(root).children) == 1


def test_a_queued_task_is_journalled_for_the_surfaces(tmp_path: Path) -> None:
    """The operator sees it arrive without the model having read anything: one
    line in the transcript and the graph snapshot every viewer folds."""
    curator, sink, root = _run_dir(tmp_path)
    queue_task(sink.path.parent, "note it")
    wf = _workflow(curator, sink)

    wf._drain_queued_tasks(_state(root))  # pyright: ignore[reportPrivateUsage]

    kinds = [e["type"] for e in _events(sink)]
    assert "loop.task.queued" in kinds
    assert "graph.update" in kinds
    queued = next(e for e in _events(sink) if e["type"] == "loop.task.queued")
    assert queued["title"] == "note it"


def test_an_empty_queue_journals_nothing(tmp_path: Path) -> None:
    curator, sink, root = _run_dir(tmp_path)
    wf = _workflow(curator, sink)

    wf._drain_queued_tasks(_state(root))  # pyright: ignore[reportPrivateUsage]

    assert _events(sink) == []


def _node(**kw: Any) -> TaskNode:
    base: dict[str, Any] = {
        "id": "01M2M10ABAH7YRMW5YXKY4MDEX",
        "parent_id": "01M2M10AB8RER75QYT5QYHQ2JK",
        "title": "Add a --json flag",
        "created_by": "worker",
        "created_at": _NOW,
        "updated_at": _NOW,
    }
    return TaskNode(**(base | kw))


def test_the_banner_gives_an_operator_task_its_full_text(tmp_path: Path) -> None:
    del tmp_path
    banner = current_task_banner(
        "01M2M10ABAH7YRMW5YXKY4MDEX", _node(created_by="user", rationale=SPEC)
    )

    assert "Same fields as the table, keyed by name." in banner
    assert "The operator queued this task" in banner


def test_the_banner_leaves_the_models_own_tasks_alone(tmp_path: Path) -> None:
    del tmp_path
    banner = current_task_banner("01M2M10ABAH7YRMW5YXKY4MDEX", _node(rationale="my own note"))

    assert "The operator queued this task" not in banner
    assert "my own note" not in banner
