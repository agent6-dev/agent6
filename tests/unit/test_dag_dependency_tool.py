# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""DAG tools' LLM-facing layer: schema exposure, dispatch wiring, wire shapes.

Curator-level semantics (cycle rejection, journal op, focus gating on
depends_on) are covered by test_graph_curator.py and test_workflow.py; these
tests cover the LLM-facing layer added on top.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

from agent6.config import Config, load_config
from agent6.graph.curator import CuratorError, GraphCurator
from agent6.graph.models import (
    AddDependencyIntent,
    AddSubtaskIntent,
    TaskNode,
    TaskNodeDraft,
    UpdateStatusIntent,
)
from agent6.runs.layout import RunLayout
from agent6.tools.dispatch import ToolDispatcher, ToolError
from agent6.tools.schema import LOOP_EXTRA_TOOLS, PLAN_EXTRA_TOOLS, DagAddDependencyInput
from agent6.workflows import loop as loopmod

_VALID_TOML = """
[agent6]
config_version = 1
[providers.anthropic]
api_format = "anthropic"
api_key_env = "ANTHROPIC_API_KEY"
[models.worker]
provider = "anthropic"
model = "x"
[models.reviewer]
provider = "anthropic"
model = "x"
[workflow]
verify_command = ["true"]
"""

_A = "01" + "A" * 24
_B = "01" + "B" * 24


def _config(tmp_path: Path) -> Config:
    p = tmp_path / "agent6.toml"
    p.write_text(_VALID_TOML, encoding="utf-8")
    return load_config(p)


def _node(node_id: str, depends_on: tuple[str, ...]) -> TaskNode:
    now = datetime.now(tz=UTC)
    return TaskNode(
        id=node_id,
        parent_id=None,
        title="t",
        depends_on=depends_on,
        created_at=now,
        updated_at=now,
        created_by="worker",
    )


class _StubGraph:
    """Duck-typed GraphCurator standing in for the in-process curator."""

    def __init__(self, *, fail: str = "") -> None:
        self.fail = fail
        self.seen: list[AddDependencyIntent] = []

    def add_dependency(self, intent: AddDependencyIntent) -> TaskNode:
        self.seen.append(intent)
        if self.fail:
            raise CuratorError(self.fail)
        return _node(intent.id, (intent.depends_on,))


def test_add_dependency_in_run_and_plan_tool_lists(tmp_path: Path) -> None:
    assert DagAddDependencyInput in LOOP_EXTRA_TOOLS
    assert DagAddDependencyInput in PLAN_EXTRA_TOOLS
    cfg = _config(tmp_path)
    d = ToolDispatcher(root=tmp_path, config=cfg)
    for mode in ("run", "plan"):
        names = {t.name for t in loopmod.tool_definitions(d, mode=mode)}  # pyright: ignore[reportPrivateUsage]
        assert "add_dependency" in names, mode
    for mode in ("ask", "machine", "agent"):
        names = {t.name for t in loopmod.tool_definitions(d, mode=mode)}  # pyright: ignore[reportPrivateUsage]
        assert "add_dependency" not in names, mode


def test_dispatch_add_dependency_roundtrip(tmp_path: Path) -> None:
    stub = _StubGraph()
    d = ToolDispatcher(root=tmp_path, config=_config(tmp_path), curator=cast(GraphCurator, stub))
    out = d.dispatch("add_dependency", {"id": _A, "depends_on": _B})
    assert out == {"id": _A, "title": "t", "depends_on": [_B]}
    assert stub.seen[0].id == _A and stub.seen[0].depends_on == _B


def test_dispatch_add_dependency_requires_curator(tmp_path: Path) -> None:
    d = ToolDispatcher(root=tmp_path, config=_config(tmp_path))
    with pytest.raises(ToolError, match="DAG curator not available"):
        d.dispatch("add_dependency", {"id": _A, "depends_on": _B})


def test_dispatch_add_dependency_surfaces_curator_rejection(tmp_path: Path) -> None:
    stub = _StubGraph(fail="would introduce cycle")
    d = ToolDispatcher(root=tmp_path, config=_config(tmp_path), curator=cast(GraphCurator, stub))
    with pytest.raises(ToolError, match="cycle"):
        d.dispatch("add_dependency", {"id": _A, "depends_on": _B})


def test_dispatch_add_dependency_validates_ids(tmp_path: Path) -> None:
    stub = _StubGraph()
    d = ToolDispatcher(root=tmp_path, config=_config(tmp_path), curator=cast(GraphCurator, stub))
    with pytest.raises(ToolError):
        d.dispatch("add_dependency", {"id": "short", "depends_on": _B})
    assert not stub.seen  # rejected at the schema, never reached the curator


def test_list_tasks_wire_shape_is_stable(tmp_path: Path) -> None:
    """FROZEN wire surface: the list_tasks result dict is JSON'd verbatim to the
    model. Each task projects to exactly {id, parent_id, title, status,
    acceptance, relevant_paths, depends_on} with the sequence fields as JSON
    lists (not tuples), under a top-level {tasks, count}. Interface-independent:
    drives a real curator + real dispatcher, so it pins the returned shape
    regardless of how the curator hands state to the tool internally."""
    cur = GraphCurator(RunLayout(state_dir=tmp_path / ".agent6", run_id="run1"))
    root = cur.add_subtask(
        AddSubtaskIntent(parent_id=None, draft=TaskNodeDraft(title="root", created_by="planner"))
    )
    a = cur.add_subtask(
        AddSubtaskIntent(
            parent_id=root.id,
            draft=TaskNodeDraft(
                title="audit providers",
                acceptance="no bugs left",
                relevant_paths=("a.py",),
                created_by="worker",
            ),
        )
    )
    b = cur.add_subtask(
        AddSubtaskIntent(
            parent_id=root.id,
            draft=TaskNodeDraft(title="audit sandbox", depends_on=(a.id,), created_by="worker"),
        )
    )
    cur.update_status(UpdateStatusIntent(id=a.id, new_status="in_progress"))

    d = ToolDispatcher(root=tmp_path, config=_config(tmp_path), curator=cur)
    out = d.dispatch("list_tasks", {})
    # Exact equality also pins list-vs-tuple: ("a.py",) != ["a.py"].
    assert out == {
        "tasks": [
            {
                "id": root.id,
                "parent_id": None,
                "title": "root",
                "status": "pending",
                "acceptance": "",
                "relevant_paths": [],
                "depends_on": [],
            },
            {
                "id": a.id,
                "parent_id": root.id,
                "title": "audit providers",
                "status": "in_progress",
                "acceptance": "no bugs left",
                "relevant_paths": ["a.py"],
                "depends_on": [],
            },
            {
                "id": b.id,
                "parent_id": root.id,
                "title": "audit sandbox",
                "status": "pending",
                "acceptance": "",
                "relevant_paths": [],
                "depends_on": [a.id],
            },
        ],
        "count": 3,
    }
    json.dumps(out)  # the loop JSONs the result for the model; must not raise

    # The status filter narrows tasks and count together.
    filtered = d.dispatch("list_tasks", {"status": "in_progress"})
    assert filtered["count"] == 1
    assert [t["id"] for t in filtered["tasks"]] == [a.id]


def test_dag_prompt_blocks_mention_add_dependency() -> None:
    from agent6.prompts.loop import DAG_RULES_DECOMPOSE, DAG_RULES_OPTIONAL

    assert "add_dependency" in DAG_RULES_OPTIONAL
    assert "add_dependency" in DAG_RULES_DECOMPOSE
