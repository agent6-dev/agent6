# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""DAG dependency edges at the LLM-facing layer: `depends_on` rides `add_task` and `update_task`.

Curator semantics (cycles, the journal op, focus gating) are covered by test_graph_curator.py and
test_harness.py.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from agent6.config import Config, load_config
from agent6.graph import curator as graph_curator
from agent6.graph import models
from agent6.harness import loop as loopmod
from agent6.sessions import layout
from agent6.tools import dispatch, errors, schema

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
[harness]
verify_command = ["true"]
"""

_B = "01" + "B" * 24


def _config(tmp_path: pathlib.Path) -> Config:
    p = tmp_path / "agent6.toml"
    p.write_text(_VALID_TOML, encoding="utf-8")
    return load_config(p)


def _curator(tmp_path: pathlib.Path) -> graph_curator.GraphCurator:
    return graph_curator.GraphCurator(
        layout.SessionLayout(state_dir=tmp_path / ".agent6", session_id="run1")
    )


def test_no_separate_dependency_tool_and_both_carriers_expose_depends_on(
    tmp_path: pathlib.Path,
) -> None:
    """No mode lists an `add_dependency` tool; both carriers' schemas expose `depends_on`."""
    d = dispatch.ToolDispatcher(root=tmp_path, config=_config(tmp_path))
    for mode in ("run", "plan", "ask", "machine", "agent"):
        names = {t.name for t in loopmod.tool_definitions(d, mode=mode)}  # pyright: ignore[reportPrivateUsage]
        assert "add_dependency" not in names, mode
    assert "depends_on" in schema.DagAddTaskInput.model_json_schema()["properties"]
    assert "depends_on" in schema.DagUpdateTaskInput.model_json_schema()["properties"]


def test_add_task_carries_depends_on_to_the_node(tmp_path: pathlib.Path) -> None:
    cur = _curator(tmp_path)
    root = cur.add_subtask(
        models.AddSubtaskIntent(
            parent_id=None, draft=models.TaskNodeDraft(title="root", created_by="planner")
        )
    )
    d = dispatch.ToolDispatcher(root=tmp_path, config=_config(tmp_path), curator=cur)
    d.set_run_root_node_id(root.id)
    a = d.dispatch("add_task", {"title": "first"}).to_wire()
    b = d.dispatch("add_task", {"title": "second", "depends_on": [a["id"]]}).to_wire()
    assert list(cur.nodes()[b["id"]].depends_on) == [a["id"]]


def test_update_task_appends_edges_without_a_status(tmp_path: pathlib.Path) -> None:
    cur = _curator(tmp_path)
    root = cur.add_subtask(
        models.AddSubtaskIntent(
            parent_id=None, draft=models.TaskNodeDraft(title="root", created_by="planner")
        )
    )
    d = dispatch.ToolDispatcher(root=tmp_path, config=_config(tmp_path), curator=cur)
    d.set_run_root_node_id(root.id)
    a = d.dispatch("add_task", {"title": "first"}).to_wire()
    b = d.dispatch("add_task", {"title": "second"}).to_wire()
    out = d.dispatch("update_task", {"id": b["id"], "depends_on": [a["id"]]}).to_wire()
    # Status untouched, the edge landed, and the result names it.
    assert out == {
        "id": b["id"],
        "status": "pending",
        "title": "second",
        "depends_on": [a["id"]],
    }
    json.dumps(out)  # the loop JSONs the result for the model; must not raise


def test_update_task_with_neither_status_nor_edges_refuses(tmp_path: pathlib.Path) -> None:
    cur = _curator(tmp_path)
    root = cur.add_subtask(
        models.AddSubtaskIntent(
            parent_id=None, draft=models.TaskNodeDraft(title="root", created_by="planner")
        )
    )
    d = dispatch.ToolDispatcher(root=tmp_path, config=_config(tmp_path), curator=cur)
    d.set_run_root_node_id(root.id)
    a = d.dispatch("add_task", {"title": "first"}).to_wire()
    with pytest.raises(errors.ToolError, match="status and/or depends_on"):
        d.dispatch("update_task", {"id": a["id"]})


def test_update_task_surfaces_cycle_rejection(tmp_path: pathlib.Path) -> None:
    cur = _curator(tmp_path)
    root = cur.add_subtask(
        models.AddSubtaskIntent(
            parent_id=None, draft=models.TaskNodeDraft(title="root", created_by="planner")
        )
    )
    d = dispatch.ToolDispatcher(root=tmp_path, config=_config(tmp_path), curator=cur)
    d.set_run_root_node_id(root.id)
    a = d.dispatch("add_task", {"title": "first"}).to_wire()
    b = d.dispatch("add_task", {"title": "second", "depends_on": [a["id"]]}).to_wire()
    with pytest.raises(errors.ToolError, match="cycle"):
        d.dispatch("update_task", {"id": a["id"], "depends_on": [b["id"]]})


def test_depends_on_ids_validate_at_the_schema(tmp_path: pathlib.Path) -> None:
    cur = _curator(tmp_path)
    d = dispatch.ToolDispatcher(root=tmp_path, config=_config(tmp_path), curator=cur)
    with pytest.raises(errors.ToolError):
        d.dispatch("update_task", {"id": _B, "depends_on": ["short"]})
    assert not cur.nodes()  # rejected at the schema, never reached the curator


def test_status_and_edges_apply_together(tmp_path: pathlib.Path) -> None:
    cur = _curator(tmp_path)
    root = cur.add_subtask(
        models.AddSubtaskIntent(
            parent_id=None, draft=models.TaskNodeDraft(title="root", created_by="planner")
        )
    )
    d = dispatch.ToolDispatcher(root=tmp_path, config=_config(tmp_path), curator=cur)
    d.set_run_root_node_id(root.id)
    a = d.dispatch("add_task", {"title": "first"}).to_wire()
    b = d.dispatch("add_task", {"title": "second"}).to_wire()
    out = d.dispatch(
        "update_task", {"id": b["id"], "status": "in_progress", "depends_on": [a["id"]]}
    ).to_wire()
    assert out["status"] == "in_progress" and out["depends_on"] == [a["id"]]


def test_list_tasks_wire_shape_is_stable(tmp_path: pathlib.Path) -> None:
    """The list_tasks result dict is a frozen wire surface, JSON'd verbatim to the model.

    Each task projects to exactly {id, parent_id, title, status, acceptance, relevant_paths,
    depends_on} with JSON lists, under a top-level {tasks, count}; a real curator and dispatcher
    drive it.
    """
    cur = _curator(tmp_path)
    root = cur.add_subtask(
        models.AddSubtaskIntent(
            parent_id=None, draft=models.TaskNodeDraft(title="root", created_by="planner")
        )
    )
    a = cur.add_subtask(
        models.AddSubtaskIntent(
            parent_id=root.id,
            draft=models.TaskNodeDraft(
                title="review providers",
                acceptance="no bugs left",
                relevant_paths=("a.py",),
                created_by="worker",
            ),
        )
    )
    b = cur.add_subtask(
        models.AddSubtaskIntent(
            parent_id=root.id,
            draft=models.TaskNodeDraft(
                title="review sandbox", depends_on=(a.id,), created_by="worker"
            ),
        )
    )
    cur.update_status(models.UpdateStatusIntent(id=a.id, new_status="in_progress"))

    d = dispatch.ToolDispatcher(root=tmp_path, config=_config(tmp_path), curator=cur)
    out = d.dispatch("list_tasks", {}).to_wire()
    # Exact equality pins list-vs-tuple; `standing` rides along since the finish gate excludes it.
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
                "standing": False,
            },
            {
                "id": a.id,
                "parent_id": root.id,
                "title": "review providers",
                "status": "in_progress",
                "acceptance": "no bugs left",
                "relevant_paths": ["a.py"],
                "depends_on": [],
                "standing": False,
            },
            {
                "id": b.id,
                "parent_id": root.id,
                "title": "review sandbox",
                "status": "pending",
                "acceptance": "",
                "relevant_paths": [],
                "depends_on": [a.id],
                "standing": False,
            },
        ],
        "count": 3,
    }
    json.dumps(out)  # the loop JSONs the result for the model; must not raise

    # The status filter narrows tasks and count together.
    filtered = d.dispatch("list_tasks", {"status": "in_progress"}).to_wire()
    assert filtered["count"] == 1
    assert [t["id"] for t in filtered["tasks"]] == [a.id]


def test_dag_prompt_blocks_teach_depends_on_not_a_tool() -> None:
    from agent6.prompts import loop

    for block in (loop.DAG_RULES_OPTIONAL, loop.DAG_RULES_DECOMPOSE):
        assert "depends_on" in block
        assert "add_dependency" not in block


def test_update_task_refuses_a_note_without_a_status(tmp_path: pathlib.Path) -> None:
    """update_task refuses a note without a status.

    The graph records a note on a status change only.
    """
    from agent6.tools import _dag_tools

    curator = graph_curator.GraphCurator(
        layout.SessionLayout(state_dir=tmp_path / ".agent6", session_id="run1")
    )
    root = curator.add_subtask(
        models.AddSubtaskIntent(
            parent_id=None,
            draft=models.TaskNodeDraft(title="root", depends_on=(), created_by="planner"),
        )
    )
    with pytest.raises(errors.ToolError, match="a note rides along with a status"):
        _dag_tools.update_task(curator, {"id": root.id, "note": "worth remembering"})
