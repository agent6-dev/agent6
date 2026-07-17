# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""DAG-as-tool handlers: add_task, update_task, set_cursor, add_dependency,
list_tasks. All raise ToolError if no graph_client was wired so standalone
test instantiation works unchanged."""

from __future__ import annotations

from typing import Any

from agent6.graph.client import GraphClient
from agent6.graph.models import (
    AddDependencyIntent,
    AddSubtaskIntent,
    SetCursorIntent,
    TaskNodeDraft,
    UpdateStatusIntent,
)
from agent6.tools.errors import ToolError
from agent6.tools.schema import (
    DagAddDependencyInput,
    DagAddTaskInput,
    DagListTasksInput,
    DagSetCursorInput,
    DagUpdateTaskInput,
)


def add_task(
    graph_client: GraphClient | None, run_root_node_id: str | None, raw: dict[str, Any]
) -> dict[str, Any]:
    if graph_client is None:
        raise ToolError("add_task: DAG curator not available in this run")
    args = DagAddTaskInput.model_validate(raw)
    parent_id = args.parent_id or run_root_node_id
    draft = TaskNodeDraft(
        title=args.title,
        rationale=args.rationale,
        acceptance=args.acceptance,
        relevant_paths=args.relevant_paths,
        created_by="worker",
    )
    intent = AddSubtaskIntent(parent_id=parent_id, draft=draft)
    node = graph_client.add_subtask(intent)
    return {
        "id": node.id,
        "parent_id": node.parent_id,
        "title": node.title,
        "status": node.status,
    }


def update_task(graph_client: GraphClient | None, raw: dict[str, Any]) -> dict[str, Any]:
    if graph_client is None:
        raise ToolError("update_task: DAG curator not available in this run")
    args = DagUpdateTaskInput.model_validate(raw)
    intent = UpdateStatusIntent(
        id=args.id,
        new_status=args.status,  # type: ignore[arg-type]  # pydantic validates the literal
        note=args.note,
    )
    node = graph_client.update_status(intent)
    return {"id": node.id, "status": node.status, "title": node.title}


def set_cursor(graph_client: GraphClient | None, raw: dict[str, Any]) -> dict[str, Any]:
    if graph_client is None:
        raise ToolError("set_cursor: DAG curator not available in this run")
    args = DagSetCursorInput.model_validate(raw)
    graph_client.set_cursor(SetCursorIntent(id=args.id))
    return {"acknowledged": True, "cursor": args.id}


def add_dependency(graph_client: GraphClient | None, raw: dict[str, Any]) -> dict[str, Any]:
    if graph_client is None:
        raise ToolError("add_dependency: DAG curator not available in this run")
    args = DagAddDependencyInput.model_validate(raw)
    intent = AddDependencyIntent(id=args.id, depends_on=args.depends_on)
    # Unknown ids and cycles are rejected by the curator; dispatch()'s
    # generic wrapper surfaces that rejection to the model as a ToolError.
    node = graph_client.add_dependency(intent)
    return {"id": node.id, "title": node.title, "depends_on": list(node.depends_on)}


def list_tasks(graph_client: GraphClient | None, raw: dict[str, Any]) -> dict[str, Any]:
    if graph_client is None:
        raise ToolError("list_tasks: DAG curator not available in this run")
    args = DagListTasksInput.model_validate(raw)
    state = graph_client.get_state()
    nodes = state.get("nodes", {})
    out: list[dict[str, Any]] = []
    for node_id, raw_node in nodes.items():
        if not isinstance(raw_node, dict):
            continue
        if args.status and raw_node.get("status") != args.status:
            continue
        out.append(
            {
                "id": node_id,
                "parent_id": raw_node.get("parent_id"),
                "title": raw_node.get("title", ""),
                "status": raw_node.get("status", "pending"),
                "acceptance": raw_node.get("acceptance", ""),
                "relevant_paths": list(raw_node.get("relevant_paths", ())),
                "depends_on": list(raw_node.get("depends_on", ())),
            }
        )
    return {"tasks": out, "count": len(out)}
