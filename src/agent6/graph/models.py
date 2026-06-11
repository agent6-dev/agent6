# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Pydantic models for the persistent task graph and curator IPC.

These cross trust boundaries (LLM-emitted intents, disk reload, IPC), so they
are pydantic per project convention. Internal-only value types remain frozen
dataclasses in `agent6.types`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True)

# ---- domain types ---------------------------------------------------------

NodeStatus = Literal[
    "pending",
    "in_progress",
    "passed",
    "failed",
    "skipped",
    "obsolete",
]

NodeActor = Literal[
    "planner",
    "worker",
    "steering",
    "alignment_guard",
    "user",
    "reviewer",
    "critic",
]


class TaskNodeDraft(BaseModel):
    """A new-node payload, id is assigned by the curator on insert."""

    model_config = _MODEL_CONFIG

    title: str = Field(min_length=1)
    rationale: str = ""
    acceptance: str = ""
    relevant_paths: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()
    created_by: NodeActor


class TaskNode(BaseModel):
    """A persisted task graph node."""

    model_config = _MODEL_CONFIG

    id: str = Field(min_length=26, max_length=26)
    parent_id: str | None
    title: str = Field(min_length=1)
    rationale: str = ""
    acceptance: str = ""
    relevant_paths: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()
    children: tuple[str, ...] = ()
    status: NodeStatus = "pending"
    created_at: datetime
    updated_at: datetime
    created_by: NodeActor
    commit_sha: str = ""
    notes: str = ""


class TouchedFile(BaseModel):
    """One uncommitted file the agent touched, captured in a snapshot."""

    model_config = _MODEL_CONFIG

    path: str
    sha256: str
    size: int
    mtime: float


class NodeSnapshot(BaseModel):
    """`.agent6/runs/<id>/snapshots/<node-id>.json` content."""

    model_config = _MODEL_CONFIG

    head_sha: str = Field(min_length=4)
    branch: str
    uncommitted_touched: tuple[TouchedFile, ...] = ()
    graph_version: int


class CommittedDelta(BaseModel):
    """Changes the user (or anyone) committed between snapshot and resume."""

    model_config = _MODEL_CONFIG

    from_sha: str
    to_sha: str
    files: tuple[str, ...] = ()


class UncommittedFileDiff(BaseModel):
    """One uncommitted file whose state changed since the snapshot."""

    model_config = _MODEL_CONFIG

    path: str
    expected_sha256: str
    actual_sha256: str
    note: str = ""


class ResumeDiff(BaseModel):
    """Aggregate of what changed in the workspace since a run was paused."""

    model_config = _MODEL_CONFIG

    run_id: str
    snapshot_head: str
    current_head: str
    committed_delta: CommittedDelta
    uncommitted_diff: tuple[UncommittedFileDiff, ...] = ()
    snapshot_missing: bool = False
    guard_summary: str = ""


# ---- curator intent payloads ---------------------------------------------


class AddSubtaskIntent(BaseModel):
    model_config = _MODEL_CONFIG

    op: Literal["add_subtask"] = "add_subtask"
    parent_id: str | None
    draft: TaskNodeDraft


class UpdateStatusIntent(BaseModel):
    model_config = _MODEL_CONFIG

    op: Literal["update_status"] = "update_status"
    id: str
    new_status: NodeStatus
    note: str = ""


class AddDependencyIntent(BaseModel):
    model_config = _MODEL_CONFIG

    op: Literal["add_dependency"] = "add_dependency"
    id: str
    depends_on: str


class ObsoleteIntent(BaseModel):
    model_config = _MODEL_CONFIG

    op: Literal["obsolete"] = "obsolete"
    id: str
    reason: str


class ReorderChildrenIntent(BaseModel):
    model_config = _MODEL_CONFIG

    op: Literal["reorder_children"] = "reorder_children"
    parent_id: str
    new_order: tuple[str, ...]


class RecordCommitIntent(BaseModel):
    model_config = _MODEL_CONFIG

    op: Literal["record_commit"] = "record_commit"
    id: str
    sha: str


class SnapshotNodeIntent(BaseModel):
    model_config = _MODEL_CONFIG

    op: Literal["snapshot_node"] = "snapshot_node"
    id: str
    head_sha: str
    branch: str
    uncommitted_touched: tuple[TouchedFile, ...] = ()


class SetCursorIntent(BaseModel):
    model_config = _MODEL_CONFIG

    op: Literal["set_cursor"] = "set_cursor"
    id: str | None
