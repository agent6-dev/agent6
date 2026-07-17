# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Pydantic models for the persistent task graph.

These cross trust boundaries (LLM-emitted intents, disk reload), so they
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


class SetCursorIntent(BaseModel):
    model_config = _MODEL_CONFIG

    op: Literal["set_cursor"] = "set_cursor"
    id: str | None
