# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The task-graph nodes and the intents that mutate them.

The tree is doubly linked: a parent lists each child in `children` and each child names its
`parent_id`, a symmetry the curator keeps on every mutation. These cross trust boundaries
(model-emitted intents, disk reload), so they are pydantic.
"""

from __future__ import annotations

import datetime
from typing import Literal

import pydantic

from agent6.graph import ulid

_MODEL_CONFIG = pydantic.ConfigDict(extra="forbid", frozen=True)

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
]


def owner_note(*, created_by: str, parent_id: str | None, standing: bool) -> str:
    """Return what every surface says beside a task the operator owns.

    Args:
        created_by: The node's actor.
        parent_id: The node's parent; None for the root.
        standing: The node is the standing goal, which only the operator's steering sets.

    Returns:
        `standing goal`, `queued by you` for a task the operator added to a live run, or ""
        for the model's own tasks and the root.
    """
    if standing:
        return "standing goal"
    if created_by == "user" and parent_id is not None:
        return "queued by you"
    return ""


def queued_by_operator(node: TaskNode) -> bool:
    """Return whether the operator added the task to a live run.

    Such a task is theirs to withdraw: the model may pass it or leave it open, never retire it.

    Args:
        node: The node.

    Returns:
        True for an operator-queued task.
    """
    return owner_note(created_by=node.created_by, parent_id=node.parent_id, standing=False) != ""


class TaskNodeDraft(pydantic.BaseModel):
    """A new node before the curator assigns its id.

    Attributes:
        title: The task, one line.
        rationale: Why it exists.
        acceptance: What done looks like.
        relevant_paths: The files it touches.
        depends_on: The ids that must be done first.
        created_by: The actor adding it.
        standing: The run's fallback: it never passes, and the frontier selects it only when
            no ordinary subtask is ready.
    """

    model_config = _MODEL_CONFIG

    title: str = pydantic.Field(min_length=1)
    rationale: str = ""
    acceptance: str = ""
    relevant_paths: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()
    created_by: NodeActor
    standing: bool = False


class TaskNode(pydantic.BaseModel):
    """A persisted task-graph node.

    Attributes:
        id: The run's task count, zero-padded.
        parent_id: The parent's id; None for the root.
        title: The task, one line.
        rationale: Why it exists.
        acceptance: What done looks like.
        relevant_paths: The files it touches.
        depends_on: The ids that must be done first.
        children: The child ids, in execution order.
        status: The node's status.
        created_at: When it was added.
        updated_at: When it was last written.
        created_by: The actor that added it.
        commit_sha: The commit that landed it, or "".
        notes: Appended prose.
        standing: The never-passing fallback node.
        graph_version: The version of the mutation that last wrote the node, the number its
            journal entry carries; 0 unstamped. A node stamped newer than the journal's max
            version is a journal that lost its tail.
    """

    model_config = _MODEL_CONFIG

    id: str = pydantic.Field(min_length=1, max_length=26)
    parent_id: str | None
    title: str = pydantic.Field(min_length=1)
    rationale: str = ""
    acceptance: str = ""
    relevant_paths: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()
    children: tuple[str, ...] = ()
    status: NodeStatus = "pending"
    created_at: datetime.datetime
    updated_at: datetime.datetime
    created_by: NodeActor
    commit_sha: str = ""
    notes: str = ""
    standing: bool = False
    graph_version: int = 0

    @pydantic.field_validator("id")
    @classmethod
    def _id_is_crockford(cls, v: str) -> str:
        """Refuse an id outside the Crockford alphabet at the reload trust boundary.

        The id becomes a path component under graph_dir, so a crafted id carrying separators
        would let the next write escape it; a bad-id file fails validation and `load_graph`
        skips it with a warning.

        Args:
            v: The id.

        Returns:
            The id unchanged.

        Raises:
            ValueError: A character is outside the alphabet.
        """
        if any(ch not in ulid.CROCKFORD for ch in v):
            raise ValueError(f"node id is not Crockford base32: {v!r}")
        return v


class AddSubtaskIntent(pydantic.BaseModel):
    """Add a node under a parent.

    Attributes:
        op: The intent name.
        parent_id: The parent; None for the root.
        draft: The new node.
        after: A sibling to place the child directly after, instead of appending; the
            children list is the order the frontier executes.
    """

    model_config = _MODEL_CONFIG

    op: Literal["add_subtask"] = "add_subtask"
    parent_id: str | None
    draft: TaskNodeDraft
    after: str | None = None


class UpdateStatusIntent(pydantic.BaseModel):
    """Set a node's status, with an optional note."""

    model_config = _MODEL_CONFIG

    op: Literal["update_status"] = "update_status"
    id: str
    new_status: NodeStatus
    note: str = ""


class AddDependencyIntent(pydantic.BaseModel):
    """Make a node wait on another."""

    model_config = _MODEL_CONFIG

    op: Literal["add_dependency"] = "add_dependency"
    id: str
    depends_on: str


class RecordCommitIntent(pydantic.BaseModel):
    """Record the commit that landed a node."""

    model_config = _MODEL_CONFIG

    op: Literal["record_commit"] = "record_commit"
    id: str
    sha: str


class SetCursorIntent(pydantic.BaseModel):
    """Focus a node, or clear the focus with None."""

    model_config = _MODEL_CONFIG

    op: Literal["set_cursor"] = "set_cursor"
    id: str | None
