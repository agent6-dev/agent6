# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""What the model may do to the operator's own nodes.

A task the operator queued is finished, never dismissed, and the standing slot belongs to the
operator alone, so `--standing` stays the one way to set one.
"""

from __future__ import annotations

import pathlib

import pydantic
import pytest

from agent6.graph import curator, models
from agent6.sessions import layout
from agent6.tools import _dag_tools, errors, schema


def _curator(tmp_path: pathlib.Path) -> tuple[curator.GraphCurator, str]:
    c = curator.GraphCurator(
        layout.SessionLayout(state_dir=tmp_path / ".agent6", session_id="run1")
    )
    root = c.add_subtask(
        models.AddSubtaskIntent(
            parent_id=None, draft=models.TaskNodeDraft(title="the run", created_by="user")
        )
    )
    return c, root.id


def _queued(c: curator.GraphCurator, root: str, title: str = "add a --json flag") -> str:
    return c.add_subtask(
        models.AddSubtaskIntent(
            parent_id=root, draft=models.TaskNodeDraft(title=title, created_by="user")
        )
    ).id


@pytest.mark.parametrize("retirement", ["skipped", "obsolete"])
def test_the_model_cannot_retire_a_task_the_operator_queued(
    tmp_path: pathlib.Path, retirement: models.NodeStatus
) -> None:
    """The model's route refuses; the curator stays permissive as the operator's `/retire` route."""
    c, root = _curator(tmp_path)
    queued = _queued(c, root)

    with pytest.raises(errors.ToolError, match="queued by the operator"):
        _dag_tools.update_task(c, {"id": queued, "status": retirement})

    assert c.nodes()[queued].status == "pending"
    assert c.update_status(models.UpdateStatusIntent(id=queued, new_status=retirement)).status == (
        retirement
    )


def test_an_operator_task_still_passes(tmp_path: pathlib.Path) -> None:
    c, root = _curator(tmp_path)
    queued = _queued(c, root)

    assert (
        c.update_status(models.UpdateStatusIntent(id=queued, new_status="passed")).status
        == "passed"
    )


def test_the_root_is_retirable_though_the_operator_owns_it(tmp_path: pathlib.Path) -> None:
    """An abandoned run retires the seeded root; only queued subtasks are the operator's to keep."""
    c, root = _curator(tmp_path)

    assert (
        c.update_status(models.UpdateStatusIntent(id=root, new_status="obsolete")).status
        == "obsolete"
    )


def test_the_models_standing_task_lands_as_an_ordinary_one(tmp_path: pathlib.Path) -> None:
    """`--standing` is the one way to set a standing goal; another actor's draft loses the flag."""
    c, root = _curator(tmp_path)

    node = c.add_subtask(
        models.AddSubtaskIntent(
            parent_id=root,
            draft=models.TaskNodeDraft(
                title="keep hunting defects", standing=True, created_by="worker"
            ),
        )
    )

    assert not node.standing
    assert node.title == "keep hunting defects"
    assert node.status == "pending"


def test_add_task_offers_no_standing_flag(tmp_path: pathlib.Path) -> None:
    """add_task carries no standing argument, and a stale call naming it is refused."""
    c, root = _curator(tmp_path)

    assert "standing" not in schema.DagAddTaskInput.TOOL_DESCRIPTION
    assert "standing" not in schema.DagAddTaskInput.model_fields
    with pytest.raises(pydantic.ValidationError):
        _dag_tools.add_task(c, root, {"title": "keep hunting defects", "standing": True})

    assert _dag_tools.add_task(c, root, {"title": "ordinary"}).to_wire()["status"] == "pending"


def test_a_new_sibling_lands_before_the_standing_goal(tmp_path: pathlib.Path) -> None:
    """The standing goal is seeded right after the root, so every later task queues ahead of it."""
    c, root = _curator(tmp_path)
    standing = c.add_subtask(
        models.AddSubtaskIntent(
            parent_id=root,
            draft=models.TaskNodeDraft(
                title="keep the suite green", standing=True, created_by="steering"
            ),
        )
    ).id
    first = _queued(c, root, "first")
    second = _queued(c, root, "second")

    assert c.get(root).children == (first, second, standing)


def test_a_named_position_still_wins_over_the_standing_goal(tmp_path: pathlib.Path) -> None:
    c, root = _curator(tmp_path)
    first = _queued(c, root, "first")
    c.add_subtask(
        models.AddSubtaskIntent(
            parent_id=root,
            draft=models.TaskNodeDraft(
                title="keep the suite green", standing=True, created_by="steering"
            ),
        )
    )
    middle = c.add_subtask(
        models.AddSubtaskIntent(
            parent_id=root,
            after=first,
            draft=models.TaskNodeDraft(title="middle", created_by="worker"),
        )
    ).id

    assert c.get(root).children[:2] == (first, middle)


def test_the_operators_standing_goal_keeps_its_flag(tmp_path: pathlib.Path) -> None:
    c, root = _curator(tmp_path)

    node = c.add_subtask(
        models.AddSubtaskIntent(
            parent_id=root,
            draft=models.TaskNodeDraft(
                title="keep the suite green", standing=True, created_by="steering"
            ),
        )
    )

    assert node.standing
