# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""What the model may do to the operator's own nodes.

Two rules, both in the curator so every caller gets them: a task the operator
queued is finished, never dismissed, and the standing slot belongs to the
operator alone, so `--standing` stays the one and only way to set one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent6.graph.curator import CuratorError, GraphCurator
from agent6.graph.models import (
    AddSubtaskIntent,
    NodeStatus,
    TaskNodeDraft,
    UpdateStatusIntent,
)
from agent6.sessions.layout import SessionLayout
from agent6.tools._dag_tools import add_task


def _curator(tmp_path: Path) -> tuple[GraphCurator, str]:
    c = GraphCurator(SessionLayout(state_dir=tmp_path / ".agent6", session_id="run1"))
    root = c.add_subtask(
        AddSubtaskIntent(parent_id=None, draft=TaskNodeDraft(title="the run", created_by="user"))
    )
    return c, root.id


def _queued(c: GraphCurator, root: str, title: str = "add a --json flag") -> str:
    return c.add_subtask(
        AddSubtaskIntent(parent_id=root, draft=TaskNodeDraft(title=title, created_by="user"))
    ).id


@pytest.mark.parametrize("retirement", ["skipped", "obsolete"])
def test_the_model_cannot_retire_a_task_the_operator_queued(
    tmp_path: Path, retirement: NodeStatus
) -> None:
    c, root = _curator(tmp_path)
    queued = _queued(c, root)

    with pytest.raises(CuratorError, match="queued by the operator"):
        c.update_status(UpdateStatusIntent(id=queued, new_status=retirement))

    assert c.nodes()[queued].status == "pending"


def test_an_operator_task_still_passes(tmp_path: Path) -> None:
    c, root = _curator(tmp_path)
    queued = _queued(c, root)

    assert c.update_status(UpdateStatusIntent(id=queued, new_status="passed")).status == "passed"


def test_the_root_is_retirable_though_the_operator_owns_it(tmp_path: Path) -> None:
    """The seeded root is `created_by="user"` too, and a run that is abandoned
    retires it; only queued SUBTASKS are the operator's to keep."""
    c, root = _curator(tmp_path)

    assert c.update_status(UpdateStatusIntent(id=root, new_status="obsolete")).status == "obsolete"


def test_the_models_standing_task_lands_as_an_ordinary_one(tmp_path: Path) -> None:
    """`--standing` is the one way to set a standing goal, so there is exactly
    one and it is the operator's. A model asking for another keeps the task and
    loses only the flag."""
    c, root = _curator(tmp_path)

    node = c.add_subtask(
        AddSubtaskIntent(
            parent_id=root,
            draft=TaskNodeDraft(title="keep hunting defects", standing=True, created_by="worker"),
        )
    )

    assert not node.standing
    assert node.title == "keep hunting defects"
    assert node.status == "pending"


def test_add_task_says_the_standing_flag_was_dropped(tmp_path: Path) -> None:
    """An argument silently ignored is hidden state: the tool result names it."""
    c, root = _curator(tmp_path)

    result = add_task(c, root, {"title": "keep hunting defects", "standing": True})

    assert result.to_wire()["note"].startswith("standing is the operator's to set")
    assert "note" not in add_task(c, root, {"title": "ordinary"}).to_wire()


def test_the_operators_standing_goal_keeps_its_flag(tmp_path: Path) -> None:
    c, root = _curator(tmp_path)

    node = c.add_subtask(
        AddSubtaskIntent(
            parent_id=root,
            draft=TaskNodeDraft(title="keep the suite green", standing=True, created_by="steering"),
        )
    )

    assert node.standing
