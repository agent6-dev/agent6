# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`--standing` is not a run-only flag.

A fork carries the source's task graph, so forking keeps the unfinished tasks
but inherited whatever standing goal the source had, with no way to give one to
a session that started without it. Resume and fork take the flag now, and the
one-per-run invariant decides what happens when there is already one.
"""

from __future__ import annotations

from pathlib import Path

from agent6.graph.curator import GraphCurator
from agent6.graph.models import AddSubtaskIntent, TaskNodeDraft
from agent6.sessions.layout import SessionLayout
from agent6.ui.cli.parser import build_parser
from agent6.workflows.loop import Workflow
from tests.unit.test_task_queue_drain import _workflow  # pyright: ignore[reportPrivateUsage]


def _seeded(tmp_path: Path) -> tuple[GraphCurator, str, Workflow]:
    layout = SessionLayout(state_dir=tmp_path / ".agent6", session_id="run1")
    curator = GraphCurator(layout)
    root = curator.add_subtask(
        AddSubtaskIntent(parent_id=None, draft=TaskNodeDraft(title="the run", created_by="user"))
    ).id
    from agent6.events import EventSink

    return curator, root, _workflow(curator, EventSink(layout.session_dir / "logs.jsonl"))


def test_resume_and_fork_take_the_flag() -> None:
    parser = build_parser()

    assert parser.parse_args(["resume", "id", "--standing", "keep it green"]).standing
    assert parser.parse_args(["fork", "id", "--standing", "keep it green"]).standing


def test_a_continuation_seeds_the_goal_the_run_never_had(tmp_path: Path) -> None:
    curator, root, wf = _seeded(tmp_path)
    wf.standing_goal = "keep the suite green"

    wf._seed_standing_goal(root)  # pyright: ignore[reportPrivateUsage]

    standing = [n for n in curator.nodes().values() if n.standing]
    assert [n.title for n in standing] == ["keep the suite green"]
    assert standing[0].created_by == "steering"


def test_a_run_that_has_one_keeps_it(tmp_path: Path) -> None:
    """One standing goal per run: a resume naming another changes nothing."""
    curator, root, wf = _seeded(tmp_path)
    wf.standing_goal = "first goal"
    wf._seed_standing_goal(root)  # pyright: ignore[reportPrivateUsage]
    wf.standing_goal = "second goal"

    wf._seed_standing_goal(root)  # pyright: ignore[reportPrivateUsage]

    assert [n.title for n in curator.nodes().values() if n.standing] == ["first goal"]


def test_no_flag_seeds_nothing(tmp_path: Path) -> None:
    curator, root, wf = _seeded(tmp_path)

    wf._seed_standing_goal(root)  # pyright: ignore[reportPrivateUsage]

    assert not [n for n in curator.nodes().values() if n.standing]


def test_a_retired_goal_leaves_room_for_a_new_one(tmp_path: Path) -> None:
    """`/standing` retires the goal it replaces, and a retired goal keeps its
    flag: "the run has a goal" has to mean a live one, or a later
    `resume --standing` would find the dead one and seed nothing."""
    from agent6.graph.models import UpdateStatusIntent

    curator, root, wf = _seeded(tmp_path)
    wf.standing_goal = "first goal"
    wf._seed_standing_goal(root)  # pyright: ignore[reportPrivateUsage]
    dead = next(n for n in curator.nodes().values() if n.standing)
    curator.update_status(UpdateStatusIntent(id=dead.id, new_status="obsolete"))

    wf.standing_goal = "second goal"
    wf._seed_standing_goal(root)  # pyright: ignore[reportPrivateUsage]

    live = [n.title for n in curator.nodes().values() if n.standing and n.status == "pending"]
    assert live == ["second goal"]
