# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""One way to set a standing goal at a run's start, one way to change it after.

`run --standing` seeds the goal a fresh run returns to. A continuation takes no
such flag: a fork carries the source's graph, goal included, and `/standing`
is what replaces a goal on a run already going. Two flags for one decision is
what this pins against.
"""

from __future__ import annotations

from pathlib import Path

from agent6.graph.curator import GraphCurator
from agent6.graph.models import AddSubtaskIntent, TaskNodeDraft
from agent6.sessions.layout import SessionLayout
from agent6.ui.cli.parser import build_parser
from agent6.workflows.loop import Workflow
from tests.unit.test_task_queue_drain import (
    _workflow,  # pyright: ignore[reportPrivateUsage]
)


def _seeded(tmp_path: Path) -> tuple[GraphCurator, str, Workflow]:
    layout = SessionLayout(state_dir=tmp_path / ".agent6", session_id="run1")
    curator = GraphCurator(layout)
    root = curator.add_subtask(
        AddSubtaskIntent(parent_id=None, draft=TaskNodeDraft(title="the run", created_by="user"))
    ).id
    from agent6.events import EventSink

    return curator, root, _workflow(curator, EventSink(layout.session_dir / "logs.jsonl"))


def test_only_a_fresh_run_takes_the_flag() -> None:
    """A continuation offered `--standing` and then kept the goal it already
    had, which is a flag that does nothing on a session that has one."""
    parser = build_parser()

    assert parser.parse_args(["run", "do it", "--standing", "keep it green"]).standing
    for verb in ("resume", "fork"):
        args = parser.parse_args([verb, "id"])
        assert not hasattr(args, "standing"), f"{verb} still offers --standing"


def test_a_fresh_run_seeds_the_goal(tmp_path: Path) -> None:
    curator, root, wf = _seeded(tmp_path)
    wf.standing_goal = "keep the suite green"

    wf._seed_standing_goal(root)  # pyright: ignore[reportPrivateUsage]

    standing = [n for n in curator.nodes().values() if n.standing]
    assert [(n.title, n.created_by) for n in standing] == [("keep the suite green", "steering")]


def test_no_flag_seeds_nothing(tmp_path: Path) -> None:
    curator, root, wf = _seeded(tmp_path)

    wf._seed_standing_goal(root)  # pyright: ignore[reportPrivateUsage]

    assert not [n for n in curator.nodes().values() if n.standing]
