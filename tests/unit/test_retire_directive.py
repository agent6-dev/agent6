# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`/retire <task id>`: the operator drops a task from a live run's graph.

The model has `update_task`; the operator had nothing, on any surface. A task
is named by the id `/tasks` prints, which is the run's own count: `/retire 2`.
An id this run does not hold is refused with the ids it does.
"""

from __future__ import annotations

from pathlib import Path

from agent6.directive import LIVE_RUN_COMMANDS, STEER_COMMANDS, parse_retire
from agent6.graph.curator import GraphCurator
from agent6.graph.models import AddSubtaskIntent, NodeActor, TaskNodeDraft
from agent6.sessions.ipc import take_retired_tasks
from agent6.sessions.layout import SessionLayout
from agent6.ui.directives import act_on_directive
from agent6.viewmodel.format import short_task_id
from agent6.workflows.loop import Workflow
from tests.unit.test_task_queue_drain import (
    _workflow,  # pyright: ignore[reportPrivateUsage]
)


def _graph(tmp_path: Path) -> tuple[GraphCurator, Path, str, list[str], Workflow]:
    layout = SessionLayout(state_dir=tmp_path / ".agent6", session_id="run1")
    curator = GraphCurator(layout)
    root = curator.add_subtask(
        AddSubtaskIntent(parent_id=None, draft=TaskNodeDraft(title="the run", created_by="user"))
    ).id
    actors: tuple[tuple[str, NodeActor], ...] = (
        ("the model's own", "worker"),
        ("queued by you", "user"),
    )
    kids = [
        curator.add_subtask(
            AddSubtaskIntent(parent_id=root, draft=TaskNodeDraft(title=title, created_by=by))
        ).id
        for title, by in actors
    ]
    from agent6.events import EventSink

    sink = EventSink(layout.session_dir / "logs.jsonl")
    return curator, layout.session_dir, root, kids, _workflow(curator, sink)


def test_the_grammar_takes_the_id() -> None:
    assert parse_retire("/retire 01M2ABC") == "01M2ABC"
    assert parse_retire("/retire") == ""
    assert parse_retire("/retirement of the parser") is None


def test_it_is_offered_only_on_a_live_run() -> None:
    assert "/retire" in STEER_COMMANDS
    assert "/retire" in LIVE_RUN_COMMANDS


def test_a_full_id_retires_at_the_next_turn(tmp_path: Path) -> None:
    curator, session_dir, root, kids, wf = _graph(tmp_path)

    did, said = act_on_directive(session_dir, f"/retire {kids[0]}") or (False, "")

    assert did and "the model's own" in said
    assert take_retired_tasks(session_dir) == [kids[0]]
    del curator, root, wf


def test_the_id_is_the_runs_own_count(tmp_path: Path) -> None:
    """A task is `1`, `2`, `3` within its run, which is what an operator reads
    off `/tasks` and types back."""
    _curator, session_dir, root, kids, _wf = _graph(tmp_path)
    assert [root, *kids] == ["0001", "0002", "0003"]
    assert [short_task_id(nid) for nid in kids] == ["2", "3"]

    did, said = act_on_directive(session_dir, "/retire 2") or (False, "")

    assert did and "the model's own" in said
    assert take_retired_tasks(session_dir) == [kids[0]]


def test_the_padded_id_works_too(tmp_path: Path) -> None:
    """Whichever form a listing or a log showed."""
    _curator, session_dir, _root, kids, _wf = _graph(tmp_path)

    did, _said = act_on_directive(session_dir, f"/retire {kids[1]}") or (False, "")

    assert did and take_retired_tasks(session_dir) == [kids[1]]


def test_an_unknown_id_is_refused_with_the_ones_that_exist(tmp_path: Path) -> None:
    _curator, session_dir, _root, _kids, _wf = _graph(tmp_path)

    did, said = act_on_directive(session_dir, "/retire 9") or (True, "")

    assert not did
    assert "no task '9' here" in said
    assert "1 the run" in said and "2 the model's own" in said
    assert take_retired_tasks(session_dir) == []


def test_a_bare_directive_retires_nothing(tmp_path: Path) -> None:
    _curator, session_dir, _root, _kids, _wf = _graph(tmp_path)

    did, said = act_on_directive(session_dir, "/retire") or (True, "")

    assert not did and "/retire needs the task" in said


def test_the_loop_retires_what_the_operator_named(tmp_path: Path) -> None:
    """Including a task the operator queued, which `update_task` refuses to the
    model: the curator is the operator's own route."""
    curator, session_dir, root, kids, wf = _graph(tmp_path)
    for task_id in kids:
        act_on_directive(session_dir, f"/retire {task_id}")

    wf._drain_retired_tasks()  # pyright: ignore[reportPrivateUsage]

    assert [curator.nodes()[k].status for k in kids] == ["obsolete", "obsolete"]
    assert take_retired_tasks(session_dir) == []
    del root


def test_an_id_that_vanished_is_logged_and_skipped(tmp_path: Path) -> None:
    curator, session_dir, root, kids, wf = _graph(tmp_path)
    from agent6.sessions.ipc import retire_task

    retire_task(session_dir, "01MISSINGMISSINGMISSINGMIS")

    wf._drain_retired_tasks()  # pyright: ignore[reportPrivateUsage]

    assert [curator.nodes()[k].status for k in kids] == ["pending", "pending"]
    del root
