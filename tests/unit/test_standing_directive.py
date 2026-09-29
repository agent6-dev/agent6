# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`/standing <text>`: the operator sets a live run's standing goal.

`--standing` starts an execution with one, and only the CLI has flags, so before this
the TUI and web could not set a goal at all and no surface could change one
mid-run. The directive replaces the goal the run has, because an operator
typing one means "this is the goal now".
"""

from __future__ import annotations

import os
import pathlib

import pytest

from agent6 import directive, paths
from agent6.graph import curator as graph_curator
from agent6.graph import models
from agent6.harness import loop
from agent6.sessions import ipc
from agent6.sessions import layout as sessions_layout
from agent6.ui import directives
from agent6.ui.cli import main
from tests.unit.test_task_queue_drain import (
    _workflow,  # pyright: ignore[reportPrivateUsage]
)


def test_the_grammar_takes_the_goal() -> None:
    assert directive.parse_standing("/standing keep the suite green") == "keep the suite green"
    assert directive.parse_standing("/standing") == ""
    assert directive.parse_standing("/standingfoo") is None
    assert directive.parse_standing("tell me about /standing") is None


def test_it_is_offered_only_on_a_live_run() -> None:
    assert "/standing" in directive.STEER_COMMANDS
    assert "/standing" in directive.LIVE_RUN_COMMANDS


def test_a_bare_directive_sets_nothing(tmp_path: pathlib.Path) -> None:
    did, said = directives.act_on_directive(tmp_path, "/standing") or (True, "")

    assert not did
    assert "/standing needs the goal" in said
    assert next((r.text for r in ipc.drain_requests(tmp_path)), None) is None


def test_the_directive_writes_the_goal(tmp_path: pathlib.Path) -> None:
    did, said = directives.act_on_directive(tmp_path, "/standing keep the suite green") or (
        False,
        "",
    )

    assert did and "standing goal set" in said
    assert next((r.text for r in ipc.drain_requests(tmp_path)), None) == "keep the suite green"


def test_agent6_steer_takes_it_too(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / ".state"))
    monkeypatch.chdir(tmp_path)
    d = paths.state_dir(tmp_path) / "sessions" / "runs" / "tiny-run-AAAA11"
    d.mkdir(parents=True)
    (d / "logs.jsonl").write_text("", encoding="utf-8")
    ipc.write_worker_pid(d, os.getpid())

    assert main(["steer", "tiny-run", "/standing keep hunting defects"]) == 0

    assert "standing goal set" in capsys.readouterr().out
    assert next((r.text for r in ipc.drain_requests(d)), None) == "keep hunting defects"


def _run(tmp_path: pathlib.Path) -> tuple[graph_curator.GraphCurator, str, loop.Harness]:
    layout = sessions_layout.SessionLayout(state_dir=tmp_path / ".agent6", session_id="run1")
    curator = graph_curator.GraphCurator(layout)
    root = curator.add_subtask(
        models.AddSubtaskIntent(
            parent_id=None, draft=models.TaskNodeDraft(title="the run", created_by="user")
        )
    ).id
    from agent6 import events

    return curator, root, _workflow(curator, events.EventSink(layout.session_dir / "logs.jsonl"))


def test_the_loop_adopts_the_goal(tmp_path: pathlib.Path) -> None:
    curator, root, wf = _run(tmp_path)
    ipc.queue_request(curator.layout.session_dir, "standing", "keep the suite green")

    wf.operator_tasks.take(root)

    standing = [n for n in curator.nodes().values() if n.standing]
    assert [(n.title, n.created_by) for n in standing] == [("keep the suite green", "steering")]
    assert standing[0].status == "pending"


def test_a_new_goal_retires_the_old_one(tmp_path: pathlib.Path) -> None:
    """A retired goal is not made ordinary; an ordinary task of that shape is worked once."""
    curator, root, wf = _run(tmp_path)
    for goal in ("first goal", "second goal"):
        ipc.queue_request(curator.layout.session_dir, "standing", goal)
        wf.operator_tasks.take(root)

    by_title = {n.title: n for n in curator.nodes().values()}
    assert by_title["first goal"].status == "obsolete"
    assert by_title["second goal"].standing and by_title["second goal"].status == "pending"
    # The retired goal keeps its flag and its place in the tree; "the run's
    # standing goal" is the live one.
    live = [n.title for n in curator.nodes().values() if n.standing and n.status == "pending"]
    assert live == ["second goal"]


def test_no_goal_waiting_changes_nothing(tmp_path: pathlib.Path) -> None:
    curator, root, wf = _run(tmp_path)

    wf.operator_tasks.take(root)

    assert not [n for n in curator.nodes().values() if n.standing]
