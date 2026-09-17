# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`/standing <text>`: the operator sets a live run's standing goal.

`--standing` starts a leg with one, and only the CLI has flags, so before this
the TUI and web could not set a goal at all and no surface could change one
mid-run. The directive replaces the goal the run has, because an operator
typing one means "this is the goal now".
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent6.directive import LIVE_RUN_COMMANDS, STEER_COMMANDS, parse_standing
from agent6.graph.curator import GraphCurator
from agent6.graph.models import AddSubtaskIntent, TaskNodeDraft
from agent6.paths import state_dir
from agent6.sessions.ipc import set_standing_goal, take_standing_goal, write_worker_pid
from agent6.sessions.layout import SessionLayout
from agent6.ui.cli import main
from agent6.ui.directives import act_on_directive
from agent6.workflows.loop import Workflow
from tests.unit.test_task_queue_drain import (
    _state,  # pyright: ignore[reportPrivateUsage]
    _workflow,  # pyright: ignore[reportPrivateUsage]
)


def test_the_grammar_takes_the_goal() -> None:
    assert parse_standing("/standing keep the suite green") == "keep the suite green"
    assert parse_standing("/standing") == ""
    assert parse_standing("/standingfoo") is None
    assert parse_standing("tell me about /standing") is None


def test_it_is_offered_only_on_a_live_run() -> None:
    assert "/standing" in STEER_COMMANDS
    assert "/standing" in LIVE_RUN_COMMANDS


def test_a_bare_directive_sets_nothing(tmp_path: Path) -> None:
    did, said = act_on_directive(tmp_path, "/standing") or (True, "")

    assert not did
    assert "/standing needs the goal" in said
    assert take_standing_goal(tmp_path) is None


def test_the_directive_writes_the_goal(tmp_path: Path) -> None:
    did, said = act_on_directive(tmp_path, "/standing keep the suite green") or (False, "")

    assert did and "standing goal set" in said
    assert take_standing_goal(tmp_path) == "keep the suite green"


def test_agent6_steer_takes_it_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / ".state"))
    monkeypatch.chdir(tmp_path)
    d = state_dir(tmp_path) / "sessions" / "runs" / "tiny-run-AAAA11"
    d.mkdir(parents=True)
    (d / "logs.jsonl").write_text("", encoding="utf-8")
    write_worker_pid(d, os.getpid())

    assert main(["steer", "tiny-run", "/standing keep hunting defects"]) == 0

    assert "standing goal set" in capsys.readouterr().out
    assert take_standing_goal(d) == "keep hunting defects"


def _run(tmp_path: Path) -> tuple[GraphCurator, str, Workflow]:
    layout = SessionLayout(state_dir=tmp_path / ".agent6", session_id="run1")
    curator = GraphCurator(layout)
    root = curator.add_subtask(
        AddSubtaskIntent(parent_id=None, draft=TaskNodeDraft(title="the run", created_by="user"))
    ).id
    from agent6.events import EventSink

    return curator, root, _workflow(curator, EventSink(layout.session_dir / "logs.jsonl"))


def test_the_loop_adopts_the_goal(tmp_path: Path) -> None:
    curator, root, wf = _run(tmp_path)
    set_standing_goal(curator.layout.session_dir, "keep the suite green")

    wf._adopt_standing_goal(_state(root))  # pyright: ignore[reportPrivateUsage]

    standing = [n for n in curator.nodes().values() if n.standing]
    assert [(n.title, n.created_by) for n in standing] == [("keep the suite green", "steering")]
    assert standing[0].status == "pending"


def test_a_new_goal_retires_the_old_one(tmp_path: Path) -> None:
    """Retired, not made ordinary: a goal reads as an activity, and an ordinary
    task of that shape is worked once and marked passed."""
    curator, root, wf = _run(tmp_path)
    for goal in ("first goal", "second goal"):
        set_standing_goal(curator.layout.session_dir, goal)
        wf._adopt_standing_goal(_state(root))  # pyright: ignore[reportPrivateUsage]

    by_title = {n.title: n for n in curator.nodes().values()}
    assert by_title["first goal"].status == "obsolete"
    assert by_title["second goal"].standing and by_title["second goal"].status == "pending"
    # The retired goal keeps its flag and its place in the tree; "the run's
    # standing goal" is the live one.
    live = [n.title for n in curator.nodes().values() if n.standing and n.status == "pending"]
    assert live == ["second goal"]


def test_no_goal_waiting_changes_nothing(tmp_path: Path) -> None:
    curator, root, wf = _run(tmp_path)

    wf._adopt_standing_goal(_state(root))  # pyright: ignore[reportPrivateUsage]

    assert not [n for n in curator.nodes().values() if n.standing]
