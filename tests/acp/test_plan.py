# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The run's task list reaches the editor as ACP's plan, live and on replay."""

from __future__ import annotations

import io
import pathlib
from typing import Any

from agent6.sessions import event_log, journal
from agent6.ui.acp import runner, updates
from agent6.ui.acp import server as acp_server
from agent6.ui.acp import session as session_mod
from agent6.viewmodel import tail
from tests.acp import schema_check


def _node(
    title: str, status: str = "pending", parent: str | None = None, children: tuple[str, ...] = ()
) -> dict[str, Any]:
    """Return one task as `graph.update` journals it."""
    return {
        "title": title,
        "status": status,
        "parent_id": parent,
        "children": list(children),
        "created_by": "worker",
        "standing": False,
    }


def _graph(nodes: dict[str, Any], cursor: str | None = None) -> dict[str, Any]:
    return {"type": "graph.update", "nodes": nodes, "cursor": cursor}


_STOP = {"type": "session.stop", "reason": "finish_session", "verification": "passed"}


def _entries(sent: list[dict[str, Any]]) -> list[list[tuple[str, str]]]:
    """Return each plan update's (content, status) rows, every frame checked against the schema."""
    plans: list[list[tuple[str, str]]] = []
    for body in sent:
        assert not schema_check.errors("SessionNotification", body["params"]), body
        update = body["params"]["update"]
        if update["sessionUpdate"] == "plan":
            assert all(entry["priority"] == "medium" for entry in update["entries"])
            plans.append([(entry["content"], entry["status"]) for entry in update["entries"]])
    return plans


def _fed(*events: dict[str, Any]) -> list[list[tuple[str, str]]]:
    plan = updates.TaskPlan("s")
    return _entries([body for event in events for body in plan.feed(event)])


def test_each_task_status_maps_to_a_plan_status_in_depth_first_order() -> None:
    """ACP has no failed, skipped or obsolete status: a failed task reads pending, marked."""
    nodes = {
        "0001": _node("Ship it", "in_progress", children=("0002", "0005", "0003")),
        "0002": _node("Read the code", "passed", "0001", children=("0004",)),
        "0003": _node("Write the fix", "pending", "0001"),
        "0004": _node("Find the caller", "failed", "0002"),
        "0005": _node("Run the bench", "in_progress", "0001"),
        "0006": _node("Dropped idea", "skipped", "0001"),
        "0007": _node("Old plan", "obsolete", "0001"),
    }

    assert _fed(_graph(nodes, cursor="0003")) == [
        [
            ("Ship it", "in_progress"),
            ("Read the code", "completed"),
            ("Find the caller (failed)", "pending"),
            ("Run the bench", "in_progress"),
            # The cursor's task is the one the worker is on, whatever its recorded status.
            ("Write the fix", "in_progress"),
        ]
    ]


def test_a_plan_update_replaces_the_list_and_can_be_empty() -> None:
    """Every update is the complete list; a graph with nothing left to show sends none."""
    one = {"0001": _node("First")}
    two = {"0001": _node("First", "passed"), "0002": _node("Second")}
    none = {"0001": _node("First", "obsolete"), "0002": _node("Second", "skipped")}

    assert _fed(_graph(one), _graph(two), _graph(two), _graph(none)) == [
        [("First", "pending")],
        [("First", "completed"), ("Second", "pending")],
        [],
    ]


def test_a_stop_clears_the_focus_from_the_plan() -> None:
    """The worker is on no task after the end, so the cursor's task reads its own status."""
    nodes = {
        "0001": _node("Root", "in_progress", children=("0002",)),
        "0002": _node("Next", parent="0001"),
    }

    assert _fed(_graph(nodes, cursor="0002"), _STOP) == [
        [("Root", "in_progress"), ("Next", "in_progress")],
        [("Root", "in_progress"), ("Next", "pending")],
    ]
    # An execution that ended before it journaled a graph leaves the editor's plan alone.
    assert _fed(_STOP) == []


def test_a_task_title_is_scrubbed_like_every_model_written_string() -> None:
    assert _fed(_graph({"0001": _node("Fix\x1b[2J the\x07 parser")})) == [
        [("Fix[2J the parser", "pending")]
    ]


def test_the_plan_is_sent_live_and_again_on_replay(tmp_path: pathlib.Path) -> None:
    """`session/load` rebuilds the plan from the journal the live tail read."""
    log = tmp_path / "logs.jsonl"
    sink = event_log.EventSink(log)
    sink.emit(journal.SessionStart(mode="run", user_task="t", execution_id="one"))
    sink.emit(journal.GraphUpdate(nodes={"0001": _node("Root", "in_progress")}, cursor="0001"))
    sink.emit(
        journal.GraphUpdate(
            nodes={
                "0001": _node("Root", "in_progress", children=("0002",)),
                "0002": _node("Step", parent="0001"),
            },
            cursor="0002",
        )
    )
    sink.emit(
        journal.SessionStop(
            execution_id="one", reason="finish_session", turns=1, verification="passed"
        )
    )

    live: list[dict[str, Any]] = []
    server = acp_server.ACPServer(stdin=io.BytesIO(), stdout=io.BytesIO())
    server.notify_raw = live.append  # pyrefly: ignore[bad-assignment, method-assign]
    session = session_mod.Session(acp_id="run-x", cwd=tmp_path, session_id="run-x")
    runner.RunBridge(server=server)._stream(session, log, lambda: True, 0, runner.Announced())
    replayed: list[dict[str, Any]] = []
    updates.replay(
        tail.tail_events(log, follow=False),
        acp_session_id="run-x",
        cwd=tmp_path,
        objects_dir=tmp_path / "objects",
        notify=replayed.append,
    )

    assert _entries(live) == [
        [("Root", "in_progress")],
        [("Root", "in_progress"), ("Step", "in_progress")],
        [("Root", "in_progress"), ("Step", "pending")],
    ]
    assert _entries(replayed) == _entries(live)
