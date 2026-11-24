# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Retire the task selected in the dashboard's live tree."""

from __future__ import annotations

import asyncio
import pathlib

from textual import widgets

from agent6.graph import storage
from agent6.sessions import event_log, ipc, journal
from agent6.ui.tui import app, modals
from tests import task_runs
from tests.tui import waits


def test_confirm_retirement_of_the_selected_task(tmp_path: pathlib.Path) -> None:
    """Confirming retires the selected open task and never queues a worker stop."""
    title = "Link \x1b[1mpages\x1b[0m"
    run = task_runs.live_graph(tmp_path, child_title=title)

    async def scenario() -> None:
        tui = app.Agent6TUI(run.session_dir)
        async with tui.run_test(size=(120, 40)) as pilot:
            await pilot.press("ctrl+d")
            dashboard = tui.screen
            tree = dashboard.query_one("#plan", widgets.Tree)
            tree.focus()
            await pilot.press("home", "down", "enter")
            await pilot.pause()
            assert tree.render_line(tree.cursor_line).text.rstrip().endswith("Link pages")
            assert tree.cursor_node is not None and tree.cursor_node.data == "0002"
            assert dashboard.query_one("#log").border_title == "log · task: Link pages"
            await pilot.press("r")
            assert isinstance(tui.screen, modals.ConfirmModal)
            question = tui.screen.query_one(widgets.Static).render_line(0).text.strip()
            assert question == "Retire task 2: Link pages?"
            assert tui.focused is tui.screen.query_one("#no")
            await pilot.press("escape")
            assert not ipc.drain_requests(run.session_dir)
            await pilot.press("r", "y")
            assert [(r.kind, r.text) for r in ipc.drain_requests(run.session_dir)] == [
                ("retire", "0002")
            ]
            assert not ipc.steer_request_pending(run.session_dir)
            assert not ipc.stop_request_pending(run.session_dir)
            await pilot.press("down", "enter", "r")
            assert tui.screen is dashboard

    asyncio.run(scenario())
    assert storage.load_graph(run)["0002"].title == title


def test_a_run_ending_during_confirmation_retires_nothing(tmp_path: pathlib.Path) -> None:
    """A confirmation from a live tree cannot retire a task after the run ends."""
    run = task_runs.live_graph(tmp_path)

    async def scenario() -> None:
        tui = app.Agent6TUI(run.session_dir)
        async with tui.run_test(size=(120, 40)) as pilot:
            await pilot.press("ctrl+d")
            tree = tui.screen.query_one("#plan", widgets.Tree)
            tree.focus()
            tree.move_cursor(tree.root.children[1])
            await pilot.press("enter", "r")
            assert isinstance(tui.screen, modals.ConfirmModal)
            event_log.EventSink(run.logs_path).emit(
                journal.SessionStop(execution_id="first", reason="finish_session", turns=1)
            )
            await waits.wait_for(pilot, lambda: tui.state.finished, "the journaled end")
            await pilot.press("y")
            assert not ipc.drain_requests(run.session_dir)
            assert not tui.screen.check_action("retire_task", ())

    asyncio.run(scenario())
