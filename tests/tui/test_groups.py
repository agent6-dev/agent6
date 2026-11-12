from __future__ import annotations

import asyncio
import pathlib

from textual import widgets

from agent6 import paths
from agent6.sessions import layout
from agent6.ui.tui import app, dashboard
from tests.unit import test_sessions_show_group

group = test_sessions_show_group.group


def test_dashboard_children_refresh_and_open(group: pathlib.Path) -> None:
    asyncio.run(_scenario(group))


async def _scenario(repo: pathlib.Path) -> None:
    parent = layout.SessionLayout(state_dir=paths.state_dir(repo), session_id="fan")
    child = layout.SessionLayout(state_dir=paths.state_dir(repo), session_id="fan-l1")
    tui = app.Agent6TUI(parent.session_dir)
    async with tui.run_test(size=(140, 45)) as pilot:
        tui.action_toggle_dashboard()
        await pilot.pause()
        table = tui.screen.query_one("#children", widgets.DataTable)
        assert table.display and table.row_count == 2
        assert table.border_title == "children: 2"
        assert str(table.get_row_at(0)[1]) == "openai/child-1"
        with child.logs_path.open("a") as log:
            log.write('{"type":"budget.update","usd_total":1.25}\n')
            log.write('{"type":"session.stop","reason":"steer_abort"}\n')
        await pilot.pause(1.2)
        assert str(table.get_row_at(0)[3]) == "$1.25"
        assert "stopped" in str(table.get_row_at(0)[2])
        assert isinstance(tui.screen, dashboard.DashboardScreen)
        table.focus()
        await pilot.press("enter")
    assert tui.return_value == app.TuiExit(open_next=child.session_dir)


ordinary_parent = test_sessions_show_group.ordinary_parent


def test_dashboard_shows_a_child_that_appears_after_the_view_opens(
    tmp_path: pathlib.Path, monkeypatch: object
) -> None:
    asyncio.run(_ordinary_scenario(tmp_path))


async def _ordinary_scenario(tmp_path: pathlib.Path) -> None:
    """An ordinary live run has no children at mount; its first child appears while it waits."""
    import json
    import os

    from agent6.sessions import process

    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    os.environ["XDG_STATE_HOME"] = str(tmp_path / "state")
    os.chdir(repo)
    state = paths.state_dir(repo)
    parent = layout.SessionLayout(state_dir=state, session_id="solo")
    parent.ensure()
    parent.manifest_path.write_text(
        json.dumps({"version": 3, "session_id": "solo", "mode": "run"}), encoding="utf-8"
    )
    parent.logs_path.write_text(
        json.dumps({"type": "session.start", "mode": "run", "user_task": "t"}) + "\n",
        encoding="utf-8",
    )
    process.write_worker_pid(parent.session_dir, os.getpid())  # live, silent
    tui = app.Agent6TUI(parent.session_dir)
    async with tui.run_test(size=(140, 45)) as pilot:
        tui.action_toggle_dashboard()
        await pilot.pause()
        table = tui.screen.query_one("#children", widgets.DataTable)
        assert not table.display
        child = layout.SessionLayout(state_dir=state, session_id="solo-p1-l1")
        child.ensure()
        child.manifest_path.write_text(
            json.dumps(
                {
                    "version": 3,
                    "session_id": "solo-p1-l1",
                    "mode": "run",
                    "models": {"driver": {"provider": "openai", "model": "child-1"}},
                    "parallel": {"group": "solo-p1", "number": 1, "parent": "solo"},
                }
            ),
            encoding="utf-8",
        )
        child.logs_path.write_text(
            json.dumps({"type": "session.start", "mode": "run", "user_task": "audit"}) + "\n",
            encoding="utf-8",
        )
        await pilot.pause(1.3)
        assert table.display and table.row_count == 1
        assert table.border_title == "children: 1"
        assert str(table.get_row_at(0)[1]) == "openai/child-1"
        with child.logs_path.open("a") as log:
            log.write('{"type":"budget.update","usd_total":0.75}\n')
            log.write('{"type":"session.stop","reason":"finish_session"}\n')
        await pilot.pause(1.3)
        assert str(table.get_row_at(0)[3]) == "$0.75"
        table.focus()
        await pilot.press("enter")
    assert tui.return_value == app.TuiExit(open_next=child.session_dir)
