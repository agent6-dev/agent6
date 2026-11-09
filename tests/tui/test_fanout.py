from __future__ import annotations

import asyncio
import pathlib

from textual import widgets

from agent6 import paths
from agent6.sessions import layout
from agent6.ui.tui import app, dashboard
from tests.unit import test_sessions_show_fanout

fan_out = test_sessions_show_fanout.fan_out


def test_dashboard_lanes_refresh_and_open(fan_out: pathlib.Path) -> None:
    asyncio.run(_scenario(fan_out))


async def _scenario(fan_out: pathlib.Path) -> None:
    group = layout.SessionLayout(state_dir=paths.state_dir(fan_out), session_id="fan")
    lane = layout.SessionLayout(state_dir=paths.state_dir(fan_out), session_id="fan-l1")
    tui = app.Agent6TUI(group.session_dir)
    async with tui.run_test(size=(140, 45)) as pilot:
        tui.action_toggle_dashboard()
        await pilot.pause()
        table = tui.screen.query_one("#lanes", widgets.DataTable)
        assert table.display and table.row_count == 2
        assert table.border_title == "fan-out: 2 lanes"
        assert str(table.get_row_at(0)[1]) == "openai/lane-1"
        with lane.logs_path.open("a") as log:
            log.write('{"type":"budget.update","usd_total":1.25}\n')
            log.write('{"type":"session.stop","reason":"steer_abort"}\n')
        await pilot.pause(1.2)
        assert str(table.get_row_at(0)[3]) == "$1.25"
        assert "stopped" in str(table.get_row_at(0)[2])
        assert isinstance(tui.screen, dashboard.DashboardScreen)
        table.focus()
        await pilot.press("enter")
    assert tui.return_value == app.TuiExit(open_next=lane.session_dir)
