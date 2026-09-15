# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The footer on a narrow terminal: its key hints clip at the right edge, the
menus keep the rest; the app-wide 1-cell scrollbar rule used to give the
1-row footer a horizontal scrollbar that replaced every hint at 80 columns."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from textual.widgets import DataTable, Footer

from agent6.ui.tui.app import Agent6TUI
from agent6.ui.tui.home import Agent6HomeApp, HomeScreen


def test_the_footer_clips_instead_of_scrolling_at_80_columns(tmp_path: Path) -> None:
    run = tmp_path / "sessions" / "runs" / "narrow-run-AAAAAA"
    run.mkdir(parents=True)
    (run / "logs.jsonl").write_text(
        json.dumps(
            {"type": "session.start", "session_id": run.name, "mode": "run", "user_task": "t"}
        )
        + "\n",
        encoding="utf-8",
    )

    async def scenario() -> list[tuple[str, int, bool]]:
        seen: list[tuple[str, int, bool]] = []
        hub = Agent6HomeApp(tmp_path, tmp_path)
        async with hub.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            f = hub.screen.query_one(Footer)
            seen.append(("hub", f.styles.scrollbar_size_horizontal, f.virtual_size.width > 80))
        view = Agent6TUI(run)
        async with view.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            f = view.screen.query_one(Footer)
            seen.append(("run", f.styles.scrollbar_size_horizontal, f.virtual_size.width > 80))
        return seen

    for name, scrollbar, overflows in asyncio.run(scenario()):
        assert overflows, f"{name}: the footer fits in 80 columns; the rule is moot"
        assert scrollbar == 0, f"{name}: the footer's scrollbar would replace its hints"


def _hub_with_a_fan_out(a6: Path) -> None:
    start = {"type": "session.start", "mode": "run", "user_task": "t"}
    for name, manifest in (
        ("plain-run-AAAAAA", {"mode": "run"}),
        ("fan", {"mode": "run", "fanout": {"lanes": 1, "spec": "1"}}),
        ("fan-l1", {"mode": "run", "parallel": {"group": "fan", "lane": 1, "coordinator": "fan"}}),
    ):
        d = a6 / "sessions" / "runs" / name
        d.mkdir(parents=True)
        (d / "logs.jsonl").write_text(json.dumps(start) + "\n", encoding="utf-8")
        (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_the_hub_footer_shows_lanes_only_on_a_fan_out_and_keeps_logs_and_refresh_off_it(
    tmp_path: Path,
) -> None:
    """Lanes sat dimmed in the footer of a hub with nothing to fold, and View
    logs and Refresh took the room the row actions need at 100 columns; both
    stay in the menus."""
    a6 = tmp_path / ".agent6"
    _hub_with_a_fan_out(a6)

    async def scenario() -> None:
        hub = Agent6HomeApp(a6, tmp_path)
        async with hub.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            screen = hub.screen
            assert isinstance(screen, HomeScreen)
            table = screen.query_one("#sessions", DataTable)

            async def select(name: str) -> None:
                runs = screen._runs  # pyright: ignore[reportPrivateUsage]
                table.move_cursor(row=[rd.name for rd in runs].index(name))
                await pilot.pause()
                await pilot.pause()

            await select("plain-run-AAAAAA")
            keys = screen.active_bindings
            assert "space" not in keys
            assert keys["l"].binding.show is False and keys["r"].binding.show is False
            await select("fan")
            keys = screen.active_bindings
            assert "space" in keys and keys["space"].enabled

    asyncio.run(scenario())
