# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The footer on a narrow terminal: its key hints clip at the right edge, the
menus keep the rest; the app-wide 1-cell scrollbar rule used to give the
1-row footer a horizontal scrollbar that replaced every hint at 80 columns."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from textual.widgets import Footer

from agent6.ui.tui.app import Agent6TUI
from agent6.ui.tui.home import Agent6HomeApp


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
