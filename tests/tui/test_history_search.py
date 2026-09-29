# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Ctrl-R searches the session's past messages into the composer on both run views."""

from __future__ import annotations

import asyncio
import json
import pathlib

from agent6.ui.tui import app as tui_app
from agent6.ui.tui import composer, modals


def _run_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    run = tmp_path / "run"
    run.mkdir()
    (run / "logs.jsonl").write_text(
        "".join(
            json.dumps(e) + "\n"
            for e in (
                {"type": "session.start", "mode": "run", "user_task": "polish the\nTUI"},
                {"type": "loop.steer.injected", "chars": 14, "text": "focus on tests"},
                {"type": "loop.steer.injected", "chars": 12, "text": "fix the docs"},
            )
        ),
        encoding="utf-8",
    )
    return run


def test_ctrl_r_fills_the_conversation_composer(tmp_path: pathlib.Path) -> None:
    """Ctrl-R opens the picker; typing narrows, ↓ highlights, Enter puts the pick in the composer.

    The task is included, newlines flattened, and nothing is sent.
    """

    async def scenario() -> None:
        app = tui_app.Agent6TUI(_run_dir(tmp_path))
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("ctrl+r")
            await pilot.pause()
            assert isinstance(app.screen, modals.HistorySearchModal)
            for c in "poli":
                await pilot.press(c)
            await pilot.press("down")  # highlight the one match
            await pilot.press("enter")
            await pilot.pause()
            field = app._conv.query_one("#conv-input", composer.SteerInput)
            assert field.text == "polish the TUI"

    asyncio.run(scenario())


def test_ctrl_r_works_from_the_dashboard_and_esc_cancels(tmp_path: pathlib.Path) -> None:
    async def scenario() -> None:
        app = tui_app.Agent6TUI(_run_dir(tmp_path))
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("ctrl+d")  # over to the dashboard view
            await pilot.pause()
            await pilot.press("ctrl+r")
            await pilot.pause()
            assert isinstance(app.screen, modals.HistorySearchModal)
            await pilot.press("escape")
            await pilot.pause()
            assert not isinstance(app.screen, modals.HistorySearchModal)
            assert app.screen.query_one("#dash-input", composer.SteerInput).text == ""

    asyncio.run(scenario())


def test_ctrl_r_with_no_recorded_messages_opens_nothing(tmp_path: pathlib.Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "logs.jsonl").write_text(
        json.dumps({"type": "tool.call", "name": "read_file", "args": {}}) + "\n",
        encoding="utf-8",
    )

    async def scenario() -> None:
        app = tui_app.Agent6TUI(run)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("ctrl+r")
            await pilot.pause()
            assert not isinstance(app.screen, modals.HistorySearchModal)

    asyncio.run(scenario())
