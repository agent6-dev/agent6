# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Ctrl-Z detaches the session TUI (the run keeps going); Ctrl-_ is text undo."""

from __future__ import annotations

import asyncio
import json
import pathlib

from textual import app as textual_app

from agent6.sessions import layout
from agent6.ui.tui import app as tui_app
from agent6.ui.tui import composer


def _session_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    d = tmp_path / "run-x"
    d.mkdir()
    events = [
        {"type": "session.start", "user_task": "t", "ts": 1.0},
        {"type": "session.end", "reason": "finish_session", "all_passed": True, "ts": 2.0},
    ]
    (d / layout.LOGS_NAME).write_text(
        "".join(json.dumps(e) + "\n" for e in events), encoding="utf-8"
    )
    return d


def test_ctrl_z_detaches_and_says_so(tmp_path: pathlib.Path) -> None:
    """Ctrl-Z exits the app with `detached` set, even with the composer's undo binding focused."""
    app = tui_app.Agent6TUI(_session_dir(tmp_path))

    async def drive() -> None:
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("ctrl+z")
            await pilot.pause()

    asyncio.run(drive())
    assert app.detached is True


def test_ctrl_underscore_undoes_composer_typing(tmp_path: pathlib.Path) -> None:
    class _Harness(textual_app.App[None]):
        def compose(self) -> textual_app.ComposeResult:
            yield composer.SteerInput(id="conv-input")

    app = _Harness()

    async def drive() -> None:
        async with app.run_test() as pilot:
            await pilot.pause()
            box = app.query_one("#conv-input", composer.SteerInput)
            box.focus()
            await pilot.pause()
            box.insert("hello")
            await pilot.pause()
            await pilot.press("ctrl+underscore")
            await pilot.pause()
            assert box.text == ""

    asyncio.run(drive())
