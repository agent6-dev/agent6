# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`/shells` on the TUI composer shows the roster every surface reads off disk."""

from __future__ import annotations

import asyncio
import json
import pathlib

from textual import widgets

from agent6.ui.tui import app as tui_app
from agent6.ui.tui import modals


def _mk(d: pathlib.Path) -> None:
    d.mkdir(parents=True)
    evs = [
        {"type": "session.start", "session_id": d.name, "mode": "run", "user_task": "t"},
        {"type": "session.end", "reason": "finish_session", "all_passed": True},
    ]
    (d / "logs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in evs), encoding="utf-8")
    shell = d / "shells" / "bg-1"
    shell.mkdir(parents=True)
    (shell / "meta.json").write_text(json.dumps({"command": "sleep 5"}), encoding="utf-8")
    (shell / "result.json").write_text(json.dumps({"returncode": 0}) + "\n", encoding="utf-8")


def test_shells_opens_the_roster_as_a_text_view(tmp_path: pathlib.Path) -> None:
    d = tmp_path / "s1"
    _mk(d)

    async def scenario() -> None:
        app = tui_app.Agent6TUI(d)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.submit_instruction("/shells")
            await pilot.pause()
            assert isinstance(app.screen, modals.TextModal)
            shown = app.screen.query_one("#text-view", widgets.TextArea).text
            assert "[bg-1] exited 0: sleep 5" in shown

    asyncio.run(scenario())
