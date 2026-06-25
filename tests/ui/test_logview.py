# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Headless drive of the scrollable run-log viewer (LogScreen)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from textual.app import App
from textual.widgets import RichLog

from agent6.ui.logview import LogScreen
from agent6.ui.state import format_log_line


class _Host(App[None]):
    def __init__(self, logs_path: Path) -> None:
        super().__init__()
        self._logs = logs_path

    def on_mount(self) -> None:
        self.push_screen(LogScreen(self._logs, title="logs · test"))


def _write_log(path: Path, events: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")


def test_logscreen_renders_every_event(tmp_path: Path) -> None:
    logs = tmp_path / "logs.jsonl"
    events: list[dict[str, object]] = [
        {"type": "run.start", "mode": "run", "user_task": "do x", "ts": "2026-06-22T01:02:03.4Z"},
        {"type": "tool.call", "name": "read_file", "args": {"path": "a.py"}, "ts": "t"},
        {"type": "verify.end", "exit_code": 0, "duration_s": 1.2, "ts": "t"},
        {"type": "run.end", "all_passed": True, "ts": "t"},
    ]
    _write_log(logs, events)

    async def scenario() -> None:
        app = _Host(logs)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, LogScreen)
            body = screen.query_one("#logview-body", RichLog)
            # One rendered line per event (no wrap) -> the whole history is present.
            assert len(body.lines) == len(events)

    asyncio.run(scenario())


def test_logscreen_reload_picks_up_appended_lines(tmp_path: Path) -> None:
    logs = tmp_path / "logs.jsonl"
    _write_log(logs, [{"type": "run.start", "mode": "run", "user_task": "x", "ts": "t"}])

    async def scenario() -> None:
        app = _Host(logs)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, LogScreen)
            body = screen.query_one("#logview-body", RichLog)
            assert len(body.lines) == 1
            # A live run keeps appending; reload pulls the new lines in.
            with logs.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"type": "run.end", "all_passed": True, "ts": "t"}) + "\n")
            screen.action_reload()
            await pilot.pause()
            assert len(body.lines) == 2

    asyncio.run(scenario())


def test_logscreen_empty_log(tmp_path: Path) -> None:
    logs = tmp_path / "logs.jsonl"  # does not exist

    async def scenario() -> None:
        app = _Host(logs)
        async with app.run_test() as pilot:
            await pilot.pause()
            body = app.screen.query_one("#logview-body", RichLog)
            assert len(body.lines) == 1  # the "(no events yet …)" placeholder

    asyncio.run(scenario())


def test_format_log_line_is_public_and_compact() -> None:
    line = format_log_line({"type": "tool.call", "name": "grep", "args": {}, "ts": "t"})
    assert "tool.call" in line and "grep" in line
