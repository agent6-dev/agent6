# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Headless drive of the TUI conversation viewer (ConversationScreen)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from textual.app import App
from textual.widgets import RichLog

from agent6.tui.conversation import ConversationScreen

_EVENTS: list[dict[str, object]] = [
    {"type": "run.start", "user_task": "do X"},
    {"type": "role.call", "role": "worker"},
    {"type": "role.thinking_delta", "role": "worker", "text": "thinking hard here"},
    {"type": "role.text_delta", "role": "worker", "text": "on it"},
    {"type": "role.result", "role": "worker"},
    {"type": "tool.call", "name": "read_file", "args": {"path": "a"}},
    {"type": "tool.result", "name": "read_file", "ok": True, "summary": "12 bytes"},
    {"type": "run.end", "all_passed": True, "reason": "finish_run"},
]


class _Host(App[None]):
    def __init__(self, logs_path: Path) -> None:
        super().__init__()
        self._logs = logs_path

    def on_mount(self) -> None:
        self.push_screen(ConversationScreen(self._logs, title="conversation · test"))


def _write(logs: Path, events: list[dict[str, object]]) -> None:
    logs.write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")


def test_conversation_screen_renders_and_toggles_thinking(tmp_path: Path) -> None:
    logs = tmp_path / "logs.jsonl"
    _write(logs, _EVENTS)

    async def scenario() -> None:
        app = _Host(logs)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ConversationScreen)
            body = screen.query_one("#conv-body", RichLog)
            with_thinking = len(body.lines)
            assert with_thinking > 0  # the conversation rendered
            screen.action_toggle_thinking()  # hide the thinking block
            await pilot.pause()
            assert len(body.lines) < with_thinking  # fewer lines without thinking
            screen.action_reload()  # reload must not raise
            await pilot.pause()

    asyncio.run(scenario())


def test_conversation_screen_follows_live(tmp_path: Path) -> None:
    """Events appended after mount (a live run / a resume) show up via the poll."""
    logs = tmp_path / "logs.jsonl"
    logs.write_text("", encoding="utf-8")

    async def scenario() -> None:
        app = _Host(logs)
        async with app.run_test() as pilot:
            await pilot.pause()
            body = app.screen.query_one("#conv-body", RichLog)
            before = len(body.lines)
            with logs.open("a", encoding="utf-8") as fh:
                for event in _EVENTS:
                    fh.write(json.dumps(event) + "\n")
            await asyncio.sleep(0.7)  # let the 0.5s follow poll fire
            await pilot.pause()
            assert len(body.lines) > before  # the appended turns appeared

    asyncio.run(scenario())


def test_conversation_screen_empty(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = _Host(tmp_path / "missing.jsonl")  # no log file
        async with app.run_test() as pilot:
            await pilot.pause()
            body = app.screen.query_one("#conv-body", RichLog)
            assert len(body.lines) == 1  # the "(no conversation yet …)" placeholder

    asyncio.run(scenario())


def test_conversation_screen_q_backs_out(tmp_path: Path) -> None:
    """q (like Esc) closes the pager -- backs out one level."""
    logs = tmp_path / "logs.jsonl"
    _write(logs, _EVENTS)

    async def scenario() -> None:
        app = _Host(logs)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert isinstance(app.screen, ConversationScreen)
            await pilot.press("q")
            await pilot.pause()
            assert not isinstance(app.screen, ConversationScreen)  # backed out

    asyncio.run(scenario())
