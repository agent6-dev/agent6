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

_TRANSCRIPTS = [
    {
        "seq": 1,
        "request": {
            "body": {
                "messages": [
                    {"role": "system", "content": "SYSTEM"},
                    {"role": "user", "content": "do X"},
                ]
            }
        },
        "response": {
            "body": {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "on it",
                            "reasoning_content": "thinking hard here",
                            "tool_calls": [
                                {
                                    "id": "c1",
                                    "function": {"name": "read_file", "arguments": '{"path":"a"}'},
                                }
                            ],
                        }
                    }
                ]
            }
        },
    },
    {
        "seq": 2,
        "request": {
            "body": {
                "messages": [
                    {"role": "system", "content": "SYSTEM"},
                    {"role": "user", "content": "do X"},
                    {
                        "role": "assistant",
                        "content": "on it",
                        "tool_calls": [
                            {
                                "id": "c1",
                                "function": {"name": "read_file", "arguments": '{"path":"a"}'},
                            }
                        ],
                    },
                    {"role": "tool", "content": "FILE BODY", "tool_call_id": "c1"},
                ]
            }
        },
        "response": {"body": {"choices": [{"message": {"role": "assistant", "content": "done"}}]}},
    },
]


class _Host(App[None]):
    def __init__(self, transcripts_dir: Path) -> None:
        super().__init__()
        self._dir = transcripts_dir

    def on_mount(self) -> None:
        self.push_screen(ConversationScreen(self._dir, title="conversation · test"))


def _write(tdir: Path) -> None:
    tdir.mkdir(parents=True)
    for i, t in enumerate(_TRANSCRIPTS, 1):
        (tdir / f"2026-00000{i}.json").write_text(json.dumps(t), encoding="utf-8")


def test_conversation_screen_renders_and_toggles_thinking(tmp_path: Path) -> None:
    tdir = tmp_path / "transcripts"
    _write(tdir)

    async def scenario() -> None:
        app = _Host(tdir)
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


def test_conversation_screen_empty(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = _Host(tmp_path / "missing")  # no transcripts dir
        async with app.run_test() as pilot:
            await pilot.pause()
            body = app.screen.query_one("#conv-body", RichLog)
            assert len(body.lines) == 1  # the "(no transcripts yet …)" placeholder

    asyncio.run(scenario())


def test_conversation_screen_q_backs_out(tmp_path: Path) -> None:
    """q (like Esc) closes the pager -- backs out one level (Option 3)."""
    tdir = tmp_path / "transcripts"
    _write(tdir)

    async def scenario() -> None:
        app = _Host(tdir)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert isinstance(app.screen, ConversationScreen)
            await pilot.press("q")
            await pilot.pause()
            assert not isinstance(app.screen, ConversationScreen)  # backed out

    asyncio.run(scenario())
