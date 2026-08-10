# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Composer slash completion: the hint line and Tab, honest to what the
composer parses (/pin always, /compact live-only, /parallel always)."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from agent6.ui.tui.app import Agent6TUI
from agent6.ui.tui.conversation import (
    SteerInput,
    SteerSuggest,
    complete_steer,
    steer_suggestion_rows,
)


def test_rows_match_the_typed_prefix() -> None:
    assert [c for c, _ in steer_suggestion_rows("/", live=True)] == [
        "/pin",
        "/compact",
        "/parallel",
        "/restate",
        "/undo",
    ]
    assert [c for c, _ in steer_suggestion_rows("/p", live=True)] == ["/pin", "/parallel"]
    assert steer_suggestion_rows("fix it", live=True) == []
    assert steer_suggestion_rows("/pin keep this", live=True) == []  # args typed: hints gone


def test_compact_is_live_only() -> None:
    assert [c for c, _ in steer_suggestion_rows("/", live=False)] == [
        "/pin",
        "/parallel",
        "/restate",
        "/undo",
    ]
    assert complete_steer("/c", live=False) is None  # Tab keeps its focus-move meaning
    assert complete_steer("/c", live=True) == "/compact "


def test_tab_completes_unique_and_stalls_ambiguous() -> None:
    assert complete_steer("/pa", live=True) == "/parallel "
    assert complete_steer("/pin", live=True) == "/pin "
    # Ambiguous with no common-prefix progress: consumed but unchanged, so Tab
    # never yanks focus away mid-command.
    assert complete_steer("/p", live=True) == "/p"
    assert complete_steer("q", live=True) is None


def test_typing_slash_shows_hints_and_tab_completes(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "logs.jsonl").write_text(
        json.dumps({"type": "session.start", "mode": "run", "user_task": "t"}) + "\n",
        encoding="utf-8",
    )
    (run / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")  # live

    async def scenario() -> None:
        app = Agent6TUI(run)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("slash")
            await pilot.pause()
            sug = app.screen.query_one("#conv-suggest", SteerSuggest)
            assert sug.display is True
            shown = str(sug.render())
            assert "/parallel" in shown and "/compact" in shown
            await pilot.press("p", "i", "tab")
            await pilot.pause()
            assert app.screen.query_one("#conv-input", SteerInput).text == "/pin "
            assert sug.display is False  # a space follows the word: hints gone

    asyncio.run(scenario())
