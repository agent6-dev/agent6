# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Composer slash completion is honest to what the composer parses."""

from __future__ import annotations

import asyncio
import json
import os
import pathlib

from agent6.ui.tui import app as tui_app
from agent6.ui.tui import composer


def test_rows_match_the_typed_prefix() -> None:
    assert [c for c, _ in composer.steer_suggestion_rows("/", mode="steer")] == [
        "/pin",
        "/compact",
        "/parallel",
        "/restate",
        "/undo",
        "/btw",
        "/task",
        "/standing",
        "/retire",
        "/now",
        "/stop",
        "/shells",
    ]
    assert [c for c, _ in composer.steer_suggestion_rows("/p", mode="steer")] == [
        "/pin",
        "/parallel",
    ]
    assert composer.steer_suggestion_rows("fix it", mode="steer") == []
    assert (
        composer.steer_suggestion_rows("/pin keep this", mode="steer") == []
    )  # args typed: hints gone


def test_compact_and_btw_are_live_only() -> None:
    assert [c for c, _ in composer.steer_suggestion_rows("/", mode="resume")] == [
        "/pin",
        "/parallel",
        "/restate",
        "/undo",
        "/shells",
    ]
    assert composer.complete_steer("/c", mode="resume") is None  # Tab keeps its focus-move meaning
    assert composer.complete_steer("/c", mode="steer") == "/compact "


def test_a_draft_offers_only_the_fan_out() -> None:
    assert [c for c, _ in composer.steer_suggestion_rows("/", mode="start")] == ["/parallel"]
    assert composer.complete_steer("/p", mode="start") == "/parallel "


def test_tab_completes_unique_and_stalls_ambiguous() -> None:
    assert composer.complete_steer("/pa", mode="steer") == "/parallel "
    assert composer.complete_steer("/pin", mode="steer") == "/pin "
    # Ambiguous with no common-prefix progress: consumed but unchanged, so Tab never yanks focus.
    assert composer.complete_steer("/p", mode="steer") == "/p"
    assert composer.complete_steer("q", mode="steer") is None


def test_typing_slash_shows_hints_and_tab_completes(tmp_path: pathlib.Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "logs.jsonl").write_text(
        json.dumps({"type": "session.start", "mode": "run", "user_task": "t"}) + "\n",
        encoding="utf-8",
    )
    (run / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")  # live

    async def scenario() -> None:
        app = tui_app.Agent6TUI(run)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("slash")
            await pilot.pause()
            sug = app.screen.query_one("#conv-suggest", composer.SteerSuggest)
            assert sug.display is True
            shown = str(sug.render())
            assert "/parallel" in shown and "/compact" in shown
            await pilot.press("p", "i", "tab")
            await pilot.pause()
            assert app.screen.query_one("#conv-input", composer.SteerInput).text == "/pin "
            assert sug.display is False  # a space follows the word: hints gone

    asyncio.run(scenario())
