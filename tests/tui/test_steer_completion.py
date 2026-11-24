# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Composer completion is honest to what the composer parses: commands and `@path` files."""

from __future__ import annotations

import asyncio
import json
import os
import pathlib
import subprocess
import warnings

import pytest

from agent6 import directive
from agent6.ui.tui import app as tui_app
from agent6.ui.tui import composer


def test_each_composer_offers_its_registered_rows() -> None:
    for mode in ("start", "plan/ask", "steer", "resume"):
        assert composer.steer_suggestion_rows("/", mode=mode) == [
            (c.name, c.help) for c in directive.commands_for(mode)
        ]
    assert not composer.steer_suggestion_rows("/", mode="draft")


def test_rows_match_the_typed_prefix() -> None:
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
        "/review",
        "/undo",
        "/shells",
    ]
    assert composer.complete_steer("/c", mode="resume") is None  # Tab keeps its focus-move meaning
    assert composer.complete_steer("/c", mode="steer") == "/compact "


def test_a_start_offers_parallel_and_review() -> None:
    assert [c for c, _ in composer.steer_suggestion_rows("/", mode="start")] == [
        "/parallel",
        "/review",
    ]
    assert composer.complete_steer("/p", mode="start") == "/parallel "
    assert composer.complete_steer("/rev", mode="start") == "/review "


def test_tab_completes_unique_and_stalls_ambiguous() -> None:
    assert composer.complete_steer("/pa", mode="steer") == "/parallel "
    assert composer.complete_steer("/pin", mode="steer") == "/pin "
    # Ambiguous with no common-prefix progress: consumed but unchanged, so Tab never yanks focus.
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)  # os.path.commonprefix warns on 3.15
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
            assert not sug.display  # a space follows the word: hints gone

    asyncio.run(scenario())


def test_unsent_parallel_shows_each_task_in_the_conversation(tmp_path: pathlib.Path) -> None:
    """A spec repeats one task. A second directive starts another task."""
    run = tmp_path / "run"
    run.mkdir()
    (run / "logs.jsonl").write_text('{"type":"session.start","mode":"run","user_task":"t"}\n')
    (run / "worker.pid").write_text(str(os.getpid()))

    async def scenario() -> None:
        app = tui_app.Agent6TUI(run)
        async with app.run_test(size=(100, 35)) as pilot:
            await pilot.pause()
            bar = app.screen.query_one("#conv-input", composer.SteerInput)
            bar.load_text("/parallel 2 -- fix parser /parallel p/m -- test CLI")
            await pilot.pause()
            hint = app.screen.query_one("#conv-suggest", composer.SteerSuggest)
            rendered = "\n".join(hint.render_line(y).text for y in range(hint.size.height))
            assert "3 children for 2 tasks" in rendered, rendered
            assert "2 attempts (2): fix parser" in rendered
            assert "1 attempt (p/m): test CLI" in rendered
            bar.load_text("plain instruction")
            await pilot.pause()
            assert not hint.display

    asyncio.run(scenario())


def test_an_at_reference_lists_files_and_tab_accepts_one(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tab extends an ambiguous fragment and replaces a unique one with the plain path."""
    repo = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    for name in ("notes.md", "src/main.py", "src/model.py"):
        (repo / name).parent.mkdir(exist_ok=True)
        (repo / name).write_text("", encoding="utf-8")
    monkeypatch.chdir(repo)
    run = tmp_path / "run"
    run.mkdir()
    (run / "logs.jsonl").write_text('{"type":"session.start","mode":"run","user_task":"t"}\n')
    (run / "worker.pid").write_text(str(os.getpid()))

    async def scenario() -> None:
        app = tui_app.Agent6TUI(run)
        async with app.run_test(size=(100, 35)) as pilot:
            await pilot.pause()
            bar = app.screen.query_one("#conv-input", composer.SteerInput)
            hint = app.screen.query_one("#conv-suggest", composer.SteerSuggest)
            bar.insert("read @src")
            await pilot.pause()
            assert hint.display
            assert "files: src/main.py  src/model.py" in str(hint.render())
            await pilot.press("tab")  # two matches: Tab keeps what they share
            assert bar.text == "read @src/m"
            await pilot.press("a", "tab")
            await pilot.pause()
            assert bar.text == "read src/main.py "
            assert not hint.display
            bar.insert("and mail a@no")  # an address is no reference
            await pilot.pause()
            assert not hint.display

    asyncio.run(scenario())
