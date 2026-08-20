# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""/undo in the TUI: the fork is the continuation, and the message taken back is
the operator's to edit and resend. The fold's `undone_to` (a live /undo) and
this view's own `undo_fork` on a finished run both route the composer there."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from textual.widgets import Static

from agent6.ui.tui import app as app_mod
from agent6.ui.tui.app import Agent6TUI
from agent6.ui.tui.conversation import SteerInput


def _undone_run(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    evs = [
        {"type": "session.start", "session_id": d.name, "mode": "run", "user_task": "add it"},
        {"type": "loop.steer.injected", "chars": 14, "text": "name it better"},
        {
            "type": "session.undone",
            "new_session_id": "fork-child-AAAAAA",
            "undone_text": "name it better",
        },
        {"type": "session.end", "reason": "undone", "all_passed": False},
    ]
    (d / "logs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in evs), encoding="utf-8")


def test_a_live_undo_hands_the_follow_up_to_the_fork(tmp_path: Path, monkeypatch: Any) -> None:
    """The composer holds the undone text, its title names the fork, and Enter
    resumes the fork (this view's run is over; the fork carries on). The
    view used to keep an empty composer pointed at the undone run, whose
    resume would have started a leg on the wrong session."""
    spawned: list[tuple[str, str]] = []

    def _fake_resume(
        _cwd: Path, rid: str, *, steer: str = "", preset: str = "", config_path: object = None
    ) -> str:
        spawned.append((rid, steer))
        return ""

    monkeypatch.setattr(app_mod, "spawn_detached_resume", _fake_resume)
    run = tmp_path / "undone-run-AAAAAA"
    _undone_run(run)

    async def scenario() -> None:
        app = Agent6TUI(run)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            bar = app._conv.query_one("#conv-input", SteerInput)  # pyright: ignore[reportPrivateUsage]
            assert app.continue_as == "fork-child-AAAAAA"
            assert bar.text == "name it better"
            assert "continue as fork-child-AAAAAA" in str(bar.border_title)
            app.submit_instruction("name it much better")
            assert spawned == [("fork-child-AAAAAA", "name it much better")]
            # The dashboard's bar agrees.
            await pilot.press("ctrl+d")
            await pilot.pause()
            app._heartbeat_at = 0.0  # pyright: ignore[reportPrivateUsage]
            app._tick()  # pyright: ignore[reportPrivateUsage]
            await pilot.pause()
            dash_bar = app._dash.query_one("#dash-input", SteerInput)  # pyright: ignore[reportPrivateUsage]
            assert "continue as fork-child-AAAAAA" in str(dash_bar.border_title)
            assert isinstance(app._dash.query_one("#top", Static), Static)  # pyright: ignore[reportPrivateUsage]

    asyncio.run(scenario())


def test_undo_of_a_finished_run_fills_the_composer_and_routes_to_the_child(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """The non-live path forks in-process (no event lands in this run's log);
    the view still hands the follow-up to the child."""
    spawned: list[tuple[str, str]] = []

    def _fake_resume(
        _cwd: Path, rid: str, *, steer: str = "", preset: str = "", config_path: object = None
    ) -> str:
        spawned.append((rid, steer))
        return ""

    def _fake_undo_fork(*_a: object, **_k: object) -> tuple[str, str]:
        return ("fork-child-BBBBBB", "the message taken back")

    monkeypatch.setattr(app_mod, "spawn_detached_resume", _fake_resume)
    monkeypatch.setattr(app_mod, "undo_fork", _fake_undo_fork)
    run = tmp_path / "done-run-AAAAAA"
    run.mkdir()
    evs = [
        {"type": "session.start", "session_id": run.name, "mode": "run", "user_task": "t"},
        {"type": "session.end", "reason": "finish_session", "all_passed": True},
    ]
    (run / "logs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in evs), encoding="utf-8")

    async def scenario() -> None:
        app = Agent6TUI(run)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            app.submit_instruction("/undo")
            await pilot.pause()
            bar = app._conv.query_one("#conv-input", SteerInput)  # pyright: ignore[reportPrivateUsage]
            assert bar.text == "the message taken back"
            assert app.continue_as == "fork-child-BBBBBB"
            app.submit_instruction("the message, edited")
            assert spawned == [("fork-child-BBBBBB", "the message, edited")]

    asyncio.run(scenario())


@pytest.mark.parametrize("continue_as", ["", "fork-child-AAAAAA"])
def test_composer_labels_name_the_fork(continue_as: str) -> None:
    from agent6.ui.tui.conversation import composer_labels

    title, _ = composer_labels("resume", continue_as=continue_as)
    assert title == ("continue as fork-child-AAAAAA" if continue_as else "continue this session")
