# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Run > Review this run… shells out to `sessions review` off the UI thread into a modal.

A live run is refused before any call, as the CLI refuses it.
"""

from __future__ import annotations

import asyncio
import json
import os
import pathlib
from typing import Any

from textual import widgets

from agent6.sessions import ipc
from agent6.ui.tui import app as app_mod
from agent6.ui.tui import modals


def _run_dir(tmp_path: pathlib.Path, name: str, *, ended: bool) -> pathlib.Path:
    run = tmp_path / name
    run.mkdir()
    events: list[dict[str, Any]] = [
        {"type": "session.start", "session_id": name, "mode": "run", "user_task": "t"}
    ]
    if ended:
        events.append({"type": "session.end", "reason": "finish_session", "all_passed": True})
    (run / "logs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    return run


def test_review_this_run_opens_the_cli_review_in_a_modal(
    tmp_path: pathlib.Path, monkeypatch: Any
) -> None:
    calls: list[list[str]] = []

    def _fake_output(
        argv: list[str], _cwd: pathlib.Path, *, timeout_s: float = 120.0
    ) -> tuple[bool, str]:
        calls.append(argv[-3:])
        return True, "## Outcome\nfinished green"

    monkeypatch.setattr(app_mod, "run_cli_output", _fake_output)
    run = _run_dir(tmp_path, "done-run-AAAAAA", ended=True)

    async def scenario() -> None:
        app = app_mod.Agent6TUI(run)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            app.action_review_run()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert calls == [["review", "--", "done-run-AAAAAA"]]
            assert isinstance(app.screen, modals.TextModal)
            assert "finished green" in app.screen.query_one("#text-view", widgets.TextArea).text

    asyncio.run(scenario())


def test_review_of_a_live_run_is_refused_without_a_call(
    tmp_path: pathlib.Path, monkeypatch: Any
) -> None:
    calls: list[list[str]] = []

    def _fake_output(
        argv: list[str], _cwd: pathlib.Path, *, timeout_s: float = 120.0
    ) -> tuple[bool, str]:
        calls.append(argv)
        return True, "never"

    monkeypatch.setattr(app_mod, "run_cli_output", _fake_output)
    run = _run_dir(tmp_path, "live-run-AAAAAA", ended=False)
    ipc.write_worker_pid(run, os.getpid())

    async def scenario() -> None:
        app = app_mod.Agent6TUI(run)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.action_review_run()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert calls == []
            assert not isinstance(app.screen, modals.TextModal)

    asyncio.run(scenario())


def test_a_refused_review_is_a_notice_not_a_modal(tmp_path: pathlib.Path, monkeypatch: Any) -> None:
    def _fake_output(
        _argv: list[str], _cwd: pathlib.Path, *, timeout_s: float = 120.0
    ) -> tuple[bool, str]:
        return False, "no reviewer route"

    monkeypatch.setattr(app_mod, "run_cli_output", _fake_output)
    run = _run_dir(tmp_path, "done-run-BBBBBB", ended=True)

    async def scenario() -> None:
        app = app_mod.Agent6TUI(run)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            app.action_review_run()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert not isinstance(app.screen, modals.TextModal)

    asyncio.run(scenario())


def test_a_second_pick_while_a_review_runs_starts_no_second_call(
    tmp_path: pathlib.Path, monkeypatch: Any
) -> None:
    """Run > Review this run… picked twice during a call runs one reviewer and opens one modal."""
    import threading

    gate = threading.Event()
    calls: list[list[str]] = []

    def _slow_output(
        argv: list[str], _cwd: pathlib.Path, *, timeout_s: float = 120.0
    ) -> tuple[bool, str]:
        calls.append(argv[-3:])
        gate.wait(5)
        return True, "## Outcome\nfinished green"

    monkeypatch.setattr(app_mod, "run_cli_output", _slow_output)
    run = _run_dir(tmp_path, "done-run-CCCCCC", ended=True)

    async def scenario() -> None:
        app = app_mod.Agent6TUI(run)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            app.action_review_run()
            await pilot.pause()
            app.action_review_run()
            await pilot.pause()
            gate.set()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert calls == [["review", "--", "done-run-CCCCCC"]]
            assert isinstance(app.screen, modals.TextModal)

    asyncio.run(scenario())
