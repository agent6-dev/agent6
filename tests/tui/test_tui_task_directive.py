# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""`/task` on the TUI composer queues work into the run's graph.

The run is not steered and not interrupted: the point of queueing is that the
turn in flight never sees it.
"""

from __future__ import annotations

import asyncio
import json
import os
import pathlib

from agent6.sessions import ipc
from agent6.ui.tui import app as tui_app


def _live_run(d: pathlib.Path) -> None:
    d.mkdir(parents=True)
    evs = [
        {"type": "session.start", "session_id": d.name, "mode": "run", "user_task": "t"},
        {"type": "role.call", "role": "worker", "model": "m"},
    ]
    (d / "logs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in evs), encoding="utf-8")
    ipc.write_worker_pid(d, os.getpid())


def test_the_composer_queues_a_task_without_steering(tmp_path: pathlib.Path) -> None:
    run = tmp_path / "live-run-AAAAAA"
    _live_run(run)

    async def scenario() -> None:
        app = tui_app.Agent6TUI(run)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.submit_instruction("/task add a --json flag to the stats report")
            await pilot.pause()
            assert [r.text for r in ipc.drain_requests(run)] == [
                "add a --json flag to the stats report"
            ]
            assert not ipc.steer_request_pending(run)

    asyncio.run(scenario())


def test_a_bare_directive_queues_nothing(tmp_path: pathlib.Path) -> None:
    run = tmp_path / "live-run-BBBBBB"
    _live_run(run)

    async def scenario() -> None:
        app = tui_app.Agent6TUI(run)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.submit_instruction("/task")
            await pilot.pause()
            assert [r.text for r in ipc.drain_requests(run)] == []

    asyncio.run(scenario())
