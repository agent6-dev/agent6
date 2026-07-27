# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The TUI run views read run status from THE dir decision (status_for_run_dir).

Before this, the TUI derived status three separate ways (a pure event fold for
the label, a one-way run_ended latch for liveness, the conversation's own
event-tracked _live) and each lied somewhere: a parked run rendered a blank
label over "(waiting for the model…)" with a steer composer nobody would ever
read; a dead worker was labelled "worker exited" where the hub says "stale";
a crash->resume kept "worker exited" painted over the live leg forever; and
the two composer bars disagreed with each other live.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

from textual.widgets import Static

from agent6.ui.tui.app import Agent6TUI
from agent6.ui.tui.conversation import SteerInput


def _mk_parked(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(
        json.dumps(
            {
                "version": 2,
                "run_id": d.name,
                "mode": "run",
                "user_task": "fix the flaky test",
                "parked_task": "fix the flaky test",
            }
        ),
        encoding="utf-8",
    )


def _mk_crashed(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    evs = [
        {"type": "run.start", "run_id": d.name, "mode": "run", "user_task": "t"},
        {"type": "role.call", "role": "worker", "model": "m", "provider": "p"},
    ]
    (d / "logs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in evs), encoding="utf-8")
    (d / "worker.pid").write_text("999999999", encoding="utf-8")


async def _wait_for(pilot: Any, cond: Any, what: str, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        await pilot.pause(0.05)


async def _open_dash(app: Agent6TUI, pilot: Any) -> None:
    await _wait_for(pilot, lambda: app.screen is app._conv, "the conversation screen")
    await pilot.press("ctrl+d")
    await _wait_for(pilot, lambda: app.screen is app._dash, "the dashboard screen")
    app._heartbeat_at = 0.0  # age the throttle so the dir-status probe fires now
    app._tick()
    await pilot.pause()


def test_parked_run_tells_the_truth_on_every_pane(tmp_path: Path, monkeypatch: Any) -> None:
    """A parked run's dashboard leads with the hub's words ("parked · resume to
    start"), the stream pane says parked (never the "(waiting for the model…)"
    lie -- no model is coming), and the composer routes to resume, exactly like
    a finished run's."""
    from agent6.ui.tui import app as app_mod

    spawned: list[tuple[str, str]] = []

    def _fake_resume(_cwd: Path, rid: str, *, steer: str = "") -> str:
        spawned.append((rid, steer))
        return ""

    monkeypatch.setattr(app_mod, "spawn_detached_resume", _fake_resume)
    _mk_parked(tmp_path / "parked1")

    async def scenario() -> None:
        app = Agent6TUI(tmp_path / "parked1")
        async with app.run_test(size=(140, 40)) as pilot:
            await _open_dash(app, pilot)
            assert app.run_controllable() is False  # resume is the one action
            top = str(app._dash.query_one("#top", Static).render())
            assert "parked · resume to start" in top
            assert "task: fix the flaky test" in top  # manifest fallback, not a blank line
            body = str(app._dash.query_one("#stream-body", Static).render())
            assert "parked" in body
            assert "waiting for the model" not in body
            assert "working…" not in body
            # Both composer bars offer the resume action, not a dead-end steer.
            assert app._dash.query_one("#dash-input", SteerInput).border_title == (
                "continue the run"
            )
            app.submit_instruction("go ahead")
            assert spawned == [("parked1", "go ahead")]

    asyncio.run(scenario())


def test_dead_worker_leads_with_the_hub_word_stale(tmp_path: Path) -> None:
    """The top-line label for a lost worker is "stale" -- the word the hub row
    shows for the same probe -- with the explanatory sentence kept in the
    stream pane. Two surfaces, one word."""
    _mk_crashed(tmp_path / "crashed1")

    async def scenario() -> None:
        app = Agent6TUI(tmp_path / "crashed1")
        async with app.run_test(size=(140, 40)) as pilot:
            await _open_dash(app, pilot)
            await _wait_for(pilot, lambda: app.worker_lost, "the dead-worker probe")
            app._tick()
            await pilot.pause()
            top = str(app._dash.query_one("#top", Static).render())
            assert "stale" in top
            assert "worker exited" not in top  # the label is the hub's word now
            body = str(app._dash.query_one("#stream-body", Static).render())
            assert "worker exited without finishing" in body  # the detail stays

    asyncio.run(scenario())


def test_crash_then_resume_recovers_liveness(tmp_path: Path) -> None:
    """The dead-worker state is DERIVED, not latched: after the operator
    resumes (new leg appends events, live worker.pid), the dashboard label
    clears, the composers relabel to steer, and submits steer the live leg --
    the one-way run_ended latch kept "worker exited" painted over the live
    resumed leg and silently dropped operator input."""
    d = tmp_path / "revived1"
    _mk_crashed(d)

    async def scenario() -> None:
        app = Agent6TUI(d)
        async with app.run_test(size=(140, 40)) as pilot:
            await _open_dash(app, pilot)
            await _wait_for(pilot, lambda: app.worker_lost, "the dead-worker probe")
            # The operator resumes: a new leg appends to the log and records a
            # live worker pid.
            with (d / "logs.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"type": "loop.resume.start", "iteration": 2}) + "\n")
                fh.write(json.dumps({"type": "role.call", "role": "worker", "model": "m"}) + "\n")
            (d / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")
            await _wait_for(pilot, lambda: not app.worker_lost, "liveness to recover after resume")
            assert app.run_controllable() is True
            app._heartbeat_at = 0.0
            app._tick()
            await pilot.pause()
            top = str(app._dash.query_one("#top", Static).render())
            assert "stale" not in top and "worker exited" not in top
            # BOTH bars agree on the live mode -- the covered conversation too.
            assert "steer" in (app._dash.query_one("#dash-input", SteerInput).border_title or "")
            assert "steer" in (app._conv.query_one("#conv-input", SteerInput).border_title or "")

    asyncio.run(scenario())


def test_conversation_bar_tells_the_truth_about_a_dead_worker(tmp_path: Path) -> None:
    """The PRIMARY conversation view keys its composer on the host's liveness,
    not its own event tracking: a worker killed without a run.end relabels the
    bar to resume (its old event-only _live stayed True forever, and typed
    steers went to a corpse with a success toast)."""
    d = tmp_path / "convdead1"
    _mk_crashed(d)

    async def scenario() -> None:
        app = Agent6TUI(d)
        async with app.run_test(size=(140, 40)) as pilot:
            await _wait_for(pilot, lambda: app.screen is app._conv, "the conversation screen")
            app._heartbeat_at = 0.0
            app._tick()
            await _wait_for(pilot, lambda: app.worker_lost, "the dead-worker probe")
            app._heartbeat_at = 0.0
            app._tick()
            await pilot.pause()
            bar = app._conv.query_one("#conv-input", SteerInput)
            assert bar.border_title == "continue the run"

    asyncio.run(scenario())


def test_conversation_composer_routes_through_the_host_parser(tmp_path: Path) -> None:
    """A composer line on the PRIMARY conversation view routes through the
    host's submit_instruction, so `/compact <focus>` becomes an out-of-band
    compaction request exactly as on the dashboard -- not a literal steer the
    model is told to obey (the bar's own title advertises /compact)."""
    d = tmp_path / "convcompact1"
    d.mkdir()
    (d / "logs.jsonl").write_text(
        "".join(
            json.dumps(e) + "\n"
            for e in (
                {"type": "run.start", "run_id": d.name, "mode": "run", "user_task": "t"},
                {"type": "role.call", "role": "worker", "model": "m", "provider": "p"},
            )
        ),
        encoding="utf-8",
    )
    (d / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")  # live

    async def scenario() -> None:
        app = Agent6TUI(d)
        async with app.run_test(size=(140, 40)) as pilot:
            await _wait_for(pilot, lambda: app.screen is app._conv, "the conversation screen")
            app._heartbeat_at = 0.0
            app._tick()
            await pilot.pause()
            bar = app._conv.query_one("#conv-input", SteerInput)
            bar.post_message(SteerInput.Submitted("/compact keep the auth decisions"))
            await pilot.pause()
            await pilot.pause()
            assert (d / "compact.request").read_text(encoding="utf-8") == (
                "keep the auth decisions"
            )
            assert not (d / "steer.answer").exists()
            assert not (d / "steer.request").exists()

    asyncio.run(scenario())
