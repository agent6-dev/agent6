# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The TUI run views read run status from THE dir decision (status_for_session_dir).

Before this, the TUI derived status three separate ways (a pure event fold for
the label, a one-way run_ended latch for liveness, the conversation's own
event-tracked _live) and each lied somewhere: a parked run rendered a blank
label over "(waiting for the model…)" with a steer composer nobody would ever
read; a dead worker was labelled "worker exited" where the hub says "stale";
a crash->resume kept "worker exited" painted over the live execution forever; and
the two composer bars disagreed with each other live.
"""

from __future__ import annotations

import asyncio
import json
import os
import pathlib
import subprocess
import threading
import time
from typing import Any

import pytest
from textual import app as textual_app
from textual import screen, widgets

from agent6.config import layer
from agent6.models import choices
from agent6.ui import spawn
from agent6.ui.tui import _dashboard_header, composer
from agent6.ui.tui import app as tui_app
from agent6.viewmodel import state, tail
from tests.tui._waits import answerable, focus_answers, wait_for


def _mk_parked(d: pathlib.Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(
        json.dumps(
            {
                "version": 2,
                "session_id": d.name,
                "mode": "run",
                "user_task": "fix the flaky test",
                "parked_task": "fix the flaky test",
                "parked_reason": "checkout busy",
            }
        ),
        encoding="utf-8",
    )


def _mk_crashed(d: pathlib.Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    evs = [
        {"type": "session.start", "session_id": d.name, "mode": "run", "user_task": "t"},
        {"type": "role.call", "role": "worker", "model": "m", "provider": "p"},
    ]
    (d / "logs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in evs), encoding="utf-8")
    (d / "worker.pid").write_text("999999999", encoding="utf-8")


def _mk_unreadable(d: pathlib.Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text("{not json", encoding="utf-8")


def _screen_is(app: tui_app.Agent6TUI, name: str) -> bool:
    """`app.screen` raising on a transiently empty stack reads as "not yet", never an error."""
    try:
        current = app.screen
    except textual_app.ScreenStackError:
        return False
    return current is getattr(app, name)


async def _open_dash(app: tui_app.Agent6TUI, pilot: Any) -> None:
    await wait_for(pilot, lambda: _screen_is(app, "_conv"), "the conversation screen")
    await pilot.press("ctrl+d")
    await wait_for(pilot, lambda: _screen_is(app, "_dash"), "the dashboard screen")
    app._heartbeat_at = 0.0  # age the throttle so the dir-status probe fires
    app._tick()
    await pilot.pause()


def test_the_dashboard_title_word_is_the_sessions_mode(tmp_path: pathlib.Path) -> None:
    """The menu-bar title leads with the manifest's mode, as the web panel heading does."""
    d = tmp_path / "plan1"
    d.mkdir()
    (d / "manifest.json").write_text(
        json.dumps({"version": 2, "session_id": d.name, "mode": "plan", "user_task": "lay it out"}),
        encoding="utf-8",
    )
    (d / "logs.jsonl").write_text(
        json.dumps(
            {
                "type": "session.start",
                "session_id": d.name,
                "mode": "plan",
                "user_task": "lay it out",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    async def scenario() -> None:
        app = tui_app.Agent6TUI(d)
        async with app.run_test(size=(140, 40)) as pilot:
            await _open_dash(app, pilot)
            assert app.run_title().startswith("plan · lay it out")

    asyncio.run(scenario())


def test_the_task_count_credits_an_obsolete_task_as_done(tmp_path: pathlib.Path) -> None:
    """The top line's `tasks: N/M` counts a retired task as done, like a skipped one."""
    d = tmp_path / "obsolete1"
    d.mkdir()
    evs = [
        {"type": "session.start", "session_id": d.name, "mode": "run", "user_task": "t"},
        {
            "type": "graph.update",
            "cursor": "t3",
            "nodes": {
                "t1": {"title": "first", "parent_id": None, "status": "obsolete"},
                "t2": {"title": "second", "parent_id": None, "status": "passed"},
                "t3": {"title": "third", "parent_id": None, "status": "pending"},
            },
        },
    ]
    (d / "logs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in evs), encoding="utf-8")

    async def scenario() -> None:
        app = tui_app.Agent6TUI(d)
        async with app.run_test(size=(140, 40)) as pilot:
            await _open_dash(app, pilot)
            for _ in range(80):  # the reader thread folds the graph
                if len(app.state.tasks) == 3:
                    break
                await pilot.pause(0.05)
            top = str(app._dash.query_one("#top", widgets.Static).render())
            assert "tasks: 2/3" in top

    asyncio.run(scenario())


def test_a_finished_plans_deliverable_is_in_the_stream_pane(tmp_path: pathlib.Path) -> None:
    """A plan's end story shows plan.md, as the CLI and the web do."""
    d = tmp_path / "plan2"
    d.mkdir()
    (d / "manifest.json").write_text(
        json.dumps({"version": 2, "session_id": d.name, "mode": "plan", "user_task": "lay it out"}),
        encoding="utf-8",
    )
    evs = [
        {"type": "session.start", "session_id": d.name, "mode": "plan", "user_task": "lay it out"},
        {"type": "tool.call", "name": "finish_planning", "args": {"summary": "Plan seeded."}},
        {"type": "tool.result", "name": "finish_planning", "ok": True, "summary": "ok"},
        {"type": "session.end", "reason": "finish_planning", "all_passed": True},
    ]
    (d / "logs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in evs), encoding="utf-8")
    (d / "plan.md").write_text(
        "# Plan: lay it out\n\n## Tasks\n1. do the thing\n", encoding="utf-8"
    )

    async def scenario() -> None:
        app = tui_app.Agent6TUI(d)
        async with app.run_test(size=(140, 40)) as pilot:
            await _open_dash(app, pilot)
            body = str(app._dash.query_one("#stream-body", widgets.Static).render())
            assert "planned" in body
            assert "Plan seeded." in body
            assert "1. do the thing" in body

    asyncio.run(scenario())


def test_a_failed_finish_attempt_is_not_the_runs_end_story(tmp_path: pathlib.Path) -> None:
    """A rejected finish tool carries a proposed summary, not the run's end.

    When the execution later fails, the stream pane shows the failure without presenting that
    abandoned summary as its closing story.
    """
    d = tmp_path / "failed-finish"
    d.mkdir()
    events = [
        {"type": "session.start", "session_id": d.name, "mode": "run", "user_task": "t"},
        {"type": "role.call", "role": "worker", "model": "m", "provider": "p"},
        {"type": "role.result", "role": "worker", "ok": True, "text": "done"},
        {
            "type": "tool.call",
            "name": "finish_session",
            "args": {"summary": "Everything passed."},
        },
        {
            "type": "tool.result",
            "name": "finish_session",
            "ok": False,
            "summary": "the verify gate failed",
        },
        {"type": "session.end", "reason": "provider_error", "all_passed": False},
    ]
    (d / "logs.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )

    async def scenario() -> None:
        app = tui_app.Agent6TUI(d)
        async with app.run_test(size=(140, 40)) as pilot:
            await _open_dash(app, pilot)
            body = str(app._dash.query_one("#stream-body", widgets.Static).render())
            assert "failed · provider error" in body
            assert "Everything passed." not in body

    asyncio.run(scenario())


def test_the_header_names_the_pins_in_force(tmp_path: pathlib.Path) -> None:
    """The dashboard header lists the pinned instructions, as the web header does."""
    d = tmp_path / "pinned1"
    d.mkdir()
    evs = [
        {"type": "session.start", "session_id": d.name, "mode": "run", "user_task": "t"},
        {"type": "loop.pin.restored", "pins": ["never touch tests"], "count": 1},
        {"type": "loop.pin.added", "text": "keep the API stable", "chars": 19, "count": 2},
    ]
    (d / "logs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in evs), encoding="utf-8")

    async def scenario() -> None:
        app = tui_app.Agent6TUI(d)
        async with app.run_test(size=(140, 40)) as pilot:
            await _open_dash(app, pilot)
            top = str(app._dash.query_one("#top", widgets.Static).render())
            assert "pins: never touch tests | keep the API stable" in top

    asyncio.run(scenario())


def test_parked_run_tells_the_truth_on_every_pane(tmp_path: pathlib.Path, monkeypatch: Any) -> None:
    """A parked run's dashboard leads with the hub's words, and the composer routes to resume.

    The stream pane says parked, never "(waiting for the model…)".
    """
    spawned: list[tuple[str, str]] = []

    def _fake_resume(
        _cwd: pathlib.Path,
        rid: str,
        *,
        steer: str = "",
        preset: str = "",
        model: str = "",
        config_path: object = None,
    ) -> str:
        spawned.append((rid, steer))
        return ""

    monkeypatch.setattr(spawn, "spawn_detached_resume", _fake_resume)
    _mk_parked(tmp_path / "parked1")

    async def scenario() -> None:
        app = tui_app.Agent6TUI(tmp_path / "parked1")
        async with app.run_test(size=(140, 40)) as pilot:
            await _open_dash(app, pilot)
            assert app.session_controllable() is False  # resume is the one action
            top = str(app._dash.query_one("#top", widgets.Static).render())
            assert "parked · checkout busy" in top
            assert "task: fix the flaky test" in top  # manifest fallback, not a blank line
            body = str(app._dash.query_one("#stream-body", widgets.Static).render())
            assert "parked" in body
            assert "waiting for the model" not in body
            assert "working…" not in body
            # Both composer bars offer the resume action, not a dead-end steer.
            assert app._dash.query_one("#dash-input", composer.SteerInput).border_title == (
                "continue this session"
            )
            app.submit_instruction("go ahead")
            await app.workers.wait_for_complete()
            assert spawned == [("parked1", "go ahead")]

    asyncio.run(scenario())


def test_an_unreadable_run_tells_the_truth_on_the_stream_pane(tmp_path: pathlib.Path) -> None:
    """A session whose manifest will not parse reads "unreadable" on the header and the pane."""
    _mk_unreadable(tmp_path / "corrupt1")

    async def scenario() -> None:
        app = tui_app.Agent6TUI(tmp_path / "corrupt1")
        async with app.run_test(size=(140, 40)) as pilot:
            await _open_dash(app, pilot)
            top = str(app._dash.query_one("#top", widgets.Static).render())
            assert "unreadable" in top
            body = str(app._dash.query_one("#stream-body", widgets.Static).render())
            assert "unreadable" in body
            assert "waiting for the model" not in body
            assert "working…" not in body

    asyncio.run(scenario())


def test_a_resume_from_the_composer_carries_the_picked_preset(
    tmp_path: pathlib.Path, monkeypatch: Any
) -> None:
    """A run that is not live shows the preset and model pickers above its composer.

    The picks ride the detached resume as `--preset` and `--model`; a refused spawn says why.
    """
    spawned: list[tuple[str, str, str, str]] = []
    notes: list[str] = []

    def _fake_resume(
        _cwd: pathlib.Path,
        rid: str,
        *,
        steer: str = "",
        preset: str = "",
        model: str = "",
        config_path: object = None,
    ) -> str:
        spawned.append((rid, steer, preset, model))
        return "the checkout is busy" if steer == "refuse me" else ""

    monkeypatch.setattr(spawn, "spawn_detached_resume", _fake_resume)

    def _presets(_cwd: pathlib.Path, _cp: object) -> list[str]:
        return ["quick", "ultra"]

    def _routes(_cwd: pathlib.Path, _cp: object) -> list[str]:
        return ["o/a", "o/b"]

    monkeypatch.setattr(layer, "available_preset_names", _presets)
    monkeypatch.setattr(choices, "available_routes", _routes)
    _mk_parked(tmp_path / "parked2")

    async def scenario() -> None:
        app = tui_app.Agent6TUI(tmp_path / "parked2")
        original = app.notify

        def spy(message: Any, *args: Any, **kwargs: Any) -> None:
            notes.append(str(message))
            original(message, *args, **kwargs)

        monkeypatch.setattr(app, "notify", spy)
        async with app.run_test(size=(140, 40)) as pilot:
            await wait_for(pilot, lambda: _screen_is(app, "_conv"), "the conversation screen")
            row = app._conv.query_one("#conv-resume", composer.ResumeOptions)
            await wait_for(pilot, lambda: row.display, "the resume row")
            preset_options = row.query_one("#resume-preset", widgets.Select)._options  # pyright: ignore[reportPrivateUsage]
            assert [value for _label, value in preset_options] == ["", "quick", "ultra"]
            model_options = row.query_one("#resume-model", widgets.Select)._options  # pyright: ignore[reportPrivateUsage]
            assert [value for _label, value in model_options] == ["", "o/a", "o/b"]
            row.query_one("#resume-preset", widgets.Select).value = "quick"
            row.query_one("#resume-model", widgets.Select).value = "o/b"
            await pilot.pause()
            assert (app.resume_preset, app.resume_model) == ("quick", "o/b")
            app.submit_instruction("go ahead")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert spawned == [("parked2", "go ahead", "quick", "o/b")]
            assert any(
                "under preset quick, model o/b with your instruction" in note for note in notes
            )
            app.action_resume()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert spawned[-1] == ("parked2", "", "quick", "o/b")
            assert any("under preset quick, model o/b in the background" in note for note in notes)
            # The dashboard's pickers show the same choices.
            await _open_dash(app, pilot)
            dash_row = app._dash.query_one("#dash-resume", composer.ResumeOptions)
            await wait_for(pilot, lambda: dash_row.display, "the dashboard's row")
            preset = dash_row.query_one("#resume-preset", widgets.Select)
            model = dash_row.query_one("#resume-model", widgets.Select)
            assert preset.value == "quick"
            assert model.value == "o/b"
            preset.value = ""
            model.value = ""
            await pilot.pause()
            app.submit_instruction("use the recorded choices")
            await app.workers.wait_for_complete()
            assert spawned[-1] == ("parked2", "use the recorded choices", "", "")
            app.submit_instruction("refuse me")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert any("the checkout is busy" in note for note in notes)
            assert dash_row.display
            assert app._dash.query_one("#dash-input", composer.SteerInput).mode == "resume"

    asyncio.run(scenario())


def test_the_resume_rows_name_what_a_bare_resume_runs_under(
    tmp_path: pathlib.Path, monkeypatch: Any
) -> None:
    """Each resume picker's first entry names what a resume without the flag runs under."""
    from textual.widgets._select import SelectCurrent

    def _presets(_cwd: pathlib.Path, _cp: object) -> list[str]:
        return ["quick"]

    def _routes(_cwd: pathlib.Path, _cp: object) -> list[str]:
        return ["o/a", "o/b"]

    def _defaults(
        _cwd: pathlib.Path, _cp: object, _dir: pathlib.Path, *, preset: str = ""
    ) -> tuple[str, str]:
        return "fast (as recorded)", f"o/{preset or 'a'} (config default)"

    monkeypatch.setattr(layer, "available_preset_names", _presets)
    monkeypatch.setattr(choices, "available_routes", _routes)
    monkeypatch.setattr(choices, "resume_defaults", _defaults)
    _mk_parked(tmp_path / "parked3")

    def labels(row: composer.ResumeOptions) -> tuple[str, str]:
        """Each picker's first entry, whatever is picked."""
        pickers = (row.query_one(f"#resume-{name}", widgets.Select) for name in ("preset", "model"))
        first = tuple(str(p._options[0][0]) for p in pickers)  # pyright: ignore[reportPrivateUsage]
        return first[0], first[1]

    async def scenario() -> None:
        app = tui_app.Agent6TUI(tmp_path / "parked3")
        async with app.run_test(size=(140, 40)) as pilot:
            await wait_for(pilot, lambda: _screen_is(app, "_conv"), "the conversation screen")
            row = app._conv.query_one("#conv-resume", composer.ResumeOptions)
            await wait_for(pilot, lambda: row.display, "the resume row")
            await pilot.pause()
            assert labels(row) == ("fast (as recorded)", "o/a (config default)")
            preset = row.query_one("#resume-preset", widgets.Select)
            assert str(preset.query_one(SelectCurrent).label) == "fast (as recorded)"
            row.query_one("#resume-model", widgets.Select).value = "o/b"
            row.query_one("#resume-preset", widgets.Select).value = "quick"
            await pilot.pause()
            await pilot.pause()
            assert labels(row) == ("fast (as recorded)", "o/quick (config default)")
            assert (app.resume_preset, app.resume_model) == ("quick", "o/b")
            assert row.query_one("#resume-model", widgets.Select).value == "o/b"
            await _open_dash(app, pilot)
            dash_row = app._dash.query_one("#dash-resume", composer.ResumeOptions)
            await wait_for(pilot, lambda: dash_row.display, "the dashboard's row")
            await pilot.pause()
            assert labels(dash_row) == ("fast (as recorded)", "o/quick (config default)")
            assert dash_row.query_one("#resume-model", widgets.Select).value == "o/b"

    asyncio.run(scenario())


def test_dead_worker_leads_with_the_hub_word_stale(tmp_path: pathlib.Path) -> None:
    """The top-line label for a lost worker is "stale", the hub row's word for the same probe."""
    _mk_crashed(tmp_path / "crashed1")

    async def scenario() -> None:
        app = tui_app.Agent6TUI(tmp_path / "crashed1")
        async with app.run_test(size=(140, 40)) as pilot:
            await _open_dash(app, pilot)
            await wait_for(pilot, lambda: app.worker_lost, "the dead-worker probe")
            app._tick()
            await pilot.pause()
            top = str(app._dash.query_one("#top", widgets.Static).render())
            assert "stale" in top
            assert "worker exited" not in top  # the label is the hub's word
            body = str(app._dash.query_one("#stream-body", widgets.Static).render())
            assert "worker exited without finishing" in body  # the detail stays

    asyncio.run(scenario())


def test_dead_worker_stream_pane_drops_stale_partial_text(tmp_path: pathlib.Path) -> None:
    """A partial response left in flight by a worker death is not live output."""
    d = tmp_path / "crashed-stream"
    _mk_crashed(d)
    with (d / "logs.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {"type": "role.text_delta", "role": "worker", "text": "partial stale answer"}
            )
            + "\n"
        )

    async def scenario() -> None:
        app = tui_app.Agent6TUI(d)
        async with app.run_test(size=(140, 40)) as pilot:
            await _open_dash(app, pilot)
            await wait_for(pilot, lambda: app.worker_lost, "the dead-worker probe")
            await wait_for(
                pilot,
                lambda: app.state.last_role is not None and bool(app.state.last_role.streamed_text),
                "the partial response",
            )
            app._dash.render_heartbeat()  # pyright: ignore[reportPrivateUsage]
            await pilot.pause()
            body = str(app._dash.query_one("#stream-body", widgets.Static).render())
            assert "worker exited without finishing" in body
            assert "partial stale answer" not in body

    asyncio.run(scenario())


def test_crash_then_resume_recovers_liveness(tmp_path: pathlib.Path) -> None:
    """The dead-worker state is derived, not latched: a resume in place clears it.

    A one-way latch kept "worker exited" painted over the live execution and dropped input.
    """
    d = tmp_path / "revived1"
    _mk_crashed(d)

    async def scenario() -> None:
        app = tui_app.Agent6TUI(d)
        async with app.run_test(size=(140, 40)) as pilot:
            await _open_dash(app, pilot)
            await wait_for(pilot, lambda: app.worker_lost, "the dead-worker probe")
            # The operator resumes: a new execution appends to the log with a live worker pid.
            with (d / "logs.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"type": "loop.resume.start", "iteration": 2}) + "\n")
                fh.write(json.dumps({"type": "role.call", "role": "worker", "model": "m"}) + "\n")
            (d / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")
            await wait_for(pilot, lambda: not app.worker_lost, "liveness to recover after resume")
            assert app.session_controllable() is True
            app._heartbeat_at = 0.0
            app._tick()
            await pilot.pause()
            top = str(app._dash.query_one("#top", widgets.Static).render())
            assert "stale" not in top and "worker exited" not in top
            # Both bars agree on the live mode, the covered conversation's too.
            assert "steer" in (
                app._dash.query_one("#dash-input", composer.SteerInput).border_title or ""
            )
            assert "steer" in (
                app._conv.query_one("#conv-input", composer.SteerInput).border_title or ""
            )

    asyncio.run(scenario())


def test_conversation_bar_tells_the_truth_about_a_dead_worker(tmp_path: pathlib.Path) -> None:
    """The conversation view keys its composer on the host's liveness, not its own events.

    A worker killed without a session.end relabels the bar to resume.
    """
    d = tmp_path / "convdead1"
    _mk_crashed(d)

    async def scenario() -> None:
        app = tui_app.Agent6TUI(d)
        async with app.run_test(size=(140, 40)) as pilot:
            await wait_for(pilot, lambda: _screen_is(app, "_conv"), "the conversation screen")
            app._heartbeat_at = 0.0
            app._tick()
            await wait_for(pilot, lambda: app.worker_lost, "the dead-worker probe")
            app._heartbeat_at = 0.0
            app._tick()
            await pilot.pause()
            bar = app._conv.query_one("#conv-input", composer.SteerInput)
            assert bar.border_title == "continue this session"

    asyncio.run(scenario())


def test_a_dead_workers_open_call_settles_into_the_scrollback(tmp_path: pathlib.Path) -> None:
    """A worker killed mid-command settles its open call as one that never returned."""
    d = tmp_path / "convdead2"
    _mk_crashed(d)
    with (d / "logs.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {"type": "tool.call", "name": "run_command", "args": {"argv": ["sleep", "60"]}}
            )
            + "\n"
        )

    async def scenario() -> None:
        app = tui_app.Agent6TUI(d)
        async with app.run_test(size=(140, 40)) as pilot:
            await wait_for(pilot, lambda: _screen_is(app, "_conv"), "the conversation screen")
            app._heartbeat_at = 0.0
            app._tick()
            await wait_for(pilot, lambda: app.worker_lost, "the dead-worker probe")
            app._conv._poll()  # pyright: ignore[reportPrivateUsage]
            await pilot.pause()
            body = "\n".join(
                str(w.content)
                for w in app._conv.query(".conv-chunk").results(widgets.Static)  # pyright: ignore[reportPrivateUsage]
            )
            assert "→ run_command  sleep 60" in body
            assert "no result (the run died)" in body
            assert "running" not in body

    asyncio.run(scenario())


def test_conversation_composer_routes_through_the_host_parser(tmp_path: pathlib.Path) -> None:
    """A composer line on the conversation view routes through the host's submit_instruction.

    `/compact <focus>` becomes a compaction request, not a literal steer.
    """
    d = tmp_path / "convcompact1"
    d.mkdir()
    (d / "logs.jsonl").write_text(
        "".join(
            json.dumps(e) + "\n"
            for e in (
                {"type": "session.start", "session_id": d.name, "mode": "run", "user_task": "t"},
                {"type": "role.call", "role": "worker", "model": "m", "provider": "p"},
            )
        ),
        encoding="utf-8",
    )
    (d / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")  # live

    async def scenario() -> None:
        app = tui_app.Agent6TUI(d)
        async with app.run_test(size=(140, 40)) as pilot:
            await wait_for(pilot, lambda: _screen_is(app, "_conv"), "the conversation screen")
            app._heartbeat_at = 0.0
            app._tick()
            await pilot.pause()
            bar = app._conv.query_one("#conv-input", composer.SteerInput)
            bar.post_message(composer.SteerInput.Submitted("/compact keep the auth decisions"))
            await pilot.pause()
            await pilot.pause()
            assert (d / "compact.request").read_text(encoding="utf-8") == (
                "keep the auth decisions"
            )
            assert not (d / "steer.answer").exists()
            assert not (d / "steer.request").exists()

    asyncio.run(scenario())


def _mk_blocked(d: pathlib.Path, *, alive: bool) -> None:
    """A run blocked on an unanswered approval, with a live or dead worker."""
    d.mkdir(parents=True, exist_ok=True)
    evs = [
        {"type": "session.start", "session_id": d.name, "mode": "run", "user_task": "t"},
        {"type": "approval.prompt", "id": "ap1", "prompt": "Allow run_command: pytest"},
    ]
    (d / "logs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in evs), encoding="utf-8")
    (d / "worker.pid").write_text(str(os.getpid()) if alive else "999999999", encoding="utf-8")


def test_dead_run_pops_no_approval_modal(tmp_path: pathlib.Path) -> None:
    """Allow/Deny is not offered over a dead worker's unanswered prompt.

    The fold keeps the prompt past the death, clearing only on an answer or an execution boundary.
    """
    d = tmp_path / "ghost1"
    _mk_blocked(d, alive=False)

    async def scenario() -> None:
        app = tui_app.Agent6TUI(d)
        async with app.run_test(size=(140, 40)) as pilot:
            await wait_for(pilot, lambda: _screen_is(app, "_conv"), "the conversation screen")
            await wait_for(pilot, lambda: bool(app.state.pending_approvals), "the prompt to fold")
            app._heartbeat_at = 0.0
            app._tick()
            await pilot.pause()
            assert not isinstance(app.screen, screen.ModalScreen)
            assert not (d / "approvals" / "ap1.answer").exists()

    asyncio.run(scenario())


def _approval_ready(app: tui_app.Agent6TUI) -> bool:
    # The conversation screen renders an approval inline, the composer keeping focus.
    bar = app._conv.query_one("#conv-input", composer.SteerInput)  # pyright: ignore[reportPrivateUsage]
    return (
        _screen_is(app, "_conv")
        and answerable(app._conv)  # pyright: ignore[reportPrivateUsage]
        and app.focused is bar
    )


def test_screen_probe_tolerates_an_empty_stack(tmp_path: pathlib.Path) -> None:
    """`app.screen` raising ScreenStackError inside a poll reads as "not yet"."""
    app = tui_app.Agent6TUI(tmp_path)  # never run: the screen stack is empty
    with pytest.raises(textual_app.ScreenStackError):
        _ = app.screen
    assert _screen_is(app, "_conv") is False
    # The app's own probe answers None instead of raising.
    assert app._screen_or_none() is None  # pyright: ignore[reportPrivateUsage]


def test_live_run_still_gets_the_inline_approval(tmp_path: pathlib.Path) -> None:
    # The converse: gating on liveness must not cost the live run its approval row.
    d = tmp_path / "blocked1"
    _mk_blocked(d, alive=True)

    async def scenario() -> None:
        app = tui_app.Agent6TUI(d)
        async with app.run_test(size=(140, 40)) as pilot:
            await wait_for(pilot, lambda: _approval_ready(app), "the approval row")

    asyncio.run(scenario())


def test_answer_after_death_reports_instead_of_writing(tmp_path: pathlib.Path) -> None:
    """The approval row is withdrawn when the worker dies; the prompt stays visible as a fact."""
    d = tmp_path / "dies-mid-modal"
    _mk_blocked(d, alive=True)

    async def scenario() -> None:
        app = tui_app.Agent6TUI(d)
        async with app.run_test(size=(140, 40)) as pilot:
            await wait_for(pilot, lambda: _approval_ready(app), "the approval row")
            (d / "worker.pid").write_text("999999999", encoding="utf-8")
            app._heartbeat_at = 0.0
            app._tick()
            await wait_for(pilot, lambda: app.worker_lost, "the dead-worker probe")
            app._conv._poll()  # pyright: ignore[reportPrivateUsage]
            await pilot.pause()
            await pilot.press("a")  # the row is gone: the key answers nothing
            await pilot.pause()
            assert not (d / "approvals" / "ap1.answer").exists()
            assert not app._conv.query(composer.ApprovalRow)  # pyright: ignore[reportPrivateUsage]
            item = app._conv.query_one("#conv-approval", widgets.Static)  # pyright: ignore[reportPrivateUsage]
            assert "approval pending when the run ended" in str(item.render())

    asyncio.run(scenario())


def test_exit_on_end_holds_over_a_ghost_prompt_and_ctrl_q_leaves(tmp_path: pathlib.Path) -> None:
    """A dead run's dashboard holds deliberately: the header names the state and the leave key."""
    d = tmp_path / "ghost2"
    _mk_blocked(d, alive=False)

    async def scenario() -> None:
        app = tui_app.Agent6TUI(d, exit_on_end=True)
        async with app.run_test(size=(140, 40)) as pilot:
            await wait_for(pilot, lambda: app._end_hold, "the end hold")
            assert "Ctrl+Q to leave" in app.sub_title
            assert app.is_running
            await pilot.press("ctrl+q")
            deadline = time.monotonic() + 5.0
            while app.is_running and time.monotonic() < deadline:
                await asyncio.sleep(0.05)
            assert not app.is_running, "ctrl+q did not leave the held dashboard"

    asyncio.run(scenario())


def test_end_hold_header_keeps_the_shared_status_reason(tmp_path: pathlib.Path) -> None:
    """The run header cannot shorten the hub's qualified status to one word."""
    d = tmp_path / "failed"
    d.mkdir()
    app = tui_app.Agent6TUI(d)
    app.dir_status = ("failed", "provider_error")
    app._end_hold = True

    assert "failed · provider error" in app.run_title()


def test_finished_run_holds_the_dashboard_until_the_user_leaves(tmp_path: pathlib.Path) -> None:
    """The dashboard holds on session.end, so the payoff stays on screen.

    The header says how to leave, and the composer routes a typed follow-up to resume.
    """
    d = tmp_path / "done1"
    d.mkdir(parents=True)
    evs = [
        {"type": "session.start", "session_id": d.name, "mode": "run", "user_task": "t"},
        {"type": "session.end", "reason": "finish_session", "iterations": 2, "all_passed": True},
    ]
    (d / "logs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in evs), encoding="utf-8")

    async def scenario() -> None:
        app = tui_app.Agent6TUI(d, exit_on_end=True)
        async with app.run_test(size=(140, 40)) as pilot:
            await wait_for(pilot, lambda: app._end_hold, "the end hold")
            assert app.is_running
            # The hold leads with the hub's own status word ("passed" here).
            assert "passed" in app.sub_title and "Ctrl+Q to leave" in app.sub_title
            # A screen stamping its title AFTER the hold began (the mount /
            # A title stamped after the hold began must not wipe it: titles compute at stamp time.
            app._conv.on_screen_resume()
            assert "passed" in app.sub_title and "Ctrl+Q to leave" in app.sub_title
            assert "· t ·" in app.sub_title, "the live task name, not the dir fallback"
            await _open_dash(app, pilot)
            assert app._dash.query_one("#dash-input", composer.SteerInput).border_title == (
                "what should it do next"  # finish_session over a green tree: new work only
            )

    asyncio.run(scenario())


def test_a_finished_log_is_folded_before_the_first_paint(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A finished run opens with its last role and end story.

    Seeding only the status left the dashboard's first paint on an empty fold, so it called the role
    idle and promised a model was still coming until the reader replayed the journal.
    """
    d = tmp_path / "seeded"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(
        json.dumps(
            {
                "mode": "run",
                "session_id": d.name,
                "user_task": "t",
                "models": {"driver": {"provider": "p", "model": "manifest-model"}},
            }
        ),
        encoding="utf-8",
    )
    evs = [
        {"type": "session.start", "session_id": d.name, "mode": "run", "user_task": "t"},
        {"type": "role.call", "role": "worker", "model": "last-model", "provider": "p"},
        {"type": "role.result", "role": "worker", "ok": True, "text": "done"},
        {"type": "tool.call", "name": "finish_session", "args": {"summary": "All done."}},
        {"type": "tool.result", "name": "finish_session", "ok": True, "summary": "ok"},
        {"type": "session.end", "reason": "finish_session", "iterations": 1, "all_passed": True},
    ]
    (d / "logs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in evs), encoding="utf-8")

    reader_go = threading.Event()
    real_tail_events = tail.tail_events

    def held_reader(path: pathlib.Path, **kwargs: Any) -> Any:
        if kwargs.get("follow"):
            reader_go.wait(timeout=5)
        yield from real_tail_events(path, **kwargs)

    monkeypatch.setattr(tail, "tail_events", held_reader)

    async def scenario() -> None:
        app = tui_app.Agent6TUI(d, exit_on_end=True)
        try:
            async with app.run_test(size=(140, 40)) as pilot:
                await _open_dash(app, pilot)
                assert state.status_facts(app.state).finished
                top = str(app._dash.query_one("#top", widgets.Static).render())
                body = str(app._dash.query_one("#stream-body", widgets.Static).render())
                assert "role: worker / last-model" in top
                assert "passed" in top
                assert "passed" in body and "All done." in body
                assert "waiting for the model" not in body
                reader_go.set()
        finally:
            reader_go.set()

    asyncio.run(scenario())


def test_a_resumed_execution_drops_the_prior_executions_role_and_finish_story(
    tmp_path: pathlib.Path,
) -> None:
    """An execution boundary makes the prior call and finish summary historical.

    Until the resumed execution calls a model, its header falls back to the manifest; if the new
    execution then stops, its end story does not repeat the prior execution's summary.
    """
    d = tmp_path / "resumed-story"
    d.mkdir()
    logs = d / "logs.jsonl"
    (d / "manifest.json").write_text(
        json.dumps(
            {
                "mode": "run",
                "session_id": d.name,
                "user_task": "t",
                "models": {"driver": {"provider": "p", "model": "next-model"}},
            }
        ),
        encoding="utf-8",
    )
    first_execution = [
        {"type": "session.start", "session_id": d.name, "mode": "run", "user_task": "t"},
        {"type": "role.call", "role": "worker", "model": "old-model", "provider": "p"},
        {"type": "role.result", "role": "worker", "ok": True, "text": "done"},
        {
            "type": "tool.call",
            "name": "finish_session",
            "args": {"summary": "First execution done."},
        },
        {"type": "tool.result", "name": "finish_session", "ok": True, "summary": "ok"},
        {"type": "session.end", "reason": "finish_session", "all_passed": True},
    ]
    logs.write_text(
        "".join(json.dumps(event) + "\n" for event in first_execution), encoding="utf-8"
    )

    def append(*events: dict[str, object]) -> None:
        with logs.open("a", encoding="utf-8") as fh:
            for event in events:
                fh.write(json.dumps(event) + "\n")

    async def scenario() -> None:
        app = tui_app.Agent6TUI(d)
        async with app.run_test(size=(140, 40)) as pilot:
            await _open_dash(app, pilot)
            assert "First execution done." in str(
                app._dash.query_one("#stream-body", widgets.Static).render()
            )

            (d / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")
            append({"type": "loop.resume.start", "iteration": 2, "mode": "run"})
            await wait_for(pilot, lambda: not app.state.finished, "the resumed execution")
            app._tick()
            await pilot.pause()
            top = str(app._dash.query_one("#top", widgets.Static).render())
            body = str(app._dash.query_one("#stream-body", widgets.Static).render())
            assert "role: worker / next-model" in top
            assert "old-model" not in top
            assert "First execution done." not in body

            append(
                {"type": "role.call", "role": "worker", "model": "new-model", "provider": "p"},
                {"type": "role.result", "role": "worker", "ok": True, "text": "stopping"},
                {"type": "session.end", "reason": "steer_abort", "all_passed": None},
            )
            await wait_for(pilot, lambda: app.state.finished, "the resumed execution's end")
            app._tick()
            await pilot.pause()
            top = str(app._dash.query_one("#top", widgets.Static).render())
            body = str(app._dash.query_one("#stream-body", widgets.Static).render())
            assert "role: worker / new-model" in top
            assert "stopped" in body
            assert "First execution done." not in body

    asyncio.run(scenario())


def test_dead_pane_hints_point_at_controls_that_exist(tmp_path: pathlib.Path) -> None:
    """The dead, parked and created hints point at the composer's Enter, not a removed r key."""
    d = tmp_path / "crashed-hint"
    _mk_crashed(d)

    async def scenario() -> None:
        app = tui_app.Agent6TUI(d)
        async with app.run_test(size=(140, 40)) as pilot:
            await _open_dash(app, pilot)
            await wait_for(pilot, lambda: app.worker_lost, "the dead-worker probe")
            app._tick()
            await pilot.pause()
            body = str(app._dash.query_one("#stream-body", widgets.Static).render())
            assert "press r" not in body
            assert "Enter resumes" in body

    asyncio.run(scenario())


def test_spinners_run_only_during_a_model_call_and_the_composer_follows_liveness(
    tmp_path: pathlib.Path,
) -> None:
    """A live worker is not proof that a model call is running.

    Before its first call and after its last result, both views stay still; session.end, not
    role.result, changes both composers from steer to resume.
    """
    d = tmp_path / "call-edges"
    d.mkdir()
    logs = d / "logs.jsonl"
    logs.write_text("", encoding="utf-8")
    (d / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")
    (d / "manifest.json").write_text(
        json.dumps(
            {
                "mode": "run",
                "session_id": d.name,
                "user_task": "t",
                "models": {"driver": {"provider": "p", "model": "m"}},
            }
        ),
        encoding="utf-8",
    )

    def append(*events: dict[str, object]) -> None:
        with logs.open("a", encoding="utf-8") as fh:
            for event in events:
                fh.write(json.dumps(event) + "\n")

    async def scenario() -> None:
        app = tui_app.Agent6TUI(d)
        async with app.run_test(size=(140, 40)) as pilot:
            await wait_for(pilot, lambda: _screen_is(app, "_conv"), "the conversation screen")
            conv_live = app._conv.query_one("#conv-live", widgets.Static)
            conv_bar = app._conv.query_one("#conv-input", composer.SteerInput)
            await wait_for(pilot, lambda: conv_bar.mode == "steer", "the starting steer bar")
            app._conv._poll()  # pyright: ignore[reportPrivateUsage]
            assert not conv_live.display
            conv_spin = app._conv._spin  # pyright: ignore[reportPrivateUsage]
            app._conv._poll()  # pyright: ignore[reportPrivateUsage]
            assert app._conv._spin == conv_spin  # pyright: ignore[reportPrivateUsage]
            dash_spin = app.spin
            app._heartbeat_at = 0.0
            app._tick()
            assert app.spin == dash_spin

            append(
                {"type": "session.start", "session_id": d.name, "mode": "run", "user_task": "t"},
                {"type": "role.call", "role": "worker", "model": "m", "provider": "p"},
            )
            await wait_for(
                pilot,
                lambda: app.state.last_role is not None and app.state.last_role.in_flight,
                "the model call",
            )
            dash_spin = app.spin
            app._heartbeat_at = 0.0
            app._tick()
            assert app.spin == dash_spin + 1
            conv_spin = app._conv._spin  # pyright: ignore[reportPrivateUsage]
            app._conv._poll()  # pyright: ignore[reportPrivateUsage]
            assert app._conv._spin == conv_spin + 1  # pyright: ignore[reportPrivateUsage]

            append({"type": "role.result", "role": "worker", "ok": True, "text": "done"})
            await wait_for(
                pilot,
                lambda: app.state.last_role is not None and not app.state.last_role.in_flight,
                "the model result",
            )
            assert conv_bar.mode == "steer"
            dash_spin = app.spin
            app._heartbeat_at = 0.0
            app._tick()
            assert app.spin == dash_spin
            app._conv._poll()  # pyright: ignore[reportPrivateUsage]
            assert not conv_live.display

            append(
                {
                    "type": "session.end",
                    "reason": "finish_session",
                    "iterations": 1,
                    "all_passed": True,
                }
            )
            await wait_for(pilot, lambda: app.state.finished, "the session end")
            assert conv_bar.mode == "resume"
            await _open_dash(app, pilot)
            assert app._dash.query_one("#dash-input", composer.SteerInput).mode == "resume"

    asyncio.run(scenario())


def test_waiting_run_pane_says_waiting_not_working(tmp_path: pathlib.Path) -> None:
    """A run blocked on a prompt says it is waiting on the operator in the stream pane too."""
    d = tmp_path / "blocked-pane"
    _mk_blocked(d, alive=True)

    async def scenario() -> None:
        app = tui_app.Agent6TUI(d)
        async with app.run_test(size=(140, 40)) as pilot:
            # Deny writes only the bridge file; no answer event lands, so the run stays waiting.
            await wait_for(pilot, lambda: _approval_ready(app), "the approval row")
            await focus_answers(app._conv, pilot)  # pyright: ignore[reportPrivateUsage]
            await pilot.press("d")
            await _open_dash(app, pilot)
            await wait_for(pilot, lambda: app.dir_status[0] == "waiting", "the waiting word")

            def pane() -> str:
                # The fold lands in the reader thread, so the pane follows a tick later.
                app._tick()  # pyright: ignore[reportPrivateUsage]
                return str(app._dash.query_one("#stream-body", widgets.Static).render())  # pyright: ignore[reportPrivateUsage]

            await wait_for(pilot, lambda: "waiting · needs answer" in pane(), "the waiting pane")
            assert "working…" not in pane()

    asyncio.run(scenario())


def test_prompt_and_answer_events_update_the_chip_immediately(tmp_path: pathlib.Path) -> None:
    """The header chip flips on the prompt/answer event itself, never a heartbeat later.

    Filmed on the dashboard: the log pane already showed approval.answer + verify.end while the chip
    still read "waiting · needs answer": the synchronous dir-status refresh covered only session
    boundaries, so the chip (and both composer bars) lagged the fold by up to ~1s. Asserted with NO
    awaits between the event and the read, so the heartbeat cannot mask the regression.
    """
    d = tmp_path / "live1"
    d.mkdir(parents=True)
    evs = [
        {"type": "session.start", "session_id": d.name, "mode": "run", "user_task": "t"},
        {"type": "role.call", "role": "worker", "model": "m", "provider": "p"},
    ]
    (d / "logs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in evs), encoding="utf-8")
    (d / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")  # a live worker

    async def scenario() -> None:
        app = tui_app.Agent6TUI(d)
        async with app.run_test(size=(140, 40)) as pilot:
            await _open_dash(app, pilot)
            assert app.dir_status[1] != "needs answer"
            # The prompt arrives: the chip says so at once.
            app._handle_event({"type": "approval.prompt", "id": "approval-1", "prompt": "run x?"})
            assert app.dir_status == ("waiting", "needs answer")
            # The answer lands: the chip clears at once.
            app._handle_event({"type": "approval.answer", "id": "approval-1", "approved": True})
            assert app.dir_status[1] != "needs answer"

    asyncio.run(scenario())


def test_dashboard_header_says_where_the_changes_are(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The header carries the run's branch line, as the web header and `sessions show` do."""
    repo = tmp_path / "repo"
    repo.mkdir()
    git = ["git", "-C", str(repo)]
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t"}
    env["GIT_COMMITTER_EMAIL"] = "t@t"
    subprocess.run([*git, "init", "-q", "-b", "main"], check=True)
    subprocess.run([*git, "commit", "-q", "--allow-empty", "-m", "base"], check=True, env=env)
    monkeypatch.chdir(repo)  # the dashboard reads the branch facts from the cwd checkout
    d = tmp_path / "branched"
    d.mkdir()
    evs = [
        {"type": "session.start", "session_id": d.name, "mode": "run", "user_task": "t"},
        {"type": "session.end", "all_passed": True, "reason": "finish_session"},
    ]
    (d / "logs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in evs), encoding="utf-8")
    manifest: dict[str, Any] = {
        "mode": "run",
        "run_branch": "agent6/branched",
        "base_branch": "main",
    }
    (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    async def header() -> str:
        app = tui_app.Agent6TUI(d)
        async with app.run_test(size=(140, 40)) as pilot:
            await _open_dash(app, pilot)
            return str(app._dash.query_one("#top", widgets.Static).render())

    assert "branch:" not in asyncio.run(header())  # no commit yet: no branch to name
    subprocess.run([*git, "branch", "agent6/branched"], check=True)
    assert "branch: agent6/branched → merges into main" in asyncio.run(header())
    tip = subprocess.run(
        [*git, "rev-parse", "agent6/branched"], check=True, capture_output=True, text=True
    ).stdout.strip()

    async def held_header() -> tuple[str, str, str]:
        """The merge stamp lands while the finished screen is held, and the header re-reads it.

        A resume in place then commits past the stamp, and the header follows.
        """
        app = tui_app.Agent6TUI(d)
        async with app.run_test(size=(140, 40)) as pilot:
            await _open_dash(app, pilot)
            before = str(app._dash.query_one("#top", widgets.Static).render())
            manifest["merged"] = {"into": "main", "sha": tip, "tip": tip}
            (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            app._dash.query_one(_dashboard_header.RunHeader)._branch_recheck_at = 0.0  # pyright: ignore[reportPrivateUsage]
            app._dash.render_heartbeat()
            await pilot.pause()
            after = str(app._dash.query_one("#top", widgets.Static).render())
            with (d / "logs.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"type": "loop.resume.start", "iteration": 2}) + "\n")
            subprocess.run([*git, "checkout", "-q", "agent6/branched"], check=True)
            subprocess.run(
                [*git, "commit", "-q", "--allow-empty", "-m", "past the stamp"], check=True, env=env
            )
            subprocess.run([*git, "checkout", "-q", "main"], check=True)
            await wait_for(pilot, lambda: not app.state.finished, "the resume to fold")
            app._dash.render_heartbeat()
            await pilot.pause()
            return before, after, str(app._dash.query_one("#top", widgets.Static).render())

    before, after, resumed = asyncio.run(held_header())
    assert "branch: agent6/branched → merges into main" in before
    assert "branch: agent6/branched (merged into main)" in after
    assert "branch: agent6/branched → merges into main" in resumed
    assert "branch: agent6/branched → merges into main" in asyncio.run(header())  # a reopen agrees


def test_dashboard_header_says_what_the_run_serves(tmp_path: pathlib.Path) -> None:
    """The header names a forwarded port and the `agent6 forward` command that reaches it."""
    import os
    import socket

    from agent6.sessions import ipc

    d = tmp_path / "serving"
    d.mkdir()
    evs = [{"type": "session.start", "session_id": d.name, "mode": "run", "user_task": "t"}]
    (d / "logs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in evs), encoding="utf-8")
    with socket.socket() as srv:
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port = srv.getsockname()[1]
        ipc.write_session_netns_pid(d, os.getpid())

        async def header() -> str:
            app = tui_app.Agent6TUI(d)
            async with app.run_test(size=(140, 40)) as pilot:
                await _open_dash(app, pilot)
                return str(app._dash.query_one("#top", widgets.Static).render())

        top = asyncio.run(header())
    # This process stands in for the network holder, so the test socket shows among the listeners.
    serving = next(line for line in top.splitlines() if line.startswith("serving: "))
    assert str(port) in serving and "· agent6 forward serving " in serving


def test_a_clipped_table_cell_says_it_was_clipped() -> None:
    """The tools table marks sliced args, so `background=True` and `preview=True` stay visible."""
    from agent6.viewmodel import format

    assert (
        format.clip_cell("argv=/bin/sh -c 'echo hi', background=True", 20)
        == "argv=/bin/sh -c 'ec\u2026"
    )
    assert format.clip_cell("short", 20) == "short"
    assert format.clip_cell("two\nlines", 20) == "two lines"


def test_a_parked_sessions_empty_view_names_the_reason() -> None:
    """The parked placeholder carries the parked reason and the way forward."""
    from agent6.ui.tui import conversation

    note = conversation.empty_conversation_note("parked", "uncommitted changes", ended=False)
    assert "parked" in note and "uncommitted changes" in note and "below" in note
    assert conversation.empty_conversation_note("parked", "", ended=False).startswith("parked")
    assert (
        conversation.empty_conversation_note("", "", ended=True)
        == "this session made no conversation"
    )
    assert "appears as the session streams" in conversation.empty_conversation_note(
        "", "", ended=False
    )
    # A crashed run and one that never started are named, as on the dashboard.
    assert "crashed or killed" in conversation.empty_conversation_note("stale", "", ended=True)
    assert "has not started" in conversation.empty_conversation_note("created", "", ended=True)


def _mk_created(d: pathlib.Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(
        json.dumps({"version": 2, "session_id": d.name, "mode": "run", "user_task": "t"}),
        encoding="utf-8",
    )


@pytest.mark.parametrize("make", [_mk_created, _mk_parked, _mk_crashed])
def test_both_run_views_word_a_dead_state_the_same(tmp_path: pathlib.Path, make: Any) -> None:
    """One owner words a created, parked or crashed run for the dashboard and the conversation."""
    from agent6.ui.tui import conversation

    d = tmp_path / "dead1"
    make(d)
    out: list[str] = []
    status: list[tuple[str, str]] = []

    async def scenario() -> None:
        app = tui_app.Agent6TUI(d)
        async with app.run_test(size=(140, 40)) as pilot:
            await _open_dash(app, pilot)
            status.append(app.dir_status)
            out.append(str(app._dash.query_one("#stream-body", widgets.Static).render()))

    asyncio.run(scenario())
    word, detail = status[0]
    assert word in ("created", "parked", "stale")
    first = out[0].split("\n")[0].strip()
    assert first and first in conversation.empty_conversation_note(word, detail, ended=True)


def test_a_session_that_never_commits_shows_no_commit_pane(tmp_path: pathlib.Path) -> None:
    """An ask or a plan has no diff pane; the log pane takes the width."""
    for mode, name in (("ask", "asks"), ("run", "runs")):
        d = tmp_path / name / f"{mode}-one-AAAAAA"
        d.mkdir(parents=True)
        (d / "manifest.json").write_text(
            json.dumps({"version": 3, "session_id": d.name, "mode": mode, "user_task": "t"}),
            encoding="utf-8",
        )
        evs = [
            {"type": "session.start", "session_id": d.name, "mode": mode, "user_task": "t"},
            {"type": "session.end", "reason": "answered", "iterations": 1, "all_passed": None},
        ]
        (d / "logs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in evs), encoding="utf-8")

        async def scenario(d: pathlib.Path = d, mode: str = mode) -> None:
            app = tui_app.Agent6TUI(d)
            async with app.run_test(size=(140, 40)) as pilot:
                await _open_dash(app, pilot)
                assert app._dash.query_one("#diff").display is (mode == "run")

        asyncio.run(scenario())
