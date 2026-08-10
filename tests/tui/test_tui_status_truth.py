# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The TUI run views read run status from THE dir decision (status_for_session_dir).

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
from agent6.ui.tui.modals import ApprovalModal


def _mk_parked(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(
        json.dumps(
            {
                "version": 2,
                "session_id": d.name,
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
        {"type": "session.start", "session_id": d.name, "mode": "run", "user_task": "t"},
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
            assert app.session_controllable() is False  # resume is the one action
            top = str(app._dash.query_one("#top", Static).render())
            assert "parked · resume to start" in top
            assert "task: fix the flaky test" in top  # manifest fallback, not a blank line
            body = str(app._dash.query_one("#stream-body", Static).render())
            assert "parked" in body
            assert "waiting for the model" not in body
            assert "working…" not in body
            # Both composer bars offer the resume action, not a dead-end steer.
            assert app._dash.query_one("#dash-input", SteerInput).border_title == (
                "continue this session"
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
            assert app.session_controllable() is True
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
    not its own event tracking: a worker killed without a session.end relabels the
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
            assert bar.border_title == "continue this session"

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
                {"type": "session.start", "session_id": d.name, "mode": "run", "user_task": "t"},
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


def _mk_blocked(d: Path, *, alive: bool) -> None:
    """A run blocked on an unanswered approval, with a live or dead worker."""
    d.mkdir(parents=True, exist_ok=True)
    evs = [
        {"type": "session.start", "session_id": d.name, "mode": "run", "user_task": "t"},
        {"type": "approval.prompt", "id": "ap1", "prompt": "Allow run_command: pytest"},
    ]
    (d / "logs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in evs), encoding="utf-8")
    (d / "worker.pid").write_text(str(os.getpid()) if alive else "999999999", encoding="utf-8")


def test_dead_run_pops_no_approval_modal(tmp_path: Path) -> None:
    """The fold keeps an unanswered prompt past a worker death (it clears only
    on an answer event or a leg boundary), so the dashboard popped live-looking
    Allow/Deny over a corpse and wrote the answer where nobody polls."""
    d = tmp_path / "ghost1"
    _mk_blocked(d, alive=False)

    async def scenario() -> None:
        app = Agent6TUI(d)
        async with app.run_test(size=(140, 40)) as pilot:
            await _wait_for(pilot, lambda: app.screen is app._conv, "the conversation screen")
            await _wait_for(pilot, lambda: app.state.pending_approvals, "the prompt to fold")
            app._heartbeat_at = 0.0
            app._tick()
            await pilot.pause()
            assert not isinstance(app.screen, ApprovalModal)
            assert not (d / "approvals" / "ap1.answer").exists()

    asyncio.run(scenario())


def _modal_ready(app: Agent6TUI) -> bool:
    # app.screen flips to the modal synchronously at push; wait for its buttons
    # to MOUNT too, or run_test teardown races the modal's own mount lifecycle.
    return isinstance(app.screen, ApprovalModal) and bool(app.screen.query("#yes"))


def test_live_run_still_pops_the_approval_modal(tmp_path: Path) -> None:
    # The converse: gating on liveness must not cost the live run its modal.
    d = tmp_path / "blocked1"
    _mk_blocked(d, alive=True)

    async def scenario() -> None:
        app = Agent6TUI(d)
        async with app.run_test(size=(140, 40)) as pilot:
            await _wait_for(pilot, lambda: _modal_ready(app), "the modal")

    asyncio.run(scenario())


def test_answer_after_death_reports_instead_of_writing(tmp_path: Path) -> None:
    """The worker dies while the modal is open: the answer reaches nothing, so
    say so instead of silently writing a file the next resume drops."""
    d = tmp_path / "dies-mid-modal"
    _mk_blocked(d, alive=True)

    async def scenario() -> None:
        app = Agent6TUI(d)
        async with app.run_test(size=(140, 40)) as pilot:
            await _wait_for(pilot, lambda: _modal_ready(app), "the modal")
            (d / "worker.pid").write_text("999999999", encoding="utf-8")
            app._heartbeat_at = 0.0
            app._tick()
            await _wait_for(pilot, lambda: app.worker_lost, "the dead-worker probe")
            await pilot.press("y")
            await pilot.pause()
            assert not (d / "approvals" / "ap1.answer").exists()
            notes = [str(n.message) for n in app._notifications]
            assert any("reached nothing" in n for n in notes)

    asyncio.run(scenario())


def test_exit_on_end_is_not_pinned_by_a_ghost_prompt(tmp_path: Path) -> None:
    """exit_on_end required every prompt answered before closing; a dead run's
    ghost prompt can never be answered, so the auto-spawned dashboard sat open
    forever. It must close once the run is over and no modal is up."""
    d = tmp_path / "ghost2"
    _mk_blocked(d, alive=False)

    async def scenario() -> None:
        app = Agent6TUI(d, exit_on_end=True)
        async with app.run_test(size=(140, 40)) as pilot:
            del pilot  # the app must exit on its own tick, unprompted
            deadline = time.monotonic() + 10.0
            while app.is_running and time.monotonic() < deadline:
                await asyncio.sleep(0.05)
            assert not app.is_running, "exit_on_end never fired over the ghost prompt"

    asyncio.run(scenario())


def test_dead_pane_hints_point_at_controls_that_exist(tmp_path: Path) -> None:
    """The dead/parked/created hints said "press r to resume", but the r
    binding was removed (no plain-letter shortcuts) and the composer holds
    focus, so pressing r typed the letter into the box. Point at the
    composer's Enter, the action that exists."""
    d = tmp_path / "crashed-hint"
    _mk_crashed(d)

    async def scenario() -> None:
        app = Agent6TUI(d)
        async with app.run_test(size=(140, 40)) as pilot:
            await _open_dash(app, pilot)
            await _wait_for(pilot, lambda: app.worker_lost, "the dead-worker probe")
            app._tick()
            await pilot.pause()
            body = str(app._dash.query_one("#stream-body", Static).render())
            assert "press r" not in body
            assert "Enter resumes" in body

    asyncio.run(scenario())


def test_waiting_run_pane_says_waiting_not_working(tmp_path: Path) -> None:
    """A run blocked on an unanswered prompt read "waiting · needs answer" on
    the top line while the stream pane ticked a live "worker working…"
    spinner beside it -- two lines, two claims. The pane now says what the
    run is doing: waiting on the operator."""
    d = tmp_path / "blocked-pane"
    _mk_blocked(d, alive=True)

    async def scenario() -> None:
        app = Agent6TUI(d)
        async with app.run_test(size=(140, 40)) as pilot:
            # Dismiss the prompt modal (Esc = deny writes only the bridge file;
            # no answer EVENT lands, so the fold keeps the run "waiting").
            await _wait_for(pilot, lambda: _modal_ready(app), "the modal")
            await pilot.press("escape")
            await _open_dash(app, pilot)
            await _wait_for(pilot, lambda: app.dir_status[0] == "waiting", "the waiting word")
            app._tick()
            await pilot.pause()
            body = str(app._dash.query_one("#stream-body", Static).render())
            assert "waiting for your answer" in body
            assert "working…" not in body

    asyncio.run(scenario())


def test_prompt_and_answer_events_update_the_chip_immediately(tmp_path: Path) -> None:
    """The header chip flips on the prompt/answer event itself, never a
    heartbeat later. Filmed on the dashboard: the log pane already showed
    approval.answer + verify.end while the chip still read "waiting · needs
    answer" -- the synchronous dir-status refresh covered only session
    boundaries, so the chip (and both composer bars) lagged the fold by up to
    ~1s. Asserted with NO awaits between the event and the read, so the
    heartbeat cannot mask the regression."""
    d = tmp_path / "live1"
    d.mkdir(parents=True)
    evs = [
        {"type": "session.start", "session_id": d.name, "mode": "run", "user_task": "t"},
        {"type": "role.call", "role": "worker", "model": "m", "provider": "p"},
    ]
    (d / "logs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in evs), encoding="utf-8")
    (d / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")  # a live worker

    async def scenario() -> None:
        app = Agent6TUI(d)
        async with app.run_test(size=(140, 40)) as pilot:
            await _open_dash(app, pilot)
            assert app.dir_status[1] != "needs answer"
            # The prompt arrives: the chip must say so NOW (no pause between).
            app._handle_event({"type": "approval.prompt", "id": "approval-1", "prompt": "run x?"})
            assert app.dir_status == ("waiting", "needs answer")
            # The answer lands: the chip must clear NOW.
            app._handle_event({"type": "approval.answer", "id": "approval-1", "approved": True})
            assert app.dir_status[1] != "needs answer"

    asyncio.run(scenario())
