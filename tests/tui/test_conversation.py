# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Headless drive of the TUI conversation viewer (ConversationScreen)."""

from __future__ import annotations

import asyncio
import bisect
import json
import os
import time
from pathlib import Path
from typing import Any

from textual.containers import VerticalScroll
from textual.widgets import Static

from agent6.ui.tui.app import Agent6TUI
from agent6.ui.tui.composer import ApprovalRow
from agent6.ui.tui.conversation import ConversationScreen, SteerInput


async def _wait_for(pilot: Any, cond: Any, what: str, timeout: float = 10.0) -> None:
    """Wait for the 0.5s follow poll (and rendering) by condition, not by a
    fixed sleep that loses the race on a loaded machine."""
    deadline = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        await pilot.pause(0.05)


def _following(scroll: VerticalScroll) -> bool:
    return scroll.max_scroll_y - scroll.scroll_y <= 2.0


def _body_text(app: Agent6TUI) -> str:
    # The scrollback is a sequence of chunk Statics (selectable), in DOM order.
    return "\n".join(str(w.content) for w in app.screen.query(".conv-chunk").results(Static))


def _nlines(app: Agent6TUI) -> int:
    return len([ln for ln in _body_text(app).splitlines() if ln.strip()])


_EVENTS: list[dict[str, object]] = [
    {"type": "session.start", "user_task": "do X"},
    {"type": "role.call", "role": "worker"},
    # Multi-line on purpose: collapsed shows only the first line, so expanded
    # is distinguishable from collapsed by the second line's presence.
    {"type": "role.thinking_delta", "role": "worker", "text": "thinking hard here\nsecond thought"},
    {"type": "role.text_delta", "role": "worker", "text": "on it"},
    {"type": "role.result", "role": "worker"},
    {"type": "tool.call", "name": "read_file", "args": {"path": "a"}},
    {"type": "tool.result", "name": "read_file", "ok": True, "summary": "12 bytes"},
    {"type": "session.end", "all_passed": True, "reason": "finish_session"},
]


def _write(logs: Path, events: list[dict[str, object]]) -> None:
    logs.write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")


def _hosted(run: Path, events: list[dict[str, object]] | None, *, live: bool = False) -> Agent6TUI:
    """The run app over *run*, the host whose fold, dir status and prompt
    dispatcher the conversation screen reads; the screen is its first view.
    *live* plants this process as the run's worker, so the dir status reads
    live; None writes no journal at all."""
    run.mkdir(parents=True, exist_ok=True)
    if events is not None:
        _write(run / "logs.jsonl", events)
    if live:
        (run / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")
    return Agent6TUI(run)


def test_conversation_screen_cycles_detail_level(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = _hosted(tmp_path / "run", _EVENTS)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ConversationScreen)

            def body_text() -> str:
                return _body_text(app)

            assert _nlines(app) > 0  # the conversation rendered
            # Collapsed default: the first line of the reasoning as a one-line
            # summary (with a more-count when it spans lines), not the bulk.
            assert "thinking hard here" in body_text()
            assert body_text().count("thinking hard here") == 1
            assert "second thought" not in body_text()  # the bulk stays folded
            screen.action_cycle_detail()  # collapsed -> expanded
            await pilot.pause()
            assert "second thought" in body_text()  # the bulk is now shown
            screen.action_cycle_detail()  # expanded -> hidden
            await pilot.pause()
            assert "thinking" not in body_text()  # thinking omitted entirely
            screen.action_reload()  # reload must not raise
            await pilot.pause()

    asyncio.run(scenario())


def test_an_unrendered_finish_call_with_bad_args_does_not_crash_the_screen(
    tmp_path: Path,
) -> None:
    """A corrupt finish call is not a transcript item, so its non-object args
    must not take down the conversation screen while it tries to find a summary."""
    events: list[dict[str, object]] = [
        {"type": "session.start", "user_task": "do X"},
        {"type": "tool.call", "name": "finish_session", "args": "not an object"},
        {"type": "session.end", "all_passed": True, "reason": "finish_session"},
    ]

    async def scenario() -> None:
        app = _hosted(tmp_path / "run", events)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert isinstance(app.screen, ConversationScreen)
            assert "● passed" in _body_text(app)

    asyncio.run(scenario())


def test_conversation_screen_follows_live(tmp_path: Path) -> None:
    """Events appended after mount (a live run / a resume) show up via the poll."""
    run = tmp_path / "run"
    logs = run / "logs.jsonl"

    async def scenario() -> None:
        app = _hosted(run, [], live=True)
        async with app.run_test() as pilot:
            await pilot.pause()
            before = _nlines(app)
            with logs.open("a", encoding="utf-8") as fh:
                for event in _EVENTS:
                    fh.write(json.dumps(event) + "\n")
            await _wait_for(pilot, lambda: _nlines(app) > before, "the appended turns")

    asyncio.run(scenario())


def test_steer_bar_stays_for_a_finished_run_as_the_resume_composer(tmp_path: Path) -> None:
    """The conversation view is the run app's main screen: after the run
    ends the bar stays, and Enter resumes the run with the typed follow-up."""

    async def scenario() -> None:
        app = _hosted(tmp_path / "run", _EVENTS)  # ends with session.end: finished
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.screen.query_one("#conv-input", SteerInput).display

    asyncio.run(scenario())


def test_steer_bar_shows_for_a_live_run_and_submits_over_the_bridge(tmp_path: Path) -> None:
    from agent6.sessions.ipc import STEER_ANSWER_FILE, steer_request_pending

    run = tmp_path / "run"

    async def scenario() -> None:
        app = _hosted(run, _EVENTS[:-1], live=True)  # no session.end, a live worker
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.screen.query_one("#conv-input", SteerInput)
            assert bar.display  # a live run shows the steer bar
            bar.post_message(SteerInput.Submitted("go left"))
            await pilot.pause()

    asyncio.run(scenario())
    assert steer_request_pending(run)  # the run was asked to steer
    assert (run / STEER_ANSWER_FILE).read_text(encoding="utf-8") == "go left"


def test_resumed_leg_is_live_and_steers_over_the_bridge(tmp_path: Path) -> None:
    """A resumed leg emits ONLY loop.resume.start (never a second session.start).
    The screen must read it as live, as the host's dir status does once the
    leg's worker is up, so a submit routes to the steer bridge; a leg read as
    finished had Enter spawn a second resume that died on the run lock while
    the toast claimed the instruction was delivered."""
    from agent6.sessions.ipc import STEER_ANSWER_FILE, steer_request_pending

    run = tmp_path / "run"
    logs = run / "logs.jsonl"

    async def scenario() -> None:
        app = _hosted(run, _EVENTS)  # ends with session.end: finished, no worker
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ConversationScreen)
            assert not app.session_controllable()  # finished leg
            with logs.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"type": "loop.resume.start", "iteration": 1}) + "\n")
                fh.write(json.dumps({"type": "role.call", "role": "worker"}) + "\n")
            (run / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")
            await _wait_for(pilot, app.session_controllable, "the resumed leg to read live")
            bar = screen.query_one("#conv-input", SteerInput)
            bar.post_message(SteerInput.Submitted("also update docs"))
            await pilot.pause()

    asyncio.run(scenario())
    # Routed over the steer bridge, not a second doomed `agent6 resume`.
    assert steer_request_pending(run)
    assert (run / STEER_ANSWER_FILE).read_text(encoding="utf-8") == "also update docs"


def test_live_run_auto_focuses_the_steer_bar(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = _hosted(tmp_path / "run", _EVENTS[:-1], live=True)  # bar ready to type
        async with app.run_test() as pilot:
            await pilot.pause()
            assert isinstance(app.focused, SteerInput)

    asyncio.run(scenario())


def test_esc_backs_out_even_with_the_bar_focused(tmp_path: Path) -> None:
    # A live run auto-focuses the bar; Esc is a priority binding, so it still
    # leaves the view (the host's to_hub) instead of the bar eating the key.
    async def scenario() -> None:
        app = _hosted(tmp_path / "run", _EVENTS[:-1], live=True)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert isinstance(app.focused, SteerInput)
            await pilot.press("escape")
        assert app.return_code == 0  # to_hub: back to the hub loop

    asyncio.run(scenario())


def test_follow_survives_the_live_pane_growing(tmp_path: Path) -> None:
    # A live turn that only THINKS (no completed turn appended) still grows the live
    # pane, shrinking the scroll viewport. Follow mode must survive that nudge.
    logs = tmp_path / "logs.jsonl"
    events: list[dict[str, object]] = [{"type": "session.start", "user_task": "x"}]
    for i in range(20):  # overflow a short viewport
        more: list[dict[str, object]] = [
            {"type": "tool.call", "name": "read_file", "args": {"path": f"f{i}"}},
            {"type": "tool.result", "name": "read_file", "ok": True, "summary": f"{i} bytes"},
        ]
        events += more
    run = tmp_path / "run"
    logs = run / "logs.jsonl"

    async def scenario() -> None:
        app = _hosted(run, events, live=True)  # no session.end, a live worker
        async with app.run_test(size=(60, 12)) as pilot:
            await pilot.pause()
            scroll = app.screen.query_one("#conv-scroll", VerticalScroll)
            assert _following(scroll)  # _reload pins to the bottom
            # Expanded detail streams the reasoning tail into the live pane, so
            # a thinking burst grows it by whole lines (collapsed keeps the
            # pane a constant one-liner and nothing would move).
            conv_screen = app.screen
            assert isinstance(conv_screen, ConversationScreen)
            conv_screen._detail = "expanded"
            overflow_before = scroll.max_scroll_y
            with logs.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"type": "role.thinking_delta", "text": "x " * 300}) + "\n")
            # The growing live pane shrinks the viewport, so the overflow grows.
            await _wait_for(
                pilot, lambda: scroll.max_scroll_y > overflow_before, "the live pane to grow"
            )
            # Follow re-pins on the app's next tick, one frame after the growth
            # is first visible: wait for it to settle rather than asserting on
            # that first frame (a real follow break never re-pins, so a broken
            # regression still times this out).
            await _wait_for(pilot, lambda: _following(scroll), "follow to re-pin after growth")

    asyncio.run(scenario())


def test_detail_cycle_keeps_the_top_block_anchored(tmp_path: Path) -> None:
    # Expanding a big failed-tool block above the viewport must not carry your place
    # away: the block at the top of the viewport stays put across the re-render.
    events: list[dict[str, object]] = [{"type": "session.start", "user_task": "x"}]
    big: list[dict[str, object]] = [
        {"type": "tool.call", "name": "apply_edit", "args": {"path": "b"}},
        {
            "type": "tool.result",
            "name": "apply_edit",
            "ok": False,
            "summary": "\n".join(f"line {i}" for i in range(100)),
        },
    ]
    events += big
    for i in range(15):
        row: list[dict[str, object]] = [
            {"type": "tool.call", "name": "grep", "args": {"pattern": f"m{i}"}},
            {"type": "tool.result", "name": "grep", "ok": True, "summary": f"{i} hits"},
        ]
        events += row

    async def scenario() -> None:
        app = _hosted(tmp_path / "run", events)
        async with app.run_test(size=(80, 16)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ConversationScreen)
            scroll = screen.query_one("#conv-scroll", VerticalScroll)
            scroll.scroll_to(y=18, animate=False)  # a mid position, past the failed tool
            await pilot.pause()
            starts = screen._item_visual_starts()
            anchor = bisect.bisect_right(starts, scroll.scroll_y) - 1
            offset_before = scroll.scroll_y - starts[anchor]
            screen.action_cycle_detail()  # collapsed -> expanded: the failed tool grows above
            await pilot.pause()
            offset_after = scroll.scroll_y - screen._item_visual_starts()[anchor]
            assert abs(offset_after - offset_before) <= 2  # the anchored block held its place

    asyncio.run(scenario())


def test_conversation_live_pane_shows_the_in_progress_turn(tmp_path: Path) -> None:
    # A turn that is still thinking (no role.result yet) shows in the live pane,
    # so a long reasoning generation doesn't look frozen.
    run = tmp_path / "run"
    logs = run / "logs.jsonl"
    events: list[dict[str, object]] = [
        {"type": "session.start", "user_task": "do X"},
        {"type": "role.call", "role": "worker"},
        {"type": "role.thinking_delta", "role": "worker", "text": "still reasoning"},
    ]

    async def scenario() -> None:
        app = _hosted(run, events, live=True)
        async with app.run_test() as pilot:
            await pilot.pause()
            live = app.screen.query_one("#conv-live", Static)
            assert live.display  # the in-progress turn is shown live
            # a completed turn (role.result) hands its prose off to the
            # scrollback; the pane stays up as the animated working line (the
            # run is still live -- tools execute, the next call is coming).
            with logs.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"type": "role.result", "role": "worker"}) + "\n")
            await _wait_for(
                pilot,
                lambda: "working…" in str(live.render()),
                "the live pane handoff",
            )
            assert live.display

    asyncio.run(scenario())


def test_conversation_screen_empty(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = _hosted(tmp_path / "run", None, live=True)  # a live run, no journal yet
        async with app.run_test() as pilot:
            await pilot.pause()
            assert "no conversation yet" in _body_text(app)  # the placeholder

    asyncio.run(scenario())


def test_conversation_screen_esc_backs_out(tmp_path: Path) -> None:
    """Esc leaves the conversation view: the host's to_hub exits with the
    back-to-hub code."""

    async def scenario() -> None:
        app = _hosted(tmp_path / "run", _EVENTS)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert isinstance(app.screen, ConversationScreen)
            await pilot.press("escape")
        assert app.return_code == 0

    asyncio.run(scenario())


def test_jump_to_bottom_pill_shows_when_scrolled_up(tmp_path: Path) -> None:
    """The floating jump pill appears only while the transcript is scrolled up
    (never displacing layout: it overlays), and clicking home again via its
    action returns to the tail and hides it."""
    from agent6.ui.tui.conversation import _JumpButton

    many = [dict(e) for _ in range(30) for e in _EVENTS[:-1]]  # a tall transcript

    async def scenario() -> None:
        app = _hosted(tmp_path / "run", [*many, _EVENTS[-1]])
        async with app.run_test(size=(90, 24)) as pilot:
            await pilot.pause()
            await pilot.pause()
            jump = app.screen.query_one("#conv-jump", _JumpButton)
            scroll = app.screen.query_one("#conv-scroll", VerticalScroll)
            assert scroll.max_scroll_y > 0  # tall enough to scroll
            assert not jump.display  # following the tail: hidden
            await pilot.press("ctrl+home")
            await pilot.pause()
            await pilot.pause()
            assert jump.display  # scrolled up: shown
            await pilot.press("ctrl+end")
            await pilot.pause()
            await pilot.pause()
            assert not jump.display  # back at the tail: hidden

    asyncio.run(scenario())


def test_live_pane_is_dropped_over_a_dead_worker(tmp_path: Path) -> None:
    """A worker killed mid-stream leaves its deltas in the buffers forever (only
    role.call/role.result clear them), so the pane kept saying "thinking…" over a
    corpse -- on the primary view, which carries no status label to contradict
    it. The host's dir status knows the corpse; the composer already read it."""
    events: list[dict[str, object]] = [
        {"type": "session.start", "user_task": "fix it"},
        {"type": "role.call", "role": "worker"},
        {"type": "role.thinking_delta", "text": "planning the edit"},
        {"type": "role.text_delta", "text": "I will now edit"},
        # killed here: no role.result, no session.end
    ]

    async def scenario(live: bool) -> tuple[bool, str]:
        app = _hosted(tmp_path / ("live" if live else "dead"), events, live=live)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            pane = app.screen.query_one("#conv-live", Static)
            return bool(pane.display), str(pane.content)

    shown_live, text_live = asyncio.run(scenario(True))
    assert shown_live and "thinking" in text_live  # a live turn still streams

    shown_dead, _ = asyncio.run(scenario(False))
    assert not shown_dead, "a dead worker has no in-progress turn to show"


def test_live_pane_says_waiting_while_the_operator_holds_the_answer(tmp_path: Path) -> None:
    """Blocked on an approval or a question, the run is neither thinking nor
    running a tool; the pane kept pulsing "thinking…" (the last turn's streamed
    reasoning) under the very modal asking. The host's dir status says
    "waiting"; the pane says so too."""
    events: list[dict[str, object]] = [
        {"type": "session.start", "user_task": "fix it"},
        {"type": "role.call", "role": "worker"},
        {"type": "role.thinking_delta", "text": "I will run the tests"},
        {"type": "role.result", "role": "worker"},
        {"type": "approval.prompt", "id": "approval-1", "prompt": "Allow run_command: pytest"},
    ]

    async def scenario() -> str:
        app = _hosted(tmp_path / "run", events, live=True)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            pane = app.screen.query_one("#conv-live", Static)
            assert pane.display
            return str(pane.content)

    text = asyncio.run(scenario())
    assert "waiting · needs answer" in text
    assert "thinking" not in text and "working" not in text


def test_an_ended_run_with_no_conversation_says_so_in_the_past_tense(tmp_path: Path) -> None:
    """The conversation pane is the first thing a run opens on, and it promised
    a dead run's output "appears as the session streams": the host's dir status
    knows the worker died, and the placeholder says so."""
    events: list[dict[str, object]] = [{"type": "session.start", "user_task": "do X"}]

    async def scenario() -> None:
        app = _hosted(tmp_path / "run", events)  # no worker, no end: a crash
        async with app.run_test() as pilot:
            await pilot.pause()
            body = _body_text(app)
            assert "worker exited without finishing" in body
            assert "as the session streams" not in body

    asyncio.run(scenario())


def test_live_pane_keeps_moving_between_events(tmp_path: Path) -> None:
    """Between a turn's end and the next delta the live pane VANISHED, so the
    primary view read frozen for the whole model-call/tool-run stretch. Mid-run
    with empty stream buffers it now shows an animated "working…" line, and
    the spinner frame advances on data-less polls."""
    import os

    from agent6.ui.tui.app import Agent6TUI

    d = tmp_path / "live-spin"
    d.mkdir()
    evs = [
        {"type": "session.start", "session_id": d.name, "mode": "run", "user_task": "t"},
        {"type": "role.call", "role": "worker", "model": "m", "provider": "p"},
        {"type": "role.result", "ok": True},
    ]
    (d / "logs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in evs), encoding="utf-8")
    (d / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")

    async def scenario() -> None:
        app = Agent6TUI(d)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            conv = app._conv
            conv._poll()
            live = conv.query_one("#conv-live", Static)
            assert live.display, "the live pane hid between events on a live run"
            first = str(live.render())
            assert "working…" in first
            conv._poll()  # a data-less poll still turns the spinner
            assert str(live.render()) != first

    asyncio.run(scenario())


def test_an_in_flight_tool_call_shows_in_the_live_pane_then_settles(tmp_path: Path) -> None:
    """A long run_command read as a bare "working…" until its result. The
    call shows in the live pane as soon as it is seen; its settled item lands
    in the scrollback and the pane line goes with it."""
    run = tmp_path / "run"
    logs = run / "logs.jsonl"
    call = {"type": "tool.call", "name": "run_command", "args": {"argv": ["sleep", "60"]}}

    async def scenario() -> None:
        app = _hosted(run, [_EVENTS[0], {**call, "call_id": 1}], live=True)
        async with app.run_test() as pilot:
            await pilot.pause()
            live = app.screen.query_one("#conv-live", Static)
            await _wait_for(pilot, lambda: "running" in str(live.render()), "the call in the pane")
            pane = str(live.render())
            assert "→ run_command" in pane and "sleep 60" in pane
            assert "run_command" not in _body_text(app)
            result = {"type": "tool.result", "name": "run_command", "ok": True, "summary": "exit 0"}
            with logs.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({**result, "call_id": 1}) + "\n")
            await _wait_for(pilot, lambda: "exit 0" in _body_text(app), "the settled call")
            assert "running" not in str(live.render())
            assert _body_text(app).count("→ run_command") == 1

    asyncio.run(scenario())


def test_the_live_pane_says_awaiting_approval_under_an_open_prompt(tmp_path: Path) -> None:
    """The dispatcher journals tool.call before the approval gate, so the
    call is in flight while its prompt is open: the host's dir status says
    the run waits on the answer, the pane says so, and the call never reads
    "running"."""
    events: list[dict[str, object]] = [
        _EVENTS[0],
        {"type": "tool.call", "name": "run_command", "args": {"argv": ["ls"]}, "call_id": 1},
        {"type": "approval.prompt", "id": "ap1", "prompt": "Allow run_command: ls", "call_id": 1},
    ]

    async def scenario() -> None:
        app = _hosted(tmp_path / "run", events, live=True)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ConversationScreen)
            await _wait_for(pilot, lambda: bool(screen.query(ApprovalRow)), "the approval row")
            live = screen.query_one("#conv-live", Static)
            assert "waiting · needs answer" in str(live.render())
            assert "running" not in str(live.render())

    asyncio.run(scenario())
