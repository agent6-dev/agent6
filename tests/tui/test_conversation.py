# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Headless drive of the TUI conversation viewer (ConversationScreen)."""

from __future__ import annotations

import asyncio
import bisect
import json
import time
from pathlib import Path
from typing import Any

from textual.app import App
from textual.containers import VerticalScroll
from textual.widgets import Static

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


def _body_text(app: App[None]) -> str:
    # The scrollback is a sequence of chunk Statics (selectable), in DOM order.
    return "\n".join(str(w.content) for w in app.screen.query(".conv-chunk").results(Static))


def _nlines(app: App[None]) -> int:
    return len([ln for ln in _body_text(app).splitlines() if ln.strip()])


_EVENTS: list[dict[str, object]] = [
    {"type": "run.start", "user_task": "do X"},
    {"type": "role.call", "role": "worker"},
    {"type": "role.thinking_delta", "role": "worker", "text": "thinking hard here"},
    {"type": "role.text_delta", "role": "worker", "text": "on it"},
    {"type": "role.result", "role": "worker"},
    {"type": "tool.call", "name": "read_file", "args": {"path": "a"}},
    {"type": "tool.result", "name": "read_file", "ok": True, "summary": "12 bytes"},
    {"type": "run.end", "all_passed": True, "reason": "finish_run"},
]


class _Host(App[None]):
    def __init__(self, logs_path: Path) -> None:
        super().__init__()
        self._logs = logs_path

    def on_mount(self) -> None:
        self.push_screen(ConversationScreen(self._logs, title="conversation · test"))


def _write(logs: Path, events: list[dict[str, object]]) -> None:
    logs.write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")


def test_conversation_screen_cycles_detail_level(tmp_path: Path) -> None:
    logs = tmp_path / "logs.jsonl"
    _write(logs, _EVENTS)

    async def scenario() -> None:
        app = _Host(logs)
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
            screen.action_cycle_detail()  # collapsed -> expanded
            await pilot.pause()
            assert "thinking hard here" in body_text()  # full reasoning now shown
            screen.action_cycle_detail()  # expanded -> hidden
            await pilot.pause()
            assert "thinking" not in body_text()  # thinking omitted entirely
            screen.action_reload()  # reload must not raise
            await pilot.pause()

    asyncio.run(scenario())


def test_conversation_screen_follows_live(tmp_path: Path) -> None:
    """Events appended after mount (a live run / a resume) show up via the poll."""
    logs = tmp_path / "logs.jsonl"
    logs.write_text("", encoding="utf-8")

    async def scenario() -> None:
        app = _Host(logs)
        async with app.run_test() as pilot:
            await pilot.pause()
            before = _nlines(app)
            with logs.open("a", encoding="utf-8") as fh:
                for event in _EVENTS:
                    fh.write(json.dumps(event) + "\n")
            await _wait_for(pilot, lambda: _nlines(app) > before, "the appended turns")

    asyncio.run(scenario())


def test_steer_bar_hidden_for_a_finished_run(tmp_path: Path) -> None:
    logs = tmp_path / "logs.jsonl"
    _write(logs, _EVENTS)  # _EVENTS ends with run.end -> nothing to steer

    async def scenario() -> None:
        app = _Host(logs)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert not app.screen.query_one("#conv-input", SteerInput).display

    asyncio.run(scenario())


def test_steer_bar_shows_for_a_live_run_and_submits_over_the_bridge(tmp_path: Path) -> None:
    from agent6.runs.ipc import STEER_ANSWER_FILE, steer_request_pending

    logs = tmp_path / "logs.jsonl"
    _write(logs, _EVENTS[:-1])  # drop run.end -> the run is live

    async def scenario() -> None:
        app = _Host(logs)
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.screen.query_one("#conv-input", SteerInput)
            assert bar.display  # a live run shows the steer bar
            bar.post_message(SteerInput.Submitted("go left"))
            await pilot.pause()

    asyncio.run(scenario())
    assert steer_request_pending(tmp_path)  # the run was asked to steer
    assert (tmp_path / STEER_ANSWER_FILE).read_text(encoding="utf-8") == "go left"


def test_live_run_auto_focuses_the_steer_bar(tmp_path: Path) -> None:
    logs = tmp_path / "logs.jsonl"
    _write(logs, _EVENTS[:-1])  # live -> bar ready to type

    async def scenario() -> None:
        app = _Host(logs)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert isinstance(app.focused, SteerInput)

    asyncio.run(scenario())


def test_esc_backs_out_even_with_the_bar_focused(tmp_path: Path) -> None:
    # A live run auto-focuses the bar; Esc is a priority binding, so it still closes
    # the view (back to the dashboard) instead of the bar eating the key.
    logs = tmp_path / "logs.jsonl"
    _write(logs, _EVENTS[:-1])

    async def scenario() -> None:
        app = _Host(logs)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert isinstance(app.focused, SteerInput)
            await pilot.press("escape")
            await pilot.pause()
            assert not isinstance(app.screen, ConversationScreen)

    asyncio.run(scenario())


def test_follow_survives_the_live_pane_growing(tmp_path: Path) -> None:
    # A live turn that only THINKS (no completed turn appended) still grows the live
    # pane, shrinking the scroll viewport. Follow mode must survive that nudge.
    logs = tmp_path / "logs.jsonl"
    events: list[dict[str, object]] = [{"type": "run.start", "user_task": "x"}]
    for i in range(20):  # overflow a short viewport
        more: list[dict[str, object]] = [
            {"type": "tool.call", "name": "read_file", "args": {"path": f"f{i}"}},
            {"type": "tool.result", "name": "read_file", "ok": True, "summary": f"{i} bytes"},
        ]
        events += more
    _write(logs, events)  # no run.end -> live

    async def scenario() -> None:
        app = _Host(logs)
        async with app.run_test(size=(60, 12)) as pilot:
            await pilot.pause()
            scroll = app.screen.query_one("#conv-scroll", VerticalScroll)
            assert _following(scroll)  # _reload pins to the bottom
            overflow_before = scroll.max_scroll_y
            with logs.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"type": "role.thinking_delta", "text": "x " * 300}) + "\n")
            # The growing live pane shrinks the viewport, so the overflow grows.
            await _wait_for(
                pilot, lambda: scroll.max_scroll_y > overflow_before, "the live pane to grow"
            )
            assert _following(scroll)  # still following despite the live pane growing

    asyncio.run(scenario())


def test_detail_cycle_keeps_the_top_block_anchored(tmp_path: Path) -> None:
    # Expanding a big failed-tool block above the viewport must not carry your place
    # away: the block at the top of the viewport stays put across the re-render.
    logs = tmp_path / "logs.jsonl"
    events: list[dict[str, object]] = [{"type": "run.start", "user_task": "x"}]
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
    _write(logs, events)

    async def scenario() -> None:
        app = _Host(logs)
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
    logs = tmp_path / "logs.jsonl"
    _write(
        logs,
        [
            {"type": "run.start", "user_task": "do X"},
            {"type": "role.call", "role": "worker"},
            {"type": "role.thinking_delta", "role": "worker", "text": "still reasoning"},
        ],
    )

    async def scenario() -> None:
        app = _Host(logs)
        async with app.run_test() as pilot:
            await pilot.pause()
            live = app.screen.query_one("#conv-live", Static)
            assert live.display  # the in-progress turn is shown live
            # a completed turn (role.result) hands off to the scrollback and hides it
            with logs.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"type": "role.result", "role": "worker"}) + "\n")
            await _wait_for(pilot, lambda: not live.display, "the live pane handoff")

    asyncio.run(scenario())


def test_conversation_screen_empty(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = _Host(tmp_path / "missing.jsonl")  # no log file
        async with app.run_test() as pilot:
            await pilot.pause()
            assert "no conversation yet" in _body_text(app)  # the placeholder

    asyncio.run(scenario())


def test_conversation_screen_esc_backs_out(tmp_path: Path) -> None:
    """Esc closes the conversation view -- backs out one level."""
    logs = tmp_path / "logs.jsonl"
    _write(logs, _EVENTS)

    async def scenario() -> None:
        app = _Host(logs)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert isinstance(app.screen, ConversationScreen)
            await pilot.press("escape")
            await pilot.pause()
            assert not isinstance(app.screen, ConversationScreen)  # backed out

    asyncio.run(scenario())


def test_jump_to_bottom_pill_shows_when_scrolled_up(tmp_path: Path) -> None:
    """The floating jump pill appears only while the transcript is scrolled up
    (never displacing layout: it overlays), and clicking home again via its
    action returns to the tail and hides it."""
    from agent6.ui.tui.conversation import _JumpButton

    logs = tmp_path / "logs.jsonl"
    many = [dict(e) for _ in range(30) for e in _EVENTS[:-1]]  # a tall transcript
    _write(logs, [*many, _EVENTS[-1]])

    async def scenario() -> None:
        app = _Host(logs)
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
