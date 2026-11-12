# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Headless drive of the TUI conversation viewer (ConversationScreen)."""

from __future__ import annotations

import asyncio
import bisect
import json
import os
import pathlib

import pytest
from textual import containers, widgets

from agent6.ui.tui import app as tui_app
from agent6.ui.tui import composer, conversation
from tests.tui import waits


def _following(scroll: containers.VerticalScroll) -> bool:
    return scroll.max_scroll_y - scroll.scroll_y <= 2.0


def _body_text(app: tui_app.Agent6TUI) -> str:
    # The scrollback is a sequence of chunk Statics (selectable), in DOM order.
    return "\n".join(
        str(w.content) for w in app.screen.query(".conv-chunk").results(widgets.Static)
    )


def _nlines(app: tui_app.Agent6TUI) -> int:
    return len([ln for ln in _body_text(app).splitlines() if ln.strip()])


_EVENTS: list[dict[str, object]] = [
    {"type": "session.start", "user_task": "do X"},
    {"type": "model.call", "role": "worker"},
    # Multi-line on purpose: collapsed shows only the first line, so expanded is distinguishable.
    {
        "type": "model.thinking_delta",
        "role": "worker",
        "text": "thinking hard here\nsecond thought",
    },
    {"type": "model.text_delta", "role": "worker", "text": "on it"},
    {"type": "model.result", "role": "worker"},
    {"type": "tool.call", "name": "read_file", "args_preview": {"path": "a"}},
    {"type": "tool.result", "name": "read_file", "ok": True, "summary": "12 bytes"},
    {"type": "session.stop", "verification": "passed", "reason": "finish_session"},
]


def _write(logs: pathlib.Path, events: list[dict[str, object]]) -> None:
    logs.write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")


def _hosted(
    run: pathlib.Path, events: list[dict[str, object]] | None, *, live: bool = False
) -> tui_app.Agent6TUI:
    """Return the run app over the run, with the conversation screen as its first view.

    `live` plants this process as the run's worker, so the dir status reads live; None writes
    no journal at all.
    """
    run.mkdir(parents=True, exist_ok=True)
    if events is not None:
        _write(run / "logs.jsonl", events)
    if live:
        (run / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")
    return tui_app.Agent6TUI(run)


def test_conversation_opens_with_the_task_and_policy_line(tmp_path: pathlib.Path) -> None:
    """The TUI conversation opens with the same task and policy as the CLI stream."""
    from agent6.viewmodel import policy

    async def scenario() -> None:
        app = _hosted(tmp_path / "run", _EVENTS)
        (app.session_dir / "manifest.json").write_text(
            json.dumps({"mode": "run", "policy": {"isolation": "full", "run_commands": "ask"}}),
            encoding="utf-8",
        )
        assert policy.session_policy(app.session_dir).line()
        async with app.run_test() as pilot:
            await pilot.pause()
            # The policy sits directly under the headline, no blank line between them.
            lines = [line.strip() for line in _body_text(app).splitlines()]
            assert lines[:3] == ["● do X", policy.session_policy(app.session_dir).line(), ""]

    asyncio.run(scenario())


def test_expanded_conversation_keeps_full_tasks_and_steers(tmp_path: pathlib.Path) -> None:
    """The detail toggle reveals whole prompts rather than only their headlines."""
    words = "Keep <file>literal</file>. " + "words " * 900 + "\nLAST PROMPT LINE"
    steer = "Continue with " + "details " * 900 + "\nLAST STEER LINE"

    async def scenario() -> None:
        app = _hosted(
            tmp_path / "run",
            [
                {"type": "session.start", "user_task": words},
                {"type": "steer.injected", "text": steer},
            ],
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, conversation.ConversationScreen)
            assert "LAST PROMPT LINE" not in _body_text(app)
            assert "LAST STEER LINE" not in _body_text(app)
            screen.action_cycle_detail()
            await pilot.pause()
            body = _body_text(app).replace("\n  ", "\n")
            assert words in body and steer in body

    asyncio.run(scenario())


def test_conversation_screen_cycles_detail_level(tmp_path: pathlib.Path) -> None:
    async def scenario() -> None:
        app = _hosted(tmp_path / "run", _EVENTS)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, conversation.ConversationScreen)

            def body_text() -> str:
                return _body_text(app)

            assert _nlines(app) > 0  # the conversation rendered
            # Collapsed default: the first line of the reasoning with a more-count, not the bulk.
            assert "thinking hard here" in body_text()
            assert body_text().count("thinking hard here") == 1
            assert "second thought" not in body_text()  # the bulk stays folded
            screen.action_cycle_detail()  # collapsed -> expanded
            await pilot.pause()
            assert "second thought" in body_text()  # the bulk is shown
            screen.action_cycle_detail()  # expanded -> hidden
            await pilot.pause()
            assert "thinking" not in body_text()  # thinking omitted entirely
            screen.action_reload()  # reload must not raise
            await pilot.pause()

    asyncio.run(scenario())


def test_an_unrendered_finish_call_with_bad_args_does_not_crash_the_screen(
    tmp_path: pathlib.Path,
) -> None:
    """A corrupt finish call's non-object args do not take down the conversation screen."""
    events: list[dict[str, object]] = [
        {"type": "session.start", "user_task": "do X"},
        {"type": "tool.call", "name": "finish_session", "args_preview": "not an object"},
        {"type": "session.stop", "verification": "passed", "reason": "finish_session"},
    ]

    async def scenario() -> None:
        app = _hosted(tmp_path / "run", events)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert isinstance(app.screen, conversation.ConversationScreen)
            assert "● passed" in _body_text(app)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("reason", "label"),
    [
        ("silent_finish", "finished · no finish call"),
        ("metric_plateau", "finished · metric plateau"),
    ],
)
def test_harness_end_labels_reach_the_conversation_and_dashboard(
    tmp_path: pathlib.Path, reason: str, label: str
) -> None:
    """A harness-declared end must not read passed in the done line, title or dashboard."""
    events: list[dict[str, object]] = [
        {"type": "session.start", "mode": "run", "user_task": "t"},
        {"type": "session.stop", "reason": reason, "verification": "passed"},
    ]

    async def scenario() -> None:
        run = tmp_path / "run"
        run.mkdir()
        _write(run / "logs.jsonl", events)
        app = tui_app.Agent6TUI(run, exit_on_end=True)
        async with app.run_test(size=(140, 40)) as pilot:
            await waits.wait_for(
                pilot, lambda: label in _body_text(app), "the harness end's done line"
            )
            await waits.wait_for(pilot, lambda: label in app.run_title(), "the end-hold title")
            assert label in app.run_title()
            await pilot.press("ctrl+d")
            await pilot.pause()
            assert label in str(app.screen.query_one("#stream-body", widgets.Static).render())
            assert label in str(app.screen.query_one("#top", widgets.Static).render())

    asyncio.run(scenario())


def test_conversation_screen_follows_live(tmp_path: pathlib.Path) -> None:
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
            await waits.wait_for(pilot, lambda: _nlines(app) > before, "the appended turns")

    asyncio.run(scenario())


def test_steer_bar_stays_for_a_finished_run_as_the_resume_composer(tmp_path: pathlib.Path) -> None:
    """After the run ends the bar stays, and Enter resumes the run with the typed follow-up."""

    async def scenario() -> None:
        app = _hosted(tmp_path / "run", _EVENTS)  # ends with session.stop: finished
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.screen.query_one("#conv-input", composer.SteerInput).display

    asyncio.run(scenario())


def test_steer_bar_shows_for_a_live_run_and_submits_over_the_bridge(tmp_path: pathlib.Path) -> None:
    from agent6.sessions import ipc

    run = tmp_path / "run"

    async def scenario() -> None:
        app = _hosted(run, _EVENTS[:-1], live=True)  # no session.stop, a live worker
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.screen.query_one("#conv-input", composer.SteerInput)
            assert bar.display  # a live run shows the steer bar
            bar.post_message(composer.SteerInput.Submitted("go left"))
            await pilot.pause()

    asyncio.run(scenario())
    assert ipc.steer_request_pending(run)  # the run was asked to steer
    assert (run / ipc.STEER_ANSWER_FILE).read_text(encoding="utf-8") == "go left"


def test_resumed_execution_is_live_and_steers_over_the_bridge(tmp_path: pathlib.Path) -> None:
    """A resumed execution emits ONLY session.resume (never a second session.start).

    The screen must read it as live, as the host's dir status does once the execution's worker is
    up, so a submit routes to the steer bridge; an execution read as finished had Enter spawn a
    second resume that died on the run lock while the toast claimed the instruction was delivered.
    """
    from agent6.sessions import ipc

    run = tmp_path / "run"
    logs = run / "logs.jsonl"

    async def scenario() -> None:
        app = _hosted(run, _EVENTS)  # ends with session.stop: finished, no worker
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, conversation.ConversationScreen)
            assert not app.session_controllable()  # finished execution
            with logs.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"type": "session.resume", "turn": 1}) + "\n")
                fh.write(json.dumps({"type": "model.call", "role": "worker"}) + "\n")
            (run / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")
            await waits.wait_for(
                pilot, app.session_controllable, "the resumed execution to read live"
            )
            bar = screen.query_one("#conv-input", composer.SteerInput)
            bar.post_message(composer.SteerInput.Submitted("also update docs"))
            await pilot.pause()

    asyncio.run(scenario())
    # Routed over the steer bridge, not a second doomed `agent6 resume`.
    assert ipc.steer_request_pending(run)
    assert (run / ipc.STEER_ANSWER_FILE).read_text(encoding="utf-8") == "also update docs"


def test_live_run_auto_focuses_the_steer_bar(tmp_path: pathlib.Path) -> None:
    async def scenario() -> None:
        app = _hosted(tmp_path / "run", _EVENTS[:-1], live=True)  # bar ready to type
        async with app.run_test() as pilot:
            await pilot.pause()
            assert isinstance(app.focused, composer.SteerInput)

    asyncio.run(scenario())


def test_esc_backs_out_even_with_the_bar_focused(tmp_path: pathlib.Path) -> None:
    # Esc is a priority binding, so it leaves the view instead of the focused bar eating the key.
    async def scenario() -> None:
        app = _hosted(tmp_path / "run", _EVENTS[:-1], live=True)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert isinstance(app.focused, composer.SteerInput)
            await pilot.press("escape")
        assert app.return_code == 0  # to_hub: back to the hub loop

    asyncio.run(scenario())


def test_follow_survives_the_live_pane_growing(tmp_path: pathlib.Path) -> None:
    # A turn that only thinks grows the live pane and shrinks the viewport; follow survives it.
    logs = tmp_path / "logs.jsonl"
    events: list[dict[str, object]] = [{"type": "session.start", "user_task": "x"}]
    for i in range(20):  # overflow a short viewport
        more: list[dict[str, object]] = [
            {"type": "tool.call", "name": "read_file", "args_preview": {"path": f"f{i}"}},
            {"type": "tool.result", "name": "read_file", "ok": True, "summary": f"{i} bytes"},
        ]
        events += more
    run = tmp_path / "run"
    logs = run / "logs.jsonl"

    async def scenario() -> None:
        app = _hosted(run, events, live=True)  # no session.stop, a live worker
        async with app.run_test(size=(60, 12)) as pilot:
            await pilot.pause()
            scroll = app.screen.query_one("#conv-scroll", containers.VerticalScroll)
            assert _following(scroll)  # _reload pins to the bottom
            # Expanded detail streams the reasoning tail into the live pane, growing it.
            conv_screen = app.screen
            assert isinstance(conv_screen, conversation.ConversationScreen)
            conv_screen._detail = "expanded"
            overflow_before = scroll.max_scroll_y
            with logs.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"type": "model.call", "role": "worker", "model": "m"}) + "\n")
                fh.write(json.dumps({"type": "model.thinking_delta", "text": "x " * 300}) + "\n")
            # The growing live pane shrinks the viewport, so the overflow grows.
            await waits.wait_for(
                pilot, lambda: scroll.max_scroll_y > overflow_before, "the live pane to grow"
            )
            # Follow re-pins on the next tick; a real follow break never re-pins.
            await waits.wait_for(pilot, lambda: _following(scroll), "follow to re-pin after growth")

    asyncio.run(scenario())


def test_detail_cycle_keeps_the_top_block_anchored(tmp_path: pathlib.Path) -> None:
    # Expanding a big block above the viewport keeps the block at the top of the viewport in place.
    events: list[dict[str, object]] = [{"type": "session.start", "user_task": "x"}]
    big: list[dict[str, object]] = [
        {"type": "tool.call", "name": "apply_edit", "args_preview": {"path": "b"}},
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
            {"type": "tool.call", "name": "grep", "args_preview": {"pattern": f"m{i}"}},
            {"type": "tool.result", "name": "grep", "ok": True, "summary": f"{i} hits"},
        ]
        events += row

    async def scenario() -> None:
        app = _hosted(tmp_path / "run", events)
        async with app.run_test(size=(80, 16)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, conversation.ConversationScreen)
            scroll = screen.query_one("#conv-scroll", containers.VerticalScroll)
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


def test_conversation_live_pane_shows_the_in_progress_turn(tmp_path: pathlib.Path) -> None:
    # A turn still thinking shows in the live pane, so a long generation does not look frozen.
    run = tmp_path / "run"
    logs = run / "logs.jsonl"
    events: list[dict[str, object]] = [
        {"type": "session.start", "user_task": "do X"},
        {"type": "model.call", "role": "worker"},
        {"type": "model.thinking_delta", "role": "worker", "text": "still reasoning"},
    ]

    async def scenario() -> None:
        app = _hosted(run, events, live=True)
        async with app.run_test() as pilot:
            await pilot.pause()
            live = app.screen.query_one("#conv-live", widgets.Static)
            assert live.display  # the in-progress turn is shown live
            # A completed turn hands its prose to the scrollback; no next call is implied.
            with logs.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"type": "model.result", "role": "worker"}) + "\n")
            await waits.wait_for(pilot, lambda: not live.display, "the live pane handoff")

    asyncio.run(scenario())


def test_conversation_screen_empty(tmp_path: pathlib.Path) -> None:
    async def scenario() -> None:
        app = _hosted(tmp_path / "run", None, live=True)  # a live run, no journal yet
        async with app.run_test() as pilot:
            await pilot.pause()
            assert "no conversation yet" in _body_text(app)  # the placeholder

    asyncio.run(scenario())


def test_conversation_screen_esc_backs_out(tmp_path: pathlib.Path) -> None:
    """Esc leaves the conversation view: the host's to_hub exits with the back-to-hub code."""

    async def scenario() -> None:
        app = _hosted(tmp_path / "run", _EVENTS)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert isinstance(app.screen, conversation.ConversationScreen)
            await pilot.press("escape")
        assert app.return_code == 0

    asyncio.run(scenario())


def test_jump_to_bottom_pill_shows_when_scrolled_up(tmp_path: pathlib.Path) -> None:
    """The jump pill overlays only while scrolled up; its action returns to the tail."""
    many = [dict(e) for _ in range(30) for e in _EVENTS[:-1]]  # a tall transcript

    async def scenario() -> None:
        app = _hosted(tmp_path / "run", [*many, _EVENTS[-1]])
        async with app.run_test(size=(90, 24)) as pilot:
            await pilot.pause()
            await pilot.pause()
            jump = app.screen.query_one("#conv-jump", conversation._JumpButton)
            scroll = app.screen.query_one("#conv-scroll", containers.VerticalScroll)
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


def test_live_pane_is_dropped_over_a_dead_worker(tmp_path: pathlib.Path) -> None:
    """A worker killed mid-stream leaves no "thinking…" over a corpse on the primary view.

    Only model.call and model.result clear the delta buffers; the host's dir status knows the
    corpse.
    """
    events: list[dict[str, object]] = [
        {"type": "session.start", "user_task": "fix it"},
        {"type": "model.call", "role": "worker"},
        {"type": "model.thinking_delta", "text": "planning the edit"},
        {"type": "model.text_delta", "text": "I will now edit"},
        # killed here: no model.result, no session.stop
    ]

    async def scenario(live: bool) -> tuple[bool, str]:
        app = _hosted(tmp_path / ("live" if live else "dead"), events, live=live)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            pane = app.screen.query_one("#conv-live", widgets.Static)
            return bool(pane.display), str(pane.content)

    shown_live, text_live = asyncio.run(scenario(True))
    assert shown_live and "thinking" in text_live  # a live turn still streams

    shown_dead, _ = asyncio.run(scenario(False))
    assert not shown_dead, "a dead worker has no in-progress turn to show"


def test_live_pane_says_waiting_while_the_operator_holds_the_answer(tmp_path: pathlib.Path) -> None:
    """Blocked on an approval or a question, the pane says waiting, not "thinking…"."""
    events: list[dict[str, object]] = [
        {"type": "session.start", "user_task": "fix it"},
        {"type": "model.call", "role": "worker"},
        {"type": "model.thinking_delta", "text": "I will run the tests"},
        {"type": "model.result", "role": "worker"},
        {"type": "approval.prompt", "id": "approval-1", "prompt": "Allow run_command: pytest"},
    ]

    async def scenario() -> str:
        app = _hosted(tmp_path / "run", events, live=True)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.pause()
            pane = app.screen.query_one("#conv-live", widgets.Static)
            assert pane.display
            return str(pane.content)

    text = asyncio.run(scenario())
    assert "waiting · needs answer" in text
    assert "thinking" not in text and "working" not in text


def test_an_ended_run_with_no_conversation_says_so_in_the_past_tense(
    tmp_path: pathlib.Path,
) -> None:
    """The placeholder of a dead run says the worker died, not that output will stream."""
    events: list[dict[str, object]] = [{"type": "session.start", "user_task": "do X"}]

    async def scenario() -> None:
        app = _hosted(tmp_path / "run", events)  # no worker, no end: a crash
        async with app.run_test() as pilot:
            await pilot.pause()
            body = _body_text(app)
            assert "worker exited without finishing" in body
            assert "as the session streams" not in body

    asyncio.run(scenario())


def test_live_pane_moves_only_while_a_model_call_is_in_flight(tmp_path: pathlib.Path) -> None:
    """A live worker between calls is not a model at work.

    The live pane stays hidden after model.result, then appears and advances when the next
    model.call enters flight.
    """
    import os

    d = tmp_path / "live-spin"
    d.mkdir()
    evs = [
        {"type": "session.start", "session_id": d.name, "mode": "run", "user_task": "t"},
        {"type": "model.call", "role": "worker", "model": "m", "provider": "p"},
        {"type": "model.result", "ok": True},
    ]
    logs = d / "logs.jsonl"
    logs.write_text("".join(json.dumps(e) + "\n" for e in evs), encoding="utf-8")
    (d / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")

    async def scenario() -> None:
        app = tui_app.Agent6TUI(d)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            conv = app._conv
            conv._poll()
            live = conv.query_one("#conv-live", widgets.Static)
            assert not live.display
            with logs.open("a", encoding="utf-8") as fh:
                fh.write(
                    json.dumps(
                        {"type": "model.call", "role": "worker", "model": "m", "provider": "p"}
                    )
                    + "\n"
                )
            await waits.wait_for(
                pilot,
                lambda: app.state.last_role is not None and app.state.last_role.in_flight,
                "the next model call",
            )
            conv._poll()
            assert live.display
            first = str(live.render())
            assert "working…" in first
            conv._poll()
            assert str(live.render()) != first

    asyncio.run(scenario())


def test_an_in_flight_tool_call_shows_in_the_live_pane_then_settles(tmp_path: pathlib.Path) -> None:
    """A long run_command read as a bare "working…" until its result.

    The call shows in the live pane as soon as it is seen; its settled item lands in the scrollback
    and the pane line goes with it.
    """
    run = tmp_path / "run"
    logs = run / "logs.jsonl"
    call = {"type": "tool.call", "name": "run_command", "args_preview": {"argv": ["sleep", "60"]}}

    async def scenario() -> None:
        app = _hosted(run, [_EVENTS[0], {**call, "call_id": 1}], live=True)
        async with app.run_test() as pilot:
            await pilot.pause()
            live = app.screen.query_one("#conv-live", widgets.Static)
            await waits.wait_for(
                pilot, lambda: "running" in str(live.render()), "the call in the pane"
            )
            pane = str(live.render())
            assert "→ run_command" in pane and "sleep 60" in pane
            assert "run_command" not in _body_text(app)
            result = {"type": "tool.result", "name": "run_command", "ok": True, "summary": "exit 0"}
            with logs.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({**result, "call_id": 1}) + "\n")
            await waits.wait_for(pilot, lambda: "exit 0" in _body_text(app), "the settled call")
            assert not live.display
            assert _body_text(app).count("→ run_command") == 1

    asyncio.run(scenario())


def test_verification_updates_one_item_in_both_tui_views(tmp_path: pathlib.Path) -> None:
    """Baseline approval, execution and result share one visible marker in both TUI views."""
    from tests.unit import test_transcript_verify

    rows = test_transcript_verify.attempt()
    run = tmp_path / "run"

    async def scenario() -> None:
        app = _hosted(run, [_EVENTS[0], *rows[:2]], live=True)
        async with app.run_test() as pilot:
            await pilot.pause()
            conv = app._conv
            live = conv.query_one("#conv-live", widgets.Static)
            for row, expected in [
                (None, "verify: waiting for approval"),
                (rows[2], "verify: running `pytest -q`"),
                (rows[3], "verify: ✓ pass · 0.2s"),
            ]:
                if row is not None:
                    with (run / "logs.jsonl").open("a") as stream:
                        stream.write(json.dumps(row) + "\n")
                await waits.wait_for(
                    pilot,
                    lambda expected=expected: (
                        expected in _body_text(app) + (str(live.render()) if live.display else "")
                        and expected in app._dash._stream_story(app.state, active=False).plain
                    ),
                    expected,
                )
                shown = _body_text(app) + (str(live.render()) if live.display else "")
                assert shown.count("verify:") == 1
                assert expected in app._dash._stream_story(app.state, active=False).plain

    asyncio.run(scenario())


def test_the_live_pane_says_awaiting_approval_under_an_open_prompt(tmp_path: pathlib.Path) -> None:
    """A call in flight while its approval prompt is open reads as waiting, never running.

    The dispatcher journals tool.call before the approval gate.
    """
    events: list[dict[str, object]] = [
        _EVENTS[0],
        {
            "type": "tool.call",
            "name": "run_command",
            "args_preview": {"argv": ["ls"]},
            "call_id": 1,
        },
        {"type": "approval.prompt", "id": "ap1", "prompt": "Allow run_command: ls", "call_id": 1},
    ]

    async def scenario() -> None:
        app = _hosted(tmp_path / "run", events, live=True)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, conversation.ConversationScreen)
            await waits.wait_for(
                pilot, lambda: bool(screen.query(composer.ApprovalRow)), "the approval row"
            )
            live = screen.query_one("#conv-live", widgets.Static)
            assert "waiting · needs answer" in str(live.render())
            assert "running" not in str(live.render())

    asyncio.run(scenario())
