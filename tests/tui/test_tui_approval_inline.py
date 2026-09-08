# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""An approval on the conversation screen is an inline item plus a docked key
row, never a modal: the conversation stays scrollable and readable while the
command under judgment sits at its tail, one key answers, and the item
collapses to a dim line once answered."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

from textual.widgets import Static

from agent6.ui.tui.app import Agent6TUI
from agent6.ui.tui.composer import ApprovalRow, SteerInput
from agent6.ui.tui.modals import ApprovalModal


def _live_run(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "approvals").mkdir(exist_ok=True)
    (d / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")
    evs = [
        {"type": "session.start", "session_id": d.name, "mode": "run", "user_task": "add it"},
        {"type": "role.call", "role": "worker", "model": "m"},
    ]
    (d / "logs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in evs), encoding="utf-8")


def _append(d: Path, ev: dict[str, object]) -> None:
    with (d / "logs.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(ev) + "\n")


def test_an_approval_is_an_inline_item_with_a_key_row(tmp_path: Path) -> None:
    run = tmp_path / "live-run-AAAAAA"
    _live_run(run)

    async def scenario() -> None:
        app = Agent6TUI(run)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            _append(
                run,
                {
                    "type": "approval.prompt",
                    "id": "ap1",
                    "prompt": "Allow run_command: pytest -q tests/unit/test_x.py",
                    "standing": True,
                },
            )
            app._conv._poll()  # pyright: ignore[reportPrivateUsage]
            await pilot.pause()
            await pilot.pause()
            assert not isinstance(app.screen, ApprovalModal)
            assert await _row_shown(app, pilot)
            item = app._conv.query_one("#conv-approval", Static)  # pyright: ignore[reportPrivateUsage]
            assert item.display
            text = str(item.render())
            assert "approval needed" in text and "pytest -q tests/unit/test_x.py" in text
            # The composer keeps focus: an empty one lets the row's keys answer.
            assert app.focused is app._conv.query_one("#conv-input", SteerInput)  # pyright: ignore[reportPrivateUsage]
            await pilot.press("a")
            assert await _answer_written(run, pilot) == "yes"
            _append(run, {"type": "approval.answer", "id": "ap1", "approved": True})
            app._conv._poll()  # pyright: ignore[reportPrivateUsage]
            await pilot.pause()
            await pilot.pause(0.3)
            app._tick()  # pyright: ignore[reportPrivateUsage]
            app._conv._poll()  # pyright: ignore[reportPrivateUsage]
            await pilot.pause()
            assert not app._conv.query(ApprovalRow)  # pyright: ignore[reportPrivateUsage]
            assert "allowed" in str(item.render())

    asyncio.run(scenario())


async def _answer_written(run: Path, pilot: Any, name: str = "ap1") -> str:
    """The answer file's text once the click's or key's answer has landed
    through the host: a single pause is not enough under a loaded gate."""
    path = run / "approvals" / f"{name}.answer"
    for _ in range(80):
        if path.exists():
            break
        await pilot.pause(0.05)
    return path.read_text(encoding="utf-8")


async def _row_shown(app: Agent6TUI, pilot: Any) -> bool:
    """Whether the approval row is up. The host folds the journal in its own
    thread, so the row follows within a few ticks rather than one pause."""
    for _ in range(80):
        if app._conv.query(ApprovalRow):  # pyright: ignore[reportPrivateUsage]
            return True
        app._conv._poll()  # pyright: ignore[reportPrivateUsage]
        await pilot.pause(0.05)
    return bool(app._conv.query(ApprovalRow))  # pyright: ignore[reportPrivateUsage]


async def _open_approval(app: Agent6TUI, pilot: Any, run: Path) -> None:
    await pilot.pause()
    await pilot.pause()
    prompt = {"type": "approval.prompt", "id": "ap1", "prompt": "Allow run_command: ls"}
    _append(run, {**prompt, "standing": True})
    app._conv._poll()  # pyright: ignore[reportPrivateUsage]
    await pilot.pause()
    await pilot.pause()
    assert await _row_shown(app, pilot)


def test_a_typed_message_never_answers_the_approval(tmp_path: Path) -> None:
    """The row took focus, so a sentence typed at the composer answered the
    approval on its first `s`. The composer keeps focus and the keys fire only
    while it is empty: the text lands, nothing is granted."""
    run = tmp_path / "live-run-CCCCCC"
    _live_run(run)

    async def scenario() -> None:
        app = Agent6TUI(run)
        async with app.run_test(size=(120, 40)) as pilot:
            await _open_approval(app, pilot, run)
            await pilot.press("slash", "b", "t", "w", "space", "s", "u", "r", "e")
            await pilot.pause()
            assert not (run / "approvals" / "ap1.answer").exists()
            bar = app._conv.query_one("#conv-input", SteerInput)  # pyright: ignore[reportPrivateUsage]
            assert bar.text == "/btw sure"
            assert await _row_shown(app, pilot)

    asyncio.run(scenario())


def test_a_resumed_leg_drops_the_previous_legs_approval(tmp_path: Path) -> None:
    """A leg boundary must withdraw an unanswered approval from the dead leg."""
    run = tmp_path / "live-run-HHHHHH"
    _live_run(run)

    async def scenario() -> None:
        app = Agent6TUI(run)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            prompt: dict[str, object] = {
                "type": "approval.prompt",
                "id": "ap1",
                "prompt": "Allow run_command: ls",
            }
            # The journal is the one feed: the host app and this screen both
            # tail it in order. Feeding the app by hand as well let its tail
            # re-deliver the prompt after a hand-fed boundary under load, and
            # the row flapped.
            _append(run, prompt)
            app._conv._poll()  # pyright: ignore[reportPrivateUsage]
            await pilot.pause()
            assert await _row_shown(app, pilot)
            boundary: dict[str, object] = {"type": "loop.resume.start", "iteration": 2}
            _append(run, boundary)
            # The withdrawal is an async DOM prune after the tails read the
            # boundary: poll and wait rather than trust one pause.
            for _ in range(80):
                app._conv._poll()  # pyright: ignore[reportPrivateUsage]
                await pilot.pause(0.05)
                if not app._conv.query(ApprovalRow):  # pyright: ignore[reportPrivateUsage]
                    break
            assert not app._conv.query(ApprovalRow)  # pyright: ignore[reportPrivateUsage]

    asyncio.run(scenario())


def test_an_answered_approval_stays_closed_on_reload(tmp_path: Path) -> None:
    """Reload before the worker journals its answer must not reopen the row."""
    run = tmp_path / "live-run-GGGGGG"
    _live_run(run)

    async def scenario() -> None:
        app = Agent6TUI(run)
        async with app.run_test(size=(120, 40)) as pilot:
            await _open_approval(app, pilot, run)
            await pilot.press("a")
            await pilot.pause()
            assert not app._conv.query(ApprovalRow)  # pyright: ignore[reportPrivateUsage]
            app._conv.action_reload()  # pyright: ignore[reportPrivateUsage]
            await pilot.pause()
            assert not app._conv.query(ApprovalRow)  # pyright: ignore[reportPrivateUsage]

    asyncio.run(scenario())


def test_a_click_on_a_row_label_answers(tmp_path: Path) -> None:
    run = tmp_path / "live-run-DDDDDD"
    _live_run(run)

    async def scenario() -> None:
        app = Agent6TUI(run)
        async with app.run_test(size=(120, 40)) as pilot:
            await _open_approval(app, pilot, run)
            label = app._conv.query_one(".answer-yes", Static)  # pyright: ignore[reportPrivateUsage]
            for _ in range(80):  # the row is queryable before it is laid out
                if label.region.width > 0:
                    break
                await pilot.pause(0.05)
            await pilot.click(label)
            answer = run / "approvals" / "ap1.answer"
            for _ in range(40):
                if answer.exists():
                    break
                await pilot.pause(0.05)
            else:  # a click under a loaded pilot can land before the layout settles
                await pilot.click(label)
            assert await _answer_written(run, pilot) == "yes"

    asyncio.run(scenario())


def test_a_click_after_another_surface_answered_is_refused(tmp_path: Path) -> None:
    """The web answered while the row was still up here: the click writes
    nothing, the first answer stands, and the screen says so."""
    run = tmp_path / "live-run-EEEEEE"
    _live_run(run)

    async def scenario() -> None:
        app = Agent6TUI(run)
        async with app.run_test(size=(120, 40)) as pilot:
            await _open_approval(app, pilot, run)
            (run / "approvals").mkdir(exist_ok=True)
            (run / "approvals" / "ap1.answer").write_text("no", encoding="utf-8")
            with patch.object(app._conv, "notify") as notify:  # pyright: ignore[reportPrivateUsage]
                app._conv.on_approval_row_answered(  # pyright: ignore[reportPrivateUsage]
                    ApprovalRow.Answered("yes")
                )
            assert (run / "approvals" / "ap1.answer").read_text(encoding="utf-8") == "no"
            assert "already answered" in str(notify.call_args)
            # The row settles as an answer does: nothing left to click.
            assert app._conv._approval is None  # pyright: ignore[reportPrivateUsage]
            assert str(app._conv._approval_done).startswith("answered elsewhere")  # pyright: ignore[reportPrivateUsage]
            for _ in range(40):  # the row unmounts on a later tick
                if not app._conv.query(ApprovalRow):  # pyright: ignore[reportPrivateUsage]
                    break
                await pilot.pause(0.05)
            assert not app._conv.query(ApprovalRow)  # pyright: ignore[reportPrivateUsage]

    asyncio.run(scenario())


def test_a_dead_runs_approval_is_shown_but_not_answerable(tmp_path: Path) -> None:
    """A run killed with its prompt open: the fact stays on the surface, the
    key row (whose answer would reach nothing) is not offered."""
    run = tmp_path / "dead-run-AAAAAA"
    _live_run(run)
    (run / "worker.pid").write_text("4194304", encoding="utf-8")  # past pid_max: gone
    _append(
        run,
        {"type": "approval.prompt", "id": "ap1", "prompt": "Allow run_command: rm -rf build"},
    )

    async def scenario() -> None:
        app = Agent6TUI(run)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            app._conv._poll()  # pyright: ignore[reportPrivateUsage]
            await pilot.pause()
            assert not app.session_controllable()
            item = app._conv.query_one("#conv-approval", Static)  # pyright: ignore[reportPrivateUsage]
            assert item.display
            text = str(item.render())
            assert "approval pending when the run ended" in text and "rm -rf build" in text
            assert not app._conv.query(ApprovalRow)  # pyright: ignore[reportPrivateUsage]

    asyncio.run(scenario())


def test_escape_with_a_menu_open_closes_the_menu_not_the_view(tmp_path: Path) -> None:
    run = tmp_path / "live-run-BBBBBB"
    _live_run(run)

    async def scenario() -> None:
        from agent6.ui.tui.menubar import MenuBar

        app = Agent6TUI(run)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            bar = app._conv.query_one(MenuBar)  # pyright: ignore[reportPrivateUsage]
            bar.open("r")
            await pilot.pause()
            assert bar.opened
            await pilot.press("escape")
            await pilot.pause()
            assert not bar.opened
            assert app.is_running and app.screen is app._conv  # pyright: ignore[reportPrivateUsage]

    asyncio.run(scenario())


def test_a_non_standing_approvals_session_keys_type_the_letter(tmp_path: Path) -> None:
    """An approval nobody may answer for the session offers no `s`/`x`: the
    key is the letter it is, typed into the composer."""
    run = tmp_path / "live-run-EEEEEE"
    _live_run(run)

    async def scenario() -> None:
        app = Agent6TUI(run)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            prompt = {"type": "approval.prompt", "id": "ap1", "prompt": "Allow fetch: x.io"}
            _append(run, {**prompt, "standing": False})
            app._conv._poll()  # pyright: ignore[reportPrivateUsage]
            await pilot.pause()
            await pilot.pause()
            assert await _row_shown(app, pilot)
            await pilot.press("s", "x")
            await pilot.pause()
            assert not (run / "approvals" / "ap1.answer").exists()
            bar = app._conv.query_one("#conv-input", SteerInput)  # pyright: ignore[reportPrivateUsage]
            assert bar.text == "sx"

    asyncio.run(scenario())


def test_a_key_off_the_composer_answers_nothing(tmp_path: Path) -> None:
    """The answer keys are the composer's: with focus elsewhere (the
    scrollback), a key neither answers nor types; the label's click does."""
    run = tmp_path / "live-run-FFFFFF"
    _live_run(run)

    async def scenario() -> None:
        app = Agent6TUI(run)
        async with app.run_test(size=(120, 40)) as pilot:
            await _open_approval(app, pilot, run)
            app._conv.query_one("#conv-scroll").focus()  # pyright: ignore[reportPrivateUsage]
            await pilot.pause()
            await pilot.press("a")
            await pilot.pause()
            assert not (run / "approvals" / "ap1.answer").exists()
            bar = app._conv.query_one("#conv-input", SteerInput)  # pyright: ignore[reportPrivateUsage]
            assert bar.text == ""
            assert await _row_shown(app, pilot)

    asyncio.run(scenario())
