# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""An approval on the conversation screen is an inline item plus a docked key row, never a modal.

The conversation stays scrollable, one key answers, and the item collapses once answered.
"""

from __future__ import annotations

import asyncio
import json
import os
import pathlib
import time
from typing import Any
from unittest import mock

import pytest
from rich import color as rich_color
from textual import color as textual_color
from textual import screen, widgets

from agent6.ui.tui import app as tui_app
from agent6.ui.tui import composer
from tests.tui._waits import (
    TIMEOUT_S,
    answer_written,
    answerable,
    focus_answers,
    painted,
    row_gone,
    wait_for,
)


def _live_run(d: pathlib.Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "approvals").mkdir(exist_ok=True)
    (d / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")
    evs = [
        {"type": "session.start", "session_id": d.name, "mode": "run", "user_task": "add it"},
        {"type": "role.call", "role": "worker", "model": "m"},
    ]
    (d / "logs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in evs), encoding="utf-8")


def _append(d: pathlib.Path, ev: dict[str, object]) -> None:
    with (d / "logs.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(ev) + "\n")


# textual paints a span's colour as the ANSI colour a theme maps, and a base style's as CSS.
_ANSI_YELLOW = rich_color.Color.parse("color(3)")
_ANSI_RED = rich_color.Color.parse("color(1)")
_ANSI_GREEN = rich_color.Color.parse("color(2)")


def test_an_approval_is_an_inline_item_with_a_key_row(tmp_path: pathlib.Path) -> None:
    """An approval is an inline item answered from a key row, and only its `?` paints yellow.

    A style passed to `Text(...)` is the base of the whole text, so the head and the command
    under judgment painted bold yellow too.
    """
    run = tmp_path / "live-run-AAAAAA"
    _live_run(run)

    async def scenario() -> None:
        app = tui_app.Agent6TUI(run)
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
            assert not isinstance(app.screen, screen.ModalScreen)
            assert await _row_shown(app, pilot)
            item = app._conv.query_one("#conv-approval", widgets.Static)  # pyright: ignore[reportPrivateUsage]
            assert item.display
            text = str(item.render())
            assert "approval needed" in text and "pytest -q tests/unit/test_x.py" in text
            mark, head, command = await painted(
                pilot,
                item,
                "?",
                "Allow run_command: approval needed",
                "pytest -q tests/unit/test_x.py",
            )
            assert mark.bold and head.bold and not command.bold
            assert mark.color == _ANSI_YELLOW
            assert head.color == command.color != mark.color
            # The composer keeps focus: an empty one lets the row's keys answer.
            assert app.focused is app._conv.query_one("#conv-input", composer.SteerInput)  # pyright: ignore[reportPrivateUsage]
            await focus_answers(app._conv, pilot)  # pyright: ignore[reportPrivateUsage]
            await pilot.press("y")
            assert await answer_written(run, pilot) == "yes"
            _append(run, {"type": "approval.answer", "id": "ap1", "approved": True})
            app._conv._poll()  # pyright: ignore[reportPrivateUsage]
            await pilot.pause()
            await pilot.pause(0.3)
            app._tick()  # pyright: ignore[reportPrivateUsage]
            app._conv._poll()  # pyright: ignore[reportPrivateUsage]
            assert await row_gone(app._conv, pilot)  # pyright: ignore[reportPrivateUsage]
            assert "allowed" in str(item.render())

    asyncio.run(scenario())


async def _row_shown(app: tui_app.Agent6TUI, pilot: Any) -> bool:
    """Whether the approval row is up and answerable.

    The host folds the journal in its own thread, so the row follows within a few ticks rather than
    one pause.
    """
    deadline = time.monotonic() + TIMEOUT_S
    while not answerable(app._conv):  # pyright: ignore[reportPrivateUsage]
        if time.monotonic() >= deadline:
            return False
        app._conv._poll()  # pyright: ignore[reportPrivateUsage]
        await pilot.pause(0.05)
    return True


async def _open_approval(app: tui_app.Agent6TUI, pilot: Any, run: pathlib.Path) -> None:
    await pilot.pause()
    await pilot.pause()
    prompt = {"type": "approval.prompt", "id": "ap1", "prompt": "Allow run_command: ls"}
    _append(run, {**prompt, "standing": True})
    app._conv._poll()  # pyright: ignore[reportPrivateUsage]
    await pilot.pause()
    await pilot.pause()
    assert await _row_shown(app, pilot)


def test_the_waiting_line_under_an_approval_paints_ansi_yellow(tmp_path: pathlib.Path) -> None:
    """The `waiting · needs answer` line paints the ANSI yellow of the approval's `?`.

    textual reads a base style's `yellow` as #ffff00, at 1.1:1 contrast on agent6-light.
    """
    run = tmp_path / "live-run-MMMMMM"
    _live_run(run)

    async def scenario() -> None:
        app = tui_app.Agent6TUI(run)
        async with app.run_test(size=(120, 40)) as pilot:
            await _open_approval(app, pilot, run)
            conv = app._conv  # pyright: ignore[reportPrivateUsage]
            live = conv.query_one("#conv-live", widgets.Static)
            (label,) = await painted(
                pilot,
                live,
                "waiting · needs answer",
                pump=conv._poll,  # pyright: ignore[reportPrivateUsage]
            )
            assert label.bold and label.color == _ANSI_YELLOW

    asyncio.run(scenario())


@pytest.mark.parametrize("theme", ["agent6-light", "agent6-dark"])
def test_the_answers_paint_ansi_green_and_red(tmp_path: pathlib.Path, theme: str) -> None:
    """The row's answers paint ANSI green and red, which the theme maps to its own shades.

    As a base style they painted CSS #008000 and #ff0000 on every theme, the red at 3.4:1 on
    agent6-light. The two single answers are bold.
    """
    run = tmp_path / "live-run-NNNNNN"
    _live_run(run)

    async def scenario() -> None:
        app = tui_app.Agent6TUI(run)
        async with app.run_test(size=(120, 40)) as pilot:
            await _open_approval(app, pilot, run)
            app.theme = theme
            await pilot.pause()
            row = app._conv.query_one(composer.ApprovalRow)  # pyright: ignore[reportPrivateUsage]
            colours: dict[str, object] = {}
            for answer, text in (
                ("yes", "[y] allow"),
                ("session", "[a] allow all"),
                ("no", "[n] deny"),
                ("session-deny", "[d] deny all"),
            ):
                label = row.query_one(f".answer-{answer}", widgets.Static)
                (style,) = await painted(pilot, label, text)
                assert label.styles.color.ansi is not None
                assert bool(style.bold) is (answer in ("yes", "no"))
                colours[answer] = style.color
            assert colours == {
                "yes": _ANSI_GREEN,
                "session": _ANSI_GREEN,
                "no": _ANSI_RED,
                "session-deny": _ANSI_RED,
            }

    asyncio.run(scenario())


@pytest.mark.parametrize("theme", ["agent6-light", "agent6-dark"])
@pytest.mark.parametrize("answer", ["yes", "session", "no", "session-deny"])
def test_focused_answer_paints_readable_theme_text(
    tmp_path: pathlib.Path, theme: str, answer: str
) -> None:
    """A focused answer kept its green or red on $primary, at 1.04:1 to 2.04:1."""
    run = tmp_path / "live-run-FOCUS1"
    _live_run(run)

    async def scenario() -> None:
        app = tui_app.Agent6TUI(run)
        async with app.run_test(size=(120, 40)) as pilot:
            await _open_approval(app, pilot, run)
            app.theme = theme
            label = app.screen.query_one(f".answer-{answer}", widgets.Static)
            label.focus()
            await pilot.pause()
            assert app.focused is label
            (style,) = await painted(pilot, label, str(label.render()))
            variables = app.get_css_variables()
            foreground = label.rich_style.color
            background = textual_color.Color.parse(variables["primary"])
            assert style.color == foreground
            assert style.bgcolor == background.rich_color
            assert style.color is not None and style.bgcolor is not None

            def luminance(colour: rich_color.Color) -> float:
                rgb = colour.get_truecolor(app.ansi_theme)
                linear = [
                    c / 255 / 12.92 if c / 255 <= 0.04045 else ((c / 255 + 0.055) / 1.055) ** 2.4
                    for c in rgb
                ]
                return sum(c * w for c, w in zip(linear, (0.2126, 0.7152, 0.0722), strict=True))

            light, dark = sorted((luminance(style.color), luminance(style.bgcolor)), reverse=True)
            assert (light + 0.05) / (dark + 0.05) >= 4.5

    asyncio.run(scenario())


def test_a_typed_message_never_answers_the_approval(tmp_path: pathlib.Path) -> None:
    """The composer keeps focus, and the answer keys fire only while it is empty.

    With the row focused, a sentence typed at the composer answered on its first letter.
    """
    run = tmp_path / "live-run-CCCCCC"
    _live_run(run)

    async def scenario() -> None:
        app = tui_app.Agent6TUI(run)
        async with app.run_test(size=(120, 40)) as pilot:
            await _open_approval(app, pilot, run)
            await pilot.press("slash", "b", "t", "w", "space", "y", "e", "s", "space", "a", "n")
            await pilot.pause()
            assert not (run / "approvals" / "ap1.answer").exists()
            bar = app._conv.query_one("#conv-input", composer.SteerInput)  # pyright: ignore[reportPrivateUsage]
            assert bar.text == "/btw yes an"
            assert await _row_shown(app, pilot)

    asyncio.run(scenario())


def test_a_resumed_execution_drops_the_previous_executions_approval(tmp_path: pathlib.Path) -> None:
    """An execution boundary must withdraw an unanswered approval from the dead execution."""
    run = tmp_path / "live-run-HHHHHH"
    _live_run(run)

    async def scenario() -> None:
        app = tui_app.Agent6TUI(run)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            prompt: dict[str, object] = {
                "type": "approval.prompt",
                "id": "ap1",
                "prompt": "Allow run_command: ls",
            }
            # The journal is the one feed; a hand-fed app re-delivered the prompt under load.
            _append(run, prompt)
            app._conv._poll()  # pyright: ignore[reportPrivateUsage]
            await pilot.pause()
            assert await _row_shown(app, pilot)
            boundary: dict[str, object] = {"type": "loop.resume.start", "turn": 2}
            _append(run, boundary)
            # The withdrawal is an async DOM prune after the boundary; the poll feeds the fold.
            conv = app._conv  # pyright: ignore[reportPrivateUsage]
            assert await row_gone(conv, pilot, conv._poll)  # pyright: ignore[reportPrivateUsage]

    asyncio.run(scenario())


def test_an_answered_approval_stays_closed_on_reload(tmp_path: pathlib.Path) -> None:
    """Reload before the worker journals its answer must not reopen the row."""
    run = tmp_path / "live-run-GGGGGG"
    _live_run(run)

    async def scenario() -> None:
        app = tui_app.Agent6TUI(run)
        async with app.run_test(size=(120, 40)) as pilot:
            await _open_approval(app, pilot, run)
            await focus_answers(app._conv, pilot)  # pyright: ignore[reportPrivateUsage]
            await pilot.press("y")
            assert await row_gone(app._conv, pilot)  # pyright: ignore[reportPrivateUsage]
            app._conv.action_reload()  # pyright: ignore[reportPrivateUsage]
            await pilot.pause()
            assert not app._conv.query(composer.ApprovalRow)  # pyright: ignore[reportPrivateUsage]

    asyncio.run(scenario())


def test_a_click_on_a_row_label_answers(tmp_path: pathlib.Path) -> None:
    run = tmp_path / "live-run-DDDDDD"
    _live_run(run)

    async def scenario() -> None:
        app = tui_app.Agent6TUI(run)
        async with app.run_test(size=(120, 40)) as pilot:
            await _open_approval(app, pilot, run)
            label = app._conv.query_one(".answer-yes", widgets.Static)  # pyright: ignore[reportPrivateUsage]
            # A click needs the label laid out, which is past mounted.
            await wait_for(pilot, lambda: label.region.width > 0, "the answer label laid out")
            await pilot.click(label)
            assert await answer_written(run, pilot) == "yes"

    asyncio.run(scenario())


def test_a_click_after_another_surface_answered_is_refused(tmp_path: pathlib.Path) -> None:
    """An answer that landed elsewhere stands: the click writes nothing and the screen says so."""
    run = tmp_path / "live-run-EEEEEE"
    _live_run(run)

    async def scenario() -> None:
        app = tui_app.Agent6TUI(run)
        async with app.run_test(size=(120, 40)) as pilot:
            await _open_approval(app, pilot, run)
            (run / "approvals").mkdir(exist_ok=True)
            (run / "approvals" / "ap1.answer").write_text("no", encoding="utf-8")
            with mock.patch.object(app._conv, "notify") as notify:  # pyright: ignore[reportPrivateUsage]
                app._conv.on_approval_row_answered(  # pyright: ignore[reportPrivateUsage]
                    composer.ApprovalRow.Answered("yes")
                )
            assert (run / "approvals" / "ap1.answer").read_text(encoding="utf-8") == "no"
            assert "already answered" in str(notify.call_args)
            # The row settles as an answer does: nothing left to click.
            assert app._conv._approval is None  # pyright: ignore[reportPrivateUsage]
            assert str(app._conv._approval_done).startswith("answered elsewhere")  # pyright: ignore[reportPrivateUsage]
            assert await row_gone(app._conv, pilot)  # pyright: ignore[reportPrivateUsage]

    asyncio.run(scenario())


def test_a_dead_runs_approval_is_shown_but_notanswerable(tmp_path: pathlib.Path) -> None:
    """A run killed with its prompt open keeps the fact on the surface and offers no key row."""
    run = tmp_path / "dead-run-AAAAAA"
    _live_run(run)
    (run / "worker.pid").write_text("4194304", encoding="utf-8")  # past pid_max: gone
    _append(
        run,
        {"type": "approval.prompt", "id": "ap1", "prompt": "Allow run_command: rm -rf build"},
    )

    async def scenario() -> None:
        app = tui_app.Agent6TUI(run)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            app._conv._poll()  # pyright: ignore[reportPrivateUsage]
            await pilot.pause()
            assert not app.session_controllable()
            item = app._conv.query_one("#conv-approval", widgets.Static)  # pyright: ignore[reportPrivateUsage]
            assert item.display
            text = str(item.render())
            assert "approval pending when the run ended" in text and "rm -rf build" in text
            assert not app._conv.query(composer.ApprovalRow)  # pyright: ignore[reportPrivateUsage]

    asyncio.run(scenario())


def test_escape_with_a_menu_open_closes_the_menu_not_the_view(tmp_path: pathlib.Path) -> None:
    run = tmp_path / "live-run-BBBBBB"
    _live_run(run)

    async def scenario() -> None:
        from agent6.ui.tui import menubar

        app = tui_app.Agent6TUI(run)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            bar = app._conv.query_one(menubar.MenuBar)  # pyright: ignore[reportPrivateUsage]
            bar.open("r")
            await pilot.pause()
            assert bar.opened
            await pilot.press("escape")
            await pilot.pause()
            assert not bar.opened
            assert app.is_running and app.screen is app._conv  # pyright: ignore[reportPrivateUsage]

    asyncio.run(scenario())


def test_a_non_standing_approvals_session_keys_type_the_letter(tmp_path: pathlib.Path) -> None:
    """An approval nobody may answer for the session offers no `a`/`d`; the letter is typed."""
    run = tmp_path / "live-run-EEEEEE"
    _live_run(run)

    async def scenario() -> None:
        app = tui_app.Agent6TUI(run)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.pause()
            prompt = {"type": "approval.prompt", "id": "ap1", "prompt": "Allow fetch: x.io"}
            _append(run, {**prompt, "standing": False})
            app._conv._poll()  # pyright: ignore[reportPrivateUsage]
            await pilot.pause()
            await pilot.pause()
            assert await _row_shown(app, pilot)
            await pilot.press("a", "d")
            await pilot.pause()
            assert not (run / "approvals" / "ap1.answer").exists()
            bar = app._conv.query_one("#conv-input", composer.SteerInput)  # pyright: ignore[reportPrivateUsage]
            assert bar.text == "ad"

    asyncio.run(scenario())


def test_a_key_answers_from_the_transcript(tmp_path: pathlib.Path) -> None:
    """With focus tabbed out of the composer, the letters answer from the transcript too."""
    run = tmp_path / "live-run-FFFFFF"
    _live_run(run)

    async def scenario() -> None:
        app = tui_app.Agent6TUI(run)
        async with app.run_test(size=(120, 40)) as pilot:
            await _open_approval(app, pilot, run)
            app._conv.query_one("#conv-scroll").focus()  # pyright: ignore[reportPrivateUsage]
            await pilot.pause()
            await pilot.press("y")
            assert await answer_written(run, pilot) == "yes"
            bar = app._conv.query_one("#conv-input", composer.SteerInput)  # pyright: ignore[reportPrivateUsage]
            assert bar.text == ""  # the transcript types nothing

    asyncio.run(scenario())


def test_a_letter_typed_as_the_approval_appears_types(tmp_path: pathlib.Path) -> None:
    """An approval takes neither the focus nor the keys: `yes…` into the composer is a message."""
    run = tmp_path / "live-run-JJJJJJ"
    _live_run(run)

    async def scenario() -> None:
        app = tui_app.Agent6TUI(run)
        async with app.run_test(size=(120, 40)) as pilot:
            await _open_approval(app, pilot, run)
            await pilot.press("y", "e", "s")
            await pilot.pause()
            assert not (run / "approvals" / "ap1.answer").exists()
            bar = app._conv.query_one("#conv-input", composer.SteerInput)  # pyright: ignore[reportPrivateUsage]
            assert bar.text == "yes"

    asyncio.run(scenario())


def test_tab_reaches_the_answers_and_enter_answers(tmp_path: pathlib.Path) -> None:
    """The answers are tab stops: Tab walks to one and Enter answers it."""
    run = tmp_path / "live-run-LLLLLL"
    _live_run(run)

    async def scenario() -> None:
        app = tui_app.Agent6TUI(run)
        async with app.run_test(size=(120, 40)) as pilot:
            await _open_approval(app, pilot, run)
            bar = app._conv.query_one("#conv-input", composer.SteerInput)  # pyright: ignore[reportPrivateUsage]
            assert app.focused is bar
            for _ in range(8):  # Tab walks the screen; stop on the first answer
                await pilot.press("tab")
                await pilot.pause()
                if isinstance(app.focused, widgets.Static) and "answer-yes" in app.focused.classes:
                    break
            assert app.focused is not None and "answer-yes" in app.focused.classes
            await pilot.press("enter")
            assert await answer_written(run, pilot) == "yes"

    asyncio.run(scenario())


def test_the_dashboard_answers_inline_and_keeps_the_focus_on_the_answers(
    tmp_path: pathlib.Path,
) -> None:
    """The dashboard popped a modal, which took the focus mid-sentence.

    It shows the same row, with the command (it has no transcript), and answering from the row
    leaves the focus there, so the next approval answers too.
    """
    run = tmp_path / "live-run-KKKKKK"
    _live_run(run)

    async def scenario() -> None:
        app = tui_app.Agent6TUI(run)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.press("ctrl+d")
            await pilot.pause()
            bar = app._dash.query_one("#dash-input", composer.SteerInput)  # pyright: ignore[reportPrivateUsage]
            bar.focus()
            _append(
                run, {"type": "approval.prompt", "id": "ap1", "prompt": "Allow run_command: ls"}
            )
            await wait_for(
                pilot,
                lambda: answerable(app._dash),  # pyright: ignore[reportPrivateUsage]
                "the dashboard row",
                pump=app._tick,  # pyright: ignore[reportPrivateUsage]
            )
            row = app._dash.query(composer.ApprovalRow).first()  # pyright: ignore[reportPrivateUsage]
            assert not isinstance(app.screen, screen.ModalScreen)
            assert app.focused is bar  # the row took nothing
            shown = str(row.query_one(widgets.Static).render())
            assert "Allow run_command" in shown and "ls" in shown
            await focus_answers(app._dash, pilot)  # pyright: ignore[reportPrivateUsage]
            await pilot.press("y")
            assert await answer_written(run, pilot) == "yes"
            _append(run, {"type": "approval.answer", "id": "ap1", "approved": True})
            _append(
                run, {"type": "approval.prompt", "id": "ap2", "prompt": "Allow run_command: rm"}
            )
            await wait_for(
                pilot,
                lambda: app.focused is not None and "answer-yes" in app.focused.classes,
                "the next approval's answers to keep the focus",
                pump=app._tick,  # pyright: ignore[reportPrivateUsage]
            )
            await pilot.press("n")
            assert await answer_written(run, pilot, "ap2") == "no"

    asyncio.run(scenario())
