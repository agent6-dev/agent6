# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Headless drive of the textual dashboard via textual's run_test Pilot.

textual ships in the base install, so these run in CI. They cover what only a human could
check otherwise: streamed reasoning and markup-hostile model output render without crashing,
the approval is keyboard-answerable, and the steer composer writes the right bridge file.
"""

from __future__ import annotations

import asyncio
import json
import os
import pathlib
import time
from typing import Any

import pytest
from rich import text as rich_text
from textual import app as textual_app
from textual import screen as textual_screen
from textual import widgets

from agent6.app import stop as app_stop
from agent6.sessions import ipc, manifest
from agent6.ui import spawn
from agent6.ui.tui import _dashboard_header, composer, modals
from agent6.ui.tui import app as tui_app
from agent6.viewmodel import state
from tests.tui._waits import answer_written, focus_answers, wait_for


def _ev(**fields: Any) -> dict[str, object]:
    return dict(fields)


def _screen_is(app: tui_app.Agent6TUI, name: str) -> bool:
    """`app.screen` raising on a transiently empty stack reads as "not yet", never an error."""
    try:
        current = app.screen
    except textual_app.ScreenStackError:
        return False
    return current is getattr(app, name)


async def _show_dashboard(pilot: Any) -> None:
    """Open on the conversation, then flip to the dashboard and wait for it to be on top.

    Startup pushes the screens asynchronously; a Ctrl+D fired early types into the wrong screen.
    """
    app = pilot.app
    await wait_for(pilot, lambda: _screen_is(app, "_conv"), "the conversation screen")
    await pilot.press("ctrl+d")
    await wait_for(pilot, lambda: _screen_is(app, "_dash"), "the dashboard screen")


async def _settle_focus(pilot: Any, widget: Any) -> None:
    """Wait for a deferred focus() (Widget.focus defers via call_later) to land."""
    await wait_for(pilot, lambda: pilot.app.focused is widget, f"focus on {widget}")


class _ModalHost(textual_app.App[None]):
    def __init__(self, modal: textual_screen.ModalScreen[Any]) -> None:
        super().__init__()
        self.modal = modal
        self.results: list[object] = []

    def on_mount(self) -> None:
        self.push_screen(self.modal, self.results.append)


def test_question_modal_digit_in_freetext_is_not_hijacked() -> None:
    """A digit typed into an answer field is plain text; an option button fills its field.

    The multi-question modal has no digit quick-select; ctrl+s submits the answers as a tuple.
    """
    result: dict[str, tuple[str, ...] | None] = {}

    class _Host(textual_app.App[None]):
        def on_mount(self) -> None:
            self.push_screen(
                modals.QuestionModal(
                    "q1", (state.Question(question="pick?", options=("alpha", "beta")),)
                ),
                lambda v: result.__setitem__("v", v),
            )

    async def scenario() -> None:
        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = app.screen
            assert isinstance(modal, modals.QuestionModal)
            modal.query_one("#ans-0", widgets.Input).focus()
            await pilot.pause()
            await pilot.press("2")  # a digit is text, not an option pick
            await pilot.pause()
            assert isinstance(app.screen, modals.QuestionModal)  # still open (no digit-select)
            assert "v" not in result
            assert modal.query_one("#ans-0", widgets.Input).value == "2"  # digit typed as text
            # An option button fills that question's field; it does not dismiss.
            modal.query_one("#opt-0-0", widgets.Button).press()
            await pilot.pause()
            assert isinstance(app.screen, modals.QuestionModal)  # still open (fill, not submit)
            assert (
                modal.query_one("#ans-0", widgets.Input).value == "alpha"
            )  # filled from the option
            await pilot.press("ctrl+s")  # submit collects the answers
            await pilot.pause()
            assert result.get("v") == ("alpha",)  # tuple of answers, aligned to questions

    asyncio.run(scenario())


def test_diff_colors_content_with_header_like_prefixes(tmp_path: pathlib.Path) -> None:
    """Added and removed content stays colored when its text begins with a file header."""
    (tmp_path / "logs.jsonl").write_text("", encoding="utf-8")

    async def scenario() -> None:
        app = tui_app.Agent6TUI(tmp_path)
        async with app.run_test(size=(100, 30)) as pilot:
            await _show_dashboard(pilot)
            app._handle_event(
                _ev(
                    type="diff.updated",
                    patch="--- a/file\n+++ b/file\n@@ -1 +1 @@\n----old\n++++new",
                )
            )
            app._tick()
            await pilot.pause()
            content = app._dash.query_one("#diff-body", widgets.Static).content
            assert isinstance(content, rich_text.Text)
            removed_style = content.get_style_at_offset(app.console, content.plain.index("----old"))
            added_style = content.get_style_at_offset(app.console, content.plain.index("++++new"))
            assert str(removed_style) == "red"
            assert str(added_style) == "green"

    asyncio.run(scenario())


def test_modal_arrow_keys_move_focus() -> None:
    """Arrow keys move focus in a modal like Tab (the app.focus_next fix).

    Tested on the button-only confirm dialog, where no text field consumes the arrows.
    """

    class _Host(textual_app.App[None]):
        def on_mount(self) -> None:
            self.push_screen(modals.ConfirmModal("Confirm", "Proceed?"), lambda _v: None)

    async def scenario() -> None:
        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = app.screen
            assert isinstance(modal, modals.ConfirmModal)
            first = modal.focused
            assert isinstance(first, widgets.Button)
            await pilot.press("right")  # arrow moves focus to the other button
            await pilot.pause()
            assert isinstance(modal.focused, widgets.Button) and modal.focused is not first
            await pilot.press("left")  # and back
            await pilot.pause()
            assert modal.focused is first

    asyncio.run(scenario())


def test_consequential_modal_buttons_name_their_answers() -> None:
    async def labels(modal: textual_screen.ModalScreen[Any]) -> list[str]:
        app = _ModalHost(modal)
        async with app.run_test() as pilot:
            await pilot.pause()
            return [str(button.label) for button in modal.query(widgets.Button)]

    async def scenario() -> None:
        assert await labels(modals.ConfirmModal("Confirm", "Proceed?", confirm_label="Delete")) == [
            "Delete (y)",
            "Cancel (n)",
        ]
        assert await labels(modals.SteerModal()) == ["Send (Ctrl+S)", "Continue"]
        assert await labels(
            modals.QuestionModal(
                "q",
                (state.Question(question="Choose one", options=("literal [one]", "two")),),
            )
        ) == ["literal [one]", "two", "Submit (ctrl+s)"]

    asyncio.run(scenario())


def test_each_consequential_modal_delivers_one_result() -> None:
    async def scenario() -> None:
        confirmation = _ModalHost(modals.ConfirmModal("Confirm", "Proceed?"))
        async with confirmation.run_test() as pilot:
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert confirmation.results == [False]

        steer_modal = modals.SteerModal()
        steer = _ModalHost(steer_modal)
        async with steer.run_test() as pilot:
            await pilot.pause()
            steer_modal.query_one(widgets.TextArea).insert("go")
            await pilot.press("ctrl+s")
            await pilot.pause()
            assert steer.results == ["go"]

        question_modal = modals.QuestionModal(
            "q",
            (state.Question(question="Choose one", options=("literal [one]", "two")),),
        )
        question = _ModalHost(question_modal)
        async with question.run_test() as pilot:
            await pilot.pause()
            question_modal.query_one("#opt-0-0", widgets.Button).press()
            await pilot.pause()
            question_modal.query_one("#question-submit", widgets.Button).press()
            await pilot.pause()
            assert question.results == [("literal [one]",)]

    asyncio.run(scenario())


def test_tools_table_maximizes_to_full_height(tmp_path: pathlib.Path) -> None:
    """`f` on the focused tool table fills the screen, not its 20% resting height."""
    (tmp_path / "logs.jsonl").write_text("", encoding="utf-8")

    async def scenario() -> None:
        app = tui_app.Agent6TUI(tmp_path)
        async with app.run_test(size=(100, 40)) as pilot:
            await _show_dashboard(pilot)
            app._handle_event(_ev(type="tool.call", name="grep", args={"pattern": "x"}))
            app._tick()
            await pilot.pause()
            table = app._dash.query_one("#tools", widgets.DataTable)
            table.focus()
            await _settle_focus(pilot, table)
            resting_h = table.size.height
            app._dash.action_fullscreen()  # View menu / palette action (no bare letter)
            await pilot.pause()
            assert app.screen.maximized is table
            assert table.has_class("-maximized")
            maxed_h = table.size.height
            assert maxed_h > resting_h * 2  # fills the screen, not the 20% resting band
            assert maxed_h >= 25  # ~full of a 40-row screen (minus menu + footer)

    asyncio.run(scenario())


def test_plan_tree_maximizes_to_full_width(tmp_path: pathlib.Path) -> None:
    """`f` on the focused task-graph pane fills the screen width, not its 32% resting width."""
    (tmp_path / "logs.jsonl").write_text("", encoding="utf-8")

    async def scenario() -> None:
        app = tui_app.Agent6TUI(tmp_path)
        async with app.run_test(size=(100, 40)) as pilot:
            await _show_dashboard(pilot)
            app._handle_event(
                _ev(
                    type="graph.update",
                    cursor="t1",
                    nodes={
                        "t1": {
                            "title": "do the thing",
                            "parent_id": None,
                            "status": "in_progress",
                            "children": [],
                        }
                    },
                )
            )
            app._tick()
            await pilot.pause()
            tree = app._dash.query_one("#plan", widgets.Tree)
            tree.focus()
            await _settle_focus(pilot, tree)
            resting_w = tree.size.width
            app._dash.action_fullscreen()  # View menu / palette action (no bare letter)
            await pilot.pause()
            assert app.screen.maximized is tree
            assert tree.has_class("-maximized")
            maxed_w = tree.size.width
            assert maxed_w > resting_w * 2  # fills the screen, not the 32% resting column
            assert maxed_w >= 90  # ~full of a 100-col screen

    asyncio.run(scenario())


def test_tool_row_enter_opens_detail_with_full_args(tmp_path: pathlib.Path) -> None:
    """Enter on a tool-calls row opens a read-only detail modal carrying the full arg value."""
    (tmp_path / "logs.jsonl").write_text("", encoding="utf-8")
    long_val = "abc/" * 100  # 400 chars, well past the 80-char preview + 90-char column

    async def scenario() -> None:
        app = tui_app.Agent6TUI(tmp_path)
        async with app.run_test() as pilot:
            await _show_dashboard(pilot)
            app._handle_event(_ev(type="tool.call", name="run_command", args={"cmd": long_val}))
            app._handle_event(
                _ev(type="tool.result", name="run_command", ok=True, summary="lots of output")
            )
            app._tick()
            await pilot.pause()
            app._dash.query_one("#tools", widgets.DataTable).focus()
            await pilot.pause()
            await pilot.press("enter")  # select the single row
            await pilot.pause()
            assert isinstance(app.screen, modals.ToolCallDetailModal)
            args_text = app.screen.query_one("#tc-args", widgets.TextArea).text
            assert long_val in args_text  # full value present, not the "…" preview
            await pilot.press("escape")  # closes past the focused read-only TextArea
            await pilot.pause()
            assert not isinstance(app.screen, modals.ToolCallDetailModal)

    asyncio.run(scenario())


def test_render_and_modals(tmp_path: pathlib.Path) -> None:
    # These drive a live run, so its pid belongs on disk as it would be.
    (tmp_path / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")
    (tmp_path / "logs.jsonl").write_text("", encoding="utf-8")

    async def scenario() -> None:
        app = tui_app.Agent6TUI(tmp_path)
        async with app.run_test() as pilot:
            await _show_dashboard(pilot)
            # Markup-hostile model output in every pane must not parse as Rich markup.
            for ev in (
                _ev(type="session.start", user_task="do [a] thing", mode="run"),
                _ev(
                    type="graph.update",
                    cursor="t1",
                    nodes={
                        "t1": {
                            "title": "fix [the] bug",
                            "parent_id": None,
                            "status": "in_progress",
                            "children": ["t2"],
                        },
                        "t2": {
                            "title": "add [/close] tag",
                            "parent_id": "t1",
                            "status": "pending",
                            "children": [],
                        },
                    },
                ),
                _ev(type="role.call", role="worker", model="kimi-k2.6"),
                _ev(type="role.thinking_delta", role="worker", text="let me [check]"),
                _ev(type="role.text_delta", role="worker", text="answer is [x]"),
                _ev(type="tool.call", name="grep", args={"pattern": "[a-z]+", "path": "x.py"}),
                _ev(type="tool.result", name="grep", ok=True, summary="3 matches in [src]"),
                _ev(type="diff.updated", index=1, patch="--- a\n+++ b\n@@ [x] @@"),
            ):
                app._handle_event(ev)
            app._tick()  # the coalesced repaint happens in the tick, not per event
            await pilot.pause()

            # The dashboard answers inline (never a modal): focus the row, key.
            app._handle_event(_ev(type="approval.prompt", id="ap1", prompt="run_command(['ls'])"))
            app._tick()
            await pilot.pause()
            assert not isinstance(app.screen, textual_screen.ModalScreen)
            await focus_answers(app._dash, pilot)
            await pilot.press("y")
            assert await answer_written(tmp_path, pilot) == "yes"

            app._handle_event(_ev(type="approval.answer", id="ap1", approved=True))
            app._handle_event(_ev(type="approval.prompt", id="ap2", prompt="rm -rf"))
            app._tick()
            await pilot.pause()
            await focus_answers(app._dash, pilot)
            await pilot.press("n")
            assert await answer_written(tmp_path, pilot, "ap2") == "no"

            # An external steer request routes to the docked bar, which takes focus.

            app._handle_event(_ev(type="session.steer_requested", source="sigint"))
            app._tick()
            bar = app._dash.query_one("#dash-input", composer.SteerInput)
            await _settle_focus(pilot, bar)
            assert app.focused is bar
            await pilot.press("f", "i", "x")
            await pilot.press("enter")
            await pilot.pause()
            assert (tmp_path / "steer.answer").read_text(encoding="utf-8") == "fix"

            # Markup-hostile options render; a click fills its field, ctrl+s writes the file.
            app._handle_event(
                _ev(
                    type="question.prompt",
                    id="q1",
                    questions=[
                        {"question": "which [approach]?", "options": ["use [A]", "use [B]"]}
                    ],
                )
            )
            app._tick()
            await pilot.pause()
            assert isinstance(app.screen, modals.QuestionModal)
            app.screen.query_one(
                "#opt-0-1", widgets.Button
            ).press()  # fill ans-0 with the 2nd option
            await pilot.pause()
            await pilot.press("ctrl+s")  # submit
            await pilot.pause()
            assert (tmp_path / "questions" / "q1.answer").read_text(encoding="utf-8") == json.dumps(
                ["use [B]"]
            )

            # Question modal: a typed free-text answer is sent verbatim.
            app._handle_event(
                _ev(
                    type="question.prompt",
                    id="q2",
                    questions=[{"question": "name?", "options": []}],
                )
            )
            app._tick()
            await pilot.pause()
            await pilot.press("z", "z")
            await pilot.press("enter")
            await pilot.pause()
            assert (tmp_path / "questions" / "q2.answer").read_text(encoding="utf-8") == json.dumps(
                ["zz"]
            )

    asyncio.run(scenario())


def test_start_question_before_session_start_is_answerable(tmp_path: pathlib.Path) -> None:
    """A run asking about uncommitted changes before session.start reads as waiting.

    The TUI shows the question and answers over the bridge, the ask_user channel.
    """
    (tmp_path / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")
    (tmp_path / "manifest.json").write_text(
        json.dumps({"mode": "run", "session_id": tmp_path.name, "user_task": "t"}),
        encoding="utf-8",
    )
    (tmp_path / "logs.jsonl").write_text(
        json.dumps(
            {
                "type": "question.prompt",
                "id": "question-1",
                "questions": [
                    {
                        "question": "1 tracked file has uncommitted changes:\n    seed.txt\n"
                        "How should this run treat them?",
                        "options": ["stash", "include", "cancel"],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    async def scenario() -> None:
        app = tui_app.Agent6TUI(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await wait_for(
                pilot,
                lambda: (
                    isinstance(app.screen, modals.QuestionModal)
                    and bool(app.screen.query("#opt-0-0"))
                ),
                "the question modal",
            )
            app._heartbeat_at = 0.0
            app._tick()
            assert app.dir_status == ("waiting", "needs answer")
            app.screen.query_one("#opt-0-0", widgets.Button).press()  # stash
            await pilot.pause()
            await pilot.press("ctrl+s")
            await pilot.pause()
            assert (tmp_path / "questions" / "question-1.answer").read_text(
                encoding="utf-8"
            ) == json.dumps(["stash"])

    asyncio.run(scenario())


def test_back_and_quit_exit_codes(tmp_path: pathlib.Path) -> None:
    """Esc leaves the run view for the hub from both views; Ctrl+Q quits from anywhere.

    The composer bars own plain letters, so there is no q alias; standalone, each just closes.
    """

    async def press(from_hub: bool, *keys: str) -> tui_app.TuiExit | None:
        (tmp_path / "logs.jsonl").write_text("", encoding="utf-8")
        app = tui_app.Agent6TUI(tmp_path, from_hub=from_hub)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            for key in keys:
                await pilot.press(key)
                await pilot.pause()
        return app.return_value

    assert asyncio.run(press(True, "escape")) == tui_app.TuiExit()  # conversation Esc -> the hub
    assert (
        asyncio.run(press(True, "ctrl+d", "escape")) == tui_app.TuiExit()
    )  # dashboard Esc -> the hub
    assert asyncio.run(press(True, "ctrl+q")) == tui_app.TuiExit(quit_hub=True)  # only Ctrl+Q quits
    assert asyncio.run(press(False, "escape")) == tui_app.TuiExit()  # standalone: just close
    assert asyncio.run(press(False, "ctrl+q")) == tui_app.TuiExit()  # standalone: just close


def test_dashboard_pane_maximize_and_restore(tmp_path: pathlib.Path) -> None:
    """F maximizes the focused pane to full screen; Esc and f both restore it.

    Esc while maximized must minimize (not also back out to the hub), and a non-default pane like
    the diff must be focusable for this to work.
    """

    async def scenario() -> None:
        (tmp_path / "logs.jsonl").write_text("", encoding="utf-8")
        app = tui_app.Agent6TUI(tmp_path, from_hub=True)
        async with app.run_test(size=(120, 40)) as pilot:
            await _show_dashboard(pilot)
            await pilot.pause()
            diff = app._dash.query_one("#diff")
            diff.focus()
            await _settle_focus(pilot, diff)
            app._dash.action_fullscreen()  # maximize the focused pane
            await pilot.pause()
            assert app.screen.maximized is diff
            await pilot.press("escape")  # Esc restores, does NOT exit to the hub
            await pilot.pause()
            assert app.screen.maximized is None
            assert app.return_value is None  # still running; Esc was consumed by minimize
            diff.focus()
            await _settle_focus(pilot, diff)
            app._dash.action_fullscreen()
            await pilot.pause()
            assert app.screen.maximized is diff
            app._dash.action_fullscreen()  # the action toggles back too
            await pilot.pause()
            assert app.screen.maximized is None

    asyncio.run(scenario())


def test_dashboard_diff_pane_scrolls(tmp_path: pathlib.Path) -> None:
    """A long diff scrolls in the diff pane, inline and while maximized."""

    async def scenario() -> None:
        (tmp_path / "logs.jsonl").write_text("", encoding="utf-8")
        app = tui_app.Agent6TUI(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await _show_dashboard(pilot)
            await pilot.pause()
            long_patch = "--- a\n+++ b\n" + "".join(f"+added line {i}\n" for i in range(200))
            app._handle_event(_ev(type="diff.updated", index=1, patch=long_patch))
            app._tick()
            await pilot.pause()
            diff = app._dash.query_one("#diff")
            assert diff.max_scroll_y > 0  # content overflows the pane -> scrollable
            diff.focus()
            await _settle_focus(pilot, diff)  # focus() defers; land it before maximizing
            app._dash.action_fullscreen()  # maximize, then it must still scroll
            await pilot.pause()
            assert app.screen.maximized is diff
            assert diff.max_scroll_y > 0

    asyncio.run(scenario())


def test_dashboard_inline_log_is_a_bounded_gapless_window(tmp_path: pathlib.Path) -> None:
    """Coalescing folds many events between paints.

    The inline log must stay a bounded window: feed a pre-burst, then a burst larger than the window
    in one tick, and the RichLog caps at MAX_LOG_TAIL: the gap-causing pre-burst lines are
    evicted, so it is the gapless recent window, not pre-burst lines + a hole + the tail.
    """

    async def scenario() -> None:
        (tmp_path / "logs.jsonl").write_text("", encoding="utf-8")
        app = tui_app.Agent6TUI(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await _show_dashboard(pilot)
            await pilot.pause()
            for i in range(100):
                app._handle_event(_ev(type="tool.call", name=f"pre{i}"))
            app._tick()
            await pilot.pause()
            for i in range(state.MAX_LOG_TAIL + 100):
                app._handle_event(_ev(type="tool.call", name=f"burst{i}"))
            app._tick()
            await pilot.pause()
            log = app._dash.query_one("#log", widgets.RichLog)
            assert len(log.lines) == state.MAX_LOG_TAIL  # bounded; pre-burst lines evicted

    asyncio.run(scenario())


def test_conversation_and_dashboard_footers_match(tmp_path: pathlib.Path) -> None:
    """The two run views share one footer scheme, Ctrl+D leftmost, no plain-letter keys.

    The conversation's one extra is ^t Detail, since the dashboard has no transcript.
    """
    from textual.widgets._footer import FooterKey

    async def scenario() -> None:
        (tmp_path / "logs.jsonl").write_text("", encoding="utf-8")
        app = tui_app.Agent6TUI(tmp_path, from_hub=True)

        async def footer_keys(pilot: Any, what: str) -> list[tuple[str, str]]:
            """The footer's entries once it renders its screen's bindings in their own order.

            The first render carries every key already, in an order of its own,
            and settles a frame later. The set, the count and two equal reads
            all read as ready during it; the screen's own `active_bindings` is
            the order it is settling towards, so that is what says when.
            """

            def settled() -> bool:
                shown = [k for k, b in app.screen.active_bindings.items() if b.binding.show]
                keys = [fk.key for fk in app.screen.query(FooterKey)]
                # The footer adds the palette itself; compare only the keys the screen offers.
                return bool(shown) and [k for k in keys if k in shown] == shown

            await wait_for(pilot, settled, what)
            return [(fk.key_display, fk.description) for fk in app.screen.query(FooterKey)]

        async with app.run_test(size=(120, 30)) as pilot:
            await wait_for(pilot, lambda: _screen_is(app, "_conv"), "the conversation screen")
            conv = await footer_keys(pilot, "the conversation footer")
            await _show_dashboard(pilot)
            dash = await footer_keys(pilot, "the dashboard footer")
            shared = [(k, d) for k, d in conv if d != "Detail"]
            assert [k for k, _ in shared] == [k for k, _ in dash]  # same keys, same order
            assert conv[0][0] == "^d" and conv[0][1] == "Dashboard"  # leftmost toggle
            assert dash[0][1] == "Conversation"
            toggles = ("Dashboard", "Conversation")
            assert [lbl for _, lbl in shared if lbl not in toggles] == [
                lbl for _, lbl in dash if lbl not in toggles
            ]
            # No plain single-letter shortcuts on either view.
            assert all(len(k) > 1 for k, _ in conv + dash)

    asyncio.run(scenario())


def test_dashboard_claims_are_per_process(tmp_path: pathlib.Path) -> None:
    """Concurrent front-ends hold independent claims; unmount removes only ours."""
    import os
    import subprocess
    import sys

    (tmp_path / "logs.jsonl").write_text("", encoding="utf-8")
    peer = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        ipc.register_frontend(tmp_path, peer.pid)  # a live web viewer's claim

        async def scenario() -> None:
            app = tui_app.Agent6TUI(tmp_path)
            async with app.run_test() as pilot:
                await pilot.pause()
                assert (tmp_path / "frontends" / str(os.getpid())).exists()  # ours
                assert (tmp_path / "frontends" / str(peer.pid)).exists()  # peer intact
            # Unmount removed only our claim; the peer keeps bridging.
            assert not (tmp_path / "frontends" / str(os.getpid())).exists()
            assert (tmp_path / "frontends" / str(peer.pid)).exists()
            assert ipc.frontend_is_live(tmp_path)

        asyncio.run(scenario())
    finally:
        peer.kill()
        peer.wait()


def test_dead_peer_claim_does_not_mask_the_live_dashboard(tmp_path: pathlib.Path) -> None:
    """A hard-killed viewer's stale claim does not affect liveness; the probe prunes it."""
    import os

    (tmp_path / "logs.jsonl").write_text("", encoding="utf-8")

    ipc.register_frontend(tmp_path, 999999999)  # dead

    async def scenario() -> None:
        app = tui_app.Agent6TUI(tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert ipc.frontend_is_live(tmp_path)  # our claim carries it
            assert not (tmp_path / "frontends" / "999999999").exists()  # pruned
            assert (tmp_path / "frontends" / str(os.getpid())).exists()

    asyncio.run(scenario())


def test_resume_reopens_the_approval_for_a_reused_prompt_id(tmp_path: pathlib.Path) -> None:
    """A dashboard held across a resume pops the new session's modal, whose ids restart at 1."""
    # These drive a live run, so its pid belongs on disk as it would be.
    (tmp_path / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")
    (tmp_path / "logs.jsonl").write_text("", encoding="utf-8")

    async def scenario() -> None:
        app = tui_app.Agent6TUI(tmp_path)
        async with app.run_test() as pilot:
            app._handle_event(_ev(type="session.start", user_task="session one", mode="run"))
            app._handle_event(_ev(type="approval.prompt", id="approval-1", prompt="first?"))
            app._tick()
            app._conv._poll()
            await pilot.pause()
            assert app._conv.query(composer.ApprovalRow)
            await focus_answers(app._conv, pilot)
            await pilot.press("y")
            assert await answer_written(tmp_path, pilot, "approval-1") == "yes"
            app._handle_event(_ev(type="approval.answer", id="approval-1", approved=True))
            # A resumed execution emits only loop.resume.start, never a second session.start.
            app._handle_event(_ev(type="loop.resume.start", iteration=2, messages=4))
            # The worker drops a stale answer as it emits the prompt.
            ipc.clear_answer(tmp_path, "approval-1")
            app._handle_event(_ev(type="approval.prompt", id="approval-1", prompt="again?"))
            app._tick()
            app._conv._poll()
            await pilot.pause()
            assert app._conv.query(composer.ApprovalRow)  # re-shown, not swallowed
            await focus_answers(app._conv, pilot)
            await pilot.press("n")
            assert await answer_written(tmp_path, pilot, "approval-1") == "no"

    asyncio.run(scenario())


def test_steer_request_marker_round_trip(tmp_path: pathlib.Path) -> None:
    assert not ipc.steer_request_pending(tmp_path)
    ipc.request_steer(tmp_path)
    assert ipc.steer_request_pending(tmp_path)  # the run's requested() sees this
    ipc.clear_steer_request(tmp_path)
    assert not ipc.steer_request_pending(tmp_path)


def test_dashboard_bar_is_default_focus_and_steers(tmp_path: pathlib.Path) -> None:
    """The dashboard opens with the composer focused, and Enter drops the steer marker and text.

    The run must be live: a dir with no worker routes the composer to resume instead.
    """
    import os

    async def scenario() -> None:
        (tmp_path / "logs.jsonl").write_text("", encoding="utf-8")
        ipc.write_worker_pid(tmp_path, os.getpid())
        app = tui_app.Agent6TUI(tmp_path)
        async with app.run_test(size=(100, 30)) as pilot:
            await _show_dashboard(pilot)
            bar = app._dash.query_one("#dash-input", composer.SteerInput)
            await _settle_focus(pilot, bar)
            assert app.focused is bar  # default focus: type at once, no popup
            assert not ipc.steer_request_pending(tmp_path)
            await pilot.press("g", "o")
            await pilot.press("enter")
            await pilot.pause()
            assert ipc.steer_request_pending(tmp_path)  # marker dropped for the run
            assert (tmp_path / "steer.answer").read_text(encoding="utf-8") == "go"

    asyncio.run(scenario())


def test_finished_run_bar_resumes_with_the_instruction(
    tmp_path: pathlib.Path, monkeypatch: Any
) -> None:
    """Typing into the composer of a finished run spawns a detached `agent6 resume --steer=<text>`.

    The follow-up rides the flag, since resume's stale-state clear would wipe a seeded file.
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
    (tmp_path / "logs.jsonl").write_text(
        "".join(
            json.dumps(e) + "\n"
            for e in (
                _ev(type="session.start", user_task="x", mode="run"),
                _ev(type="session.end", reason="completed", all_passed=True),
            )
        ),
        encoding="utf-8",
    )

    async def scenario() -> None:
        app = tui_app.Agent6TUI(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            for _ in range(50):
                await pilot.pause()
                if app.state.finished:
                    break
            await pilot.pause()
            bar = app._conv.query_one("#conv-input", composer.SteerInput)
            assert bar.display  # the primary view keeps the bar after session.end
            assert bar.border_title == "continue this session"  # relabelled for resume
            bar.post_message(composer.SteerInput.Submitted("also add tests"))
            await pilot.pause()
            await pilot.pause()
            # The instruction rides --steer on the detached resume.
            await app.workers.wait_for_complete()
            assert spawned == [(tmp_path.name, "also add tests")]

    asyncio.run(scenario())


def test_end_hold_follows_the_resumed_execution(tmp_path: pathlib.Path, monkeypatch: Any) -> None:
    """Continuing from the foreground run's end hold keeps the same view live."""
    events = (
        _ev(type="session.start", user_task="finish the parser", mode="run"),
        _ev(type="session.end", reason="finish_session", all_passed=True),
    )
    (tmp_path / "logs.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )
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
        (tmp_path / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")
        with (tmp_path / "logs.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps(_ev(type="loop.resume.start", iteration=2)) + "\n")
            log.write(json.dumps(_ev(type="role.call", role="worker", model="m")) + "\n")
        return ""

    monkeypatch.setattr(spawn, "spawn_detached_resume", _fake_resume)

    async def scenario() -> None:
        app = tui_app.Agent6TUI(tmp_path, exit_on_end=True)
        async with app.run_test(size=(120, 40)) as pilot:
            await wait_for(pilot, lambda: app._end_hold, "the end hold")
            conv_row = app._conv.query_one("#conv-resume", composer.ResumeOptions)
            assert conv_row.display
            app._conv.query_one("#conv-input", composer.SteerInput).post_message(
                composer.SteerInput.Submitted("also add tests")
            )
            await app.workers.wait_for_complete()
            await wait_for(pilot, lambda: not app.state.finished, "the resumed execution")
            assert spawned == [(tmp_path.name, "also add tests")]
            assert app.dir_status[0] == "running"
            assert not app._end_hold
            assert app._conv.query_one("#conv-input", composer.SteerInput).mode == "steer"
            assert not conv_row.display
            await _show_dashboard(pilot)
            assert not app._dash.query_one("#dash-resume", composer.ResumeOptions).display
            assert "Ctrl+Q to leave" not in str(app.sub_title)

    asyncio.run(scenario())


def _no_kill(_session_dir: pathlib.Path, _worker: object, _grace_s: float) -> tuple[bool, int]:
    """The escalation stubbed out: a test's worker pid is its own process."""
    return False, 0


def test_stop_now_aborts_via_bridge(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run > Stop now on a live run confirms, then lands the abort steer and the stop marker."""
    import os

    import agent6.app.stop as stop_mod

    monkeypatch.setattr(stop_mod, "STOP_WAIT_S", 0.5)
    monkeypatch.setattr(stop_mod, "_kill", _no_kill)  # never this process

    async def scenario() -> None:
        (tmp_path / "logs.jsonl").write_text("", encoding="utf-8")
        ipc.write_worker_pid(tmp_path, os.getpid())
        app = tui_app.Agent6TUI(tmp_path)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app.action_stop_now()
            await pilot.pause()
            assert isinstance(app.screen, modals.ConfirmModal)  # confirms before stopping
            await pilot.press("y")  # confirm
            for _ in range(50):  # the stop runs on a thread
                await pilot.pause()
                if ipc.steer_request_pending(tmp_path) and ipc.stop_request_pending(tmp_path):
                    break
            assert (tmp_path / "steer.answer").read_text(encoding="utf-8") == "abort"
            assert ipc.steer_request_pending(tmp_path) and ipc.stop_request_pending(tmp_path)
            await app.workers.wait_for_complete()
            notes = [str(n.message) for n in app._notifications]  # pyright: ignore[reportPrivateUsage]
            assert any("did not answer" in note for note in notes)

    asyncio.run(scenario())


def test_ctrl_z_on_the_run_spawned_view_detaches_the_run_itself(tmp_path: pathlib.Path) -> None:
    """Ctrl-Z on the view `agent6 run --tui` spawns steers a detach; a plain viewer leaves the run.

    That view fronts a run in the terminal's own process, so leaving it alone left the run
    streaming in the foreground.
    """
    import os

    async def spawned_view() -> None:
        (tmp_path / "logs.jsonl").write_text("", encoding="utf-8")
        ipc.write_worker_pid(tmp_path, os.getpid())
        app = tui_app.Agent6TUI(tmp_path, exit_on_end=True)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            await pilot.press("ctrl+z")
            await pilot.pause()
        assert app.detached
        assert (tmp_path / "steer.answer").read_text(encoding="utf-8") == "detach"
        assert ipc.steer_request_pending(tmp_path)

    async def viewer() -> None:
        (tmp_path / "steer.answer").unlink()
        (tmp_path / "steer.request").unlink()
        app = tui_app.Agent6TUI(tmp_path)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            await pilot.press("ctrl+z")
            await pilot.pause()
        assert app.detached
        assert not ipc.steer_request_pending(tmp_path)

    async def hub_viewer() -> None:
        app = tui_app.Agent6TUI(tmp_path, from_hub=True)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            await pilot.press("ctrl+z")
            await pilot.pause()
        assert app.detached
        assert app.return_value == tui_app.TuiExit()  # reopen the hub, do not quit its loop

    asyncio.run(spawned_view())
    asyncio.run(viewer())
    asyncio.run(hub_viewer())


def test_a_typed_stop_is_the_one_stop(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`/stop` in the composer on a live run stops without a confirm; a dead session says so."""
    import os

    from agent6.app import stop

    calls: list[bool] = []

    def _fake(session_dir: pathlib.Path, *, after_step: bool = False) -> stop.StopOutcome:
        calls.append(after_step)
        return stop.StopOutcome(session_dir.name, True, "stopped", f"{session_dir.name} stopped")

    monkeypatch.setattr(app_stop, "stop_session", _fake)

    async def scenario() -> None:
        (tmp_path / "logs.jsonl").write_text("", encoding="utf-8")
        ipc.write_worker_pid(tmp_path, os.getpid())
        app = tui_app.Agent6TUI(tmp_path)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app.submit_instruction("/stop")
            await app.workers.wait_for_complete()
            assert calls == [False]
            notes = [(str(n.message), n.severity) for n in app._notifications]  # pyright: ignore[reportPrivateUsage]
            assert (f"{tmp_path.name} stopped", "information") in notes

    async def finished() -> None:
        (tmp_path / "logs.jsonl").write_text(
            json.dumps(_ev(type="session.end", reason="finish_session")) + "\n",
            encoding="utf-8",
        )
        app = tui_app.Agent6TUI(tmp_path)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app.submit_instruction("/stop")
            await pilot.pause()
            assert calls == [False]
            notes = [(str(n.message), n.severity) for n in app._notifications]  # pyright: ignore[reportPrivateUsage]
            assert ("nothing to stop: the session is not live", "warning") in notes

    asyncio.run(scenario())
    asyncio.run(finished())


def test_stop_after_step_drops_the_marker(tmp_path: pathlib.Path) -> None:
    """Run > Stop after this step on a live run confirms, then drops the stop.request marker."""
    import os

    async def scenario() -> None:
        (tmp_path / "logs.jsonl").write_text("", encoding="utf-8")
        ipc.write_worker_pid(tmp_path, os.getpid())
        app = tui_app.Agent6TUI(tmp_path)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            assert not ipc.stop_request_pending(tmp_path)
            app.action_stop_step()
            await pilot.pause()
            assert isinstance(app.screen, modals.ConfirmModal)  # confirms before stopping
            await pilot.press("y")  # confirm
            for _ in range(50):  # the stop runs on a thread
                await pilot.pause()
                if ipc.stop_request_pending(tmp_path):
                    break
            assert ipc.stop_request_pending(tmp_path)  # marker for the boundary stop
            assert not (tmp_path / "steer.answer").exists()  # no mid-turn abort

    asyncio.run(scenario())


def test_context_pct_readout_in_top_line_and_bar(tmp_path: pathlib.Path, monkeypatch: Any) -> None:
    """With the context window known, the top line and the composer subtitle show `ctx: NN%`."""

    def _window(_provider: str, _model: str) -> int:
        return 100_000

    monkeypatch.setattr("agent6.viewmodel.state._window", _window)
    (tmp_path / "logs.jsonl").write_text("", encoding="utf-8")

    async def scenario() -> None:
        app = tui_app.Agent6TUI(tmp_path)
        async with app.run_test(size=(150, 40)) as pilot:
            await _show_dashboard(pilot)
            app._handle_event(_ev(type="session.start", user_task="x", mode="run"))
            app._handle_event(_ev(type="role.call", role="worker", model="m", provider="p"))
            app._handle_event(
                _ev(
                    type="role.result",
                    role="worker",
                    ok=True,
                    tokens_in=1_000,
                    cache_read=40_000,
                    cache_creation=0,
                )
            )
            app._tick()
            await pilot.pause()
            assert app.context_pct() == 41
            top = str(app._dash.query_one("#top", widgets.Static).render())
            assert "ctx: 41%" in top
            bar = app._dash.query_one("#dash-input", composer.SteerInput)
            assert "ctx 41%" in (bar.border_subtitle or "")
            await pilot.press("ctrl+d")
            await pilot.pause()
            conv_bar = app._conv.query_one("#conv-input", composer.SteerInput)
            assert "ctx 41%" in (conv_bar.border_subtitle or "")

    asyncio.run(scenario())


def test_working_timer_anchors_to_the_last_events_ts_not_the_attach(tmp_path: pathlib.Path) -> None:
    """The idle anchor is the last event's own ts, so a wedged run reads as wedged.

    Replayed history bumped the anchor, so a run wedged 40 minutes read "working… 3s".
    """

    async def scenario() -> None:

        (tmp_path / "logs.jsonl").write_text("", encoding="utf-8")
        ipc.write_worker_pid(tmp_path, os.getpid())  # alive-but-wedged, not stale
        app = tui_app.Agent6TUI(tmp_path)
        async with app.run_test(size=(150, 40)) as pilot:
            await _show_dashboard(pilot)
            wedged_at = time.time() - 2400
            app._handle_event(
                _ev(type="session.start", user_task="x", mode="run", ts=wedged_at - 5)
            )
            app._handle_event(_ev(type="role.call", role="worker", model="m", ts=wedged_at))
            app._tick()
            await pilot.pause()
            assert app.seconds_since_event() >= 2399
            top = str(app._dash.query_one("#top", widgets.Static).render())
            import re

            assert re.search(r" 2[34]\d\ds", top), top  # the visible beat reads the helper

    asyncio.run(scenario())


def test_budget_meter_reads_this_executions_spend_not_the_runs_total(
    tmp_path: pathlib.Path,
) -> None:
    """The budget percentage divides the execution's own spend by its cap.

    The cap re-arms each resume while usd_total stays cumulative.
    """

    async def scenario() -> None:
        (tmp_path / "logs.jsonl").write_text("", encoding="utf-8")
        app = tui_app.Agent6TUI(tmp_path)
        async with app.run_test(size=(150, 40)) as pilot:
            await _show_dashboard(pilot)
            app._handle_event(_ev(type="session.start", user_task="x", mode="run"))
            app._handle_event(_ev(type="role.call", role="worker", model="m"))
            app._handle_event(_ev(type="budget.update", usd_total=2.0, usd_cap=2.5))
            app._handle_event(_ev(type="loop.resume.start"))
            app._handle_event(_ev(type="budget.update", usd_total=0.5, usd_cap=2.5))
            app._tick()
            await pilot.pause()
            top = str(app._dash.query_one("#top", widgets.Static).render())
            assert "budget: 20%" in top, top
            assert "$2.50" in top  # the cost figure stays cumulative

    asyncio.run(scenario())


def test_compact_now_drops_the_marker_for_a_live_run(tmp_path: pathlib.Path) -> None:
    """Run > Compact context now drops the compact.request marker; a finished run refuses."""
    import os

    async def scenario() -> None:
        (tmp_path / "logs.jsonl").write_text("", encoding="utf-8")
        ipc.write_worker_pid(tmp_path, os.getpid())
        app = tui_app.Agent6TUI(tmp_path)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            assert ipc.read_compact_request(tmp_path) is None
            app.action_compact()
            await pilot.pause()
            assert ipc.read_compact_request(tmp_path) is not None  # marker dropped for the run
            # A finished run: the action refuses instead of dropping a marker.
            (tmp_path / "compact.request").unlink()
            app._handle_event(_ev(type="session.end", reason="completed", all_passed=True))
            app.action_compact()
            await pilot.pause()
            assert ipc.read_compact_request(tmp_path) is None

    asyncio.run(scenario())


def test_payload_literal_does_not_swallow_a_real_steer(tmp_path: pathlib.Path) -> None:
    # The seed counts steers as the fold does: a payload quoting the event name must not inflate it.

    (tmp_path / "logs.jsonl").write_text(
        "".join(
            json.dumps(e) + "\n"
            for e in (
                _ev(type="session.start", user_task="x"),
                _ev(
                    type="tool.call",
                    name="run_command",
                    args={"argv": ["rg", "session.steer_requested", "src/"]},
                ),
                _ev(type="role.call", role="worker", model="kimi"),
            )
        ),
        encoding="utf-8",
    )

    async def scenario() -> None:
        app = tui_app.Agent6TUI(tmp_path)
        async with app.run_test() as pilot:
            await _show_dashboard(pilot)
            app._dash.query_one("#log").focus()
            await pilot.pause()
            for _ in range(50):  # let the reader thread replay the existing log
                await pilot.pause()
            assert app._seen_steer == 0  # pyright: ignore[reportPrivateUsage]
            assert app.state.steer_requests == 0
            app._handle_event(_ev(type="session.steer_requested", source="sigint"))  # pyright: ignore[reportPrivateUsage]
            app._tick()  # pyright: ignore[reportPrivateUsage]
            bar = app._dash.query_one(composer.SteerInput)
            await _settle_focus(pilot, bar)
            assert app.focused is bar

    asyncio.run(scenario())


def test_historical_steer_request_does_not_grab_the_bar_on_open(tmp_path: pathlib.Path) -> None:
    # A stale steer request left by a detached Ctrl-C is not live; only one arriving later routes.

    (tmp_path / "logs.jsonl").write_text(
        "".join(
            json.dumps(e) + "\n"
            for e in (
                _ev(type="session.start", user_task="fix it"),
                _ev(type="session.steer_requested", source="sigint"),
                _ev(type="role.call", role="worker", model="kimi"),
            )
        ),
        encoding="utf-8",
    )

    async def scenario() -> None:
        app = tui_app.Agent6TUI(tmp_path)
        async with app.run_test() as pilot:
            await _show_dashboard(pilot)
            app._dash.query_one("#log").focus()  # park focus off the default-focused bar
            await pilot.pause()
            for _ in range(50):  # let the reader thread replay the existing log
                await pilot.pause()
                if app.state.steer_requests >= 1:
                    break
            app._tick()
            await pilot.pause()
            assert app.state.steer_requests == 1  # the historical event WAS folded
            bar = app._dash.query_one("#dash-input", composer.SteerInput)
            assert app.focused is not bar  # but it did NOT grab the composer
            # a NEW steer request (a live Ctrl-C while watching) still routes here
            app._handle_event(_ev(type="session.steer_requested", source="sigint"))
            app._tick()
            await _settle_focus(pilot, bar)
            assert app.focused is bar

    asyncio.run(scenario())


def test_toggle_and_log_viewer_keys(tmp_path: pathlib.Path) -> None:
    # Ctrl+D flips the views with a composer focused; the log viewer opens from the View menu.
    from agent6.ui.tui import dashboard, logview

    (tmp_path / "logs.jsonl").write_text(
        json.dumps({"type": "session.start", "user_task": "x"}) + "\n", encoding="utf-8"
    )

    async def scenario() -> None:
        app = tui_app.Agent6TUI(tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.screen is app._conv  # the conversation is the primary view
            await pilot.press("ctrl+d")  # show the dashboard
            await pilot.pause()
            assert isinstance(app.screen, dashboard.DashboardScreen)

            assert isinstance(app.focused, composer.SteerInput)  # the bar is the default focus
            depth = len(app.screen_stack)
            app._dash.action_view_logs()  # View menu / palette action (no bare letters)
            await pilot.pause()
            assert isinstance(app.screen, logview.LogScreen)
            await pilot.press("l")  # LogScreen (no input box) keeps its letters
            await pilot.pause()
            assert isinstance(app.screen, dashboard.DashboardScreen)
            assert len(app.screen_stack) == depth
            await pilot.press("ctrl+d")  # flip back to the conversation (bar focused)
            await pilot.pause()
            assert app.screen is app._conv
            await pilot.press("ctrl+d")  # and again to the dashboard
            await pilot.pause()
            assert isinstance(app.screen, dashboard.DashboardScreen)

    asyncio.run(scenario())


def test_task_filter_scopes_tools_log_and_diff(tmp_path: pathlib.Path) -> None:
    # Two tasks, each with a tool call and a commit; selecting one filters to its activity.
    def _nodes(cur_status: dict[str, str]) -> dict[str, object]:
        return {
            tid: {"title": t, "status": cur_status[tid], "parent_id": None, "children": []}
            for tid, t in (("t1", "Task one"), ("t2", "Task two"))
        }

    events = [
        {"type": "session.start", "user_task": "x"},
        {
            "type": "graph.update",
            "nodes": _nodes({"t1": "in_progress", "t2": "pending"}),
            "cursor": "t1",
        },
        {"type": "tool.call", "name": "read_file", "args": {"path": "a.py"}},
        {"type": "diff.updated", "sha": "aaa", "patch": "diff --git a/a.py b/a.py\n+one"},
        {
            "type": "graph.update",
            "nodes": _nodes({"t1": "passed", "t2": "in_progress"}),
            "cursor": "t2",
        },
        {"type": "tool.call", "name": "apply_edit", "args": {"path": "b.py"}},
        {"type": "diff.updated", "sha": "bbb", "patch": "diff --git a/b.py b/b.py\n+two"},
    ]
    (tmp_path / "logs.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8"
    )

    async def scenario() -> None:
        app = tui_app.Agent6TUI(tmp_path)
        async with app.run_test(size=(150, 42)) as pilot:
            for _ in range(60):
                await pilot.pause()
                if len(app.state.recent_diffs) >= 2:  # the last events emitted
                    break
            # Fold stamped each tool call + diff with the task in focus at the time.
            assert [tc.task_id for tc in app.state.tool_calls] == ["t1", "t2"]
            assert [d.task_id for d in app.state.recent_diffs] == ["t1", "t2"]

            dash = app._dash
            dash.render_state()  # the dashboard renders on demand while covered
            await pilot.pause()
            assert len(dash._visible_tools) == 2  # unfiltered: both

            def log_text() -> str:
                return " ".join(
                    strip.text for strip in dash.query_one("#log", widgets.RichLog).lines
                )

            def diff_text() -> str:
                return str(dash.query_one("#diff-body", widgets.Static).content)

            dash._selected_task_id = "t1"  # filter to task one (the handler re-renders)
            dash.render_state()
            await pilot.pause()
            assert [tc.name for tc in dash._visible_tools] == ["read_file"]
            assert "read_file" in log_text() and "apply_edit" not in log_text()
            assert "+one" in diff_text() and "+two" not in diff_text()

            dash._selected_task_id = "t2"
            dash.render_state()
            await pilot.pause()
            assert [tc.name for tc in dash._visible_tools] == ["apply_edit"]
            assert "apply_edit" in log_text() and "read_file" not in log_text()
            assert "+two" in diff_text() and "+one" not in diff_text()

    asyncio.run(scenario())


def test_question_modal_multi_collects_all_answers() -> None:
    """A multi-question prompt returns every answer as a tuple aligned to the questions."""
    result: dict[str, tuple[str, ...] | None] = {}
    qs = (
        state.Question(question="Framework?", options=("React", "Vue")),
        state.Question(question="Component name?", options=()),
    )

    class _Host(textual_app.App[None]):
        def on_mount(self) -> None:
            self.push_screen(modals.QuestionModal("q1", qs), lambda v: result.__setitem__("v", v))

    async def scenario() -> None:
        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = app.screen
            assert isinstance(modal, modals.QuestionModal)
            modal.query_one("#opt-0-1", widgets.Button).press()  # pick "Vue" for the first question
            await pilot.pause()
            assert modal.query_one("#ans-0", widgets.Input).value == "Vue"
            modal.query_one("#ans-1", widgets.Input).value = "widget"  # type the second answer
            await pilot.press("ctrl+s")
            await pilot.pause()
            assert result.get("v") == ("Vue", "widget")  # both, aligned to the questions

    asyncio.run(scenario())


def test_conversation_is_the_primary_view(tmp_path: pathlib.Path) -> None:
    """Ctrl+D toggles the dashboard and back with the same conversation instance; Esc leaves."""
    from agent6.ui.tui import conversation, dashboard

    (tmp_path / "logs.jsonl").write_text(
        json.dumps({"type": "session.start", "user_task": "x"}) + "\n", encoding="utf-8"
    )

    async def scenario() -> None:
        app = tui_app.Agent6TUI(tmp_path, from_hub=True)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            assert isinstance(app.screen, conversation.ConversationScreen)
            first = app.screen
            await pilot.press("ctrl+d")
            await pilot.pause()
            assert isinstance(app.screen, dashboard.DashboardScreen)
            await pilot.press("ctrl+d")
            await pilot.pause()
            assert app.screen is first  # the same instance: nothing was rebuilt
            await pilot.press("escape")  # Esc on the primary view leaves for the hub
            await pilot.pause()
        assert app.return_value == tui_app.TuiExit()

    asyncio.run(scenario())


def test_dashboard_detects_a_dead_worker_and_tells_the_truth(
    tmp_path: pathlib.Path, monkeypatch: Any
) -> None:
    """A worker killed without a session.end reads stale, and the composer routes to resume.

    The heartbeat probe flips the dir status; without it the dashboard spun "working…" forever,
    accepted steer with a toast and refused resume, the one correct action.
    """
    import json

    events = [
        {"type": "session.start", "session_id": "dead-01", "mode": "run", "user_task": "t"},
        {"type": "role.call", "role": "worker", "model": "m", "provider": "p"},
    ]
    (tmp_path / "logs.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in events), encoding="utf-8"
    )
    (tmp_path / "worker.pid").write_text("999999999", encoding="utf-8")  # dead pid

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

    async def scenario() -> None:
        app = tui_app.Agent6TUI(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await _show_dashboard(pilot)
            app._heartbeat_at = 0.0  # age the throttle so the probe fires
            app._tick()
            await pilot.pause()
            assert app.worker_lost is True
            assert app.session_controllable() is False
            body = str(app._dash.query_one("#stream-body", widgets.Static).render())
            assert "worker exited" in body
            assert "working…" not in body
            # The composer routes to resume (not a steer nobody will read).
            app.submit_instruction("carry on")
            await app.workers.wait_for_complete()
            assert spawned == [(tmp_path.name, "carry on")]
            # And action_resume resumes instead of refusing "still going".
            app.action_resume()
            await app.workers.wait_for_complete()
            assert len(spawned) == 2

    asyncio.run(scenario())


def test_dashboard_heartbeat_ticks_while_active(tmp_path: pathlib.Path) -> None:
    """A live but silent run shows a ticking "working… Ns" heartbeat."""
    import json

    events = [
        {"type": "session.start", "session_id": "live-01", "mode": "run", "user_task": "t"},
        {"type": "role.call", "role": "worker", "model": "m", "provider": "p"},
    ]
    (tmp_path / "logs.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in events), encoding="utf-8"
    )
    # A run with no worker.pid is one whose worker exited, and it shows no ticking heartbeat.
    (tmp_path / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")

    import re

    def _seconds(text: str) -> int:
        m = re.search(r"working… (\d+)s", text)
        return int(m.group(1)) if m else 0

    async def scenario() -> str:
        app = tui_app.Agent6TUI(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await _show_dashboard(pilot)

            def advanced() -> bool:
                app._tick()  # pyright: ignore[reportPrivateUsage]
                return (
                    _seconds(str(app._dash.query_one("#stream-body", widgets.Static).render())) >= 1
                )

            # The heartbeat needs real wall time (at least 1s since the last event), so poll for it.
            await wait_for(pilot, advanced, "the working… heartbeat to advance", timeout=15.0)
            return str(app._dash.query_one("#stream-body", widgets.Static).render())

    text = asyncio.run(scenario())
    assert "working…" in text
    assert _seconds(text) >= 1  # the heartbeat advanced


def test_tick_survives_an_empty_screen_stack(tmp_path: pathlib.Path) -> None:
    """A tick on an empty screen stack is a no-op, the steer-request routing path included.

    A tick landing in teardown's window hit the raising App.screen property and crashed the app.
    """

    async def scenario() -> None:
        (tmp_path / "logs.jsonl").write_text("", encoding="utf-8")
        app = tui_app.Agent6TUI(tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause()
            # Empty the stack only around the synchronous tick, so teardown still finds its screens.
            stack = app._screen_stack  # pyright: ignore[reportPrivateUsage]
            saved = list(stack)
            stack.clear()
            app._seen_steer = -1  # pyright: ignore[reportPrivateUsage]
            app._tick()  # pyright: ignore[reportPrivateUsage]
            stack.extend(saved)

    asyncio.run(scenario())


def test_dashboard_follows_live_appends_after_attach(tmp_path: pathlib.Path) -> None:
    """Events appended after attach appear; the view is not a frozen snapshot."""
    import json

    logs = tmp_path / "logs.jsonl"

    def append(events: list[dict[str, object]]) -> None:
        with logs.open("a", encoding="utf-8") as fh:
            for e in events:
                fh.write(json.dumps(e) + "\n")

    logs.write_text("", encoding="utf-8")
    append(
        [
            {"type": "session.start", "session_id": "live-02", "mode": "run", "user_task": "t"},
            {"type": "tool.call", "name": "read_file", "args": {"path": "a.py"}},
            {"type": "tool.result", "name": "read_file", "ok": True, "summary": "1 byte"},
        ]
    )

    async def scenario() -> int:
        app = tui_app.Agent6TUI(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await _show_dashboard(pilot)

            def rows() -> int:
                app._tick()  # pyright: ignore[reportPrivateUsage]
                return app._dash.query_one("#tools", widgets.DataTable).row_count

            # Wait through the reader thread's initial fold, not a fixed budget.
            await wait_for(pilot, lambda: rows() >= 1, "the attach-time fold", timeout=15.0)
            before = app._dash.query_one("#tools", widgets.DataTable).row_count
            # The background process appends a new turn AFTER we attached.
            append(
                [
                    {"type": "tool.call", "name": "apply_edit", "args": {"path": "a.py"}},
                    {"type": "tool.result", "name": "apply_edit", "ok": True, "summary": "applied"},
                ]
            )
            await wait_for(pilot, lambda: rows() > before, "the appended turn", timeout=15.0)
            return app._dash.query_one("#tools", widgets.DataTable).row_count

    assert asyncio.run(scenario()) == 2


def test_composer_compact_directive_routes_to_compact_request(tmp_path: pathlib.Path) -> None:
    """A composer `/compact <focus>` on a live run is a compact request; `/pin` stays a steer."""
    import json
    import os

    events = [
        {"type": "session.start", "session_id": "live-01", "mode": "run", "user_task": "t"},
        {"type": "role.call", "role": "worker", "model": "m", "provider": "p"},
    ]
    (tmp_path / "logs.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in events), encoding="utf-8"
    )
    (tmp_path / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")  # live

    async def scenario() -> None:
        app = tui_app.Agent6TUI(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await _show_dashboard(pilot)
            app.submit_instruction("/compact keep the auth decisions")
            await pilot.pause()
            assert (tmp_path / "compact.request").read_text(encoding="utf-8") == (
                "keep the auth decisions"
            )
            assert not (tmp_path / "steer.answer").exists()
            assert not (tmp_path / "steer.request").exists()
            app.submit_instruction("/pin keep the API stable")
            await pilot.pause()
            assert (tmp_path / "steer.answer").read_text(encoding="utf-8") == (
                "/pin keep the API stable"
            )

    asyncio.run(scenario())


def test_the_composer_title_shows_its_brackets(tmp_path: pathlib.Path) -> None:
    """The composer's border title is escaped, so `/compact [focus]` keeps its `[focus]`."""
    import os

    from rich import markup

    async def scenario() -> None:
        (tmp_path / "logs.jsonl").write_text("", encoding="utf-8")
        ipc.write_worker_pid(tmp_path, os.getpid())
        app = tui_app.Agent6TUI(tmp_path)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            bar = app._conv.query_one("#conv-input", composer.SteerInput)
            title, _keys = composer.composer_labels("steer")
            assert "[focus]" in title
            assert bar.border_title == markup.escape(title)

    asyncio.run(scenario())


def test_the_menu_bar_title_keeps_the_tasks_brackets(tmp_path: pathlib.Path) -> None:
    """The bar mirrors the app subtitle escaped, so a task's `[wip]` does not vanish as markup."""

    async def scenario() -> None:
        (tmp_path / "logs.jsonl").write_text(
            json.dumps(_ev(type="session.start", user_task="fix [wip] parser", mode="run")) + "\n",
            encoding="utf-8",
        )
        app = tui_app.Agent6TUI(tmp_path)
        async with app.run_test(size=(120, 30)) as pilot:
            for _ in range(30):
                await pilot.pause()
                if app.state.user_task:
                    break
            await pilot.pause()
            shown = app.screen.query_one(".app-title", widgets.Static).render()
            text = shown.plain if isinstance(shown, rich_text.Text) else str(shown)
            assert "[wip]" in text, text

    asyncio.run(scenario())


def test_the_hidden_detail_level_says_what_it_hides(tmp_path: pathlib.Path) -> None:
    """At detail "hidden" the view says what is hidden and the key that shows it.

    It painted the empty-conversation placeholder over a run in its first minutes.
    """
    import json

    events = [
        {"type": "session.start", "mode": "run", "user_task": "look"},
        {"type": "role.thinking_delta", "text": "let me look"},
        {"type": "tool.call", "name": "read_file", "args": {"path": "a.py"}},
        {"type": "tool.result", "name": "read_file", "ok": True, "summary": "40 lines"},
    ]
    (tmp_path / "logs.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in events), encoding="utf-8"
    )

    async def scenario() -> None:
        app = tui_app.Agent6TUI(tmp_path)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            screen = app._conv  # pyright: ignore[reportPrivateUsage]
            screen._detail = "hidden"
            screen._reload()
            await pilot.pause()
            tail = str(screen._tail_widget().render())
            assert "no conversation" not in tail
            assert "hidden at this detail level" in tail and "Ctrl+T" in tail
            # A call in flight renders no sealed line at any level; the note is for hidden alone.
            (tmp_path / "logs.jsonl").write_text(
                json.dumps(events[0]) + "\n" + json.dumps(events[2]) + "\n", encoding="utf-8"
            )
            screen._detail = "collapsed"
            screen._reload()
            await pilot.pause()
            assert "hidden at this detail level" not in str(screen._tail_widget().render())

    asyncio.run(scenario())


def test_dashboard_names_the_manifests_driver_before_the_first_call(tmp_path: pathlib.Path) -> None:
    """Before any role.call the header names the driver and says the worker is starting.

    The first call then supplies the role and model shown with its in-flight beat.
    """
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "mode": "plan",
                "session_id": tmp_path.name,
                "user_task": "t",
                "models": {"driver": {"provider": "p", "model": "m0"}},
            }
        ),
        encoding="utf-8",
    )
    logs = tmp_path / "logs.jsonl"
    logs.write_text("", encoding="utf-8")
    (tmp_path / "worker.pid").write_text(str(os.getpid()), encoding="utf-8")

    async def scenario() -> None:
        app = tui_app.Agent6TUI(tmp_path)
        async with app.run_test(size=(150, 40)) as pilot:
            await _show_dashboard(pilot)
            top = str(app._dash.query_one("#top", widgets.Static).render())
            body = str(app._dash.query_one("#stream-body", widgets.Static).render())
            assert "role: planner / m0" in top
            assert "starting" in top and "starting" in body
            with logs.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(_ev(type="session.start", user_task="t", mode="plan")) + "\n")
                fh.write(
                    json.dumps(_ev(type="role.call", role="planner", model="m1", provider="p"))
                    + "\n"
                )
            await wait_for(
                pilot,
                lambda: app.state.last_role is not None and app.state.last_role.in_flight,
                "the first model call",
            )
            app._tick()
            await pilot.pause()
            top = str(app._dash.query_one("#top", widgets.Static).render())
            assert "role: planner / m1" in top

    asyncio.run(scenario())


def test_dashboard_does_not_call_a_dead_driverless_run_idle(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing driver supplies no role fact, not an idle state.

    A worker that died launching reads stale, while the unknown role is a dash cached from the
    manifest rather than a manifest read on every heartbeat.
    """
    (tmp_path / "manifest.json").write_text(
        json.dumps({"mode": "run", "session_id": tmp_path.name, "user_task": "t"}),
        encoding="utf-8",
    )
    (tmp_path / "logs.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "worker.pid").write_text("999999999", encoding="utf-8")
    real = manifest.read_manifest
    reads: list[int] = []

    def _counted(session_dir: pathlib.Path) -> object:
        reads.append(1)
        return real(session_dir)

    monkeypatch.setattr(manifest, "read_manifest", _counted)

    async def scenario() -> None:
        app = tui_app.Agent6TUI(tmp_path)
        async with app.run_test(size=(150, 40)) as pilot:
            await _show_dashboard(pilot)
            before = len(reads)
            header = app._dash.query_one(_dashboard_header.RunHeader)  # pyright: ignore[reportPrivateUsage]
            lines = {header._start_role() for _ in range(4)}  # pyright: ignore[reportPrivateUsage]
            assert lines == {"(unknown)"}
            assert len(reads) - before <= 1
            top = str(app._dash.query_one("#top", widgets.Static).render())
            assert "stale" in top
            assert "idle" not in top

    asyncio.run(scenario())


def test_the_dashboard_header_leads_with_the_status_and_never_wraps(tmp_path: pathlib.Path) -> None:
    """A long model id pushed "failed · max iterations" onto a line of its own, split mid-phrase.

    The status leads line 1 and every header line ends in an ellipsis before it would wrap.
    """

    async def scenario() -> None:
        app = tui_app.Agent6TUI(tmp_path)
        async with app.run_test(size=(100, 40)) as pilot:
            await _show_dashboard(pilot)
            model = "moonshotai/kimi-k2.7-code-20260612-with-a-very-long-suffix"
            app._handle_event(_ev(type="session.start", user_task="t " * 80, mode="run"))
            app._handle_event(_ev(type="role.call", role="worker", model=model, provider="p"))
            app._handle_event(_ev(type="session.end", all_passed=False, reason="max_iterations"))
            app._tick()
            await pilot.pause()
            await pilot.pause()
            top = app._dash.query_one("#top", widgets.Static)
            lines = str(top.render()).split("\n")
            assert lines[0].startswith("agent6  failed · max iterations")
            assert lines[1].startswith("role: worker / moonshotai/kimi")
            assert all(len(line) <= top.content_size.width for line in lines)
            assert lines[1].endswith("…")

    asyncio.run(scenario())


def test_a_short_terminal_dashboard_shows_one_pane_row_at_a_time(tmp_path: pathlib.Path) -> None:
    """At 80x24 every pane got a line or two.

    Below 28 rows the dashboard folds to the row holding focus (the log and diff otherwise) plus a
    summary line, and Tab still reaches a folded pane, which unfolds it.
    """

    async def scenario() -> None:
        app = tui_app.Agent6TUI(tmp_path)
        async with app.run_test(size=(80, 24)) as pilot:
            await _show_dashboard(pilot)
            app._handle_event(_ev(type="session.start", user_task="x", mode="run"))
            app._handle_event(_ev(type="tool.call", name="read_file", args={"path": "a.py"}))
            app._tick()
            await pilot.pause()
            dash = app._dash
            assert dash.has_class("-compact")
            tools = dash.query_one("#tools", widgets.DataTable)
            assert tools.region.height == 0 and dash.query_one("#log").region.height > 3
            summary = str(dash.query_one("#summary", widgets.Static).render())
            assert "1 tool call · last read_file" in summary
            tools.focus()
            # The fold follows the focus through a relayout, a frame or more away under load.
            await wait_for(pilot, lambda: tools.region.height > 3, "the tools pane unfolded")
            assert dash.query_one("#body").region.height == 0
            await pilot.resize_terminal(120, 40)
            await wait_for(pilot, lambda: not dash.has_class("-compact"), "the wide layout")
            assert dash.query_one("#log").region.height > 3

    asyncio.run(scenario())
