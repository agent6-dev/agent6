# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Headless drive of the textual dashboard via textual's run_test Pilot.

textual ships in the base install, so these run in CI. They cover the bits
that previously could only be checked by a human: that streamed reasoning +
markup-hostile model output render without crashing, that the approval modal
is keyboard-answerable (the y/n routing bug), and that the new Ctrl-C steer
modal writes the right bridge file.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from textual.app import App
from textual.widgets import Button, Input

from agent6.ui.app import Agent6TUI
from agent6.ui.modals import ApprovalModal, QuestionModal, SteerModal


def _ev(**fields: Any) -> dict[str, object]:
    return dict(fields)


def test_question_modal_digit_in_freetext_is_not_hijacked() -> None:
    """A digit typed into the free-text answer field must land in the Input, not
    be hijacked as a numbered option pick -- while digit quick-select still works
    when an option Button is focused (regression: on_key fired over the Input)."""
    result: dict[str, str | None] = {}

    class _Host(App[None]):
        def on_mount(self) -> None:
            self.push_screen(
                QuestionModal("q1", "pick?", ("alpha", "beta")),
                lambda v: result.__setitem__("v", v),
            )

    async def scenario() -> None:
        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = app.screen
            assert isinstance(modal, QuestionModal)
            modal.query_one("#question-input", Input).focus()
            await pilot.pause()
            await pilot.press("2")  # pre-fix this dismissed the modal as option 2
            await pilot.pause()
            assert isinstance(app.screen, QuestionModal)  # still open
            assert "v" not in result
            assert modal.query_one("#question-input", Input).value == "2"  # digit typed
            modal.query_one("#opt-1", Button).focus()  # back on an option button
            await pilot.press("1")
            await pilot.pause()
            assert result.get("v") == "alpha"  # quick-select still works

    asyncio.run(scenario())


def test_render_and_modals(tmp_path: Path) -> None:
    (tmp_path / "logs.jsonl").write_text("", encoding="utf-8")

    async def scenario() -> None:
        app = Agent6TUI(tmp_path)
        async with app.run_test() as pilot:
            # Render with bracket-laden (markup-hostile) content must not crash —
            # exercises the header, the plan TREE (step titles), the tool TABLE
            # (names/args), the stream pane and the diff pane, all of which carry
            # model output that would otherwise be parsed as Rich markup.
            for ev in (
                _ev(type="run.start", user_task="do [a] thing", mode="run"),
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
            await pilot.pause()

            # Approval modal: keyboard 'y' must reach the modal (routing bug).
            app._handle_event(_ev(type="approval.prompt", id="ap1", prompt="run_command(['ls'])"))
            app._tick()
            await pilot.pause()
            assert isinstance(app.screen, ApprovalModal)
            await pilot.press("y")
            await pilot.pause()
            assert (tmp_path / "approvals" / "ap1.answer").read_text(encoding="utf-8") == "yes"

            # Approval modal: keyboard 'n'.
            app._handle_event(_ev(type="approval.prompt", id="ap2", prompt="rm -rf"))
            app._tick()
            await pilot.pause()
            await pilot.press("n")
            await pilot.pause()
            assert (tmp_path / "approvals" / "ap2.answer").read_text(encoding="utf-8") == "no"

            # Steer modal: typed instruction + Enter is sent verbatim.
            app._handle_event(_ev(type="run.steer_requested", source="sigint"))
            app._tick()
            await pilot.pause()
            assert isinstance(app.screen, SteerModal)
            await pilot.press("f", "i", "x")
            await pilot.press("enter")
            await pilot.pause()
            assert (tmp_path / "steer.answer").read_text(encoding="utf-8") == "fix"

            # Steer modal: Escape == continue (empty answer).
            app._handle_event(_ev(type="run.steer_requested", source="sigint"))
            app._tick()
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            assert (tmp_path / "steer.answer").read_text(encoding="utf-8") == ""

            # Question modal (ask_user): markup-hostile options render, and a
            # number key selects the matching option -> bridge file written.
            app._handle_event(
                _ev(
                    type="question.prompt",
                    id="q1",
                    question="which [approach]?",
                    options=["use [A]", "use [B]"],
                )
            )
            app._tick()
            await pilot.pause()
            assert isinstance(app.screen, QuestionModal)
            await pilot.press("2")
            await pilot.pause()
            assert (tmp_path / "questions" / "q1.answer").read_text(encoding="utf-8") == "use [B]"

            # Question modal: a typed free-text answer is sent verbatim.
            app._handle_event(_ev(type="question.prompt", id="q2", question="name?", options=[]))
            app._tick()
            await pilot.pause()
            await pilot.press("z", "z")
            await pilot.press("enter")
            await pilot.pause()
            assert (tmp_path / "questions" / "q2.answer").read_text(encoding="utf-8") == "zz"

    asyncio.run(scenario())


def test_dashboard_back_vs_quit(tmp_path: Path) -> None:
    """Option 3: q (like Esc) backs out to the hub (exit 0); only Ctrl+Q quits the
    hub (QUIT_HUB_CODE). Standalone, every one of them just closes (0)."""
    from agent6.ui.app import QUIT_HUB_CODE

    async def press(from_hub: bool, key: str) -> int | None:
        (tmp_path / "logs.jsonl").write_text("", encoding="utf-8")
        app = Agent6TUI(tmp_path, from_hub=from_hub)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            await pilot.press(key)
            await pilot.pause()
        return app.return_value

    assert asyncio.run(press(True, "escape")) == 0  # back to the hub
    assert asyncio.run(press(True, "q")) == 0  # q backs out to the hub too now
    assert asyncio.run(press(True, "ctrl+q")) == QUIT_HUB_CODE  # only Ctrl+Q quits the hub
    assert asyncio.run(press(False, "q")) == 0  # standalone: just close
    assert asyncio.run(press(False, "ctrl+q")) == 0  # standalone: just close


def test_dashboard_footer_shows_one_dual_back_key(tmp_path: Path) -> None:
    """Back is a single 'Esc/q' footer entry (q and Esc both back out), not two
    separate Esc/q entries -- via key_display on the shown binding + a hidden q."""
    from textual.widgets._footer import FooterKey

    async def scenario() -> None:
        (tmp_path / "logs.jsonl").write_text("", encoding="utf-8")
        app = Agent6TUI(tmp_path, from_hub=True)
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            displays = [fk.key_display for fk in app.screen.query(FooterKey)]
            assert displays.count("Esc/q") == 1  # exactly one combined Back entry
            assert "q" not in displays  # the q alias is hidden, not a 2nd entry

    asyncio.run(scenario())


def test_steer_request_marker_round_trip(tmp_path: Path) -> None:
    from agent6.ui.approval import clear_steer_request, request_steer, steer_request_pending

    assert not steer_request_pending(tmp_path)
    request_steer(tmp_path)
    assert steer_request_pending(tmp_path)  # the run's requested() sees this
    clear_steer_request(tmp_path)
    assert not steer_request_pending(tmp_path)


def test_dashboard_s_key_steers_without_ctrl_c(tmp_path: Path) -> None:
    """The dashboard 's' action drops a steer.request marker (the run picks it up
    at its next boundary) and opens the steer box -- no Ctrl-C needed -- then the
    typed instruction lands in steer.answer for the run to inject."""
    from agent6.ui.approval import steer_request_pending
    from agent6.ui.modals import SteerModal

    async def scenario() -> None:
        (tmp_path / "logs.jsonl").write_text("", encoding="utf-8")
        app = Agent6TUI(tmp_path)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            assert not steer_request_pending(tmp_path)
            app.action_steer()
            await pilot.pause()
            assert steer_request_pending(tmp_path)  # marker dropped for the run
            assert isinstance(app.screen, SteerModal)  # steer box opened
            await pilot.press("g", "o")
            await pilot.press("enter")
            await pilot.pause()
            assert (tmp_path / "steer.answer").read_text(encoding="utf-8") == "go"

    asyncio.run(scenario())
