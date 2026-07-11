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
import json
from pathlib import Path
from typing import Any

from textual.app import App
from textual.widgets import Button, DataTable, Input, RichLog, Static, TextArea, Tree

from agent6.tui.app import Agent6TUI
from agent6.tui.modals import (
    ApprovalModal,
    QuestionModal,
    SteerModal,
    ToolCallDetailModal,
)
from agent6.viewmodel.state import Question


def _ev(**fields: Any) -> dict[str, object]:
    return dict(fields)


def test_question_modal_digit_in_freetext_is_not_hijacked() -> None:
    """A digit typed into an answer field is plain text: the multi-question modal
    has no digit quick-select. An option button fills its question's field (never
    dismisses); ctrl+s submits the collected answers as a tuple."""
    result: dict[str, tuple[str, ...] | None] = {}

    class _Host(App[None]):
        def on_mount(self) -> None:
            self.push_screen(
                QuestionModal("q1", (Question(question="pick?", options=("alpha", "beta")),)),
                lambda v: result.__setitem__("v", v),
            )

    async def scenario() -> None:
        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = app.screen
            assert isinstance(modal, QuestionModal)
            modal.query_one("#ans-0", Input).focus()
            await pilot.pause()
            await pilot.press("2")  # a digit is just text now, not an option pick
            await pilot.pause()
            assert isinstance(app.screen, QuestionModal)  # still open (no digit-select)
            assert "v" not in result
            assert modal.query_one("#ans-0", Input).value == "2"  # digit typed as text
            # An option button fills that question's field; it does not dismiss.
            modal.query_one("#opt-0-0", Button).press()
            await pilot.pause()
            assert isinstance(app.screen, QuestionModal)  # still open (fill, not submit)
            assert modal.query_one("#ans-0", Input).value == "alpha"  # filled from the option
            await pilot.press("ctrl+s")  # submit collects the answers
            await pilot.pause()
            assert result.get("v") == ("alpha",)  # tuple of answers, aligned to questions

    asyncio.run(scenario())


def test_modal_arrow_keys_move_focus() -> None:
    """Arrow keys move focus in a modal like Tab (the app.focus_next fix). Tested
    on the button-only approval dialog, where no text field consumes the arrows."""

    class _Host(App[None]):
        def on_mount(self) -> None:
            self.push_screen(ApprovalModal("a", "allow?"), lambda _v: None)

    async def scenario() -> None:
        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = app.screen
            assert isinstance(modal, ApprovalModal)
            first = modal.focused
            assert isinstance(first, Button)
            await pilot.press("right")  # arrow moves focus to the other button
            await pilot.pause()
            assert isinstance(modal.focused, Button) and modal.focused is not first
            await pilot.press("left")  # and back
            await pilot.pause()
            assert modal.focused is first

    asyncio.run(scenario())


def test_tools_table_maximizes_to_full_height(tmp_path: Path) -> None:
    """Pressing `f` on the focused tool table fills the screen, not its 20%
    resting height -- regression: the explicit `height: 20%` made the maximized
    view stay short until `#tools.-maximized { height: 1fr; }` was added."""
    (tmp_path / "logs.jsonl").write_text("", encoding="utf-8")

    async def scenario() -> None:
        app = Agent6TUI(tmp_path)
        async with app.run_test(size=(100, 40)) as pilot:
            app._handle_event(_ev(type="tool.call", name="grep", args={"pattern": "x"}))
            app._tick()
            await pilot.pause()
            table = app.query_one("#tools", DataTable)
            table.focus()
            await pilot.pause()
            resting_h = table.size.height
            await pilot.press("f")  # maximize
            await pilot.pause()
            assert app.screen.maximized is table
            assert table.has_class("-maximized")
            maxed_h = table.size.height
            assert maxed_h > resting_h * 2  # fills the screen, not the 20% resting band
            assert maxed_h >= 25  # ~full of a 40-row screen (minus menu + footer)

    asyncio.run(scenario())


def test_plan_tree_maximizes_to_full_width(tmp_path: Path) -> None:
    """Pressing `f` on the focused task-graph pane fills the screen WIDTH, not its
    32% resting width -- the width analogue of the tool-table height bug; the
    explicit `width: 32%` made the maximized view stay a narrow column until
    `#plan.-maximized { width: 1fr; }` was added."""
    (tmp_path / "logs.jsonl").write_text("", encoding="utf-8")

    async def scenario() -> None:
        app = Agent6TUI(tmp_path)
        async with app.run_test(size=(100, 40)) as pilot:
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
            tree = app.query_one("#plan", Tree)
            tree.focus()
            await pilot.pause()
            resting_w = tree.size.width
            await pilot.press("f")  # maximize
            await pilot.pause()
            assert app.screen.maximized is tree
            assert tree.has_class("-maximized")
            maxed_w = tree.size.width
            assert maxed_w > resting_w * 2  # fills the screen, not the 32% resting column
            assert maxed_w >= 90  # ~full of a 100-col screen

    asyncio.run(scenario())


def test_tool_row_enter_opens_detail_with_full_args(tmp_path: Path) -> None:
    """Enter on a tool-calls row opens a read-only detail modal carrying the FULL
    arg value, not the column-truncated preview."""
    (tmp_path / "logs.jsonl").write_text("", encoding="utf-8")
    long_val = "abc/" * 100  # 400 chars, well past the 80-char preview + 90-char column

    async def scenario() -> None:
        app = Agent6TUI(tmp_path)
        async with app.run_test() as pilot:
            app._handle_event(_ev(type="tool.call", name="run_command", args={"cmd": long_val}))
            app._handle_event(
                _ev(type="tool.result", name="run_command", ok=True, summary="lots of output")
            )
            app._tick()
            await pilot.pause()
            app.query_one("#tools", DataTable).focus()
            await pilot.pause()
            await pilot.press("enter")  # select the single row
            await pilot.pause()
            assert isinstance(app.screen, ToolCallDetailModal)
            args_text = app.screen.query_one("#tc-args", TextArea).text
            assert long_val in args_text  # full value present, not the "…" preview
            await pilot.press("escape")  # closes past the focused read-only TextArea
            await pilot.pause()
            assert not isinstance(app.screen, ToolCallDetailModal)

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
            app._tick()  # the coalesced repaint happens in the tick, not per event
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

            # Steer modal: a typed (multi-line) instruction, sent with Ctrl+S.
            app._handle_event(_ev(type="run.steer_requested", source="sigint"))
            app._tick()
            await pilot.pause()
            assert isinstance(app.screen, SteerModal)
            await pilot.press("f", "i", "x")
            await pilot.press("ctrl+s")
            await pilot.pause()
            assert (tmp_path / "steer.answer").read_text(encoding="utf-8") == "fix"

            # Steer modal: Escape == continue (empty answer).
            app._handle_event(_ev(type="run.steer_requested", source="sigint"))
            app._tick()
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            assert (tmp_path / "steer.answer").read_text(encoding="utf-8") == ""

            # Question modal (ask_user): markup-hostile options render; clicking an
            # option fills its answer field, and ctrl+s writes the bridge file (a
            # JSON list of answers aligned to the questions).
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
            assert isinstance(app.screen, QuestionModal)
            app.screen.query_one("#opt-0-1", Button).press()  # fill ans-0 with the 2nd option
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


def test_dashboard_back_vs_quit(tmp_path: Path) -> None:
    """Option 3: q (like Esc) backs out to the hub (exit 0); only Ctrl+Q quits the
    hub (QUIT_HUB_CODE). Standalone, every one of them just closes (0)."""
    from agent6.tui.app import QUIT_HUB_CODE

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


def test_dashboard_pane_maximize_and_restore(tmp_path: Path) -> None:
    """f maximizes the focused pane to full screen; Esc and f both restore it. Esc
    while maximized must minimize (not also back out to the hub), and a non-default
    pane like the diff must be focusable for this to work."""

    async def scenario() -> None:
        (tmp_path / "logs.jsonl").write_text("", encoding="utf-8")
        app = Agent6TUI(tmp_path, from_hub=True)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            diff = app.query_one("#diff")
            diff.focus()
            await pilot.pause()
            await pilot.press("f")  # maximize the focused pane
            await pilot.pause()
            assert app.screen.maximized is diff
            await pilot.press("escape")  # Esc restores, does NOT exit to the hub
            await pilot.pause()
            assert app.screen.maximized is None
            assert app.return_value is None  # still running; Esc was consumed by minimize
            diff.focus()
            await pilot.press("f")
            await pilot.pause()
            assert app.screen.maximized is diff
            await pilot.press("f")  # f toggles back too
            await pilot.pause()
            assert app.screen.maximized is None

    asyncio.run(scenario())


def test_dashboard_diff_pane_scrolls(tmp_path: Path) -> None:
    """A long diff overflows the diff pane, which is a scroll container, so it can be
    scrolled -- inline and while maximized (regression: it used to be a plain Static
    that just clipped)."""

    async def scenario() -> None:
        (tmp_path / "logs.jsonl").write_text("", encoding="utf-8")
        app = Agent6TUI(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            long_patch = "--- a\n+++ b\n" + "".join(f"+added line {i}\n" for i in range(200))
            app._handle_event(_ev(type="diff.updated", index=1, patch=long_patch))
            app._tick()
            await pilot.pause()
            diff = app.query_one("#diff")
            assert diff.max_scroll_y > 0  # content overflows the pane -> scrollable
            diff.focus()
            await pilot.press("f")  # maximize, then it must still scroll
            await pilot.pause()
            assert app.screen.maximized is diff
            assert diff.max_scroll_y > 0

    asyncio.run(scenario())


def test_dashboard_inline_log_is_a_bounded_gapless_window(tmp_path: Path) -> None:
    """Coalescing folds many events between paints. The inline log must stay a bounded
    window: feed a pre-burst, then a burst larger than the window in one tick, and the
    RichLog caps at MAX_LOG_TAIL -- the gap-causing pre-burst lines are evicted, so it
    is the gapless recent window, not pre-burst lines + a hole + the tail."""
    from agent6.viewmodel.state import MAX_LOG_TAIL

    async def scenario() -> None:
        (tmp_path / "logs.jsonl").write_text("", encoding="utf-8")
        app = Agent6TUI(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            for i in range(100):
                app._handle_event(_ev(type="tool.call", name=f"pre{i}"))
            app._tick()
            await pilot.pause()
            for i in range(MAX_LOG_TAIL + 100):
                app._handle_event(_ev(type="tool.call", name=f"burst{i}"))
            app._tick()
            await pilot.pause()
            log = app.query_one("#log", RichLog)
            assert len(log.lines) == MAX_LOG_TAIL  # bounded; pre-burst lines evicted

    asyncio.run(scenario())


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


def test_dashboard_does_not_clobber_live_frontend_pid(tmp_path: Path) -> None:
    """A live peer front-end (a web viewer) already owns frontend.pid: the
    dashboard must not overwrite it on mount or clear it on exit (the web side
    ref-counts; clobbering strands its viewers)."""
    import subprocess
    import sys

    (tmp_path / "logs.jsonl").write_text("", encoding="utf-8")
    # A real same-user process stands in for the live peer front-end.
    peer = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        (tmp_path / "frontend.pid").write_text(str(peer.pid), encoding="utf-8")

        async def scenario() -> None:
            app = Agent6TUI(tmp_path)
            async with app.run_test() as pilot:
                await pilot.pause()
                app._tick()
                await pilot.pause()
                assert (tmp_path / "frontend.pid").read_text(encoding="utf-8") == str(peer.pid)
            # On unmount the pid is not ours, so it must survive.
            assert (tmp_path / "frontend.pid").read_text(encoding="utf-8") == str(peer.pid)

        asyncio.run(scenario())
    finally:
        peer.kill()
        peer.wait()


def test_dashboard_claims_stale_pid_and_self_heals(tmp_path: Path) -> None:
    """A stale frontend.pid (dead process) is claimed on mount; if a peer owner
    goes away mid-session the next tick re-claims; unmount clears only our own."""
    import os

    (tmp_path / "logs.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "frontend.pid").write_text("999999999", encoding="utf-8")  # dead

    async def scenario() -> None:
        app = Agent6TUI(tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert (tmp_path / "frontend.pid").read_text(encoding="utf-8") == str(os.getpid())
            # A peer owner appears then dies: the tick re-claims the bridge.
            (tmp_path / "frontend.pid").write_text("999999999", encoding="utf-8")
            app._tick()
            await pilot.pause()
            assert (tmp_path / "frontend.pid").read_text(encoding="utf-8") == str(os.getpid())
        assert not (tmp_path / "frontend.pid").exists()  # ours, cleared on unmount

    asyncio.run(scenario())


def test_resume_reopens_modal_for_reused_prompt_id(tmp_path: Path) -> None:
    """`agent6 resume` appends a new session whose prompt ids restart at
    approval-1; a dashboard held across the resume must pop the new session's
    modal, not swallow it as already seen."""
    (tmp_path / "logs.jsonl").write_text("", encoding="utf-8")

    async def scenario() -> None:
        app = Agent6TUI(tmp_path)
        async with app.run_test() as pilot:
            app._handle_event(_ev(type="run.start", user_task="session one", mode="run"))
            app._handle_event(_ev(type="approval.prompt", id="approval-1", prompt="first?"))
            app._tick()
            await pilot.pause()
            assert isinstance(app.screen, ApprovalModal)
            await pilot.press("y")
            await pilot.pause()
            app._handle_event(_ev(type="approval.answer", id="approval-1", approved=True))
            # The resume: a fresh run.start, then the new session's approval-1.
            app._handle_event(_ev(type="run.start", user_task="session one", mode="run"))
            app._handle_event(_ev(type="approval.prompt", id="approval-1", prompt="again?"))
            app._tick()
            await pilot.pause()
            assert isinstance(app.screen, ApprovalModal)  # re-popped, not swallowed
            await pilot.press("n")
            await pilot.pause()
            answer = (tmp_path / "approvals" / "approval-1.answer").read_text(encoding="utf-8")
            assert answer == "no"

    asyncio.run(scenario())


def test_steer_request_marker_round_trip(tmp_path: Path) -> None:
    from agent6.frontend.approval import clear_steer_request, request_steer, steer_request_pending

    assert not steer_request_pending(tmp_path)
    request_steer(tmp_path)
    assert steer_request_pending(tmp_path)  # the run's requested() sees this
    clear_steer_request(tmp_path)
    assert not steer_request_pending(tmp_path)


def test_dashboard_s_key_steers_without_ctrl_c(tmp_path: Path) -> None:
    """The dashboard 's' action drops a steer.request marker (the run picks it up
    at its next boundary) and opens the steer box -- no Ctrl-C needed -- then the
    typed instruction lands in steer.answer for the run to inject."""
    from agent6.frontend.approval import steer_request_pending
    from agent6.tui.modals import SteerModal

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
            await pilot.press("ctrl+s")
            await pilot.pause()
            assert (tmp_path / "steer.answer").read_text(encoding="utf-8") == "go"

    asyncio.run(scenario())


def test_dashboard_stop_action_aborts_via_bridge(tmp_path: Path) -> None:
    """The dedicated Stop action (x) confirms, then writes an abort over the file
    bridge -- separate from steering, which never stops the run."""
    from agent6.frontend.approval import steer_request_pending
    from agent6.tui.modals import ConfirmModal

    async def scenario() -> None:
        (tmp_path / "logs.jsonl").write_text("", encoding="utf-8")
        app = Agent6TUI(tmp_path)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app.action_stop()
            await pilot.pause()
            assert isinstance(app.screen, ConfirmModal)  # confirms before stopping
            await pilot.press("y")  # confirm
            await pilot.pause()
            assert (tmp_path / "steer.answer").read_text(encoding="utf-8") == "abort"
            assert steer_request_pending(tmp_path)

    asyncio.run(scenario())


def test_historical_steer_request_does_not_pop_a_modal_on_open(tmp_path: Path) -> None:
    # A CLI Ctrl-C that DETACHED leaves run.steer_requested in the log. Opening the
    # TUI must not pop a stale (already-handled) steer modal for it -- only a request
    # that arrives AFTER the TUI is watching should prompt.
    (tmp_path / "logs.jsonl").write_text(
        "".join(
            json.dumps(e) + "\n"
            for e in (
                _ev(type="run.start", user_task="fix it"),
                _ev(type="run.steer_requested", source="sigint"),
                _ev(type="role.call", role="worker", model="kimi"),
            )
        ),
        encoding="utf-8",
    )

    async def scenario() -> None:
        app = Agent6TUI(tmp_path)
        async with app.run_test() as pilot:
            for _ in range(50):  # let the reader thread replay the existing log
                await pilot.pause()
                if app.state.steer_requests >= 1:
                    break
            app._tick()
            await pilot.pause()
            assert app.state.steer_requests == 1  # the historical event WAS folded
            assert not isinstance(app.screen, SteerModal)  # but it did NOT pop a modal
            # a NEW steer request (a live Ctrl-C while watching) still prompts
            app._handle_event(_ev(type="run.steer_requested", source="sigint"))
            app._tick()
            await pilot.pause()
            assert isinstance(app.screen, SteerModal)

    asyncio.run(scenario())


def test_l_and_t_toggle_detail_views_without_stacking(tmp_path: Path) -> None:
    # l/t are toggles: pressing l opens the log, l again closes it (not a second
    # stacked copy needing two escapes), and t switches to the conversation.
    from agent6.tui.conversation import ConversationScreen
    from agent6.tui.logview import LogScreen

    (tmp_path / "logs.jsonl").write_text(
        json.dumps({"type": "run.start", "user_task": "x"}) + "\n", encoding="utf-8"
    )

    async def scenario() -> None:
        app = Agent6TUI(tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("l")  # open the log
            await pilot.pause()
            assert isinstance(app.screen, LogScreen)
            await pilot.press("l")  # toggle it off -> back to the dashboard (no stack)
            await pilot.pause()
            assert not isinstance(app.screen, LogScreen)
            assert len(app.screen_stack) == 1
            await pilot.press("l")  # open the log again
            await pilot.pause()
            assert isinstance(app.screen, LogScreen)
            await pilot.press("t")  # switch to conversation, not stack on top of the log
            await pilot.pause()
            assert isinstance(app.screen, ConversationScreen)
            assert len(app.screen_stack) == 2  # dashboard + conversation, not + log too

    asyncio.run(scenario())


def test_task_filter_scopes_tools_log_and_diff(tmp_path: Path) -> None:
    # Two tasks, each with a tool call + a commit. Selecting a task filters the
    # tools table / log / diff to just that task's activity; the fold stamps each
    # event with the cursor task in focus when it landed.
    def _nodes(cur_status: dict[str, str]) -> dict[str, object]:
        return {
            tid: {"title": t, "status": cur_status[tid], "parent_id": None, "children": []}
            for tid, t in (("t1", "Task one"), ("t2", "Task two"))
        }

    events = [
        {"type": "run.start", "user_task": "x"},
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
        app = Agent6TUI(tmp_path)
        async with app.run_test(size=(150, 42)) as pilot:
            for _ in range(60):
                await pilot.pause()
                if len(app.state.recent_diffs) >= 2:  # the last events emitted
                    break
            # Fold stamped each tool call + diff with the task in focus at the time.
            assert [tc.task_id for tc in app.state.tool_calls] == ["t1", "t2"]
            assert [d.task_id for d in app.state.recent_diffs] == ["t1", "t2"]

            app._tick()
            await pilot.pause()
            assert len(app._visible_tools) == 2  # unfiltered: both

            app._selected_task_id = "t1"  # filter to task one (handler also sets _dirty)
            app._dirty = True
            app._tick()
            await pilot.pause()
            assert [tc.name for tc in app._visible_tools] == ["read_file"]

            app._selected_task_id = "t2"
            app._dirty = True
            app._tick()
            await pilot.pause()
            assert [tc.name for tc in app._visible_tools] == ["apply_edit"]

    asyncio.run(scenario())


def test_question_modal_multi_collects_all_answers() -> None:
    """A prompt with several questions: each has its own answer field, option
    buttons fill their own field, and Submit (ctrl+s) returns every answer as a
    tuple aligned to the questions."""
    result: dict[str, tuple[str, ...] | None] = {}
    qs = (
        Question(question="Framework?", options=("React", "Vue")),
        Question(question="Component name?", options=()),
    )

    class _Host(App[None]):
        def on_mount(self) -> None:
            self.push_screen(QuestionModal("q1", qs), lambda v: result.__setitem__("v", v))

    async def scenario() -> None:
        app = _Host()
        async with app.run_test() as pilot:
            await pilot.pause()
            modal = app.screen
            assert isinstance(modal, QuestionModal)
            modal.query_one("#opt-0-1", Button).press()  # pick "Vue" for the first question
            await pilot.pause()
            assert modal.query_one("#ans-0", Input).value == "Vue"
            modal.query_one("#ans-1", Input).value = "widget"  # type the second answer
            await pilot.press("ctrl+s")
            await pilot.pause()
            assert result.get("v") == ("Vue", "widget")  # both, aligned to the questions

    asyncio.run(scenario())


def test_dashboard_heartbeat_ticks_while_active(tmp_path: Path) -> None:
    """An attached dashboard on a live-but-silent run shows a ticking "working…
    Ns" heartbeat, so a thinking / resuming run reads as alive, not hung. The
    elapsed count advances across ticks even with no new events."""
    import json

    events = [
        {"type": "run.start", "run_id": "live-01", "mode": "run", "user_task": "t"},
        {"type": "role.call", "role": "worker", "model": "m", "provider": "p"},
    ]
    (tmp_path / "logs.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in events), encoding="utf-8"
    )

    async def scenario() -> str:
        app = Agent6TUI(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            for _ in range(15):  # ~3s: let the reader fold + the heartbeat tick
                await pilot.pause(0.2)
            app._tick()  # pyright: ignore[reportPrivateUsage]
            await pilot.pause()
            return str(app.query_one("#stream-body", Static).render())

    text = asyncio.run(scenario())
    assert "working…" in text
    # A number of seconds is shown and it is > 0 (the heartbeat advanced).
    import re

    m = re.search(r"working… (\d+)s", text)
    assert m is not None and int(m.group(1)) >= 1


def test_dashboard_follows_live_appends_after_attach(tmp_path: Path) -> None:
    """The detach->attach symptom the user hit: after opening on a live run, NEW
    events appended by the background process must appear (not a frozen snapshot).
    Attach on a partial log, append more, and assert the new tool row shows."""
    import json

    logs = tmp_path / "logs.jsonl"

    def append(events: list[dict[str, object]]) -> None:
        with logs.open("a", encoding="utf-8") as fh:
            for e in events:
                fh.write(json.dumps(e) + "\n")

    logs.write_text("", encoding="utf-8")
    append(
        [
            {"type": "run.start", "run_id": "live-02", "mode": "run", "user_task": "t"},
            {"type": "tool.call", "name": "read_file", "args": {"path": "a.py"}},
            {"type": "tool.result", "name": "read_file", "ok": True, "summary": "1 byte"},
        ]
    )

    async def scenario() -> int:
        app = Agent6TUI(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            for _ in range(8):
                await pilot.pause(0.2)
            before = app.query_one("#tools", DataTable).row_count
            # The background process appends a new turn AFTER we attached.
            append(
                [
                    {"type": "tool.call", "name": "apply_edit", "args": {"path": "a.py"}},
                    {"type": "tool.result", "name": "apply_edit", "ok": True, "summary": "applied"},
                ]
            )
            for _ in range(10):  # > tail poll (0.25s) + tick (0.2s)
                await pilot.pause(0.2)
            app._tick()  # pyright: ignore[reportPrivateUsage]
            await pilot.pause()
            after = app.query_one("#tools", DataTable).row_count
            assert after > before, f"live append not followed: {before} -> {after}"
            return after

    assert asyncio.run(scenario()) == 2
