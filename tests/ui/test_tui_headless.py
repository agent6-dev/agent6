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

from agent6.ui.tui import Agent6TUI, _ApprovalModal, _SteerModal


def _ev(**fields: Any) -> dict[str, object]:
    return dict(fields)


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
            assert isinstance(app.screen, _ApprovalModal)
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
            assert isinstance(app.screen, _SteerModal)
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

    asyncio.run(scenario())
