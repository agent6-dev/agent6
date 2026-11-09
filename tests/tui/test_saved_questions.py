from __future__ import annotations

import asyncio
import json
import os
import pathlib
from typing import Any

import pytest
from textual import widgets

from agent6 import paths
from agent6.sessions import ipc, process
from agent6.ui.tui import app as tui_app
from agent6.ui.tui import modals
from tests.tui import waits


class _App(tui_app.Agent6TUI):
    def __init__(self, session_dir: pathlib.Path) -> None:
        super().__init__(session_dir)
        self.notices: list[str] = []

    def notify(self, message: Any, **kwargs: Any) -> None:  # type: ignore[override]
        self.notices.append(str(message))
        super().notify(message, **kwargs)


def _question(repo: pathlib.Path, *, alive: bool) -> pathlib.Path:
    saved = paths.state_dir(repo) / "sessions" / "runs" / "question-run-AAAAAA"
    saved.mkdir(parents=True)
    rows = [
        {"type": "session.start", "session_id": saved.name, "mode": "run", "user_task": "t"},
        {
            "type": "question.prompt",
            "id": "question-1",
            "questions": [{"question": "Which DB?", "options": ["pg", "sqlite"]}],
            "wait": not alive,
        },
    ]
    (saved / "logs.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    process.write_worker_pid(saved, os.getpid() if alive else 999999999)
    return saved


def test_run_view_announces_a_dead_workers_question(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The actual run view dispatches a dead worker's pending question once without a modal."""
    monkeypatch.chdir(tmp_path)
    saved = _question(tmp_path, alive=False)

    async def scenario() -> None:
        app = _App(saved)
        async with app.run_test(size=(140, 40)) as pilot:
            await waits.wait_for(pilot, lambda: app.worker_lost, "dead worker")
            await waits.wait_for(pilot, lambda: bool(app.state.questions), "question fold")
            for _ in range(3):
                app._tick()  # pyright: ignore[reportPrivateUsage]
                await pilot.pause()
            assert not isinstance(app.screen, modals.QuestionModal)
            assert [n for n in app.notices if "Run menu: Answer a question" in n] == [
                "question-1 waiting: Which DB?\nRun menu: Answer a question"
            ]

    asyncio.run(scenario())


@pytest.mark.parametrize("alive", [False, True])
def test_question_header_and_saved_notice_name_when_the_answer_is_read(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, alive: bool
) -> None:
    """The run view promises a next turn only while a worker can read the saved answer."""
    monkeypatch.chdir(tmp_path)
    saved = _question(tmp_path, alive=alive)

    async def scenario() -> None:
        app = _App(saved)
        async with app.run_test(size=(140, 40)) as pilot:
            await waits.wait_for(pilot, lambda: bool(app.state.questions), "question fold")
            if not alive:
                await waits.wait_for(pilot, lambda: app.worker_lost, "dead worker")
            app.action_answer_question()
            await pilot.pause()
            modal = app.screen
            assert isinstance(modal, modals.QuestionModal)
            header = str(modal.query(widgets.Static).first().render())
            when = "its next turn" if alive else "the resume delivers it"
            assert when in header
            if not alive:
                assert "the run went on" not in header
            modal.query_one("#opt-0-0", widgets.Button).press()
            await pilot.pause()
            await pilot.press("ctrl+s")
            await pilot.pause()
            expected = (
                "question-1 answered; the run reads it at its next turn"
                if alive
                else "question-1 answer saved; the resume delivers it"
            )
            assert expected in app.notices
            assert ipc.read_question_answers(saved, "question-1", timeout_s=1.0) == ("pg",)

    asyncio.run(scenario())


def test_stop_modal_names_fanout_wait_and_conditional_resume(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stop does not promise every session a five-second wait or a resumable checkpoint."""
    monkeypatch.chdir(tmp_path)
    saved = _question(tmp_path, alive=True)

    async def scenario() -> None:
        app = _App(saved)
        async with app.run_test(size=(140, 40)) as pilot:
            await waits.wait_for(pilot, app.session_controllable, "live worker")
            app.action_stop_now()
            await pilot.pause()
            modal = app.screen
            assert isinstance(modal, modals.ConfirmModal)
            body = str(modal.query(widgets.Static).first().render())
            assert "5 s (30 s for a fan-out)" in body
            assert "A run `agent6 resume` accepts stays resumable." in body

    asyncio.run(scenario())
