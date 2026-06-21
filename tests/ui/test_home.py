# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the `agent6 tui` hub helpers + the ask_user question bridge."""

from __future__ import annotations

import json
from pathlib import Path

from agent6.ui.approval import (
    approvals_dir,
    clear_pending_answers,
    questions_dir,
    read_question_answer,
    write_answer,
    write_question_answer,
    write_tui_pid,
)
from agent6.ui.home import _list_runs, _run_summary


def _write_run(agent6_dir: Path, sub: str, run_id: str, events: list[dict[str, object]]) -> Path:
    rd = agent6_dir / sub / run_id
    rd.mkdir(parents=True)
    (rd / "logs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    return rd


def test_run_summary_reads_mode_task_status(tmp_path: Path) -> None:
    rd = _write_run(
        tmp_path / ".agent6",
        "runs",
        "r1",
        [
            {"type": "run.start", "mode": "run", "user_task": "fix [the] bug"},
            {"type": "tool.call", "name": "read_file"},
            {"type": "run.end", "all_passed": True},
        ],
    )
    s = _run_summary(rd)  # pyright: ignore[reportPrivateUsage]
    assert s["mode"] == "run"
    assert s["task"] == "fix [the] bug"
    assert s["status"] == "ok"


def test_run_summary_running_when_no_end(tmp_path: Path) -> None:
    rd = _write_run(tmp_path / ".agent6", "runs", "r2", [{"type": "run.start", "mode": "plan"}])
    assert _run_summary(rd)["status"] == "running"  # pyright: ignore[reportPrivateUsage]


def test_list_runs_spans_runs_and_asks(tmp_path: Path) -> None:
    a6 = tmp_path / ".agent6"
    _write_run(a6, "runs", "r1", [{"type": "run.start", "mode": "run"}])
    _write_run(a6, "asks", "a1", [{"type": "run.start", "mode": "ask"}])
    names = {p.name for p in _list_runs(a6)}  # pyright: ignore[reportPrivateUsage]
    assert names == {"r1", "a1"}


def test_question_bridge_round_trip(tmp_path: Path) -> None:
    write_tui_pid(tmp_path, 999999999)  # a live-ish pid so read doesn't early-out
    write_question_answer(tmp_path, "q1", "use B")
    assert read_question_answer(tmp_path, "q1", timeout_s=1.0) == "use B"


def test_read_question_answer_returns_none_when_no_tui(tmp_path: Path) -> None:
    # No tui.pid -> tui_is_live False -> immediate None (don't block headless).
    assert read_question_answer(tmp_path, "q1", timeout_s=1.0) is None


def test_read_question_answer_consumes_the_file(tmp_path: Path) -> None:
    # The answer file is unlinked after reading, so a later prompt with the same
    # id (counters reset on resume) can't re-read a stale answer.
    write_tui_pid(tmp_path, 999999999)
    write_question_answer(tmp_path, "q1", "first")
    assert read_question_answer(tmp_path, "q1", timeout_s=1.0) == "first"
    assert not (questions_dir(tmp_path) / "q1.answer").exists()


def test_clear_pending_answers_wipes_stale_state(tmp_path: Path) -> None:
    write_answer(tmp_path, "approval-1", approved=True)
    write_question_answer(tmp_path, "question-1", "stale")
    write_tui_pid(tmp_path, 12345)
    clear_pending_answers(tmp_path)
    assert not (approvals_dir(tmp_path) / "approval-1.answer").exists()
    assert not (questions_dir(tmp_path) / "question-1.answer").exists()
    assert not (tmp_path / "tui.pid").exists()


def test_home_app_lists_runs_and_opens_new_work_modal(tmp_path: Path) -> None:
    import asyncio

    from agent6.ui.home import Agent6HomeApp, _NewWorkModal

    a6 = tmp_path / ".agent6"
    _write_run(a6, "runs", "r1", [{"type": "run.start", "mode": "run", "user_task": "do [x]"}])

    async def scenario() -> None:
        app = Agent6HomeApp(a6, tmp_path)
        async with app.run_test() as pilot:
            from textual.widgets import DataTable

            table = app.query_one("#runs", DataTable)
            assert table.row_count == 1  # the one run is listed
            # 'n' opens the new-work modal; Esc closes it without starting work.
            await pilot.press("n")
            await pilot.pause()
            assert isinstance(app.screen, _NewWorkModal)
            await pilot.press("escape")
            await pilot.pause()
            assert not isinstance(app.screen, _NewWorkModal)

    asyncio.run(scenario())
