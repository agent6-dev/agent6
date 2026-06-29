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

            from agent6.ui.home import HomeScreen

            await pilot.pause()  # let on_mount push the HomeScreen
            assert isinstance(app.screen, HomeScreen)  # hub lives on its own screen
            table = app.screen.query_one("#runs", DataTable)
            assert table.row_count == 1  # the one run is listed
            # 'n' opens the new-work modal; Esc closes it without starting work.
            await pilot.press("n")
            await pilot.pause()
            assert isinstance(app.screen, _NewWorkModal)
            await pilot.press("escape")
            await pilot.pause()
            assert not isinstance(app.screen, _NewWorkModal)

    asyncio.run(scenario())


def test_new_work_modal_is_multiline_and_starts_chosen_mode(tmp_path: Path) -> None:
    import asyncio

    from textual.widgets import TextArea

    from agent6.ui.home import Agent6HomeApp, _NewWorkModal

    a6 = tmp_path / ".agent6"
    _write_run(a6, "runs", "r1", [{"type": "run.start", "mode": "run", "user_task": "x"}])

    async def scenario() -> None:
        app = Agent6HomeApp(a6, tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause()
            result: list[object] = []
            app.push_screen(_NewWorkModal(), result.append)
            await pilot.pause()
            ta = app.screen.query_one(TextArea)
            # Enter is a NEWLINE here (multiline task), not a submit.
            await pilot.press("a")
            await pilot.press("enter")
            await pilot.press("b")
            await pilot.pause()
            assert ta.text == "a\nb"
            assert isinstance(app.screen, _NewWorkModal)  # Enter did not dismiss
            # ↓ past the last line moves to the buttons; →,Enter starts 'plan'.
            await pilot.press("down")
            await pilot.press("right")
            await pilot.press("enter")
            await pilot.pause()
            # mode + multiline task + profile ("" = config default, no --profile).
            assert result == [("plan", "a\nb", "")]

    asyncio.run(scenario())


def test_new_work_modal_yields_chosen_profile(tmp_path: Path) -> None:
    """The new-work modal carries the picked config profile in its result tuple:
    (mode, task, profile). Selecting a built-in (e.g. 'ultra') yields that name;
    the default '(config default)' would yield '' (covered above)."""
    import asyncio

    from textual.widgets import Select, TextArea

    from agent6.ui.home import Agent6HomeApp, _NewWorkModal

    a6 = tmp_path / ".agent6"
    _write_run(a6, "runs", "r1", [{"type": "run.start", "mode": "run", "user_task": "x"}])

    async def scenario() -> None:
        app = Agent6HomeApp(a6, tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause()
            result: list[object] = []
            # An explicit profile list so the dropdown offers a known value.
            app.push_screen(_NewWorkModal(["ultra"]), result.append)
            await pilot.pause()
            app.screen.query_one(TextArea).insert("do it")
            # Pick 'ultra' directly on the Select (no overlay navigation needed).
            app.screen.query_one("#new-profile", Select).value = "ultra"
            await pilot.pause()
            # Run via a button activation (Esc-free path), like the user clicking.
            from agent6.ui.widgets import ActionItem

            next(iter(app.screen.query(ActionItem))).post_message(ActionItem.Activated("run"))
            await pilot.pause()
            assert result == [("run", "do it", "ultra")]  # profile threaded through

    asyncio.run(scenario())


def test_spawn_argv_includes_profile_flag_only_when_chosen(
    tmp_path: Path, monkeypatch: object
) -> None:
    """The launch helper builds `agent6 <mode> --profile <name> <task>` when a
    profile is picked, and `agent6 <mode> <task>` (no --profile) for the
    "(config default)" choice (profile=""). Captures argv by stubbing Popen, so
    no real agent6 is spawned; the helper times out fast on the stubbed proc."""
    import subprocess

    from agent6.ui import home

    captured: list[list[str]] = []

    class _FakeProc:
        returncode = 0

        def poll(self) -> int:
            return 0  # "already exited" -> helper bails out immediately

    def _fake_popen(argv: list[str], **_kw: object) -> _FakeProc:
        captured.append(list(argv))
        return _FakeProc()

    monkeypatch.setattr(subprocess, "Popen", _fake_popen)  # type: ignore[attr-defined]
    # A stable executable name so the argv assertion isn't path-dependent.
    monkeypatch.setattr(home, "_agent6_exe", lambda: "agent6")  # type: ignore[attr-defined]

    a6 = tmp_path / ".agent6"
    a6.mkdir()

    # Chosen profile -> --profile is present, after <mode>, before <task>.
    home._spawn_and_locate(a6, tmp_path, "plan", "do it", profile="ultra")
    assert captured[-1] == ["agent6", "plan", "--profile", "ultra", "do it"]

    # "(config default)" (profile="") -> NO --profile flag at all.
    home._spawn_and_locate(a6, tmp_path, "run", "do it", profile="")
    assert captured[-1] == ["agent6", "run", "do it"]
    assert "--profile" not in captured[-1]


def test_run_merge_cli_builds_argv_and_parses_result(tmp_path: Path, monkeypatch: object) -> None:
    """The hub's merge helper shells out to `agent6 runs merge <id>` and reports the
    captured output as (ok, message) -- it never touches git_ops itself."""
    import subprocess

    from agent6.ui import home

    captured: list[list[str]] = []

    class _Proc:
        returncode = 0
        stdout = "[agent6] merged agent6/r1 into main (squash) -> abcdef123456\n"
        stderr = ""

    def _fake_run(argv: list[str], **_kw: object) -> _Proc:
        captured.append(list(argv))
        return _Proc()

    monkeypatch.setattr(subprocess, "run", _fake_run)  # type: ignore[attr-defined]
    monkeypatch.setattr(home, "_agent6_exe", lambda: "agent6")  # type: ignore[attr-defined]

    ok, msg = home._run_merge_cli(tmp_path, "r1")  # pyright: ignore[reportPrivateUsage]
    assert captured[-1] == ["agent6", "runs", "merge", "r1"]
    assert ok is True
    assert "merged agent6/r1" in msg


def test_merge_action_confirms_then_shells_out(tmp_path: Path, monkeypatch: object) -> None:
    """Pressing `m` opens a confirm modal; confirming runs `agent6 runs merge` for the
    selected run (stubbed here so no real CLI is spawned)."""
    import asyncio

    from textual.widgets import DataTable

    from agent6.ui import home
    from agent6.ui.home import Agent6HomeApp
    from agent6.ui.modals import ConfirmModal

    a6 = tmp_path / ".agent6"
    _write_run(a6, "runs", "r1", [{"type": "run.start", "mode": "run", "user_task": "x"}])

    calls: list[str] = []

    def _fake_merge(cwd: Path, run_id: str) -> tuple[bool, str]:
        calls.append(run_id)
        return True, "merged"

    monkeypatch.setattr(home, "_run_merge_cli", _fake_merge)  # type: ignore[attr-defined]

    async def scenario() -> None:
        app = Agent6HomeApp(a6, tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause()
            tbl = app.screen.query_one("#runs", DataTable)
            tbl.focus()
            tbl.move_cursor(row=0)
            await pilot.press("m")
            await pilot.pause()
            assert isinstance(app.screen, ConfirmModal)
            await pilot.press("y")  # confirm
            await pilot.pause()
            assert calls == ["r1"]  # merged the selected run

    asyncio.run(scenario())


def test_home_open_run_returns_its_dir(tmp_path: Path) -> None:
    """Selecting a run on the hub (Enter on the row) opens it: the app exits
    returning that run directory for the dashboard to watch."""
    import asyncio

    from textual.widgets import DataTable

    from agent6.ui.home import Agent6HomeApp

    a6 = tmp_path / ".agent6"
    rd = _write_run(a6, "runs", "r1", [{"type": "run.start", "mode": "run", "user_task": "x"}])

    async def scenario() -> None:
        app = Agent6HomeApp(a6, tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause()
            tbl = app.screen.query_one("#runs", DataTable)
            tbl.focus()
            tbl.move_cursor(row=0)
            await pilot.press("enter")
            await pilot.pause()
        assert app.return_value == rd  # opened the selected run

    asyncio.run(scenario())
