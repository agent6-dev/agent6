# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Tests for the `agent6 tui` hub helpers + the ask_user question bridge."""

from __future__ import annotations

import json
from pathlib import Path

from agent6.runs.ipc import (
    approvals_dir,
    clear_pending_answers,
    questions_dir,
    read_question_answers,
    register_frontend,
    write_answer,
    write_question_answers,
)
from agent6.ui.tui.home import _list_runs, run_mtime


def _write_run(agent6_dir: Path, sub: str, run_id: str, events: list[dict[str, object]]) -> Path:
    rd = agent6_dir / sub / run_id
    rd.mkdir(parents=True)
    (rd / "logs.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    return rd


def test_list_runs_spans_runs_and_asks(tmp_path: Path) -> None:
    a6 = tmp_path / ".agent6"
    _write_run(a6, "runs", "r1", [{"type": "run.start", "mode": "run"}])
    _write_run(a6, "asks", "a1", [{"type": "run.start", "mode": "ask"}])
    names = {p.name for p in _list_runs(a6)}  # pyright: ignore[reportPrivateUsage]
    assert names == {"r1", "a1"}


def test_run_mtime_is_log_activity_not_dir_mtime(tmp_path: Path) -> None:
    """A run's listed/sorted time is its logs.jsonl mtime (last run activity), not
    the run-dir mtime. Opening a run writes a front-end claim into the dir, bumping the dir
    mtime; that must NOT move the run's 'when' or its sort position."""
    import os

    a6 = tmp_path / ".agent6"
    rd = _write_run(a6, "runs", "r1", [{"type": "run.start", "mode": "run"}])
    os.utime(rd / "logs.jsonl", (1000, 1000))  # last real activity
    # Simulate opening the dashboard: it writes a front-end claim, bumping the dir
    # mtime well past the log's. Pre-fix this became the displayed/sort time.
    register_frontend(rd, 123)
    os.utime(rd, (5000, 5000))
    assert run_mtime(rd) == 1000.0  # pyright: ignore[reportPrivateUsage]


def test_run_mtime_falls_back_to_dir_before_log_exists(tmp_path: Path) -> None:
    import os

    rd = tmp_path / "runs" / "fresh"
    rd.mkdir(parents=True)
    os.utime(rd, (2000, 2000))
    assert run_mtime(rd) == 2000.0  # pyright: ignore[reportPrivateUsage] - no log yet -> dir mtime


def test_question_bridge_round_trip(tmp_path: Path) -> None:
    register_frontend(tmp_path, 999999999)  # a live-ish pid so read doesn't early-out
    write_question_answers(tmp_path, "q1", ["use B"])
    assert read_question_answers(tmp_path, "q1", timeout_s=1.0) == ("use B",)


def test_read_question_answer_returns_none_when_no_tui(tmp_path: Path) -> None:
    # No front-end claim -> frontend_is_live False -> immediate None (don't block headless).
    assert read_question_answers(tmp_path, "q1", timeout_s=1.0) is None


def test_read_question_answer_consumes_the_file(tmp_path: Path) -> None:
    # The answer file is unlinked after reading, so a later prompt with the same
    # id (counters reset on resume) can't re-read a stale answer.
    register_frontend(tmp_path, 999999999)
    write_question_answers(tmp_path, "q1", ["first"])
    assert read_question_answers(tmp_path, "q1", timeout_s=1.0) == ("first",)
    assert not (questions_dir(tmp_path) / "q1.answer").exists()


def test_clear_pending_answers_wipes_stale_state(tmp_path: Path) -> None:
    write_answer(tmp_path, "approval-1", approved=True)
    write_question_answers(tmp_path, "question-1", ["stale"])
    register_frontend(tmp_path, 12345)
    clear_pending_answers(tmp_path)
    assert not (approvals_dir(tmp_path) / "approval-1.answer").exists()
    assert not (questions_dir(tmp_path) / "question-1.answer").exists()
    assert not (tmp_path / "frontend.pid").exists()


def test_refresh_keeps_runs_list_aligned_with_table_when_a_run_vanishes(tmp_path: Path) -> None:
    """A run dir that disappears between the listing and its stat() must be dropped
    from BOTH the table and self._runs. Otherwise the two desync and every
    cursor_row-indexed action (open/logs/merge) maps to the wrong run for rows
    past the gap."""
    import asyncio
    import shutil

    from textual.widgets import DataTable

    from agent6.ui.tui.home import Agent6HomeApp, HomeScreen

    a6 = tmp_path / ".agent6"
    for rid in ("r1", "r2", "r3"):
        _write_run(a6, "runs", rid, [{"type": "run.start", "mode": "run", "user_task": rid}])

    async def scenario() -> None:
        app = Agent6HomeApp(a6, tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, HomeScreen)
            table = screen.query_one("#runs", DataTable)
            assert table.row_count == 3
            # Delete the run currently shown in the MIDDLE row, then refresh.
            vanished = screen._runs[1]  # pyright: ignore[reportPrivateUsage]
            shutil.rmtree(vanished)
            screen.action_refresh()
            await pilot.pause()
            runs = screen._runs  # pyright: ignore[reportPrivateUsage]
            assert table.row_count == 2
            assert len(runs) == 2  # pre-fix this stayed 3 (the vanished run kept)
            assert vanished not in runs
            assert all(rd.exists() for rd in runs)  # every selectable row maps to a live run

    asyncio.run(scenario())


def test_home_app_lists_runs_and_opens_new_work_modal(tmp_path: Path) -> None:
    import asyncio

    from agent6.ui.tui.home import Agent6HomeApp, _NewWorkModal

    a6 = tmp_path / ".agent6"
    _write_run(a6, "runs", "r1", [{"type": "run.start", "mode": "run", "user_task": "do [x]"}])

    async def scenario() -> None:
        app = Agent6HomeApp(a6, tmp_path)
        async with app.run_test() as pilot:
            from textual.widgets import DataTable

            from agent6.ui.tui.home import HomeScreen

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

    from agent6.ui.tui.home import Agent6HomeApp, _NewWorkModal

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

    from agent6.ui.tui.home import Agent6HomeApp, _NewWorkModal

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
            from agent6.ui.tui.widgets import ActionItem

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

    from agent6.ui.tui import home

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
    monkeypatch.setattr(home, "agent6_exe", lambda: "agent6")  # type: ignore[attr-defined]

    a6 = tmp_path / ".agent6"
    a6.mkdir()

    # Chosen profile -> --profile is present, after <mode>, before the `--` task.
    home._spawn_and_locate(a6, tmp_path, "plan", "do it", profile="ultra")
    assert captured[-1] == ["agent6", "plan", "--profile", "ultra", "--", "do it"]

    # "(config default)" (profile="") -> NO --profile flag at all.
    home._spawn_and_locate(a6, tmp_path, "run", "do it", profile="")
    assert captured[-1] == ["agent6", "run", "--", "do it"]
    assert "--profile" not in captured[-1]


def test_spawn_argv_parallel_directive(tmp_path: Path, monkeypatch: object) -> None:
    """`/parallel N <task>` (run only) fans out lanes: the hub spawns
    `agent6 run <task> --parallel N`. A malformed directive is refused before any
    spawn (same (None, message) surface a failed spawn uses)."""
    import subprocess

    from agent6.ui.tui import home

    captured: list[list[str]] = []

    class _FakeProc:
        returncode = 0

        def poll(self) -> int:
            return 0

    def _fake_popen(argv: list[str], **_kw: object) -> _FakeProc:
        captured.append(list(argv))
        return _FakeProc()

    monkeypatch.setattr(subprocess, "Popen", _fake_popen)  # type: ignore[attr-defined]
    monkeypatch.setattr(home, "agent6_exe", lambda: "agent6")  # type: ignore[attr-defined]
    a6 = tmp_path / ".agent6"
    a6.mkdir()

    home._spawn_and_locate(a6, tmp_path, "run", "/parallel 2 add a greeting", profile="")
    assert captured[-1] == ["agent6", "run", "--parallel", "2", "--", "add a greeting"]

    home._spawn_and_locate(a6, tmp_path, "run", "/parallel gpt-5,opus refactor", profile="ultra")
    assert captured[-1] == [
        "agent6",
        "run",
        "--profile",
        "ultra",
        "--parallel",
        "gpt-5,opus",
        "--",
        "refactor",
    ]

    # Omitted spec -> one isolated lane (--parallel 1).
    home._spawn_and_locate(a6, tmp_path, "run", "/parallel refactor the parser", profile="")
    assert captured[-1] == ["agent6", "run", "--parallel", "1", "--", "refactor the parser"]

    # Multi-segment: one detached fan-out spawned per segment.
    start = len(captured)
    home._spawn_and_locate(a6, tmp_path, "run", "/parallel 2 task A /parallel 3 task B", profile="")
    assert captured[start:] == [
        ["agent6", "run", "--parallel", "2", "--", "task A"],
        ["agent6", "run", "--parallel", "3", "--", "task B"],
    ]

    # Malformed: refused before any Popen (nothing new captured).
    before = len(captured)
    run_dir, err = home._spawn_and_locate(a6, tmp_path, "run", "/parallel", profile="")
    assert run_dir is None and "/parallel" in err
    assert len(captured) == before

    # All-or-nothing: a later empty segment refuses the whole message.
    before = len(captured)
    run_dir, err = home._spawn_and_locate(
        a6, tmp_path, "run", "/parallel 2 ok /parallel", profile=""
    )
    assert run_dir is None and "/parallel" in err
    assert len(captured) == before


def test_parallel_partial_spawn_failure_surfaces(tmp_path: Path, monkeypatch: object) -> None:
    """A later lane's spawn failure must fail the whole message (and an earlier
    lane's failure must not be masked by a later success): the caller's only
    surfaces are open-the-run XOR show-the-error, so a partial failure returns
    the diagnostic and stays on the hub, lanes already launched keep running."""
    from agent6.ui.tui import home

    def fake_spawn(
        agent6_dir: Path, repo_cwd: Path, mode: str, task: str, *, profile: str, spec: str
    ) -> tuple[Path | None, str]:
        if "task B" in task:
            return None, "boom"
        return tmp_path / "r1", ""

    monkeypatch.setattr(home, "_spawn_run", fake_spawn)  # type: ignore[attr-defined]
    monkeypatch.setattr(home, "_model_refusal", lambda repo_cwd, segments: None)  # type: ignore[attr-defined]
    run_dir, err = home._spawn_and_locate(  # pyright: ignore[reportPrivateUsage]
        tmp_path, tmp_path, "run", "/parallel 2 task A /parallel 3 task B", profile=""
    )
    assert run_dir is None
    assert "boom" in err and "task B" in err

    # reversed: first lane fails, second succeeds -- lane 1's diagnostic survives
    def fake_spawn_rev(
        agent6_dir: Path, repo_cwd: Path, mode: str, task: str, *, profile: str, spec: str
    ) -> tuple[Path | None, str]:
        if "task A" in task:
            return None, "boom"
        return tmp_path / "r2", ""

    monkeypatch.setattr(home, "_spawn_run", fake_spawn_rev)  # type: ignore[attr-defined]
    run_dir, err = home._spawn_and_locate(  # pyright: ignore[reportPrivateUsage]
        tmp_path, tmp_path, "run", "/parallel 2 task A /parallel 3 task B", profile=""
    )
    assert run_dir is None
    assert "boom" in err and "task A" in err


def test_spawn_parallel_refuses_unknown_model_before_spawn(
    tmp_path: Path, monkeypatch: object
) -> None:
    """A `/parallel` model the cache can't confirm is the modal's normal error
    path: refused before any Popen (nothing spawned), with a did-you-mean."""
    import json
    import subprocess

    from agent6.config import Config
    from agent6.ui.tui import home

    cache = tmp_path / "cache" / "models"
    cache.mkdir(parents=True)
    (cache / "o.json").write_text(
        json.dumps({"models": ["moonshotai/kimi-k2.6"]}), encoding="utf-8"
    )
    monkeypatch.setenv("AGENT6_CACHE_HOME", str(tmp_path / "cache"))  # type: ignore[attr-defined]

    # The miss now re-checks the live listing before refusing; stub it with the
    # same ids so the refusal rests on "fresh" evidence (no real network).
    from agent6.models import validate as models_validate

    def _listing(*_a: object) -> list[str] | None:
        return ["moonshotai/kimi-k2.6"]

    monkeypatch.setattr(models_validate, "_fresh_listing", _listing)  # type: ignore[attr-defined]

    cfg = Config.model_validate(
        {
            "providers": {"o": {"api_format": "openai", "base_url": "https://x/v1"}},
            "models": {"worker": {"provider": "o", "model": "moonshotai/kimi-k2.6"}},
        }
    )

    class _Eff:
        config = cfg

    captured: list[list[str]] = []

    def _fake_popen(argv: list[str], **_kw: object) -> object:
        captured.append(list(argv))
        raise AssertionError("no spawn on a model refusal")

    monkeypatch.setattr(subprocess, "Popen", _fake_popen)  # type: ignore[attr-defined]
    monkeypatch.setattr(home, "load_effective", lambda _cwd, _cp=None: _Eff())  # type: ignore[attr-defined]
    a6 = tmp_path / ".agent6"
    a6.mkdir()

    run_dir, err = home._spawn_and_locate(
        a6, tmp_path, "run", "/parallel moonshotai/kimi-k2.7 fix it", profile=""
    )
    assert run_dir is None
    assert "unknown model 'moonshotai/kimi-k2.7'" in err
    assert "closest: moonshotai/kimi-k2.6" in err
    assert captured == []


def test_spawn_sets_stream_to_log_env(tmp_path: Path, monkeypatch: object) -> None:
    """The hub spawns the run with AGENT6_STREAM_TO_LOG=1 so the detached, non-TTY
    process emits role.*_delta events into logs.jsonl for the dashboard to render.
    Without it the run takes the non-streaming path and the dashboard shows only
    worker status, never live thinking."""
    import subprocess

    from agent6.ui.tui import home

    captured_env: dict[str, str] = {}

    class _FakeProc:
        returncode = 0

        def poll(self) -> int:
            return 0  # "already exited" -> helper bails out immediately

    def _fake_popen(argv: list[str], **kw: object) -> _FakeProc:
        env = kw.get("env")
        if isinstance(env, dict):
            captured_env.update(env)
        return _FakeProc()

    monkeypatch.setattr(subprocess, "Popen", _fake_popen)  # type: ignore[attr-defined]
    monkeypatch.setattr(home, "agent6_exe", lambda: "agent6")  # type: ignore[attr-defined]
    a6 = tmp_path / ".agent6"
    a6.mkdir()
    home._spawn_and_locate(a6, tmp_path, "ask", "why?", profile="")
    assert captured_env.get("AGENT6_STREAM_TO_LOG") == "1"
    # Still inherits the rest of the environment (PATH etc.), not a bare env.
    assert "PATH" in captured_env


def test_run_merge_cli_builds_argv_and_parses_result(tmp_path: Path, monkeypatch: object) -> None:
    """The hub's merge helper shells out to `agent6 runs merge <id>` and reports the
    captured output as (ok, message) -- it never touches git_ops itself."""
    import subprocess

    from agent6.ui.tui import home

    captured: list[list[str]] = []

    class _Proc:
        returncode = 0
        stdout = "[agent6] merged agent6/r1 into main (squash) -> abcdef123456\n"
        stderr = ""

    def _fake_run(argv: list[str], **_kw: object) -> _Proc:
        captured.append(list(argv))
        return _Proc()

    monkeypatch.setattr(subprocess, "run", _fake_run)  # type: ignore[attr-defined]
    monkeypatch.setattr(home, "agent6_exe", lambda: "agent6")  # type: ignore[attr-defined]

    ok, msg = home._run_merge_cli(tmp_path, "r1")  # pyright: ignore[reportPrivateUsage]
    assert captured[-1] == ["agent6", "runs", "merge", "r1"]
    assert ok is True
    assert "merged agent6/r1" in msg


def test_merge_action_confirms_then_shells_out(tmp_path: Path, monkeypatch: object) -> None:
    """Pressing `m` opens a confirm modal; confirming runs `agent6 runs merge` for the
    selected run (stubbed here so no real CLI is spawned)."""
    import asyncio

    from textual.widgets import DataTable

    from agent6.ui.tui import home
    from agent6.ui.tui.home import Agent6HomeApp
    from agent6.ui.tui.modals import ConfirmModal

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

    from agent6.ui.tui.home import Agent6HomeApp

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
