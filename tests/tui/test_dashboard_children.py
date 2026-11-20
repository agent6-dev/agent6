from __future__ import annotations

import asyncio
import json
import os
import pathlib
import subprocess
import time

import pytest
import textual.widgets._footer as footer_widgets
from textual import pilot as textual_pilot
from textual import widgets

from agent6 import paths
from agent6.sessions import child_dispatch, completion, ipc, layout, process
from agent6.ui import keymap
from agent6.ui.tui import app, menubar, modals
from tests.tui import waits


def _events(directory: pathlib.Path, *rows: dict[str, object]) -> None:
    with (directory / layout.LOGS_NAME).open("a", encoding="utf-8") as journal:
        journal.writelines(json.dumps(row) + "\n" for row in rows)


@pytest.fixture
def family(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    """An active dispatch, two saved branches, a refused launch and a panel verdict."""
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    subprocess.run(["git", "init", "-q"], check=True)
    state = paths.state_dir(repo)
    start = int(time.time()) - 50
    root = layout.SessionLayout(state, "atlas")
    root.ensure()
    root.manifest_path.write_text(json.dumps({"version": 3, "session_id": "atlas", "mode": "run"}))
    _events(root.session_dir, {"type": "session.start", "mode": "run", "user_task": "Audit parser"})
    process.write_worker_pid(root.session_dir, os.getpid())
    for group, tasks in (
        ("atlas-p1", ("Check [flags]", "Read grammar", "Add tests", "Try route")),
        ("atlas-rpanel", ("Rank attempts",)),
    ):
        panel = group == "atlas-rpanel"
        children: list[child_dispatch.ChildRecord] = []
        for number, task in enumerate(tasks, 1):
            sid = f"{group}-l{number}"
            join = None
            if number == 4:
                join = child_dispatch.JoinRecord(status="failed", detail="no matching provider")
            else:
                child = layout.SessionLayout(
                    state, sid, layout.EVIDENCE_BUCKET if panel else "runs"
                )
                child.ensure()
                mode = "reviewer" if panel else "run"
                child.manifest_path.write_text(
                    json.dumps(
                        {
                            "version": 3,
                            "mode": mode,
                            "session_id": sid,
                            "models": {"driver": {"provider": "local", "model": "swift"}},
                            "parallel": {"parent": "atlas", "group": group, "number": number},
                        }
                    )
                )
                _events(
                    child.session_dir,
                    {"type": "session.start", "mode": mode, "user_task": task, "ts": start},
                    {
                        "type": "budget.update",
                        "input_total": 120,
                        "output_total": 4,
                        "usd_total": 0.03,
                        "usd_partial": True,
                    },
                    {"type": "tool.call", "turn": 2, "name": "read_file", "call_id": 1},
                )
                if panel or number > 1:
                    _events(
                        child.session_dir,
                        {"type": "session.stop", "reason": "finish_session", "ts": start + 12},
                    )
                else:
                    process.write_worker_pid(child.session_dir, os.getpid())
                if panel:
                    completion.write_completion(
                        child.session_dir,
                        completion.CompletionRecord(
                            text='{"order":["B","A"],"summary":"B checks both flags"}',
                            stop_reason="answered",
                            output_tokens=10,
                            calls=1,
                        ),
                    )
                elif number > 1:
                    join = child_dispatch.JoinRecord(
                        status="stopped" if number == 2 else "retained",
                        branch=f"agent6/{sid}",
                        commits=number - 1,
                    )
            children.append(
                child_dispatch.ChildRecord(
                    session_id=sid, number=number, task=task, model="alternate", join=join
                )
            )
        child_dispatch.write_record(
            root.session_dir,
            child_dispatch.DispatchRecord(
                identity=group,
                source="panel" if panel else "model",
                owner="exec-1",
                turn=2,
                join_policy="record_only" if panel else "rank",
                base="a" * 40,
                spec="",
                children=tuple(children),
                state="complete" if panel else "running",
            ),
            now=start,
        )
    return root.session_dir


@pytest.mark.parametrize("size", [(80, 24), (140, 40)])
def test_dashboard_child_rows(family: pathlib.Path, size: tuple[int, int]) -> None:
    """Each row shows the child's progress or its recorded result, including a refused launch."""

    async def scenario() -> None:
        tui = app.Agent6TUI(family)
        async with tui.run_test(size=size) as pilot:
            await pilot.press("ctrl+d")
            table = tui.screen.query_one("#children", widgets.DataTable)
            table.focus()
            table.move_cursor(row=table.get_row_index("atlas-p1-l1"))
            await pilot.pause()
            assert table.row_count == 5
            assert 0 < table.region.height <= 9
            live = list(map(str, table.get_row("atlas-p1-l1")))
            assert {
                "attempt 1",
                "atlas-p1-l1",
                "Check [flags]",
                "local/swift",
                "running",
                "read_file",
                "2",
                "124",
                "~$0.03",
            } <= set(live)
            assert any(s.endswith("s") and s[:-1].isdigit() and int(s[:-1]) >= 50 for s in live)
            for sid, words in (
                ("atlas-p1-l2", ("stopped", "1 commit", "agent6/atlas-p1-l2", "12s")),
                ("atlas-p1-l3", ("retained", "2 commits", "agent6/atlas-p1-l3")),
                ("atlas-p1-l4", ("failed", "not started", "no matching provider", "alternate")),
                ("atlas-rpanel-l1", ("reviewer 1", "B > A", "B checks both flags")),
            ):
                row = " ".join(map(str, table.get_row(sid)))
                assert all(word in row for word in words), row
            if directory := os.getenv("TUI_CAPTURES"):
                capture = pathlib.Path(directory)
                capture.mkdir(exist_ok=True, parents=True)
                tui.save_screenshot(f"children-{size[0]}x{size[1]}.svg", path=str(capture))
            if size[0] == 80:
                await pilot.press("right", "right", "right")
                x = table.scroll_x
                assert x > 0
                with (family.parent / "atlas-p1-l1" / layout.LOGS_NAME).open("a") as log:
                    log.write('{"type":"tool.call","name":"run_command","turn":3,"call_id":2}\n')
                await waits.wait_for(
                    pilot,
                    lambda: "run_command" in map(str, table.get_row("atlas-p1-l1")),
                    "tool change",
                )
                assert table.scroll_x >= x

    asyncio.run(scenario())


def test_enter_keeps_the_child_id_when_rows_move(family: pathlib.Path) -> None:
    """A dispatch update inserts another row before the cursor without retargeting Enter."""

    async def scenario() -> None:
        tui = app.Agent6TUI(family)
        async with tui.run_test() as pilot:
            await pilot.press("ctrl+d")
            table = tui.screen.query_one("#children", widgets.DataTable)
            table.move_cursor(row=table.get_row_index("atlas-p1-l2"))
            table.focus()
            record = child_dispatch.read_record(child_dispatch.record_path(family, "atlas-p1"))
            child_dispatch.write_record(
                family,
                record.model_copy(
                    update={
                        "identity": "atlas-p0",
                        "children": (
                            child_dispatch.ChildRecord(
                                session_id="atlas-p0-l1", number=1, task="Extra"
                            ),
                        ),
                    }
                ),
                now=time.time(),
            )
            await waits.wait_for(pilot, lambda: table.row_count == 6, "new dispatch row")
            await pilot.press("enter")
        assert tui.return_value == app.TuiExit(open_next=family.parent / "atlas-p1-l2")

    asyncio.run(scenario())


def test_enter_opens_the_reviewer_but_not_a_never_started_childs_prefix(
    family: pathlib.Path,
) -> None:
    """A missing child's id can prefix a real session in the same bucket."""
    (family.parent / "atlas-p1-l40").mkdir()
    reviewer = family.parent.parent / "reviewers" / "atlas-rpanel-l1"
    # A log without a readable manifest can still be opened for inspection.
    (reviewer / "manifest.json").unlink()

    async def scenario() -> None:
        tui = app.Agent6TUI(family)
        async with tui.run_test() as pilot:
            await pilot.press("ctrl+d")
            table = tui.screen.query_one("#children", widgets.DataTable)
            table.focus()
            table.move_cursor(row=table.get_row_index("atlas-p1-l4"))
            await pilot.press("enter")
            assert tui.return_value is None
            table.move_cursor(row=table.get_row_index("atlas-rpanel-l1"))
            await pilot.press("enter")
        assert tui.return_value == app.TuiExit(open_next=reviewer)

    asyncio.run(scenario())


async def _highlight_live(
    tui: app.Agent6TUI, pilot: textual_pilot.Pilot[app.TuiExit]
) -> widgets.DataTable[object]:
    await pilot.press("ctrl+d")
    table = tui.screen.query_one("#children", widgets.DataTable)
    table.focus()
    table.move_cursor(row=table.get_row_index("atlas-p1-l1"))
    await pilot.pause()
    return table


def test_child_keys_show_in_the_footer_and_help(family: pathlib.Path) -> None:
    """The footer fits both live actions at 80 columns. Help names the same keys."""

    async def scenario() -> None:
        tui = app.Agent6TUI(family)
        async with tui.run_test(size=(80, 24)) as pilot:
            await _highlight_live(tui, pilot)
            wanted = {"x": "Stop child", "m": "Message child"}
            assert {
                k: v.binding.description
                for k, v in tui.screen.active_bindings.items()
                if k in wanted and v.enabled and v.binding.show
            } == wanted

            def painted_keys() -> dict[str, str]:
                return {
                    key.key: key.description
                    for key in tui.screen.query(footer_widgets.FooterKey)
                    if key.key in wanted
                    and key.region.height
                    and key.region.right <= 80
                    and key.region.bottom <= 24
                }

            await waits.wait_for(pilot, lambda: painted_keys() == wanted, "visible child keys")
            await pilot.press("question_mark")
            assert isinstance(tui.screen, menubar.HelpScreen)
            lines = [str(w.render()) for w in tui.screen.query(widgets.Static)]
            for key, label in wanted.items():
                assert any(label in line and line.strip().endswith(key) for line in lines), lines

    asyncio.run(scenario())


def test_message_and_stop_reach_only_the_highlighted_child(family: pathlib.Path) -> None:
    """The dialogs name the selected child. Only that child gets either IPC request."""
    child = family.parent / "atlas-p1-l1"
    sibling = family.parent / "atlas-p1-l2"

    async def scenario() -> None:
        tui = app.Agent6TUI(family)
        async with tui.run_test() as pilot:
            table = await _highlight_live(tui, pilot)
            await pilot.press("m")
            assert isinstance(tui.screen, modals.TextInputModal)
            assert child.name in str(tui.screen.query_one(widgets.Static).render())
            table.move_cursor(row=table.get_row_index(sibling.name))
            tui.screen.query_one(widgets.Input).value = "abort"
            await pilot.press("enter")
            assert ipc.steer_request_pending(child) and not ipc.steer_answer_is_abort(child)
            assert "abort" in (child / ipc.STEER_ANSWER_FILE).read_text()
            assert not ipc.steer_request_pending(family) and not ipc.steer_request_pending(sibling)
            table.move_cursor(row=table.get_row_index(child.name))
            await pilot.press("x")
            assert isinstance(tui.screen, modals.ConfirmModal)
            assert child.name in str(tui.screen.query_one(widgets.Static).render())
            await pilot.press("escape")
            assert not ipc.stop_request_pending(child)
            await pilot.press("x", "y")
            assert ipc.stop_request_is_now(child) and ipc.steer_answer_is_abort(child)
            assert not ipc.stop_request_pending(family) and not ipc.stop_request_pending(sibling)

    asyncio.run(scenario())


def test_message_reaches_a_reviewer_in_a_later_batch(family: pathlib.Path) -> None:
    """A reviewer's number within its batch can differ from its row in the panel."""
    group = "atlas-rpanel"
    batch = group + "-b2"
    sid = batch + "-l1"
    record = child_dispatch.read_record(child_dispatch.record_path(family, group))
    reviewer = layout.SessionLayout(layout.layout_of(family).state_dir, sid, layout.EVIDENCE_BUCKET)
    reviewer.ensure()
    reviewer.manifest_path.write_text(
        json.dumps(
            {
                "version": 3,
                "mode": "reviewer",
                "session_id": sid,
                "parallel": {"group": batch, "number": 1, "parent": family.name},
            }
        )
    )
    _events(reviewer.session_dir, {"type": "session.start", "mode": "reviewer"})
    process.write_worker_pid(reviewer.session_dir, os.getpid())
    row = child_dispatch.ChildRecord(session_id=sid, number=2, task="Check ordering")
    child_dispatch.write_record(
        family,
        record.model_copy(update={"state": "running", "children": (*record.children, row)}),
        now=time.time(),
    )

    async def scenario() -> None:
        tui = app.Agent6TUI(family)
        async with tui.run_test() as pilot:
            table = await _highlight_live(tui, pilot)
            table.move_cursor(row=table.get_row_index(sid))
            await pilot.press("m")
            assert isinstance(tui.screen, modals.TextInputModal)
            tui.screen.query_one(widgets.Input).value = "Check both flags"
            await pilot.press("enter")
            assert ipc.steer_request_pending(reviewer.session_dir)

    asyncio.run(scenario())


def test_no_controls_for_children_that_never_started_or_ended(family: pathlib.Path) -> None:
    async def scenario() -> None:
        tui = app.Agent6TUI(family)
        async with tui.run_test() as pilot:
            table = await _highlight_live(tui, pilot)
            dashboard = tui.screen
            keys = keymap.SCREEN_KEYS["dashboard"]
            for sid in ("atlas-p1-l2", "atlas-p1-l3", "atlas-p1-l4", "atlas-rpanel-l1"):
                table.move_cursor(row=table.get_row_index(sid))
                for action in ("stop_child", "message_child"):
                    bound = dashboard.active_bindings.get(keys[action])
                    assert bound is None or not bound.enabled
                    await pilot.press(keys[action])
                    assert tui.screen is dashboard
            assert not (family.parent / "atlas-p1-l4").exists()

    asyncio.run(scenario())


@pytest.mark.parametrize(("key", "changed"), [("m", "ended"), ("x", "parent"), ("m", "worker")])
def test_action_rechecks_the_child_on_submit(family: pathlib.Path, key: str, changed: str) -> None:
    """An end, a different parent or a replacement worker invalidates an open dialog."""
    child = family.parent / "atlas-p1-l1"

    async def scenario() -> None:
        tui = app.Agent6TUI(family)
        async with tui.run_test() as pilot:
            await _highlight_live(tui, pilot)
            dashboard = tui.screen
            await pilot.press(key)
            assert tui.screen is not dashboard
            if changed == "ended":
                _events(child, {"type": "session.stop", "reason": "finish_session"})
            elif changed == "parent":
                path = child / "manifest.json"
                record = json.loads(path.read_text())
                record["parallel"]["parent"] = "other"
                path.write_text(json.dumps(record))
            else:
                process.write_worker_pid(child, os.getppid())
            if key == "m":
                tui.screen.query_one(widgets.Input).value = "check both flags"
            await pilot.press("enter" if key == "m" else "y")
            assert tui.screen is dashboard
            assert not ipc.steer_request_pending(child) and not ipc.stop_request_pending(child)

    asyncio.run(scenario())
