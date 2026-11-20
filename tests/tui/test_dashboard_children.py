from __future__ import annotations

import asyncio
import json
import os
import pathlib
import subprocess
import time

import pytest
from textual import widgets

from agent6 import paths
from agent6.sessions import child_dispatch, completion, layout, process
from agent6.ui.tui import app
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
                        status="stopped" if number == 2 else "unmerged",
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
            await pilot.pause()
            assert table.row_count == 5
            assert 0 < table.region.height <= 9
            live = list(map(str, table.get_row("atlas-p1-l1")))
            assert {
                "1",
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
                ("atlas-p1-l3", ("unmerged", "2 commits", "agent6/atlas-p1-l3")),
                ("atlas-p1-l4", ("failed", "not started", "no matching provider", "alternate")),
                ("atlas-rpanel-l1", ("reviewer", "B > A", "B checks both flags")),
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
        assert tui.return_value == app.TuiExit(
            open_next=family.parent.parent / "reviewers" / "atlas-rpanel-l1"
        )

    asyncio.run(scenario())
