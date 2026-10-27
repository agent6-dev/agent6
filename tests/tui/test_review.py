from __future__ import annotations

import asyncio
import json
import os
import pathlib
import subprocess
from typing import Any

import pytest
from textual import widgets

from agent6 import event_log
from agent6.app import review_actions
from agent6.sessions import ipc, process
from agent6.ui import spawn
from agent6.ui.tui import app as tui_app
from agent6.ui.tui import home, modals
from tests import review_helpers


def _shown(widget: widgets.Static) -> str:
    return str(widget.content)


@pytest.mark.parametrize("entry", ["hub", "dashboard"])
def test_open_select_and_preview_recorded_trees(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, entry: str
) -> None:
    """The keyboard opens findings and previews recorded trees, not the edited checkout."""
    monkeypatch.chdir(tmp_path)

    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=tmp_path, text=True).strip()

    git("init", "-q")
    source = tmp_path / "parser.py"
    source.write_text("one\ntwo\nold value\nfour\n", encoding="utf-8")
    git("add", ".")
    git("commit", "-qm", "base")
    base = git("rev-parse", "HEAD^{tree}")
    source.write_text("one\ntwo\nreviewed value\nfour\n", encoding="utf-8")
    git("add", ".")
    git("commit", "-qm", "head")
    head = git("rev-parse", "HEAD^{tree}")
    saved = review_helpers.reviewed_session(tmp_path)
    panel = json.loads(saved.logs_path.read_text())
    panel["target"].update(base_tree=base, head_tree=head)
    saved.logs_path.write_text(json.dumps(panel) + "\n")
    source.write_text("unreviewed workspace edit\n")
    before = git("status", "--porcelain")

    async def scenario() -> None:
        app = (
            home.Agent6HomeApp(saved.state_dir, tmp_path)
            if entry == "hub"
            else tui_app.Agent6TUI(saved.session_dir)
        )
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            if entry == "dashboard":
                await pilot.press("ctrl+d")
            await pilot.press("v" if entry == "hub" else "ctrl+f")
            await pilot.pause()
            assert type(app.screen).__name__ == "FindingsScreen"
            table = app.screen.query_one("#findings", widgets.DataTable)
            assert table.row_count == 2
            assert "f1" in _shown(app.screen.query_one("#finding-detail", widgets.Static))
            await pilot.press("down")
            await pilot.pause()
            assert "f2" in _shown(app.screen.query_one("#finding-detail", widgets.Static))
            patch = _shown(app.screen.query_one("#diff-body", widgets.Static))
            assert "+reviewed value" in patch and "-old value" in patch
            assert "unreviewed workspace edit" not in patch
            assert not app.screen.query_one("#diff-nav").display
            await pilot.press("escape")
            assert type(app.screen).__name__ in {"HomeScreen", "DashboardScreen"}

    asyncio.run(scenario())
    assert git("status", "--porcelain") == before


@pytest.mark.parametrize("kind", ["pre_existing", "changed", "unknown"])
def test_selected_finding_shows_only_pre_existing_line_origin(
    tmp_path: pathlib.Path, kind: str
) -> None:
    """Only pre-existing lines show their base range, reason and blame in the detail pane."""
    saved = review_helpers.reviewed_session(tmp_path)
    panel = json.loads(saved.logs_path.read_text())
    origin: dict[str, Any] = dict(kind=kind, reason="The cited bytes match the reviewed base.")
    if kind == "pre_existing":
        origin.update(
            base_location=dict(path="old_parser.py", side="base", start_line=7, end_line=8),
            commits=["a" * 40, "b" * 40],
        )
    panel["findings"][0]["line_origin"] = origin
    saved.logs_path.write_text(json.dumps(panel) + "\n")

    async def scenario() -> None:
        app = home.Agent6HomeApp(saved.state_dir, tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("v")
            await pilot.pause()
            detail = _shown(app.screen.query_one("#finding-detail", widgets.Static))
            assert "Defect f1" in detail
            if kind == "pre_existing":
                assert (
                    "  Pre-existing line: old_parser.py:7-8 (base). "
                    "The cited bytes match the reviewed base.\n"
                    f"  Blame: {'a' * 40}, {'b' * 40}"
                ) in detail
            else:
                assert "Pre-existing line" not in detail
                assert "Blame:" not in detail
                assert "The cited bytes match the reviewed base." not in detail

    asyncio.run(scenario())


@pytest.mark.parametrize("recorded", [True, False], ids=["recorded", "legacy"])
def test_selected_finding_shows_its_own_panels_rules(
    tmp_path: pathlib.Path, recorded: bool
) -> None:
    """Show the selected panel's sources, not a later panel's rules or an invented empty set."""
    saved = review_helpers.reviewed_repo(tmp_path)
    rows = [json.loads(line) for line in saved.logs_path.read_text().splitlines()]
    start = next(row for row in rows if row["type"] == "loop.review.start")
    blob = start["rule_sources"][0]["blob_id"]
    base_tree = start["target"]["base_tree"]
    if not recorded:
        for field in ("rule_sources", "rules_complete", "rules_reason"):
            start.pop(field)
    saved.logs_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    sink = event_log.EventSink(saved.logs_path)
    sink.emit(
        "loop.review.start",
        panel_id="later",
        execution_id="execution",
        target=start["target"],
        seats=[],
        rule_sources=[],
        rules_complete=False,
        rules_reason="Later rule set is hidden",
    )
    sink.emit(
        "loop.review.skipped",
        panel_id="later",
        execution_id="execution",
        reason="Later rule set is hidden",
    )

    async def scenario() -> None:
        app = home.Agent6HomeApp(saved.state_dir, tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("v")
            await pilot.pause()
            detail = _shown(app.screen.query_one("#finding-detail", widgets.Static))
            assert f"base tree {base_tree}" in detail
            assert "Panel panel:" in detail and "Panel later:" not in detail
            assert "Later rule set is hidden" not in detail
            if recorded:
                assert "rules complete" in detail
                assert f"REVIEW.md: blob {blob}, 35 bytes" in detail
            else:
                assert "rule sources not recorded" in detail
                assert "No base rules exist." not in detail

    asyncio.run(scenario())


def test_resolve_and_reopen_record_the_reason_and_tui_source(
    tmp_path: pathlib.Path,
) -> None:
    """The form validates reasons and records dispositions through the real action service."""
    saved = review_helpers.reviewed_session(tmp_path)

    async def scenario() -> None:
        app = home.Agent6HomeApp(saved.state_dir, tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("v")
            assert type(app.screen).__name__ == "FindingsScreen"
            await pilot.press("s")
            await pilot.click("#review-submit")
            assert review_actions.read_review(saved.session_dir).version == 1
            assert app.screen.query_one("#review-error", widgets.Static).content
            app.screen.query_one("#review-reason", widgets.TextArea).load_text(
                "Reproduced and fixed"
            )
            await pilot.press("ctrl+s")
            await app.workers.wait_for_complete()
            await pilot.pause()
            finding = review_actions.read_review(saved.session_dir).findings[0]
            assert finding.status == "resolved"
            assert finding.resolution is not None
            assert finding.resolution.reason == "Reproduced and fixed"
            assert finding.resolution.verify_id is None
            assert finding.actions[-1].source == "tui"
            await pilot.press("o")
            app.screen.query_one("#review-reason", widgets.TextArea).load_text("Still reproduces")
            await pilot.click("#review-submit")
            await app.workers.wait_for_complete()
            await pilot.pause()
            finding = review_actions.read_review(saved.session_dir).findings[0]
            assert finding.status == "open"
            assert finding.actions[-1].reason == "Still reproduces"
            detail = _shown(app.screen.query_one("#finding-detail", widgets.Static))
            assert "Still reproduces" in detail and "Reproduced and fixed" in detail

    asyncio.run(scenario())


def test_live_fix_uses_the_single_writer_bridge(tmp_path: pathlib.Path) -> None:
    """Fix queues a TUI action without writing the live worker's journal."""
    saved = review_helpers.reviewed_session(tmp_path)
    process.write_worker_pid(saved.session_dir, os.getpid())

    async def scenario() -> None:
        app = home.Agent6HomeApp(saved.state_dir, tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("v")
            assert type(app.screen).__name__ == "FindingsScreen"
            await pilot.press("f")
            await app.workers.wait_for_complete()
            pending = list(ipc.review_requests(saved.session_dir))
            assert len(pending) == 1
            request = review_actions.ActionRequest.model_validate_json(pending[0][1])
            assert request.source == "tui" and request.finding_id == "f1"
            assert review_actions.read_review(saved.session_dir).version == 1
            review_actions.drain(event_log.EventSink(saved.logs_path), execution_id="live")
            finding = review_actions.read_review(saved.session_dir).findings[0]
            assert finding.fix_state == "submitted" and finding.status == "open"
            await pilot.press("r")
            await pilot.pause()
            assert "submitted" in _shown(app.screen.query_one("#finding-detail", widgets.Static))

    try:
        asyncio.run(scenario())
    finally:
        process.clear_worker_pid(saved.session_dir)


def test_offline_fix_requires_confirmation_and_reports_resume_failure(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A confirmed Fix resumes with the stored finding and keeps failed delivery visible."""
    saved = review_helpers.reviewed_session(tmp_path)
    calls: list[dict[str, Any]] = []

    def resume(cwd: pathlib.Path, session_id: str, **kwargs: Any) -> str:
        assert cwd == tmp_path and session_id == saved.session_id
        calls.append(kwargs)
        return "resume snapshot is unavailable"

    monkeypatch.setattr(spawn, "spawn_detached_resume", resume)

    async def scenario() -> None:
        app = home.Agent6HomeApp(saved.state_dir, tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("v")
            assert type(app.screen).__name__ == "FindingsScreen"
            await pilot.press("f")
            assert isinstance(app.screen, modals.ConfirmModal)
            assert not calls
            await pilot.press("n")
            assert not calls
            await pilot.press("f", "y")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert len(calls) == 1
            assert "Reproduce this report" in calls[0]["steer"]
            assert "Defect f1" in calls[0]["steer"]
            finding = review_actions.read_review(saved.session_dir).findings[0]
            assert finding.status == "open" and finding.fix_state == "failed"
            assert "resume snapshot is unavailable" in _shown(
                app.screen.query_one("#finding-detail", widgets.Static)
            )

    asyncio.run(scenario())


def test_merge_refusal_keeps_the_finding_ids_visible(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The hub displays the merge command's refusal without dropping finding IDs."""
    saved = review_helpers.reviewed_session(tmp_path)
    notices: list[str] = []
    refusal = "merge refused: open block findings f1, f2"

    def merge_refusal(*_args: object) -> tuple[bool, str]:
        return False, refusal

    def notify(message: object, **_kwargs: object) -> None:
        notices.append(str(message))

    monkeypatch.setattr(home, "_run_merge_cli", merge_refusal)

    async def scenario() -> None:
        app = home.Agent6HomeApp(saved.state_dir, tmp_path)
        monkeypatch.setattr(app, "notify", notify)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("m", "y")
            await pilot.pause()
            assert refusal in notices

    asyncio.run(scenario())
