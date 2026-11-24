from __future__ import annotations

import pathlib
import re
from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest

from agent6.sessions import event_log, ipc, journal
from tests import task_runs
from tests.web import session_logs, test_files_preview_browser, test_web_browser

if TYPE_CHECKING:
    from playwright.sync_api import Page

pytestmark = pytest.mark.browser
browser = test_files_preview_browser.browser
open_page = test_web_browser.open_page
port = test_web_browser.port
OpenPage = Callable[[int], tuple["Page", list[str]]]


def test_directives_fit_a_short_viewport(
    open_page: OpenPage, port: int, tmp_path: pathlib.Path
) -> None:
    """The list box fits between the entry and the edge of a short screen."""
    from playwright import sync_api

    session_logs.write_run(tmp_path, "short", [], live=True)
    page, errors = open_page(port)
    page.set_viewport_size({"width": 960, "height": 240})
    page.goto(f"http://127.0.0.1:{port}/#/session/short")
    entry = page.get_by_label("Message to the run")
    entry.fill("/")
    popup = page.get_by_role("listbox", name="Directives")
    sync_api.expect(popup).to_be_in_viewport(ratio=1)
    box, field = popup.bounding_box(), entry.bounding_box()
    assert box and field
    assert box["y"] + box["height"] <= field["y"] or box["y"] >= field["y"] + field["height"]
    assert not errors


def test_arrows_reveal_the_active_directive(
    open_page: OpenPage, port: int, tmp_path: pathlib.Path
) -> None:
    """An active descendant below the scroll box has to scroll into view without taking focus."""
    from playwright import sync_api

    session_logs.write_run(tmp_path, "suggest", [], live=True)
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/session/suggest")
    entry = page.get_by_label("Message to the run")
    entry.fill("/")
    popup = page.get_by_role("listbox", name="Directives")
    options = popup.get_by_role("option")
    sync_api.expect(options.last).to_contain_text("/shells")
    for _ in range(options.count()):
        entry.press("ArrowDown")
    sync_api.expect(options.last).to_have_attribute("aria-selected", "true")
    sync_api.expect(options.last).to_be_in_viewport(ratio=1)
    sync_api.expect(entry).to_be_focused()
    entry.press("ArrowDown")
    sync_api.expect(options.first).to_be_in_viewport(ratio=1)
    assert not errors


def test_more_dismisses_and_uses_the_inline_actions_when_wide(
    open_page: OpenPage, port: int, tmp_path: pathlib.Path
) -> None:
    """Escape, a click outside and a choice close More. Widening needs no second set of buttons."""
    from playwright import sync_api

    session_logs.write_run(tmp_path, "more", [], live=True)
    page, errors = open_page(port)
    phone: sync_api.ViewportSize = {"width": 390, "height": 640}
    page.set_viewport_size(phone)
    page.goto(f"http://127.0.0.1:{port}/#/session/more")
    bar = page.locator(".run-app .actions")
    more = bar.get_by_text("More", exact=True)
    compact = bar.get_by_role("button", name="Compact context", exact=True)
    more.click()
    sync_api.expect(compact).to_be_visible()
    page.keyboard.press("Escape")
    sync_api.expect(compact).to_be_hidden()
    more.click()
    page.locator(".card-conv h2").click(position={"x": 2, "y": 2})
    sync_api.expect(compact).to_be_hidden()
    more.click()
    compact.click()
    sync_api.expect(compact).to_be_hidden()
    more.click()
    menu = bar.locator('[popover="auto"]:popover-open')
    sync_api.expect(menu).to_be_visible()
    page.set_viewport_size({"width": 1280, "height": 900})
    sync_api.expect(more).to_be_hidden()
    sync_api.expect(menu).to_have_count(0)
    sync_api.expect(compact).to_be_visible()
    sync_api.expect(bar.get_by_text("Compact context", exact=True)).to_have_count(1)
    page.set_viewport_size(phone)
    sync_api.expect(compact).to_be_hidden()
    assert not errors


def test_retire_queues_only_the_chosen_live_task(
    open_page: OpenPage, port: int, tmp_path: pathlib.Path
) -> None:
    """Retiring a selected task queues its full id without stopping or steering the worker."""
    from playwright import sync_api

    run = task_runs.live_graph(tmp_path)
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/session/{run.session_id}")
    tree = page.locator(".card-tasks")
    sync_api.expect(tree).to_contain_text("Check paths")
    sync_api.expect(tree.get_by_role("radio", name=re.compile("Check paths"))).to_have_count(0)
    retire = tree.get_by_role("button", name="Retire task")
    sync_api.expect(retire).to_be_disabled()
    tree.get_by_role("radio", name=re.compile("Link pages")).check()
    retire.click()
    dialog = page.get_by_role("alertdialog", name=re.compile("Retire task 2"))
    sync_api.expect(dialog).to_contain_text("Link pages")
    sync_api.expect(dialog).to_contain_text("obsolete")
    sync_api.expect(dialog.get_by_role("button", name="Cancel")).to_be_focused()
    page.keyboard.press("Escape")
    assert not ipc.drain_requests(run.session_dir)
    retire.click()
    dialog.get_by_role("button", name="Retire task").click()
    sync_api.expect(page.get_by_role("status").filter(has_text="retiring")).to_contain_text(
        "Link pages"
    )
    assert [(r.kind, r.text) for r in ipc.drain_requests(run.session_dir)] == [("retire", "0002")]
    assert not ipc.steer_request_pending(run.session_dir)
    assert not ipc.stop_request_pending(run.session_dir)
    retire.click()
    event_log.EventSink(run.logs_path).emit(
        journal.SessionStop(execution_id="first", reason="finish_session", turns=1)
    )
    sync_api.expect(tree.locator('input[type="radio"]')).to_have_count(0)
    dialog.get_by_role("button", name="Retire task").click()
    assert not ipc.drain_requests(run.session_dir)
    sync_api.expect(retire).to_be_hidden()
    assert not errors


def test_the_task_graph_for_a_commit_offers_no_retirement(
    open_page: OpenPage, port: int, tmp_path: pathlib.Path
) -> None:
    from playwright import sync_api

    run = task_runs.live_graph(tmp_path)
    sha = "a" * 40
    event_log.EventSink(run.logs_path).emit(
        journal.CheckpointCommitted(sha=sha, turn=1, subject="Add index")
    )
    page, errors = open_page(port)
    page.route("**/diff?*", lambda route: route.fulfill(json={"patch": ""}))
    page.goto(f"http://127.0.0.1:{port}/#/session/{run.session_id}")
    commit = page.get_by_role("combobox", name="Commit", exact=True)
    tree = page.locator(".card-tasks")
    tree.get_by_role("radio", name=re.compile("Link pages")).check()
    commit.select_option(sha)
    sync_api.expect(tree).to_contain_text("as of turn 1")
    sync_api.expect(tree.get_by_role("radio")).to_have_count(0)
    sync_api.expect(tree.get_by_role("button", name="Retire task")).to_be_hidden()
    commit.select_option("")
    sync_api.expect(tree.get_by_role("button", name="Retire task")).to_be_disabled()
    assert not errors


def test_a_cumulative_diff_failure_keeps_the_selected_commits_graph(
    open_page: OpenPage, port: int, tmp_path: pathlib.Path
) -> None:
    """Changing diff mode does not replace the selected commit's tasks with the live graph."""
    from playwright import sync_api

    run = task_runs.live_graph(tmp_path)
    sha = "a" * 40
    events = event_log.EventSink(run.logs_path)
    events.emit(journal.CheckpointCommitted(sha=sha, turn=1, subject="Add index"))
    events.emit(journal.GraphUpdate(nodes={}, cursor=None))
    page, errors = open_page(port)
    page.route("**/diff?*", lambda route: route.fulfill(json={"patch": ""}))
    page.route(
        "**/diff?*cumulative=1",
        lambda route: route.fulfill(status=503, json={"error": "Diff unavailable"}),
    )
    page.goto(f"http://127.0.0.1:{port}/#/session/{run.session_id}")
    tree = page.locator(".card-tasks")
    sync_api.expect(tree).to_contain_text("no task graph yet")
    page.get_by_role("combobox", name="Commit", exact=True).select_option(sha)
    sync_api.expect(tree).to_contain_text("as of turn 1")
    sync_api.expect(tree).to_contain_text("Link pages")
    page.get_by_label("cumulative").check()
    sync_api.expect(page.locator(".card-diff")).to_contain_text("Diff unavailable")
    sync_api.expect(tree).to_contain_text("as of turn 1")
    sync_api.expect(tree).to_contain_text("Link pages")
    assert not [e for e in errors if not e.startswith("Failed to load resource")]
