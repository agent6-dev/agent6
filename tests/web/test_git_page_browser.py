# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The Git page driven in a browser: its rows, keyboard, focus and waiting states."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest

from agent6.ui.web import git_model
from tests.web import test_web_browser

if TYPE_CHECKING:
    from playwright.sync_api import Page

pytestmark = pytest.mark.browser
browser = test_web_browser.browser
open_page = test_web_browser.open_page
git_port = test_web_browser.git_port
dirty_git_port = test_web_browser.dirty_git_port

OpenPage = Callable[[int], tuple["Page", list[str]]]


def test_a_graph_row_shows_its_first_ref_and_counts_the_rest(
    open_page: OpenPage, git_port: int
) -> None:
    """A row's ref badges filled its subject column and hid the subject on a laptop."""
    page, errors = open_page(git_port)
    page.goto(f"http://127.0.0.1:{git_port}/#/git")
    row = page.locator(".gc-row").first  # the merge: main, origin/main and v2
    row.wait_for()
    assert row.locator(".gref").all_inner_texts() == ["main", "+2"]
    assert row.locator(".gref", has_text="+2").get_attribute("title") == "origin/main, v2"
    assert errors == [], errors


def test_arrow_keys_move_between_graph_rows_and_tab_stops_at_one(
    open_page: OpenPage, git_port: int
) -> None:
    """Arrow keys scrolled the graph under a focused row, and every row was a tab stop.

    A row's name lists the refs, subject, author and date it shows; its lanes are hidden.
    """
    from playwright.sync_api import expect

    page, errors = open_page(git_port)
    page.goto(f"http://127.0.0.1:{git_port}/#/git")
    rows = page.locator(".gc-row")
    rows.first.wait_for()
    expect(page.locator(".git-graph [tabindex='0']")).to_have_count(1)
    rows.first.focus()
    page.keyboard.press("ArrowDown")
    expect(rows.nth(1)).to_be_focused()
    page.keyboard.press("End")
    expect(rows.last).to_be_focused()
    page.keyboard.press("Home")
    expect(rows.first).to_be_focused()
    expect(page.locator(".git-graph [tabindex='0']")).to_have_count(1)
    # HEAD is the merge, where main, origin/main and v2 point.
    name = rows.first.get_attribute("aria-label") or ""
    assert name.startswith("main, origin/main, v2, merge feature, "), name
    assert rows.first.locator("svg").get_attribute("aria-hidden") == "true"
    assert errors == [], errors


def test_a_working_tree_rows_buttons_sit_beside_its_path_not_inside_a_button(
    open_page: OpenPage, dirty_git_port: int
) -> None:
    """A working-tree row was a button holding Stage, Unstage and Discard: nested controls."""
    from playwright.sync_api import expect

    page, errors = open_page(dirty_git_port)
    page.goto(f"http://127.0.0.1:{dirty_git_port}/#/git/worktree")
    row = page.locator(".git-side .gf[data-path='b.txt']")
    row.get_by_role("button", name="Stage", exact=True).wait_for()
    expect(page.locator(".git-side [role=button] button")).to_have_count(0)
    expect(row.get_by_role("button", name="M b.txt", exact=True)).to_have_count(1)
    row.click(position={"x": 4, "y": 4})  # the row's corner, away from the path
    page.wait_for_selector(".git-diff .diff")
    assert errors == [], errors


def test_a_phones_commit_panel_keeps_its_back_button_in_view(
    open_page: OpenPage, git_port: int
) -> None:
    """The back button scrolled away with a long commit panel on a phone."""
    page, errors = open_page(git_port)
    page.set_viewport_size({"width": 390, "height": 360})
    page.goto(f"http://127.0.0.1:{git_port}/#/git")
    page.locator(".gc-row").first.click()
    side = page.locator(".git-side.open")
    page.wait_for_selector(".git-side .gf")
    assert side.evaluate("s => { s.scrollTo(0, s.scrollHeight); return s.scrollTop > 0; }")
    back = page.get_by_role("button", name="\u2039 graph", exact=True)
    panel, button = side.bounding_box(), back.bounding_box()
    assert panel and button and button["y"] >= panel["y"], (panel, button)
    assert errors == [], errors


def test_the_focus_follows_a_file_into_its_diff_and_back(
    open_page: OpenPage, git_port: int
) -> None:
    """Opening or closing a file's diff, or switching its mode, dropped the focus to the page."""
    from playwright.sync_api import expect

    page, errors = open_page(git_port)
    page.goto(f"http://127.0.0.1:{git_port}/#/git")
    page.locator(".gc-row").first.click()  # the merge: b.txt and img.bin
    page.locator(".git-side .gf[data-path='b.txt']").click()
    page.wait_for_selector(".git-diff .diff")
    head = page.locator(".git-diff-head")
    expect(head.get_by_role("button", name="\u2039 files", exact=True)).to_be_focused()
    head.get_by_role("button", name="split", exact=True).click()
    page.wait_for_selector(".git-diff .ds-row")
    expect(head.get_by_role("button", name="split", exact=True)).to_be_focused()
    head.get_by_role("button", name="\u2039 files", exact=True).click()
    focused = "document.activeElement.closest('.gf')?.dataset.path"
    page.wait_for_function(f"{focused} === 'b.txt'")
    expect(page.locator(".git-side .gf.sel")).to_have_attribute("data-path", "b.txt")
    assert errors == [], errors


def test_load_more_hands_the_focus_to_the_first_row_it_adds(
    open_page: OpenPage, git_port: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Load more rebuilt itself when its page came in, and the focus it held dropped to the page."""
    from playwright.sync_api import expect

    monkeypatch.setattr(git_model, "GRAPH_PAGE", 2)
    page, errors = open_page(git_port)
    page.goto(f"http://127.0.0.1:{git_port}/#/git")
    page.get_by_role("button", name="Load more", exact=True).focus()
    page.keyboard.press("Enter")
    expect(page.locator(".gc-row").nth(2)).to_be_focused()
    assert errors == [], errors


def test_closing_a_phones_panel_returns_the_focus_to_the_graph(
    open_page: OpenPage, git_port: int
) -> None:
    """Closing a phone's commit panel hid the focused back button and dropped the focus."""
    from playwright.sync_api import expect

    page, errors = open_page(git_port)
    page.set_viewport_size({"width": 390, "height": 844})
    page.goto(f"http://127.0.0.1:{git_port}/#/git")
    rows = page.locator(".gc-row")
    rows.nth(1).click()
    page.get_by_role("button", name="\u2039 graph", exact=True).click()
    expect(rows.nth(1)).to_be_focused()
    assert errors == [], errors
