from __future__ import annotations

import pathlib
from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest

from tests.unit import test_git_history
from tests.web import captures, test_files_preview_browser, test_web_browser

if TYPE_CHECKING:
    from playwright.sync_api import Page

pytestmark = pytest.mark.browser
browser = test_files_preview_browser.browser
open_page = test_web_browser.open_page
port = test_web_browser.port
git_port = test_web_browser.git_port


@pytest.mark.parametrize("width", [390, 1440])
def test_git_counts_one_commit_in_the_singular(
    open_page: Callable[[int], tuple[Page, list[str]]],
    port: int,
    tmp_path: pathlib.Path,
    width: int,
) -> None:
    """A graph containing one commit says one commit, not one commits."""
    from playwright.sync_api import expect

    test_git_history._git(tmp_path, "init", "-q", "-b", "main")
    test_git_history._git(tmp_path, "commit", "--allow-empty", "-m", "First commit")
    page, errors = open_page(port)
    page.set_viewport_size({"width": width, "height": 900})
    page.goto(f"http://127.0.0.1:{port}/#/git")
    expect(page.locator(".git-counts").first).to_contain_text("1 commit ·")
    captures.save(page, "git-singular")
    assert not errors


@pytest.mark.parametrize("width", [390, 1440])
def test_compare_fields_have_visible_labels(
    open_page: Callable[[int], tuple[Page, list[str]]], git_port: int, width: int
) -> None:
    """Compare names its populated revisions with visible Base and Head labels."""
    from playwright.sync_api import expect

    page, errors = open_page(git_port)
    page.set_viewport_size({"width": width, "height": 900})
    page.goto(f"http://127.0.0.1:{git_port}/#/git/compare/HEAD~1..HEAD")
    for name in ("Base", "Head"):
        label = page.locator(".gcmp-pick label", has_text=name)
        expect(label).to_be_visible()
        label.click()
        expect(page.get_by_role("combobox", name=name, exact=True)).to_be_focused()
        page.keyboard.press("Escape")
    captures.save(page, "compare-labels")
    assert not errors


def test_compact_compare_keeps_tab_off_the_covered_graph(
    open_page: Callable[[int], tuple[Page, list[str]]], git_port: int
) -> None:
    """The graph and toolbar leave the Tab order until Graph uncovers them."""
    from playwright import sync_api

    page, errors = open_page(git_port)
    page.set_viewport_size({"width": 390, "height": 844})
    page.goto(f"http://127.0.0.1:{git_port}/#/git")
    page.get_by_role("button", name="Compare", exact=True).press("Enter")
    sync_api.expect(page.get_by_role("heading", name="Compare")).to_be_focused()
    back = page.get_by_role("button", name="Graph", exact=True)
    back.press("Shift+Tab")
    assert not page.evaluate("document.activeElement.closest('.git-graph, .git-bar') !== null")
    for region in [".git-graph", ".git-bar"]:
        sync_api.expect(page.locator(region)).to_be_hidden()
    back.press("Enter")
    sync_api.expect(page.locator('.gc-row[tabindex="0"]')).to_be_focused()
    sync_api.expect(page.locator(".git-bar")).to_be_visible()
    assert not errors


@pytest.mark.parametrize("width", [390, 1440])
def test_git_uses_the_page_for_the_graph_and_the_open_diff(
    open_page: Callable[[int], tuple[Page, list[str]]], git_port: int, width: int
) -> None:
    """An unselected graph has no empty side pane, and a diff hides the repository toolbar."""
    from playwright.sync_api import expect

    page, errors = open_page(git_port)
    page.set_viewport_size({"width": width, "height": 900})
    page.goto(f"http://127.0.0.1:{git_port}/#/git")
    expect(page.locator(".gc-row").first).to_be_visible()
    expect(page.locator(".git-side")).to_be_hidden()
    captures.save(page, "git-graph-page")
    page.goto(f"http://127.0.0.1:{git_port}/#/git/commit/HEAD")
    page.locator(".gf-open").first.click()
    expect(page.locator(".git-diff-body .gf-patch")).to_be_visible()
    expect(page.locator(".git-bar")).to_be_hidden()
    captures.save(page, "git-diff-page")
    page.locator(".git-diff-head").get_by_role("button", name="Files", exact=True).click()
    if width == 1440:
        expect(page.locator(".git-bar")).to_be_visible()
    assert not errors
