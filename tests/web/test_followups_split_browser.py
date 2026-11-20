from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest

from tests.web import captures, test_web_browser

if TYPE_CHECKING:
    from playwright.sync_api import Page

pytestmark = pytest.mark.browser
open_page = test_web_browser.open_page
git_port = test_web_browser.git_port


def test_split_follows_the_pane_through_sidebar_and_window_resizes(
    open_page: Callable[[int], tuple[Page, list[str]]], git_port: int
) -> None:
    """A narrow diff on a desktop falls back to inline and restores the saved split choice."""
    from playwright.sync_api import expect

    page, errors = open_page(git_port)
    page.set_viewport_size({"width": 820, "height": 900})
    page.add_init_script("localStorage.setItem('a6-diff-mode', 'split')")
    page.goto(f"http://127.0.0.1:{git_port}/#/git/compare/HEAD~1..HEAD")
    page.locator(".gf-open").first.click()
    split = page.get_by_role("button", name="split", exact=True)
    expect(split).to_be_disabled()
    expect(page.locator(".git-diff-body")).to_contain_text("wider pane")
    page.get_by_role("button", name="Hide the sidebar", exact=True).click()
    expect(split).to_be_enabled()
    expect(page.locator(".dsplit")).to_be_visible()
    page.set_viewport_size({"width": 1440, "height": 900})
    captures.save(page, "split-pane")
    page.set_viewport_size({"width": 390, "height": 900})
    expect(split).to_be_disabled()
    expect(page.locator(".dsplit")).to_have_count(0)
    captures.save(page, "split-pane")
    assert not errors
