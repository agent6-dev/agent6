from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest

from tests.web import test_files_preview_browser, test_web_browser

if TYPE_CHECKING:
    from playwright.sync_api import Page

pytestmark = pytest.mark.browser
browser = test_files_preview_browser.browser
open_page = test_web_browser.open_page
dirty_git_port = test_web_browser.dirty_git_port


def test_compare_counts_commits_on_each_side(
    open_page: Callable[[int], tuple[Page, list[str]]], dirty_git_port: int
) -> None:
    """Compare counts commits rather than changed files or rows on the graph page."""
    from playwright import sync_api

    page, errors = open_page(dirty_git_port)
    for revisions, count in [
        ("v1..main", "3 commits"),
        ("v1..feature", "1 commit"),
        ("main..feature", "2 behind"),
        ("feature..main~1", "1 ahead, 1 behind"),
        ("main..HEAD", "0 commits"),
        ("v1..WORKTREE", "3 commits and uncommitted changes"),
    ]:
        page.goto(f"http://127.0.0.1:{dirty_git_port}/#/git/compare/{revisions}")
        sync_api.expect(page.locator(".gcmp-result").get_by_text(count, exact=True)).to_be_visible()
    assert not errors
