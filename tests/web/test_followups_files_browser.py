from __future__ import annotations

import pathlib
from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest

from tests.web import captures, test_web_browser

if TYPE_CHECKING:
    from playwright.sync_api import Page

pytestmark = pytest.mark.browser
open_page = test_web_browser.open_page
git_port = test_web_browser.git_port


@pytest.mark.parametrize("width", [390, 1440])
def test_refresh_reads_worktree_changes_and_keeps_the_filter(
    open_page: Callable[[int], tuple[Page, list[str]]],
    git_port: int,
    tmp_path: pathlib.Path,
    width: int,
) -> None:
    """Refresh rereads the open file and lists added and deleted files without navigation."""
    from playwright.sync_api import expect

    page, errors = open_page(git_port)
    page.set_viewport_size({"width": width, "height": 900})
    page.goto(f"http://127.0.0.1:{git_port}/#/files/WORKTREE/b.txt")
    expect(page.locator(".files-view [data-line='1']")).to_be_visible()
    (tmp_path / "b.txt").write_text("fresh bytes\n", encoding="utf-8")
    page.get_by_role("button", name="Refresh", exact=True).click()
    expect(page.locator(".files-view [data-line='1']")).to_contain_text("fresh bytes")
    expect(page).to_have_url(f"http://127.0.0.1:{git_port}/#/files/WORKTREE/b.txt")
    captures.save(page, "files-refresh-open")
    if width == 390:
        page.get_by_role("button", name="Files", exact=True).click()
    search = page.get_by_role("searchbox", name="filter files")
    search.fill("dir/")
    (tmp_path / "dir/new.txt").write_text("new\n", encoding="utf-8")
    (tmp_path / "dir/f.txt").unlink()
    page.get_by_role("button", name="Refresh", exact=True).click()
    expect(page.locator('.files-tree [data-path="dir/new.txt"]')).to_be_visible()
    expect(page.locator('.files-tree [data-path="dir/f.txt"]')).to_have_count(0)
    expect(search).to_have_value("dir/")
    captures.save(page, "files-refresh-tree")
    assert not errors
