from __future__ import annotations

import os
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
git_port = test_web_browser.git_port


def test_file_rows_link_to_the_selected_revision_in_a_new_tab(
    open_page: Callable[[int], tuple[Page, list[str]]], git_port: int, tmp_path: pathlib.Path
) -> None:
    """A middle click opens the row's encoded path without selecting it in this tab."""
    from playwright import sync_api

    (tmp_path / "dir" / "a #%.txt").write_text("committed text\n")
    test_git_history._git(tmp_path, "add", "dir/a #%.txt")
    test_git_history._git(tmp_path, "commit", "-qm", "file with a fragment delimiter")
    test_git_history._git(tmp_path, "branch", "topic/files")
    (tmp_path / "dir" / "a #%.txt").write_text("disk text\n")
    page, errors = open_page(git_port)
    base = f"http://127.0.0.1:{git_port}/"
    listing = "#/files/refs%2Fheads%2Ftopic%2Ffiles"
    href = listing + "/dir/a%20%23%25.txt"
    page.goto(base + listing)
    row = page.get_by_role("treeitem", name="dir/a #%.txt", exact=True)
    sync_api.expect(row).to_have_attribute("href", href)
    assert row.evaluate("e => e.matches('a[href]')")
    with page.context.expect_page() as opened:
        row.click(button="middle")
    tab = opened.value
    sync_api.expect(tab).to_have_url(base + href)
    sync_api.expect(tab.locator(".files-source")).to_contain_text("committed text")
    sync_api.expect(page).to_have_url(base + listing)
    sync_api.expect(row).to_have_attribute("aria-selected", "false")
    tab.close()
    row.press("Space")
    sync_api.expect(page).to_have_url(base + href)
    sync_api.expect(row).to_have_attribute("aria-selected", "true")
    assert not errors


def test_a_non_utf8_name_does_not_hide_or_link_to_another_file(
    open_page: Callable[[int], tuple[Page, list[str]]], git_port: int, tmp_path: pathlib.Path
) -> None:
    """A Git name that has no UTF-8 URL stays distinct from a literal replacement character."""
    from playwright import sync_api

    raw = os.fsdecode(b"caf\xe9.txt")
    (tmp_path / raw).write_text("raw bytes\n")
    (tmp_path / "caf\ufffd.txt").write_text("Unicode name\n")
    (tmp_path / "alias").symlink_to(raw)
    page, errors = open_page(git_port)
    page.goto(f"http://127.0.0.1:{git_port}/#/files")
    tree = page.get_by_role("tree", name="Files")
    disabled = tree.locator('[aria-disabled="true"]')
    sync_api.expect(disabled).to_be_visible()
    assert not disabled.evaluate("e => e.hasAttribute('href')")
    sync_api.expect(disabled).to_contain_text("not UTF-8")
    row = tree.get_by_role("treeitem", name="caf\ufffd.txt", exact=True)
    sync_api.expect(row).to_have_attribute("href", "#/files/WORKTREE/caf%EF%BF%BD.txt")
    row.click()
    sync_api.expect(page.locator(".files-source")).to_contain_text("Unicode name")
    tree.get_by_role("treeitem", name="alias", exact=True).click()
    sync_api.expect(page.locator(".files-text")).to_contain_text("symlink to")
    sync_api.expect(page.locator(".files-text a")).to_have_count(0)
    assert not errors


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
    if width == 1440:
        expect(page.locator(".files-view [data-line='1']")).to_contain_text("fresh bytes")
    captures.save(page, "files-refresh-tree")
    assert not errors
