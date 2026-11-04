from __future__ import annotations

import pathlib
from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest

from tests.web import test_web_browser

if TYPE_CHECKING:
    from playwright.sync_api import Page

pytestmark = pytest.mark.browser
browser = test_web_browser.browser
open_page = test_web_browser.open_page
git_port = test_web_browser.git_port


def test_show_ignored_is_remembered_and_opens_ignored_markdown(
    git_port: int,
    tmp_path: pathlib.Path,
    open_page: Callable[[int], tuple[Page, list[str]]],
) -> None:
    """The toggle reveals info/exclude files with I badges and remembers both choices."""
    from playwright.sync_api import expect

    (tmp_path / ".git" / "info" / "exclude").write_text("WORK_QUEUE.md\n", encoding="utf-8")
    (tmp_path / "WORK_QUEUE.md").write_text("# Work queue\n", encoding="utf-8")
    page, errors = open_page(git_port)
    page.goto(f"http://127.0.0.1:{git_port}/#/files/WORKTREE")
    toggle = page.get_by_role("button", name="show ignored", exact=True)
    row = page.locator(".files-tree [data-path='WORK_QUEUE.md']")
    expect(toggle).to_have_attribute("aria-pressed", "false")
    expect(row).to_have_count(0)
    toggle.click()
    expect(row).to_be_visible()
    expect(row.locator(".gf-st")).to_have_text("I")
    row.click()
    expect(page.locator(".md-view h1")).to_have_text("Work queue")
    page.reload()
    expect(page.locator(".md-view h1")).to_have_text("Work queue")
    expect(toggle).to_have_attribute("aria-pressed", "true")
    expect(row).to_be_visible()
    toggle.click()
    expect(row).to_have_count(0)
    page.reload()
    expect(toggle).to_have_attribute("aria-pressed", "false")
    expect(row).to_have_count(0)
    page.goto(f"http://127.0.0.1:{git_port}/#/files/HEAD")
    expect(toggle).to_have_count(0)
    assert not errors
