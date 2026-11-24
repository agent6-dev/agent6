"""Open a working-tree file from the session switcher."""

from __future__ import annotations

import pathlib
from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest

from agent6.ui.web import git_model
from tests.web import test_files_edit, test_files_preview_browser, test_web_browser

if TYPE_CHECKING:
    from playwright.sync_api import Page

pytestmark = pytest.mark.browser
browser = test_files_preview_browser.browser
open_page = test_web_browser.open_page
served = test_files_edit.served


def test_slash_filters_the_listed_working_tree_and_enter_opens_the_file(
    served: tuple[pathlib.Path, int],
    open_page: Callable[[int], tuple[Page, list[str]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cut listing cannot rule out a file, and typing must not issue another tree read."""
    from playwright import sync_api

    repo, port = served
    (repo / "0 Plan #1.txt").write_text("working copy\n")
    monkeypatch.setattr(git_model, "TREE_MAX", 1)
    page, errors = open_page(port)
    base = f"http://127.0.0.1:{port}/"
    page.goto(base + "#/config")
    sync_api.expect(page.locator(".cfg-bar")).to_be_visible()
    listings: list[str] = []
    page.on("request", lambda r: listings.append(r.url) if "/api/git/tree?" in r.url else None)
    page.keyboard.press("Control+k")
    page.get_by_role("combobox", name="Switch session").fill("/absent")
    picker = page.get_by_role("dialog", name="Go to file: working tree")
    note = picker.locator(".more-note")
    sync_api.expect(note).to_contain_text("no match in the listed files")
    sync_api.expect(note).to_contain_text("list is cut")
    picker.get_by_role("combobox").fill("pLaN")
    sync_api.expect(picker.get_by_role("option")).to_have_text("0 Plan #1.txt")
    sync_api.expect(note).to_contain_text("list is cut")
    assert listings == [base + "api/git/tree?rev=WORKTREE"]
    page.keyboard.press("Enter")
    sync_api.expect(page).to_have_url(base + "#/files/WORKTREE/0%20Plan%20%231.txt")
    sync_api.expect(page.locator(".files-source")).to_contain_text("working copy")
    listings.clear()
    page.keyboard.press("Control+k")
    page.get_by_role("combobox", name="Switch session").fill("/plan")
    sync_api.expect(picker.get_by_role("option")).to_have_text("0 Plan #1.txt")
    assert not listings, "Files already holds the working-tree listing"
    assert not errors
