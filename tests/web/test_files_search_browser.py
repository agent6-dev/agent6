"""Search file contents at the revision the Files page shows."""

from __future__ import annotations

import pathlib
import re
from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest

from agent6.ui.web import git_model
from tests.unit import test_git_history
from tests.web import test_files_edit, test_files_preview_browser, test_web_browser

if TYPE_CHECKING:
    from playwright.sync_api import Page

pytestmark = pytest.mark.browser
browser = test_files_preview_browser.browser
open_page = test_web_browser.open_page
served = test_files_edit.served


@pytest.mark.parametrize("revision", ["HEAD", "WORKTREE"])
def test_search_opens_the_matching_source_line(
    served: tuple[pathlib.Path, int],
    open_page: Callable[[int], tuple[Page, list[str]]],
    revision: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hit beyond the default preview opens at its source line, even in Markdown."""
    from playwright import sync_api

    repo, port = served
    monkeypatch.setattr(git_model, "TEXT_MAX_BYTES", 512)
    path = repo / "guide #1.md"
    text = "# Guide\n" + "some text\n" * 150
    path.write_text(text + "committed -[x]\n")
    test_git_history._git(repo, "add", path.name)
    test_git_history._git(repo, "commit", "-qm", "guide")
    sha = test_git_history._git(repo, "rev-parse", "HEAD")
    path.write_text(text + "draft\nunsaved -[x]\n")
    rev, line, match = (
        (sha, 152, "committed -[x]") if revision == "HEAD" else (revision, 153, "unsaved -[x]")
    )
    page, errors = open_page(port)
    base = f"http://127.0.0.1:{port}/#/files/"
    page.goto(base + revision)
    field = page.get_by_role("searchbox", name="Search file contents")
    sync_api.expect(field).to_be_visible()
    field.fill("-[x]")
    field.press("Enter")
    hit = page.get_by_role("link", name=re.compile(rf"guide #1\.md:{line}\b"))
    sync_api.expect(hit).to_contain_text(match)
    hit.click()
    sync_api.expect(page).to_have_url(base + f"{rev}/guide%20%231.md?line={line}")
    row = page.locator(f".files-source [data-line='{line}']")
    sync_api.expect(row).to_contain_text(match)
    sync_api.expect(row).to_be_in_viewport()
    sync_api.expect(row).to_have_attribute("aria-current", "location")
    sync_api.expect(row).to_be_focused()
    page.reload()
    sync_api.expect(row).to_be_in_viewport()
    assert not errors
