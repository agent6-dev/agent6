"""The Files page in a browser: its tree, its bar and the open file's head."""

from __future__ import annotations

import pathlib
from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest

from tests.unit.test_git_history import _git
from tests.web import test_web_browser

if TYPE_CHECKING:
    from playwright.sync_api import Page

pytestmark = pytest.mark.browser
browser = test_web_browser.browser
open_page = test_web_browser.open_page
git_port = test_web_browser.git_port

DEEP = "deep/alpha-folder-name/beta-folder-name/gamma"
LEAF = DEEP + "/a-long-file-name-for-a-narrow-tree.txt"
# Where a row's first item, the caret, starts: a folded caret is turned, so its own box is not.
_CONTENT_LEFT = "r => r.getBoundingClientRect().left + parseFloat(getComputedStyle(r).paddingLeft)"


@pytest.fixture
def big_tree(git_port: int, tmp_path: pathlib.Path) -> int:
    """Over 200 files, so the tree starts folded, and a deep folder with long names."""
    (tmp_path / "many").mkdir()
    for i in range(210):
        (tmp_path / "many" / f"f{i:03}.txt").write_text(f"{i}\n", encoding="utf-8")
    (tmp_path / DEEP / "delta").mkdir(parents=True)
    (tmp_path / LEAF).write_text("leaf\n", encoding="utf-8")
    (tmp_path / DEEP / "delta" / "x.txt").write_text("x\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-m", "a big tree")
    return git_port


def _row(path: str) -> str:
    return f".files-tree [data-path='{path}']"


def _open_deep(page: Page) -> None:
    """Open the folders down to the long file by clicking them."""
    parts = DEEP.split("/")
    for depth in range(1, len(parts) + 1):
        row = page.locator(_row("/".join(parts[:depth])))
        if row.get_attribute("aria-expanded") == "false":
            row.click()


def test_a_narrow_tree_keeps_each_name_on_one_line(
    open_page: Callable[[int], tuple[Page, list[str]]], big_tree: int
) -> None:
    """At the tree's narrowest, names broke down to a character per line beside the notes."""
    page, errors = open_page(big_tree)
    page.add_init_script("localStorage.setItem('a6-files-tree-w', '180px')")
    page.goto(f"http://127.0.0.1:{big_tree}/#/files/HEAD")
    page.wait_for_selector(_row("deep"))
    _open_deep(page)
    heights = page.locator(".files-tree .gtree-dir:visible, .files-tree .gf:visible").evaluate_all(
        "rows => rows.map(r => r.getBoundingClientRect().height)"
    )
    assert heights and max(heights) < 1.5 * min(heights), heights
    assert page.locator(".files-tree .gtree-note:visible").count() == 0
    assert page.locator(_row(LEAF)).get_attribute("title") == LEAF
    assert page.locator(_row(DEEP)).get_attribute("title") == DEEP
    assert not errors
