# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The Git page driven in a browser: its rows, keyboard, focus and waiting states."""

from __future__ import annotations

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
