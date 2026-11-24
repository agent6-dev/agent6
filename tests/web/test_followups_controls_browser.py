from __future__ import annotations

import pathlib
from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest

from tests.web import captures, session_logs, test_files_preview_browser, test_web_browser

if TYPE_CHECKING:
    from playwright.sync_api import Page

pytestmark = pytest.mark.browser
browser = test_files_preview_browser.browser
open_page = test_web_browser.open_page
port = test_web_browser.port


@pytest.mark.parametrize("width", [390, 1440])
def test_occasional_run_actions_fold_on_phones(
    open_page: Callable[[int], tuple[Page, list[str]]],
    port: int,
    tmp_path: pathlib.Path,
    width: int,
) -> None:
    """A live phone run keeps Stop in one row and puts occasional actions behind More."""
    from playwright.sync_api import expect

    session_logs.write_run(tmp_path, "actions-AAA111", [], live=True)
    page, errors = open_page(port)
    page.set_viewport_size({"width": width, "height": 900})
    page.goto(f"http://127.0.0.1:{port}/#/session/actions-AAA111")
    compact = page.get_by_role("button", name="Compact context", exact=True)
    if width == 390:
        expect(compact).to_be_hidden()
        bar = page.locator(".run-app > .actions").bounding_box()
        assert bar and bar["height"] < 45, bar
        page.get_by_role("button", name="More", exact=True).click()
    expect(compact).to_be_visible()
    expect(page.get_by_role("button", name="Stop now", exact=True)).to_be_visible()
    captures.save(page, "run-actions")
    assert not errors
