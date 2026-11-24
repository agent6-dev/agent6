from __future__ import annotations

import pathlib
from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest

from tests.web import session_logs, test_files_preview_browser, test_web_browser

if TYPE_CHECKING:
    from playwright.sync_api import Page

pytestmark = pytest.mark.browser
browser = test_files_preview_browser.browser
open_page = test_web_browser.open_page
port = test_web_browser.port
OpenPage = Callable[[int], tuple["Page", list[str]]]


def test_directives_fit_a_short_viewport(
    open_page: OpenPage, port: int, tmp_path: pathlib.Path
) -> None:
    """The list box fits between the entry and the edge of a short screen."""
    from playwright import sync_api

    session_logs.write_run(tmp_path, "short", [], live=True)
    page, errors = open_page(port)
    page.set_viewport_size({"width": 960, "height": 240})
    page.goto(f"http://127.0.0.1:{port}/#/session/short")
    entry = page.get_by_label("Message to the run")
    entry.fill("/")
    popup = page.get_by_role("listbox", name="Directives")
    sync_api.expect(popup).to_be_in_viewport(ratio=1)
    box, field = popup.bounding_box(), entry.bounding_box()
    assert box and field
    assert box["y"] + box["height"] <= field["y"] or box["y"] >= field["y"] + field["height"]
    assert not errors


def test_arrows_reveal_the_active_directive(
    open_page: OpenPage, port: int, tmp_path: pathlib.Path
) -> None:
    """An active descendant below the scroll box has to scroll into view without taking focus."""
    from playwright import sync_api

    session_logs.write_run(tmp_path, "suggest", [], live=True)
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/session/suggest")
    entry = page.get_by_label("Message to the run")
    entry.fill("/")
    popup = page.get_by_role("listbox", name="Directives")
    options = popup.get_by_role("option")
    sync_api.expect(options.last).to_contain_text("/shells")
    for _ in range(options.count()):
        entry.press("ArrowDown")
    sync_api.expect(options.last).to_have_attribute("aria-selected", "true")
    sync_api.expect(options.last).to_be_in_viewport(ratio=1)
    sync_api.expect(entry).to_be_focused()
    entry.press("ArrowDown")
    sync_api.expect(options.first).to_be_in_viewport(ratio=1)
    assert not errors
