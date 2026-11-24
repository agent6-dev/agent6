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


def test_more_dismisses_and_uses_the_inline_actions_when_wide(
    open_page: OpenPage, port: int, tmp_path: pathlib.Path
) -> None:
    """Escape, a click outside and a choice close More. Widening needs no second set of buttons."""
    from playwright import sync_api

    session_logs.write_run(tmp_path, "more", [], live=True)
    page, errors = open_page(port)
    phone: sync_api.ViewportSize = {"width": 390, "height": 640}
    page.set_viewport_size(phone)
    page.goto(f"http://127.0.0.1:{port}/#/session/more")
    bar = page.locator(".run-app .actions")
    more = bar.get_by_text("More", exact=True)
    compact = bar.get_by_role("button", name="Compact context", exact=True)
    more.click()
    sync_api.expect(compact).to_be_visible()
    page.keyboard.press("Escape")
    sync_api.expect(compact).to_be_hidden()
    more.click()
    page.locator(".card-conv h2").click(position={"x": 2, "y": 2})
    sync_api.expect(compact).to_be_hidden()
    more.click()
    compact.click()
    sync_api.expect(compact).to_be_hidden()
    more.click()
    menu = bar.locator('[popover="auto"]:popover-open')
    sync_api.expect(menu).to_be_visible()
    page.set_viewport_size({"width": 1280, "height": 900})
    sync_api.expect(more).to_be_hidden()
    sync_api.expect(menu).to_have_count(0)
    sync_api.expect(compact).to_be_visible()
    sync_api.expect(bar.get_by_text("Compact context", exact=True)).to_have_count(1)
    page.set_viewport_size(phone)
    sync_api.expect(compact).to_be_hidden()
    assert not errors
