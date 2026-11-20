from __future__ import annotations

import pathlib
from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest

from tests.web import captures, session_logs, test_web_browser

if TYPE_CHECKING:
    from playwright.sync_api import Page

pytestmark = pytest.mark.browser
open_page = test_web_browser.open_page
port = test_web_browser.port


@pytest.mark.parametrize("width", [390, 1440])
def test_widget_menu_fits_at_double_text_size(
    open_page: Callable[[int], tuple[Page, list[str]]],
    port: int,
    tmp_path: pathlib.Path,
    width: int,
) -> None:
    """The widget menu scrolls within the viewport at 200 percent text size."""
    from playwright.sync_api import expect

    session_logs.write_run(tmp_path, "menu-AAA111", [])
    page, errors = open_page(port)
    page.set_viewport_size({"width": width, "height": 600})
    page.context.new_cdp_session(page).send("Page.setFontSizes", {"fontSizes": {"standard": 32}})
    page.goto(f"http://127.0.0.1:{port}/#/session/menu-AAA111")
    page.get_by_role("button", name="Widgets", exact=True).click()
    menu = page.locator(".wmenu")
    box = menu.bounding_box()
    assert box and box["x"] >= 0 and box["x"] + box["width"] <= width
    assert box["y"] + box["height"] <= 600, box
    menu.get_by_role("button", name="Event log", exact=True).scroll_into_view_if_needed()
    expect(menu.get_by_role("button", name="Event log", exact=True)).to_be_in_viewport()
    captures.save(page, "widget-menu-zoom")
    menu.get_by_role("button", name="Event log", exact=True).click()
    expect(menu).to_be_hidden()
    assert not errors
