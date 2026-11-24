from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest

from tests.web import captures, test_files_preview_browser, test_web_browser

if TYPE_CHECKING:
    from playwright.sync_api import Page, Route

pytestmark = pytest.mark.browser
browser = test_files_preview_browser.browser
open_page = test_web_browser.open_page
port = test_web_browser.port


@pytest.mark.parametrize("width", [390, 1440])
def test_config_keeps_its_filter_and_names_its_preset(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, width: int
) -> None:
    """The filter stays reachable while scrolling and the preset button states its value."""
    from playwright.sync_api import expect

    page, errors = open_page(port)
    page.set_viewport_size({"width": width, "height": 900})
    page.goto(f"http://127.0.0.1:{port}/#/config")
    expect(page.get_by_role("button", name="Preset: none", exact=True)).to_be_visible()
    page.locator(".cfg-section").last.scroll_into_view_if_needed()
    expect(page.locator(".cfg-bar input.filter")).to_be_in_viewport()
    captures.save(page, "config-sticky")
    assert not errors


def test_config_focus_clears_the_wrapped_toolbar(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """Shift+Tab keeps the focused row below the toolbar as the buttons wrap."""
    from playwright import sync_api

    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/config")
    rows = page.locator(".cfg-section", has=page.locator("#cfg-section-sandbox")).locator(
        "tr[data-key]"
    )
    current = rows.nth(3).get_by_role("button")
    for width, height in [(1280, 960), (640, 480)]:  # the CSS viewport at 100% and 200% zoom
        page.set_viewport_size({"width": width, "height": height})
        current.focus()
        rows.nth(3).evaluate("""r => window.scrollBy(0, r.getBoundingClientRect().top
            - document.querySelector('.cfg-bar').getBoundingClientRect().bottom - 2)""")
        page.keyboard.press("Shift+Tab")
        sync_api.expect(rows.nth(2).get_by_role("button")).to_be_focused()
        assert rows.nth(2).evaluate("""r => r.getBoundingClientRect().top >=
            document.querySelector('.cfg-bar').getBoundingClientRect().bottom""")
    assert not errors


@pytest.mark.parametrize("width", [390, 1440])
def test_config_marks_required_fields_and_pending_saves(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, width: int
) -> None:
    """Add names its required field and Save says Saving while the response is pending."""
    from playwright.sync_api import expect

    page, errors = open_page(port)
    page.set_viewport_size({"width": width, "height": 900})
    page.goto(f"http://127.0.0.1:{port}/#/config")
    page.get_by_role("button", name="Add preset…", exact=True).click()
    expect(page.get_by_label("name (required)", exact=True)).to_have_attribute("required", "")
    page.keyboard.press("Escape")
    page.get_by_role("button", name="Add provider…", exact=True).click()
    expect(page.get_by_label("name (required)", exact=True)).to_have_attribute("name", "name")
    page.keyboard.press("Escape")
    page.get_by_role("button", name="edit web.port", exact=True).click()
    held: list[Route] = []

    def hold(route: Route) -> None:
        held.append(route)

    page.route("**/api/config", hold)
    page.get_by_role("button", name="Save", exact=True).click()
    expect(page.get_by_role("button", name="Saving…", exact=True)).to_be_disabled()
    captures.save(page, "config-saving")
    assert held
    held.pop().fulfill(status=422, json={"error": "Rejected for this test"})
    expect(page.get_by_role("button", name="Save", exact=True)).to_be_enabled()
    assert [e for e in errors if "422" not in e] == []
