from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest

from tests.web import captures, test_web_browser

if TYPE_CHECKING:
    from playwright.sync_api import Page

pytestmark = pytest.mark.browser
open_page = test_web_browser.open_page
port = test_web_browser.port


@pytest.mark.parametrize("width", [390, 1440])
def test_maintenance_folds_only_on_narrow_screens(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, width: int
) -> None:
    """Phone maintenance occupies one disclosure line until the user opens it."""
    from playwright.sync_api import expect

    page, errors = open_page(port)
    page.set_viewport_size({"width": width, "height": 900})
    page.goto(f"http://127.0.0.1:{port}/#/")
    prune = page.get_by_role("button", name="Prune merged runs", exact=True)
    if width == 390:
        expect(prune).to_be_hidden()
        page.locator(".maintenance summary").click()
    expect(prune).to_be_visible()
    captures.save(page, "maintenance")
    assert not errors
