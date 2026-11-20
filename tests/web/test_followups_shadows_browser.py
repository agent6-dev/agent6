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
def test_dialog_shadow_follows_the_theme(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, width: int
) -> None:
    """Light and dark surfaces use different shadow strength."""
    page, errors = open_page(port)
    page.set_viewport_size({"width": width, "height": 900})
    page.goto(f"http://127.0.0.1:{port}/#/")
    page.keyboard.press("?")
    dialog = page.get_by_role("dialog")
    dialog.wait_for()
    shadows: list[str] = []
    for theme in ("light", "dark"):
        page.locator("html").evaluate("(e, theme) => e.className = theme", theme)
        shadows.append(dialog.evaluate("e => getComputedStyle(e).boxShadow"))
        captures.save(page, "shadow-" + theme)
    assert shadows[0] != shadows[1], shadows
    assert not errors
