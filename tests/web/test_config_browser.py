"""The Config page in a browser: the settings grid, its bar, its editors and their states."""

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
port = test_web_browser.port

READY = '[data-key="agent6.config_version"]'


def _config(page: Page, port: int, width: int = 1280) -> None:
    page.set_viewport_size({"width": width, "height": 900})
    page.goto(f"http://127.0.0.1:{port}/#/config")
    page.locator(READY).wait_for()


@pytest.mark.parametrize("width", [1024, 390])
def test_the_config_bar_keeps_every_button_on_the_card(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, width: int
) -> None:
    """The filter and eight buttons shared one line that never wrapped.

    Below 1440px the last buttons sat past the card's edge, cut off, and the filter shrank to a
    sliver too narrow to show what was typed.
    """
    page, errors = open_page(port)
    _config(page, port, width)
    bar = page.locator(".cfg-bar")
    card = bar.evaluate("e => e.closest('.card').getBoundingClientRect().right")
    rights = bar.locator("button").evaluate_all(
        "buttons => buttons.map(b => b.getBoundingClientRect().right)"
    )
    assert len(rights) == 8
    assert max(rights) <= card
    assert bar.locator("input.filter").evaluate("e => e.offsetWidth") == bar.evaluate(
        "e => e.clientWidth"
    )
    assert not errors
