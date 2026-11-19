"""The app shell in a browser: the rail, the header, the sign-in form, dialogs and pickers."""

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


def test_a_stored_collapsed_rail_is_collapsed_when_the_rail_is_built(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """A stored collapse was applied at the end of <body>: Chromium painted the rail wide first."""
    from playwright.sync_api import expect

    page, errors = open_page(port)
    page.add_init_script("""localStorage.setItem('a6-rail', 'min');
      new MutationObserver((records, observer) => {
        if (!document.querySelector('aside.rail')) return;
        window.railBuiltCollapsed = document.documentElement.classList.contains('rail-min');
        observer.disconnect();
      }).observe(document, { childList: true, subtree: true });""")
    page.goto(f"http://127.0.0.1:{port}/#/")
    expect(page.locator("#rail-toggle")).to_have_attribute("aria-expanded", "false")
    assert page.evaluate("window.railBuiltCollapsed") is True
    assert not errors
