"""Messages in a browser: toasts, the page under a dialog, errors and stale views."""

from __future__ import annotations

import json
import pathlib
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING

import pytest

from tests.web import signin, test_web_browser

if TYPE_CHECKING:
    from playwright.sync_api import Browser, BrowserContext, Page, Playwright

pytestmark = pytest.mark.browser
open_page = test_web_browser.open_page
port = test_web_browser.port


@pytest.fixture(scope="module")
def pw() -> Iterator[Playwright]:
    """One Playwright for the module's two engines."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        yield playwright


@pytest.fixture(scope="module")
def browser(pw: Playwright) -> Iterator[Browser]:
    """One Chromium for the module."""
    chromium = pw.chromium.launch()
    try:
        yield chromium
    finally:
        chromium.close()


@pytest.fixture(scope="module")
def firefox(pw: Playwright) -> Iterator[Browser]:
    """One Firefox for the module, where the page behind a dialog scrolled."""
    if not pathlib.Path(pw.firefox.executable_path).exists():
        pytest.skip(
            "Playwright's Firefox is not installed: run `uv run playwright install firefox`"
        )
    engine = pw.firefox.launch()
    try:
        yield engine
    finally:
        engine.close()


@pytest.fixture
def open_in(request: pytest.FixtureRequest) -> Iterator[Callable[[str, int], Page]]:
    """Open a signed-in desktop page in the named engine's fixture, on a server's port."""
    contexts: list[BrowserContext] = []

    def _open(engine: str, port: int) -> Page:
        context = request.getfixturevalue(engine).new_context(
            viewport={"width": 1280, "height": 900}
        )
        contexts.append(context)
        context.add_init_script(
            f"localStorage.setItem('a6-token', {json.dumps(signin.token(port))})"
        )
        return context.new_page()

    try:
        yield _open
    finally:
        for context in contexts:
            context.close()


@pytest.mark.parametrize("engine", ["browser", "firefox"])
def test_a_dialog_holds_the_page_behind_it_still(
    engine: str, open_in: Callable[[str, int], Page], port: int
) -> None:
    """In Firefox a wheel over a dialog's backdrop, or End in it, scrolled the page behind it.

    The page stayed at the new place after the dialog closed.
    """
    from playwright.sync_api import expect

    page = open_in(engine, port)
    page.goto(f"http://127.0.0.1:{port}/#/config")
    page.locator(".cfg").first.wait_for()
    page.keyboard.press("?")
    expect(page.get_by_role("dialog", name="Keyboard shortcuts")).to_be_visible()
    page.mouse.move(80, 700)
    page.mouse.wheel(0, 1500)
    page.keyboard.press("PageDown")
    page.keyboard.press("End")
    page.wait_for_timeout(400)
    assert page.evaluate("document.scrollingElement.scrollTop") == 0
    page.keyboard.press("Escape")
    expect(page.get_by_role("dialog")).to_have_count(0)
    assert page.evaluate("document.scrollingElement.scrollTop") == 0


def test_a_dialog_moves_nothing_sideways_where_scrollbars_take_room(
    pw: Playwright, port: int
) -> None:
    """Locking a page under a dialog took its scrollbar away, and everything slid into its room.

    A page with no scrollbar must not take a gutter for one either.
    """
    from playwright.sync_api import expect

    engine = pw.chromium.launch(ignore_default_args=["--hide-scrollbars"])
    try:
        page = engine.new_page(viewport={"width": 1280, "height": 900})
        page.add_init_script(f"localStorage.setItem('a6-token', {json.dumps(signin.token(port))})")
        edges = """() => [innerWidth - document.documentElement.clientWidth,
          document.querySelector('header').getBoundingClientRect().right]"""
        for route, ready, bar in (("#/config", ".cfg", True), ("#/", ".dock textarea", False)):
            page.goto(f"http://127.0.0.1:{port}/{route}")
            page.locator(ready).first.wait_for()
            width, right = page.evaluate(edges)
            assert (width > 0) is bar
            page.keyboard.press("Control+K")
            expect(page.get_by_role("dialog", name="Switch session")).to_be_visible()
            assert page.evaluate(edges)[1] == right, route
            page.keyboard.press("Escape")
            expect(page.get_by_role("dialog")).to_have_count(0)
            assert page.evaluate(edges)[1] == right, route
    finally:
        engine.close()
