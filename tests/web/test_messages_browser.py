"""Messages in a browser: toasts, the page under a dialog, errors and stale views."""

from __future__ import annotations

import json
import pathlib
import re
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING

import pytest

from tests.web import signin, test_web_browser

if TYPE_CHECKING:
    from playwright.sync_api import Browser, BrowserContext, Page, Playwright

pytestmark = pytest.mark.browser
open_page = test_web_browser.open_page
port = test_web_browser.port

FAILURE = "merge refused ({}): the checkout has uncommitted changes in `src/stats.py`"
# Whether the element at a box's centre is in the box: nothing covers it, so a click lands.
PRESSABLE = """e => { const r = e.getBoundingClientRect();
  const hit = document.elementFromPoint(r.x + r.width / 2, r.y + r.height / 2);
  return !!hit && e.contains(hit); }"""


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


def _floor(page: Page) -> float:
    """Return where the dock begins, once its model list and the next paint have landed.

    The messages take the dock's height from a resize observation, which lands before a paint.
    """
    from playwright.sync_api import expect

    expect(page.locator(".dock select:disabled")).to_have_count(0)
    return page.evaluate("""() => new Promise(r => requestAnimationFrame(
      () => requestAnimationFrame(
        () => r(document.querySelector('.composer.dock').getBoundingClientRect().top))))""")


def test_held_messages_stay_between_the_header_and_the_dock(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """Seven held failures on a phone climbed past the screen's top edge and over the header.

    The oldest could be reached only by dismissing the newer ones, and a repeat stacked again.
    """
    from playwright.sync_api import expect

    page, errors = open_page(port)
    page.set_viewport_size({"width": 390, "height": 844})
    page.goto(f"http://127.0.0.1:{port}/#/")
    expect(page.locator(".dock textarea")).to_be_visible()
    for i in [*range(7), 3]:
        page.evaluate("text => toast(text, true)", FAILURE.format(i))
    toasts = page.locator("#toasts .toast")
    expect(toasts).to_have_count(7)
    # The repeat moved to the newest place.
    expect(toasts.first).to_contain_text("merge refused (3)")
    floor = _floor(page)
    top = page.locator("header").bounding_box()
    column = page.locator("#toasts").bounding_box()
    assert top and column
    assert column["y"] >= top["y"] + top["height"] - 1
    assert column["y"] + column["height"] <= floor + 1
    assert toasts.first.get_by_role("button", name="Dismiss").evaluate(PRESSABLE)
    oldest = toasts.last.get_by_role("button", name="Dismiss")
    page.locator("#toasts").evaluate("c => { c.scrollTop = -c.scrollHeight; }")  # the oldest's end
    assert oldest.evaluate(PRESSABLE)
    assert not errors


def test_a_failure_taller_than_the_screen_scrolls_inside_its_message(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """A 61-line merge refusal pushed its first lines and its Dismiss above the screen's top."""
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/")
    page.locator(".dock textarea").wait_for()
    lines = "\n".join(f"src/file{i}.py" for i in range(60))
    page.evaluate("text => toast(text, true)", "merge refused: conflicts in\n" + lines)
    message = page.locator("#toasts .toast")
    dismiss = message.get_by_role("button", name="Dismiss")
    floor = _floor(page)
    header = page.locator("header").bounding_box()
    box = message.bounding_box()
    assert header and box
    assert header["y"] + header["height"] - 1 <= box["y"]
    assert box["y"] + box["height"] <= floor + 1
    text = message.locator(":scope > div")
    assert text.evaluate("t => t.scrollHeight > t.clientHeight")
    text.evaluate("t => { t.scrollTop = t.scrollHeight; }")
    assert dismiss.evaluate(PRESSABLE)
    assert not errors


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


def test_a_message_raised_under_a_dialog_shows_above_it_and_takes_the_keyboard(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """A failure raised while the switcher was open sat dimmed and inert under its backdrop.

    Tab never reached its Dismiss; a message put in the dialog went with it when it closed.
    """
    from playwright.sync_api import expect

    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/")
    page.locator(".dock textarea").wait_for()
    page.keyboard.press("Control+K")
    dialog = page.get_by_role("dialog", name="Switch session")
    expect(dialog.get_by_role("combobox")).to_be_focused()
    page.evaluate("text => toast(text, true)", FAILURE.format(1))
    message = page.locator("#toasts .toast")
    assert message.evaluate(PRESSABLE)
    page.keyboard.press("Tab")
    expect(message.get_by_role("button", name="Dismiss")).to_be_focused()
    page.keyboard.press("Enter")
    expect(message).to_have_count(0)
    expect(dialog).to_be_visible()
    page.evaluate("text => toast(text, true)", FAILURE.format(2))
    page.keyboard.press("Escape")
    expect(page.get_by_role("dialog")).to_have_count(0)
    expect(message).to_contain_text("merge refused (2)")
    assert message.evaluate(PRESSABLE)
    assert not errors


@pytest.mark.parametrize("engine", ["browser", "firefox"])
def test_a_request_with_no_answer_names_the_server_in_one_sentence(
    engine: str, open_in: Callable[[str, int], Page], port: int
) -> None:
    """A stopped server read as the browser's own words, which differ between browsers."""
    from playwright.sync_api import Route, expect

    page = open_in(engine, port)

    def refuse(route: Route) -> None:
        route.abort("connectionrefused")

    page.route("**/api/**", refuse)
    page.goto(f"http://127.0.0.1:{port}/#/config")
    expect(page.locator(".problem .err")).to_have_text(
        re.compile(rf"^no answer from the agent6 web server at 127\.0\.0\.1:{port} \(\w.+\)$")
    )


def test_a_401_leaves_the_sign_in_form_alone(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """Ctrl+K on the sign-in form toasted 'not signed in: …' over the form that already says so."""
    from playwright.sync_api import expect

    page, errors = open_page(port)
    page.add_init_script("localStorage.setItem('a6-token', 'expired')")
    page.goto(f"http://127.0.0.1:{port}/#/")
    expect(page.get_by_role("button", name="Sign in", exact=True)).to_be_visible()
    with page.expect_response(lambda r: r.url.endswith("/api/hub")):
        page.keyboard.press("Control+K")
    page.wait_for_timeout(300)
    expect(page.locator("#toasts .toast")).to_have_count(0)
    expect(page.locator("#login")).to_be_visible()
    assert [e for e in errors if not e.startswith("Failed to load resource")] == []
