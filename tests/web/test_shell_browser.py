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


@pytest.mark.parametrize("route", ["#/", "#/config", "#/session/run-A"])
def test_the_sign_in_form_has_its_own_crumb_and_no_current_page(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, route: str
) -> None:
    """The sign-in form kept the crumb or the current nav link of the route it replaced.

    On `#/` the first-load probe's 401 fell through to the Sessions page, which set its crumb.
    """
    from playwright.sync_api import expect

    page, errors = open_page(port)
    page.add_init_script("localStorage.setItem('a6-token', 'expired')")
    page.goto(f"http://127.0.0.1:{port}/{route}")
    expect(page.get_by_role("button", name="Sign in", exact=True)).to_be_visible()
    page.wait_for_load_state("networkidle")
    expect(page.locator("#crumb")).to_have_text("sign in")
    expect(page.locator("[aria-current=page]")).to_have_count(0)
    assert [e for e in errors if not e.startswith("Failed to load resource")] == []


def test_the_rail_brand_sits_on_the_header_line(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """The rail's brand straddled the header's bottom line, its mark 14px below the crumb.

    Expanded and collapsed, the brand row spans the header's height and its mark shares the
    header's centre line and the nav icons' centre.
    """
    from playwright.sync_api import expect

    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/")
    expect(page.locator("#crumb")).to_have_text("sessions")
    boxes = """() => Object.fromEntries(Object.entries({ header: 'header',
      brand: 'aside.rail .rail-brand', mark: 'aside.rail .rail-brand img',
      icon: 'aside.rail .rail-nav .ic' }).map(([k, s]) => {
        const r = document.querySelector(s).getBoundingClientRect();
        return [k, { top: r.top, bottom: r.bottom,
                     cx: r.left + r.width / 2, cy: r.top + r.height / 2 }];
      }))"""
    toggle = page.locator("#rail-toggle")
    for expanded in ("true", "false"):
        if expanded == "false":
            toggle.click()
        expect(toggle).to_have_attribute("aria-expanded", expanded)
        b = page.evaluate(boxes)
        assert b["brand"]["top"] == b["header"]["top"]
        assert b["brand"]["bottom"] == b["header"]["bottom"]
        assert b["mark"]["cy"] == pytest.approx(b["header"]["cy"], abs=1)
        assert b["mark"]["cx"] == pytest.approx(b["icon"]["cx"], abs=0.5)
    assert not errors


def test_every_phone_tab_keeps_its_icon_on_one_line(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """State machines wrapped to two left-aligned lines, lifted its icon 8px and met the edge.

    Every tab's icon shares one top, and each label centres under its icon inside the bar.
    """
    from playwright.sync_api import expect

    page, errors = open_page(port)
    page.set_viewport_size({"width": 360, "height": 800})
    page.goto(f"http://127.0.0.1:{port}/#/")
    expect(page.locator("nav.tabs")).to_be_visible()
    tabs = page.evaluate("""() => [...document.querySelectorAll('nav.tabs a')].map(a => {
      const i = a.querySelector('.ic').getBoundingClientRect();
      const s = a.querySelector('span').getBoundingClientRect();
      return { top: i.top, icon: i.left + i.width / 2, label: s.left + s.width / 2,
               height: s.height, end: s.bottom, bar: a.getBoundingClientRect().bottom }; })""")
    heights = [t["height"] for t in tabs]
    assert max(heights) > 1.5 * min(heights)  # State machines wraps at this width
    assert {t["top"] for t in tabs} == {tabs[0]["top"]}
    for t in tabs:
        assert t["label"] == pytest.approx(t["icon"], abs=0.5)
        assert t["end"] < t["bar"]
    assert not errors
