"""The app shell in a browser: the rail, the header, the sign-in form, dialogs and pickers."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest

from tests.web import test_web_browser

if TYPE_CHECKING:
    from playwright.sync_api import Locator, Page

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


def _sessions(count: int, **fields: object) -> list[dict[str, object]]:
    """Return `count` finished hub rows, run-0 first, each with `fields`."""
    return [
        {
            "session_id": f"run-{i}",
            "mode": "run",
            "status": "passed",
            "status_label": "passed",
            "status_level": "ok",
            "task": f"task {i}",
            **fields,
        }
        for i in range(count)
    ]


def _open_switcher(page: Page, port: int, rows: list[dict[str, object]]) -> Locator:
    """Serve `rows` as the hub, open the Config page, then Ctrl-K's switcher."""
    page.route("**/api/hub", lambda route: route.fulfill(json={"sessions": rows}))
    page.goto(f"http://127.0.0.1:{port}/#/config")
    page.wait_for_selector(".cfg")
    page.keyboard.press("Control+K")
    dialog = page.get_by_role("dialog", name="Switch session")
    dialog.get_by_text(str(rows[0]["session_id"])).first.wait_for()
    return dialog


def test_a_long_switcher_row_ends_in_an_ellipsis_and_its_ids_line_up(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """A long row ran past the switcher's edge, cut mid-word, and each id started after its status.

    The status now ends the line, as on a Sessions row.
    """
    page, errors = open_page(port)
    rows = _sessions(2)
    rows[0].update(task="parse " * 60, status_label="passed · unmerged")
    rows[1].update(status="stale", status_label="stale", status_level="error")
    _open_switcher(page, port, rows)
    lines = page.evaluate("""() => ['run-0', 'run-1'].map(id => {
      const text = [...document.querySelectorAll('dialog[open] span')]
        .find(s => s.textContent.startsWith(id + ' · '));
      const row = text.closest('.ac-item').getBoundingClientRect();
      const box = text.getBoundingClientRect();
      return { left: box.left, right: box.right, rowRight: row.right,
               cut: text.scrollWidth > text.clientWidth,
               ellipsis: getComputedStyle(text).textOverflow }; })""")
    assert lines[0]["left"] == lines[1]["left"]
    for line in lines:
        assert line["right"] <= line["rowRight"]
    assert lines[0]["cut"] and lines[0]["ellipsis"] == "ellipsis"
    assert not errors


def test_a_narrow_switcher_row_keeps_its_session_id_whole(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """On a phone a long status cut the id beside it: "merry-fou…" by "passed · unmerged".

    The status drops under the row's text and still ends the row.
    """
    page, errors = open_page(port)
    page.set_viewport_size({"width": 320, "height": 640})
    sid = "merry-fountain-B7XAJS"
    row = _sessions(1, session_id=sid, status_label="passed · unmerged", task="parse " * 9)
    _open_switcher(page, port, row)
    box = page.evaluate(f"""() => {{
      const text = document.querySelector('dialog[open] .ac-opt .grow');
      const id = document.createRange();
      id.setStart(text.firstChild, 0); id.setEnd(text.firstChild, {len(sid)});
      return [id.getBoundingClientRect().right, text.getBoundingClientRect().right,
              text.nextElementSibling.getBoundingClientRect().right,
              text.closest('.ac-opt').getBoundingClientRect().right]; }}""")
    id_end, text_end, status_end, row_end = box
    assert id_end <= text_end
    assert status_end == pytest.approx(row_end, abs=1)
    assert not errors


def test_the_arrows_keep_the_highlighted_session_in_view_and_named(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """Ten ArrowDown presses left the switcher's highlighted row below its list, out of sight.

    The list is a list box and the field names the highlighted option as its active descendant,
    so a screen reader reads each option the arrows reach.
    """
    from playwright.sync_api import expect

    page, errors = open_page(port)
    dialog = _open_switcher(page, port, _sessions(20))
    for _ in range(10):
        page.keyboard.press("ArrowDown")
    active = dialog.get_by_role("option", selected=True)
    expect(active).to_contain_text("run-10 · run · task 10")
    expect(dialog.get_by_role("combobox")).to_have_attribute(
        "aria-activedescendant", str(active.get_attribute("id"))
    )
    box, shown = active.bounding_box(), dialog.get_by_role("listbox").bounding_box()
    assert box is not None and shown is not None
    assert shown["y"] <= box["y"] and box["y"] + box["height"] <= shown["y"] + shown["height"]
    assert page.evaluate("document.scrollingElement.scrollTop") == 0
    assert not errors


def test_enter_and_space_press_a_switcher_rows_own_button(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """Enter on the second row's Merge opened the highlighted first session; Space did nothing."""
    from playwright.sync_api import Route, expect

    page, errors = open_page(port)
    asked: list[str] = []

    def refuse(route: Route) -> None:
        asked.append(route.request.url.split("/api/", 1)[1])
        route.fulfill(status=409, json={"error": "merge preview failed"})

    page.route("**/api/session/*/merge", refuse)
    rows = _sessions(2, branch="agent6/run", unmerged=True)
    for key in ("Enter", " "):
        dialog = _open_switcher(page, port, rows)
        dialog.get_by_role("button", name="Merge").nth(1).focus()
        page.keyboard.press(key)
        expect(page.get_by_role("dialog")).to_have_count(0)
        expect(page.locator(".toast").last).to_contain_text("merge preview failed")
    assert asked == ["session/run-1/merge", "session/run-1/merge"]
    assert page.evaluate("location.hash") == "#/config"
    assert [e for e in errors if not e.startswith("Failed to load resource")] == []
