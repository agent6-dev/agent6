"""The app shell in a browser: the rail, the header, the sign-in form, dialogs and pickers."""

from __future__ import annotations

import json
import os
import pathlib
import re
from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest

from agent6 import paths
from agent6.sessions import process
from tests.web import test_web_actions, test_web_browser

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
    expect(page.locator("#crumb")).to_have_text("Sign in")
    expect(page.locator("[aria-current=page]")).to_have_count(0)
    assert [e for e in errors if not e.startswith("Failed to load resource")] == []


def test_a_route_with_no_page_says_so_under_its_own_crumb(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """`#/nope` showed a bare "not found" under a blank crumb, with the last page current.

    It offered no way back, and the browser's tab read "agent6" on every page.
    """
    from playwright.sync_api import expect

    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/config")
    expect(page).to_have_title(re.compile("^config · agent6$", re.I))
    page.evaluate("location.hash = '#/nope'")
    expect(page.locator("#view")).to_contain_text("No page lives at #/nope.")
    expect(page.locator("#crumb")).to_have_text("Not found")
    expect(page).to_have_title("Not found · agent6")
    expect(page.locator("[aria-current=page]")).to_have_count(0)
    page.get_by_role("link", name="Back to Sessions").click()
    expect(page.locator(".dock textarea")).to_be_visible()
    expect(page).to_have_title(re.compile("^sessions · agent6$", re.I))
    assert not errors


def test_an_unknown_session_reads_not_found_under_its_section(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """An unknown session opened from Config showed a bare error line under the crumb "config".

    It reads Not found under the Sessions crumb, with no Try again: a retry cannot bring it back.
    """
    from playwright.sync_api import expect

    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/config")
    expect(page.locator("#crumb")).to_have_text(re.compile("config", re.I))
    page.evaluate("location.hash = '#/session/nobody'")
    problem = page.locator("#view .problem")
    expect(problem.get_by_role("heading")).to_have_text("Not found")
    expect(problem).to_contain_text("no session 'nobody'")
    expect(page.locator("#crumb")).to_have_text("Sessions")
    expect(page.locator("[aria-current=page]")).to_have_count(2)  # the rail and the phone tabs
    expect(problem.get_by_role("button", name="Try again")).to_have_count(0)
    expect(problem.get_by_role("link", name="Back to Sessions")).to_have_attribute("href", "#/")
    assert [e for e in errors if not e.startswith("Failed to load resource")] == []


def test_a_page_that_fails_to_load_offers_a_retry(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """A failed load showed a bare error line, with no way to try again."""
    from playwright.sync_api import expect

    page, errors = open_page(port)
    page.route("**/api/config", lambda route: route.abort(), times=1)
    page.goto(f"http://127.0.0.1:{port}/#/config")
    problem = page.locator("#view .problem")
    expect(problem.get_by_role("heading")).to_have_text("Could not load this page")
    expect(page.locator("#crumb")).to_have_text(re.compile("^config$", re.I))
    problem.get_by_role("button", name="Try again").click()
    expect(page.locator(".cfg").first).to_be_visible()
    assert [e for e in errors if not e.startswith("Failed to load resource")] == []


def test_the_old_view_is_marked_busy_until_the_next_page_paints(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """Config's tab and crumb showed over the Sessions list for a slow load, with no sign of it."""
    from playwright.sync_api import Route, expect

    page, errors = open_page(port)
    held: list[Route] = []

    def hold(route: Route) -> None:
        held.append(route)

    page.route("**/api/config", hold)
    page.goto(f"http://127.0.0.1:{port}/#/")
    expect(page.locator(".dock textarea")).to_be_visible()
    page.get_by_role("link", name="Config").first.click()
    view = page.locator("#view")
    expect(view).to_have_attribute("aria-busy", "true")
    expect(view.locator(".dock textarea")).to_be_visible()
    held[0].continue_()
    expect(page.locator(".cfg").first).to_be_visible()
    expect(view).not_to_have_attribute("aria-busy", "true")
    assert not errors


def test_a_refused_secret_is_read_out_and_marks_the_field(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """The hint and the refusal showed literal backticks, and nothing announced the refusal.

    A command in either is code; the refusal is an alert the field names as its description.
    """
    from playwright.sync_api import expect

    page, errors = open_page(port)
    page.add_init_script("localStorage.setItem('a6-token', 'expired')")
    page.goto(f"http://127.0.0.1:{port}/#/")
    form = page.locator("#login")
    field = form.get_by_role("textbox", name="secret")
    expect(form.locator(".muted code")).to_have_text("agent6 web")
    field.fill("wrong words")
    form.get_by_role("button", name="Sign in").click()
    refusal = form.get_by_role("alert")
    expect(refusal).to_have_text("that is not the secret this agent6 web was started with")
    expect(refusal.locator("code")).to_have_text("agent6 web")
    expect(field).to_have_attribute("aria-invalid", "true")
    expect(field).to_have_accessible_description(
        "that is not the secret this agent6 web was started with"
    )
    field.press_sequentially("s")
    expect(field).not_to_have_attribute("aria-invalid", "true")
    assert "`" not in str(form.text_content())
    assert [e for e in errors if not e.startswith("Failed to load resource")] == []


def test_a_message_shows_a_backticked_command_as_code(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """A toast showed `agent6 sessions` with its backticks, as Markdown source."""
    from playwright.sync_api import expect

    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/")
    expect(page.locator(".dock textarea")).to_be_visible()
    page.evaluate("toast('the resume has not started yet; check `agent6 sessions`', true)")
    message = page.get_by_role("alert")
    expect(message).to_have_text("the resume has not started yet; check agent6 sessions")
    expect(message.locator("code")).to_have_text("agent6 sessions")
    assert not errors


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
    expect(page.locator("#crumb")).to_have_text(re.compile("sessions", re.I))
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


def test_collapsing_the_rail_moves_no_icon(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """A row was 41px with its label and 40px without, so collapsing lifted each icon 1px more."""
    from playwright.sync_api import expect

    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/")
    expect(page.locator("aside.rail .rail-nav a[aria-current=page]")).to_be_visible()
    tops = """() => [...document.querySelectorAll('aside.rail .rail-nav a, aside.rail .rail-row')]
      .map(row => [row.getBoundingClientRect().height,
                   row.querySelector('.ic').getBoundingClientRect().top])"""
    expanded = page.evaluate(tops)
    page.locator("#rail-toggle").click()
    expect(page.locator("#rail-toggle")).to_have_attribute("aria-expanded", "false")
    assert page.evaluate(tops) == expanded
    assert {height for height, _ in expanded} == {40}
    assert not errors


@pytest.mark.parametrize(("width", "bar"), [(1280, "aside.rail .rail-nav"), (390, "nav.tabs")])
def test_the_open_pages_link_is_marked_by_more_than_its_colour(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, width: int, bar: str
) -> None:
    """The phone tabs marked the open page by hue alone, and a hovered rail link took its fill.

    The open page's link has an accent bar and a semibold label; a hovered link has neither.
    """
    from playwright.sync_api import expect

    page, errors = open_page(port)
    page.set_viewport_size({"width": width, "height": 800})
    page.goto(f"http://127.0.0.1:{port}/#/")
    expect(page.locator(f"{bar} a[aria-current=page]")).to_have_attribute("data-tab", "hub")
    page.locator(f"{bar} a[data-tab=git]").hover()
    cues = page.evaluate(f"""() => Object.fromEntries([...document.querySelectorAll('{bar} a')]
      .map(a => [a.dataset.tab, [getComputedStyle(a, '::before').content !== 'none',
                                 +getComputedStyle(a).fontWeight >= 600]]))""")
    assert cues == {
        "hub": [True, True],
        "git": [False, False],
        "files": [False, False],
        "machines": [False, False],
        "config": [False, False],
    }
    assert not errors


def test_a_banner_sits_under_the_header_with_a_full_size_close_button(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, tmp_path: pathlib.Path
) -> None:
    """The first banner touched the header line, and its close button was a 32x20 target."""
    from playwright.sync_api import expect

    machine = paths.state_dir(tmp_path) / "machines" / "noting-machine"
    (machine / "states" / "0001-work").mkdir(parents=True)
    (machine / "machine.asm.toml").write_text(test_web_actions.TINY, encoding="utf-8")
    (machine / "journal.jsonl").write_text("", encoding="utf-8")
    process.write_worker_pid(machine, os.getpid())
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/machine/noting-machine")
    expect(page.get_by_role("button", name="Steer")).to_be_visible()
    note = {
        "type": "machine.notify",
        "ts": "2026-11-13T10:00:00Z",
        "state": "route",
        "message": "Look",
    }
    with (machine / "journal.jsonl").open("a") as stream:
        stream.write(json.dumps(note) + "\n")
    banner = page.locator(".notif-banner")
    expect(banner).to_have_count(1)
    header, box = page.locator("header").bounding_box(), banner.bounding_box()
    close = banner.get_by_role("button").bounding_box()
    assert header is not None and box is not None and close is not None
    assert box["y"] - (header["y"] + header["height"]) == 12
    assert close["width"] == close["height"] == 32
    assert close["y"] >= box["y"] and close["y"] + close["height"] <= box["y"] + box["height"]
    assert not errors


def test_the_keyboard_list_spells_each_key_one_way_as_keycaps(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """The list spelled Ctrl-K beside Ctrl+Enter, in plain text over three narrow columns.

    Each key is a keycap, a chord joins its keys with +, and the rows group under where they work.
    """
    from playwright.sync_api import expect

    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/")
    expect(page.locator(".dock textarea")).to_be_visible()
    page.keyboard.press("?")
    table = page.get_by_role("dialog", name="Keyboard shortcuts").locator("table.keys")
    rows = table.evaluate("""t => [...t.querySelectorAll('tr')].map(tr => tr.querySelector('th')
      ? tr.textContent : [...tr.querySelectorAll('td.key kbd')].map(k => k.textContent))""")
    assert rows[:3] == ["Anywhere", ["Ctrl", "K"], "Outside a field"]
    assert ["Ctrl", "Enter"] in rows and ["Ctrl", "R"] in rows and ["Enter", "Space"] in rows
    labels = table.locator("td.key").all_text_contents()
    assert labels and not [label for label in labels if re.search(r"\w-\w", label)]
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
    # One box per line of each label, from a range over its text.
    tabs = page.evaluate("""() => [...document.querySelectorAll('nav.tabs a')].map(a => {
      const icon = a.querySelector('.ic').getBoundingClientRect();
      const text = document.createRange();
      text.selectNodeContents(a.querySelector('span'));
      const lines = [...text.getClientRects()].filter(r => r.width > 0);
      return { top: icon.top, icon: icon.left + icon.width / 2,
               lines: lines.map(r => r.left + r.width / 2),
               end: Math.max(...lines.map(r => r.bottom)), bar: a.getBoundingClientRect().bottom };
    })""")
    assert max(len(t["lines"]) for t in tabs) == 2  # State machines wraps at this width
    assert {t["top"] for t in tabs} == {tabs[0]["top"]}
    for t in tabs:
        assert t["lines"] == pytest.approx([t["icon"]] * len(t["lines"]), abs=1)
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


def test_the_pointer_and_the_arrows_share_one_highlighted_row(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """A hovered row looked as picked as the arrows' row, and Enter opened the arrows' row."""
    from playwright.sync_api import expect

    page, errors = open_page(port)
    dialog = _open_switcher(page, port, _sessions(5))
    options = dialog.get_by_role("option")
    options.nth(2).hover()
    expect(options.nth(2)).to_have_attribute("aria-selected", "true")
    page.keyboard.press("ArrowDown")
    expect(options.nth(3)).to_have_attribute("aria-selected", "true")
    lit = dialog.locator(".ac-item").evaluate_all(
        "rows => rows.map(r => getComputedStyle(r).backgroundColor !== 'rgba(0, 0, 0, 0)')"
    )
    assert lit == [False, False, False, True, False]  # the pointer still rests on the third
    page.keyboard.press("Enter")
    page.wait_for_function("location.hash === '#/session/run-3'")
    assert [e for e in errors if not e.startswith("Failed to load resource")] == []


def test_the_switchers_placeholder_fits_a_phones_field(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """On a phone the placeholder stopped after "worktree…", so "> for commands" never showed.

    A 320px screen is the narrowest phone.
    """
    page, errors = open_page(port)
    page.set_viewport_size({"width": 320, "height": 640})
    dialog = _open_switcher(page, port, _sessions(1))
    field = dialog.get_by_role("combobox")
    need, room = field.evaluate("""f => { const s = getComputedStyle(f);
      const text = document.createElement('canvas').getContext('2d');
      text.font = s.fontSize + ' ' + s.fontFamily;
      return [text.measureText(f.placeholder).width,
              f.clientWidth - parseFloat(s.paddingLeft) - parseFloat(s.paddingRight)]; }""")
    assert "> for commands" in str(field.get_attribute("placeholder"))
    assert need <= room
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


@pytest.mark.parametrize(
    ("key", "title"), [("Control+K", "Switch session"), ("?", "Keyboard shortcuts")]
)
def test_a_dialog_closes_from_its_own_close_button(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, key: str, title: str
) -> None:
    """The pickers and the keyboard list had no Close button.

    A touch screen has no Esc, and nothing showed that a tap outside closes the dialog.
    """
    from playwright.sync_api import expect

    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/")
    expect(page.locator(".dock textarea")).to_be_visible()
    page.keyboard.press(key)
    dialog = page.get_by_role("dialog", name=title)
    close = dialog.get_by_role("button", name="Close", exact=True)
    expect(close).to_be_visible()
    close.click()
    expect(page.get_by_role("dialog")).to_have_count(0)
    assert not errors
