"""The app shell in a browser: the rail, the header, the sign-in form, dialogs and pickers."""

from __future__ import annotations

import json
import pathlib
import re
from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest

from agent6 import paths
from tests.unit import test_git_history
from tests.web import session_logs, signin, test_web_browser

if TYPE_CHECKING:
    from playwright.sync_api import Browser, Locator, Page

pytestmark = pytest.mark.browser
browser = test_web_browser.browser
open_page = test_web_browser.open_page
port = test_web_browser.port
git_port = test_web_browser.git_port


@pytest.mark.parametrize(
    ("stored", "name"), [("min", "Hide the sidebar"), ("hidden", "Show the sidebar")]
)
def test_a_stored_rail_state_is_applied_when_the_rail_is_built(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, stored: str, name: str
) -> None:
    """A stored collapse was applied at the end of <body>: Chromium painted the rail wide first.

    A stored hidden rail was not applied at all.
    """
    from playwright.sync_api import expect

    page, errors = open_page(port)
    page.add_init_script(f"""localStorage.setItem('a6-rail', '{stored}');
      new MutationObserver((records, observer) => {{
        if (!document.querySelector('aside.rail')) return;
        window.railBuiltAs = document.documentElement.classList.contains('rail-{stored}');
        observer.disconnect();
      }}).observe(document, {{ childList: true, subtree: true }});""")
    page.goto(f"http://127.0.0.1:{port}/#/")
    expect(page.locator("#rail-toggle")).to_have_attribute("aria-label", name)
    assert page.evaluate("window.railBuiltAs") is True
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


@pytest.mark.parametrize(
    ("route", "ready", "label"),
    [
        ("#/", ".dock textarea", "Sessions"),
        ("#/machines", ".dock textarea", "State machines"),
        ("#/config", ".cfg", "Config"),
        ("#/git", ".gc-row", "Git"),
        ("#/files", ".files-tree .gf", "Files"),
    ],
)
def test_the_crumb_names_each_page_as_its_nav_link_does(
    open_page: Callable[[int], tuple[Page, list[str]]],
    git_port: int,
    route: str,
    ready: str,
    label: str,
) -> None:
    """Each page renamed itself in lower case ("sessions", "config") beside the nav's "Sessions".

    No page had an h1, and Sessions, Git, Files and Config had no heading at all.
    """
    from playwright.sync_api import expect

    page, errors = open_page(git_port)
    page.goto(f"http://127.0.0.1:{git_port}/{route}")
    expect(page.locator(ready).first).to_be_visible()
    expect(page.locator("#crumb")).to_have_text(label)
    expect(page.get_by_role("heading", level=1)).to_have_text([label])
    expect(page).to_have_title(f"{label} · agent6")
    assert not errors


def test_a_sessions_crumb_leads_with_its_state_as_a_status_pill(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, tmp_path: pathlib.Path
) -> None:
    """A session's crumb was the text "running · <id>", with no status dot.

    It holds the rows' status pill, which follows the run, then the id on the same line; the tab's
    title reads both.
    """
    from playwright.sync_api import expect

    session = session_logs.write_run(tmp_path, "crumb-AAA111", [], live=True)
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/session/crumb-AAA111")
    state, name = page.locator("#crumb > .pill"), page.locator("#crumb > .pill + span")
    expect(state).to_have_text("running")
    expect(state).to_have_class("pill active")
    expect(name).to_have_text("crumb-AAA111")
    expect(page).to_have_title("running · crumb-AAA111 · agent6")
    first, second = state.bounding_box(), name.bounding_box()
    assert first and second
    assert first["y"] == second["y"] and first["x"] < second["x"]
    stop = {"type": "session.stop", "reason": "finish_session", "verification": "passed"}
    with (session / "logs.jsonl").open("a") as log:
        log.write(json.dumps(stop) + "\n")
    expect(state).to_have_class("pill ok")
    expect(state).to_have_text("passed")
    expect(page).to_have_title("passed · crumb-AAA111 · agent6")
    assert not errors


def test_a_phones_crumb_shows_a_sessions_state_and_id_whole(
    browser: Browser, port: int, tmp_path: pathlib.Path
) -> None:
    """A phone's crumb cut the session's id beside the brand: "passed · ready-ro…".

    The id goes under the state, in smaller text, and the crumb runs up to the buttons: at 375px
    an empty spacer before them cut "merry-fountain-B7…".
    """
    from playwright.sync_api import expect

    stop = {"type": "session.stop", "reason": "max_turns", "verification": "unverified"}
    session_logs.write_run(tmp_path, "merry-fountain-B7XAJS", [stop])
    token = json.dumps(signin.token(port))
    with browser.new_context(
        viewport={"width": 375, "height": 812}, has_touch=True, is_mobile=True
    ) as context:
        context.add_init_script(f"localStorage.setItem('a6-token', {token})")
        page = context.new_page()
        errors: list[str] = []
        page.on("pageerror", lambda exc: errors.append(str(exc)))
        page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
        page.goto(f"http://127.0.0.1:{port}/#/session/merry-fountain-B7XAJS")
        parts = page.locator("#crumb > *")
        expect(parts).to_have_text(["failed · max turns", "merry-fountain-B7XAJS"])
        # Unrounded: scrollWidth and clientWidth round, and read 140 = 140 over an ellipsis.
        over = parts.evaluate_all("""ps => ps.map(p => {
          const box = p.getBoundingClientRect().width;
          p.style.width = 'max-content';
          const need = p.getBoundingClientRect().width;
          p.style.width = '';
          return need - box; })""")
        assert max(over) <= 0, over
        state, name = parts.first.bounding_box(), parts.last.bounding_box()
        assert state and name
        assert name["y"] >= state["y"] + state["height"]
        assert [e for e in errors if not e.startswith("Failed to load resource")] == []


@pytest.mark.parametrize(("width", "height"), [(1280, 900), (390, 844)])
def test_a_focused_row_stops_clear_of_the_header_and_the_tab_bar(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, width: int, height: int
) -> None:
    """Shift+Tab left a Config row under the header, and on a phone Tab left one under the tabs."""
    page, errors = open_page(port)
    page.set_viewport_size({"width": width, "height": height})
    page.goto(f"http://127.0.0.1:{port}/#/config")
    keys = page.locator(".cfg .cfg-edit")
    keys.nth(30).wait_for()
    bars = """() => { const top = document.querySelector('header').getBoundingClientRect().bottom;
      const tabs = document.querySelector('nav.tabs');
      return [top, tabs.checkVisibility() ? tabs.getBoundingClientRect().top : innerHeight]; }"""
    top, bottom = page.evaluate(bars)
    row = "() => document.activeElement.closest('tr').getBoundingClientRect().toJSON()"
    # Row 30 just under the header, then Shift+Tab; row 30 just over the bottom bar, then Tab.
    for edge, key in ((top, "Shift+Tab"), (bottom, "Tab")):
        keys.nth(30).focus()
        box = page.evaluate(row)
        page.evaluate(
            "y => scrollBy(0, y)",
            box["top"] - top - 2 if key == "Shift+Tab" else box["bottom"] - edge + 2,
        )
        page.keyboard.press(key)
        box = page.evaluate(row)
        assert top <= box["top"] and box["bottom"] <= bottom, (key, box, top, bottom)
    assert not errors


def test_a_top_bar_control_takes_the_focus_where_the_page_stands(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """Beside a hidden rail, a top bar control taking the focus scrolled the page half a screen."""
    from playwright.sync_api import expect

    page, errors = open_page(port)
    page.add_init_script("localStorage.setItem('a6-rail', 'min')")
    page.goto(f"http://127.0.0.1:{port}/#/config")
    page.locator(".cfg .cfg-edit").nth(30).wait_for()
    page.evaluate("scrollTo(0, 1500)")
    page.evaluate("document.getElementById('rail-toggle').focus({ preventScroll: true })")
    page.keyboard.press("Enter")  # the rail hides and hands the focus to the menu button
    expect(page.locator("#rail-show")).to_be_focused()
    assert page.evaluate("scrollY") == 1500
    page.keyboard.press("Tab")
    expect(page.locator("header .brand")).to_be_focused()
    assert page.evaluate("scrollY") == 1500
    assert not errors


@pytest.mark.parametrize(("width", "lead"), [(1280, "#crumb"), (390, "header .brand")])
def test_the_top_bar_starts_and_ends_where_the_pages_text_does(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, width: int, lead: str
) -> None:
    """The crumb started 6px left of the page's text, a phone's brand and buttons 4px outside it."""
    from playwright.sync_api import expect

    page, errors = open_page(port)
    page.set_viewport_size({"width": width, "height": 800})
    page.goto(f"http://127.0.0.1:{port}/#/")
    entry = page.locator(".dock textarea")
    expect(entry).to_be_visible()
    edges = "e => { const r = e.getBoundingClientRect(); return [r.left, r.right]; }"
    left, right = entry.evaluate(edges)
    assert page.locator(lead).evaluate(edges)[0] == left
    if width < 781:
        assert page.locator("header > button").last.evaluate(edges)[1] == right
    assert not errors


def test_a_disabled_button_keeps_its_look_under_the_pointer(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """A disabled Start darkened and dropped its border under the pointer, as a live one does."""
    from playwright.sync_api import expect

    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/")
    start = page.get_by_role("button", name="Start", exact=True)
    look = "e => { const s = getComputedStyle(e); return [s.backgroundColor, s.borderColor]; }"

    def hovered() -> tuple[object, object]:
        page.mouse.move(0, 0)
        page.wait_for_timeout(300)  # past the button's colour transition
        rest = start.evaluate(look)
        box = start.bounding_box()
        assert box
        page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        page.wait_for_timeout(300)
        return rest, start.evaluate(look)

    expect(start).to_be_disabled()
    rest, hover = hovered()
    assert hover == rest
    page.locator(".dock textarea").fill("a task")
    expect(start).to_be_enabled()
    rest, hover = hovered()
    assert hover != rest
    assert not errors


def test_an_unknown_session_reads_not_found_under_its_section(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """An unknown session opened from Config showed a bare error line under the crumb "config".

    It reads Not found under its own id, as a missing machine does, with no Try again: a retry
    cannot bring it back.
    """
    from playwright.sync_api import expect

    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/config")
    expect(page.locator("#crumb")).to_have_text(re.compile("config", re.I))
    page.evaluate("location.hash = '#/session/nobody'")
    problem = page.locator("#view .problem")
    expect(problem.get_by_role("heading")).to_have_text("Not found")
    expect(problem).to_contain_text("no session 'nobody'")
    expect(page.locator("#crumb")).to_have_text("nobody")
    expect(page.locator("[aria-current=page]")).to_have_count(2)  # the rail and the phone tabs
    expect(problem.get_by_role("button", name="Try again")).to_have_count(0)
    expect(problem.get_by_role("link", name="Back to Sessions")).to_have_attribute("href", "#/")
    assert [e for e in errors if not e.startswith("Failed to load resource")] == []


def test_a_page_that_fails_to_load_offers_a_retry(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """A failed load showed a bare error line, with no way to try again.

    A failed Config offered Back to Sessions, a page it does not sit under.
    """
    from playwright.sync_api import expect

    page, errors = open_page(port)
    page.route("**/api/config", lambda route: route.abort(), times=1)
    page.goto(f"http://127.0.0.1:{port}/#/config")
    problem = page.locator("#view .problem")
    expect(problem.get_by_role("heading")).to_have_text("Could not load this page")
    expect(page.locator("#crumb")).to_have_text(re.compile("^config$", re.I))
    expect(problem.get_by_role("link")).to_have_count(0)
    problem.get_by_role("button", name="Try again").click()
    expect(page.locator(".cfg").first).to_be_visible()
    assert [e for e in errors if not e.startswith("Failed to load resource")] == []


@pytest.mark.parametrize("route", ["#/machine/nobody", "#/draft/nobody"])
def test_a_problem_views_way_back_names_the_page_it_sits_under(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, route: str
) -> None:
    """A missing machine or draft, under State machines in both bars, offered Back to Sessions."""
    from playwright.sync_api import expect

    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/{route}")
    problem = page.locator("#view .problem")
    expect(problem.get_by_role("heading")).to_have_text("Not found")
    back = problem.get_by_role("link")
    expect(back).to_have_text(["Back to State machines"])
    expect(back).to_have_attribute("href", "#/machines")
    assert [e for e in errors if not e.startswith("Failed to load resource")] == []


def test_the_old_view_is_marked_busy_until_the_next_page_paints(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """Config's tab and crumb showed over the Sessions list for a slow load, with no sign of it.

    The list and the dock over it then dimmed apart, so the rows showed through the dock.
    """
    from playwright.sync_api import Route, expect

    page, errors = open_page(port)
    held: list[Route] = []

    def hold(route: Route) -> None:
        held.append(route)

    page.route("**/api/config", hold)
    page.goto(f"http://127.0.0.1:{port}/#/")
    expect(page.locator(".dock textarea")).to_be_visible()
    page.get_by_role("link", name="Config").first.click()
    expect(page.locator("#crumb")).to_have_text(re.compile("config", re.I))
    busy = page.locator("#view > [aria-busy=true]")
    expect(busy.first).to_be_visible()
    expect(page.locator("#view > :not([aria-busy=true])")).to_have_count(0)
    page.wait_for_timeout(600)  # past the dim's 0.2 s delay and 0.15 s fade
    dims = "ss => ss.map(s => getComputedStyle(document.querySelector(s)).opacity)"
    assert page.evaluate(dims, ["#view", "#view > .grid", "#view > .dock"]) == ["0.55", "1", "1"]
    held[0].continue_()
    expect(page.locator(".cfg").first).to_be_visible()
    expect(busy).to_have_count(0)
    assert not errors


def test_a_slow_first_load_leaves_its_placeholder_undimmed(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """A slow first load dimmed "loading…" to 2.3:1 as if it were a page being replaced."""
    from playwright.sync_api import Route, expect

    page, errors = open_page(port)
    held: list[Route] = []

    def hold(route: Route) -> None:
        held.append(route)

    page.route("**/api/hub", hold)
    page.goto(f"http://127.0.0.1:{port}/#/")
    placeholder = page.locator("#view > .empty")
    expect(placeholder).to_have_text("loading…")
    page.wait_for_timeout(600)  # past the dim's 0.2 s delay and 0.15 s fade
    expect(placeholder).not_to_have_attribute("aria-busy", "true")
    assert page.evaluate("getComputedStyle(view).opacity") == "1"
    held[0].continue_()
    expect(page.locator(".dock textarea")).to_be_visible()
    assert not errors


def test_a_markdown_files_link_to_its_own_heading_leaves_it_undimmed(
    open_page: Callable[[int], tuple[Page, list[str]]], git_port: int, tmp_path: pathlib.Path
) -> None:
    """The file scrolled to the heading in place and kept the busy mark that dimmed it."""
    from playwright.sync_api import expect

    guide = "# Guide\n\n[Last part](#last-part)\n\n" + "A paragraph.\n\n" * 40 + "## Last part\n"
    (tmp_path / "guide.md").write_text(guide, encoding="utf-8")
    test_git_history._git(tmp_path, "add", "guide.md")
    test_git_history._git(tmp_path, "commit", "-m", "guide")
    page, errors = open_page(git_port)
    page.goto(f"http://127.0.0.1:{git_port}/#/files/HEAD/guide.md")
    expect(page.locator(".md-view h1")).to_have_text("Guide")
    page.get_by_role("link", name="Last part", exact=True).click()
    expect(page.locator(".md-view #last-part")).to_be_in_viewport()
    assert page.evaluate("location.hash").endswith("?anchor=last-part")
    expect(page.locator("#view > [aria-busy=true]")).to_have_count(0)
    assert not errors


def test_a_route_under_the_sign_in_form_keeps_it_whole(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """A route that met a 401 behind the sign-in form left the kept form dimmed as busy."""
    from playwright.sync_api import expect

    page, errors = open_page(port)
    page.add_init_script("localStorage.setItem('a6-token', 'expired')")
    page.goto(f"http://127.0.0.1:{port}/#/")
    field = page.locator("#login").get_by_role("textbox", name="secret")
    field.fill("half typed")
    page.wait_for_load_state("networkidle")
    page.evaluate("location.hash = '#/config'")
    page.wait_for_load_state("networkidle")
    expect(page.locator("#login")).not_to_have_attribute("aria-busy", "true")
    expect(field).to_have_value("half typed")
    assert [e for e in errors if not e.startswith("Failed to load resource")] == []


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


def test_signing_in_hands_the_focus_to_the_pages_heading(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """Signing in replaced the form that held the focus and left the focus on the page body."""
    from playwright.sync_api import expect

    page, errors = open_page(port)
    page.add_init_script("localStorage.setItem('a6-token', 'expired')")
    page.goto(f"http://127.0.0.1:{port}/#/")
    field = page.locator("#login").get_by_role("textbox", name="secret")
    field.fill(signin.SECRET)
    field.press("Enter")
    heading = page.get_by_role("heading", level=1)
    expect(heading).to_have_text("Sessions")
    expect(page.locator(".dock textarea")).to_be_visible()
    expect(heading).to_be_focused()
    assert [e for e in errors if not e.startswith("Failed to load resource")] == []


@pytest.mark.parametrize("way", ["Fork", "Ctrl+K"])
def test_a_navigation_hands_the_focus_to_the_new_pages_heading(
    open_page: Callable[[int], tuple[Page, list[str]]],
    port: int,
    tmp_path: pathlib.Path,
    way: str,
) -> None:
    """The new page's control named as the focused one took the focus: one Enter acted again.

    Fork opened the fork on its own Fork, and a switch from Review run focused the next run's.
    """
    from playwright.sync_api import expect

    for sid in ("first-AAA111", "second-BBB222"):
        session_logs.write_run(tmp_path, sid, [{"type": "session.stop"}])
    page, errors = open_page(port)
    page.route(
        "**/first-AAA111/fork",
        lambda route: route.fulfill(json={"ok": True, "new_session_id": "second-BBB222"}),
    )
    page.goto(f"http://127.0.0.1:{port}/#/session/first-AAA111")
    button = page.get_by_role("button", name="Fork" if way == "Fork" else "Review run", exact=True)
    pressed = button.element_handle()
    button.focus()
    if way == "Ctrl+K":
        page.keyboard.press("Control+K")
        page.keyboard.type("second")
        rows = page.get_by_role("dialog", name="Switch session").get_by_role("row")
        expect(rows).to_have_text([re.compile("^second-BBB222")])
    page.keyboard.press("Enter")
    page.wait_for_function("b => !b.isConnected", arg=pressed)  # the next page painted
    expect(page.locator(".run-app .composer textarea")).to_be_visible()
    assert page.evaluate("location.hash") == "#/session/second-BBB222"
    expect(page.get_by_role("heading", level=1)).to_be_focused()
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


def test_the_rail_brands_focus_ring_stands_whole_inside_the_rail(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """The brand fills the rail's scrollport, which cut its focus ring to the bottom edge."""
    from playwright.sync_api import expect

    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/")
    expect(page.locator(".dock textarea")).to_be_visible()
    ring = """() => { const b = document.querySelector('aside.rail .rail-brand');
      const s = getComputedStyle(b), r = b.getBoundingClientRect();
      const rail = document.querySelector('aside.rail');
      const out = parseFloat(s.outlineOffset) + parseFloat(s.outlineWidth);
      return [s.outlineStyle, r.top - out >= 0, r.left - out >= 0,
              r.right + out <= rail.getBoundingClientRect().left + rail.clientWidth]; }"""
    toggle = page.locator("#rail-toggle")
    for expanded in ("true", "false"):
        if expanded == "false":
            toggle.click()
        expect(toggle).to_have_attribute("aria-expanded", expanded)
        page.locator("aside.rail .rail-nav a").first.focus()
        page.keyboard.press("Shift+Tab")  # the brand, by the keyboard
        expect(page.locator("aside.rail .rail-brand")).to_be_focused()
        assert page.evaluate(ring) == ["solid", True, True, True]
    assert not errors


@pytest.mark.parametrize(("width", "rail"), [(1280, "hidden"), (390, "")])
def test_the_top_bars_icon_buttons_are_squares(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, width: int, rail: str
) -> None:
    """The menu and theme buttons stood wider than they were tall, beside a square widget button."""
    from playwright.sync_api import expect

    page, errors = open_page(port)
    page.add_init_script(f"localStorage.setItem('a6-rail', '{rail}')")
    page.set_viewport_size({"width": width, "height": 800})
    page.goto(f"http://127.0.0.1:{port}/#/")
    expect(page.locator(".dock textarea")).to_be_visible()
    sizes = page.locator("header > button").evaluate_all("""bs => bs
      .filter(b => b.checkVisibility())
      .map(b => [b.getBoundingClientRect().width, b.getBoundingClientRect().height])""")
    assert sizes and all(w == h for w, h in sizes), sizes
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


def test_the_keyboard_list_puts_each_key_under_where_it_works(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """Ctrl+R and Ctrl+Enter sat under "In the composer", and the list left out Home and End.

    On the Sessions page's composer Ctrl+R reloads the page and Ctrl+Enter starts the run.
    Home and End then sat under "In a list", where a picker's field moves its caret on them.
    """
    from playwright.sync_api import expect

    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/")
    expect(page.locator(".dock textarea")).to_be_visible()
    page.keyboard.press("?")
    table = page.get_by_role("dialog", name="Keyboard shortcuts").locator("table.keys")
    groups = table.evaluate("""t => [...t.tBodies].map(b => [b.querySelector('th').textContent,
      [...b.querySelectorAll('td.key')].map(k => k.textContent)])""")
    places = [place for place, _ in groups]
    assert len(places) == len(set(places)), places
    keys = dict(groups)
    assert keys["In the composer"] == ["Enter", "Shift+Enter", "/"]
    assert keys["In a session's composer"] == ["Ctrl+Enter", "Ctrl+R"]
    assert keys["In a list"] == ["↑ ↓"]
    assert keys["In a list on the page"] == ["Home End"]
    assert not errors


def test_a_small_buttons_label_stays_on_one_line_inside_its_card(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, tmp_path: pathlib.Path
) -> None:
    """Beside a wide details pane, "detail: collapsed" wrapped to two lines in its heading.

    The label keeps one line, and the button drops under the heading when the row is too narrow.
    """
    from playwright.sync_api import expect

    run = paths.state_dir(tmp_path) / "sessions" / "runs" / "narrow-AAA111"
    run.mkdir(parents=True)
    start = {"type": "session.start", "mode": "run", "user_task": "Look"}
    (run / "logs.jsonl").write_text(json.dumps(start) + "\n")
    page, errors = open_page(port)
    page.add_init_script("""localStorage.setItem('a6-drawer', 'open');
      localStorage.setItem('a6-drawer-w', '390px');""")
    page.set_viewport_size({"width": 820, "height": 900})
    page.goto(f"http://127.0.0.1:{port}/#/session/narrow-AAA111")
    detail = page.get_by_role("button", name=re.compile("^detail: "))
    expect(detail).to_be_visible()
    button, card = detail.bounding_box(), page.locator(".conv-card").first.bounding_box()
    assert button is not None and card is not None
    assert button["height"] == 26
    assert button["x"] + button["width"] <= card["x"] + card["width"]
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
    """Serve `rows` as the hub, open the Config page, then Ctrl+K's switcher."""
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

    The field names the highlighted session's cell as its active descendant, so a screen reader
    reads each session the arrows reach.
    """
    from playwright.sync_api import expect

    page, errors = open_page(port)
    dialog = _open_switcher(page, port, _sessions(20))
    for _ in range(10):
        page.keyboard.press("ArrowDown")
    active = dialog.get_by_role("gridcell", selected=True)
    expect(active).to_contain_text("run-10 · run · task 10")
    expect(dialog.get_by_role("combobox")).to_have_attribute(
        "aria-activedescendant", str(active.get_attribute("id"))
    )
    box, shown = active.bounding_box(), dialog.get_by_role("grid").bounding_box()
    assert box is not None and shown is not None
    assert shown["y"] <= box["y"] and box["y"] + box["height"] <= shown["y"] + shown["height"]
    assert page.evaluate("document.scrollingElement.scrollTop") == 0
    assert not errors


def test_the_pointer_and_the_arrows_share_one_highlighted_row(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """A hovered row looked as picked as the arrows' row, and Enter opened the arrows' row.

    The arrows' row was a faint tint alone; it takes the accent bar the Git pickers draw.
    """
    from playwright.sync_api import expect

    page, errors = open_page(port)
    dialog = _open_switcher(page, port, _sessions(5))
    options = dialog.get_by_role("gridcell")
    options.nth(2).hover()
    expect(options.nth(2)).to_have_attribute("aria-selected", "true")
    page.keyboard.press("ArrowDown")
    expect(options.nth(3)).to_have_attribute("aria-selected", "true")
    lit = dialog.locator(".ac-item").evaluate_all(
        "rows => rows.map(r => getComputedStyle(r).backgroundColor !== 'rgba(0, 0, 0, 0)')"
    )
    assert lit == [False, False, False, True, False]  # the pointer still rests on the third
    bars = dialog.locator(".ac-item").evaluate_all(
        "rows => rows.map(r => getComputedStyle(r).boxShadow.includes('inset'))"
    )
    assert bars == [False, False, False, True, False]
    page.keyboard.press("Enter")
    page.wait_for_function("location.hash === '#/session/run-3'")
    assert [e for e in errors if not e.startswith("Failed to load resource")] == []


def test_a_switcher_rows_buttons_sit_in_a_grid_not_a_list_box(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """The switcher's list box held its rows' Review and Merge buttons.

    A list box may hold only options, so the switcher is a grid: each row a cell for its session
    and a cell for its buttons. The palette, whose rows have no buttons, stays a list box.
    """
    from playwright.sync_api import expect

    page, errors = open_page(port)
    dialog = _open_switcher(page, port, _sessions(2, branch="agent6/run", unmerged=True))
    expect(dialog.get_by_role("combobox")).to_have_attribute("aria-haspopup", "grid")
    rows = dialog.get_by_role("grid").get_by_role("row")
    expect(rows).to_have_count(2)
    buttons = rows.nth(1).get_by_role("gridcell").nth(1).get_by_role("button")
    expect(buttons).to_have_text(["Review", "Merge"])  # the Sessions row's words
    expect(dialog.get_by_role("listbox")).to_have_count(0)
    page.keyboard.type(">")
    expect(page.get_by_role("dialog", name="Commands").get_by_role("listbox")).to_be_visible()
    assert not errors


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


def test_keys_typed_while_the_switchers_rows_load_land_in_its_field(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """Ctrl+K opened the switcher only once /api/hub answered.

    The keys typed meanwhile went into the composer behind it, and Enter started a run with them.
    """
    from playwright.sync_api import Route, expect

    page, errors = open_page(port)
    posts: list[str] = []
    page.route("**/api/new", lambda route: (posts.append(route.request.url), route.abort()))
    page.goto(f"http://127.0.0.1:{port}/#/")
    entry = page.locator(".dock textarea")
    expect(entry).to_be_visible()
    page.evaluate("clearInterval(hubTimer)")  # only Ctrl+K may fetch the hub
    held: list[Route] = []

    def hold(route: Route) -> None:
        held.append(route)

    page.route("**/api/hub", hold)
    entry.focus()
    page.keyboard.press("Control+K")
    page.keyboard.type("run-1", delay=30)
    page.keyboard.press("Enter")  # before the rows: picks nothing
    switcher = page.get_by_role("dialog", name="Switch session")
    expect(switcher.get_by_role("combobox")).to_have_value("run-1")
    expect(switcher).to_contain_text("loading…")
    assert posts == [] and entry.input_value() == ""
    held[0].fulfill(json={"sessions": _sessions(3)})
    expect(switcher.get_by_role("row")).to_have_count(1)
    page.keyboard.press("Enter")
    page.wait_for_function("location.hash === '#/session/run-1'")
    assert posts == []
    assert [e for e in errors if not e.startswith("Failed to load resource")] == []


def test_keys_typed_while_past_messages_load_land_in_the_search(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, tmp_path: pathlib.Path
) -> None:
    """Ctrl+R opened the search only once the past messages came; the keys went to the composer."""
    from playwright.sync_api import Route, expect

    session_logs.write_run(tmp_path, "done-AAA111", [{"type": "session.stop"}])
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/session/done-AAA111")
    entry = page.locator(".run-app .composer textarea")
    expect(page.locator(".conv .ci").first).to_be_visible()  # the conversation's first fetch
    held: list[Route] = []

    def hold(route: Route) -> None:
        held.append(route)

    page.route("**/conversation", hold)
    entry.focus()
    page.keyboard.press("Control+r")
    page.keyboard.type("parse", delay=30)
    search = page.get_by_role("dialog", name="Search past messages")
    expect(search.get_by_role("combobox")).to_have_value("parse")
    assert entry.input_value() == ""
    for route in held:
        route.continue_()
    expect(search.get_by_role("option")).to_have_text(["Parse CSV rows"])
    assert not errors


def test_rows_that_come_after_the_switcher_closed_open_nothing(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """Rows that answered after Esc opened the switcher; a failed load closes it and says why."""
    from playwright.sync_api import Route, expect

    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/config")
    page.wait_for_selector(".cfg")
    held: list[Route] = []

    def hold(route: Route) -> None:
        held.append(route)

    page.route("**/api/hub", hold)
    switcher = page.get_by_role("dialog", name="Switch session")
    page.keyboard.press("Control+K")
    expect(switcher).to_be_visible()
    page.keyboard.press("Escape")
    expect(switcher).to_have_count(0)
    held.pop().fulfill(json={"sessions": _sessions(2)})
    page.wait_for_timeout(300)
    expect(page.locator("dialog[open]")).to_have_count(0)
    page.keyboard.press("Control+K")
    expect(switcher).to_be_visible()
    held.pop().abort()
    expect(switcher).to_have_count(0)
    expect(page.locator(".toast.bad")).to_be_visible()
    assert [e for e in errors if not e.startswith("Failed to load resource")] == []


def test_the_palette_lists_every_command(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """The palette hid its thirteenth command behind "… 1 more" with room to spare."""
    from playwright.sync_api import expect

    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/")
    expect(page.locator(".dock textarea")).to_be_visible()
    page.keyboard.press("Control+K")
    page.keyboard.type(">")
    palette = page.get_by_role("dialog", name="Commands")
    commands = page.evaluate("paletteItems().length")
    assert commands > 12
    expect(palette.get_by_role("option")).to_have_count(commands)
    expect(palette.locator(".more-note")).to_have_text("")
    assert not errors


@pytest.mark.parametrize(("route", "ready"), [("#/config", ".cfg"), ("#/", ".dock textarea")])
def test_the_palettes_new_session_focuses_the_composer_from_any_page(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, route: str, ready: str
) -> None:
    """New session left the focus on the page body unless the Sessions page was already open."""
    from playwright.sync_api import expect

    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/{route}")
    expect(page.locator(ready).first).to_be_visible()
    page.keyboard.press("Control+K")
    page.keyboard.type(">new session")
    expect(page.get_by_role("dialog", name="Commands").get_by_role("option")).to_have_count(1)
    page.keyboard.press("Enter")
    expect(page.locator(".dock textarea")).to_be_focused()
    assert page.evaluate("location.hash") == "#/"
    assert not errors


@pytest.mark.parametrize("pick", ["New session", "Review"])
def test_a_task_handed_to_the_sessions_composer_keeps_its_picks(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, pick: str
) -> None:
    """The palette's New session and the switcher's Review painted the Sessions page afresh.

    The mode went back to run, and where the browser keeps no draft, the typed task went too.
    """
    from playwright.sync_api import expect

    page, errors = open_page(port)
    page.add_init_script("""Storage.prototype.setItem = (set => function (key, value) {
      if (key.startsWith('a6-draft')) throw new DOMException('full', 'QuotaExceededError');
      return set.call(this, key, value);
    })(Storage.prototype.setItem);""")
    rows = _sessions(1, branch="agent6/run-0")
    page.route("**/api/hub", lambda route: route.fulfill(json={"sessions": rows}))
    page.goto(f"http://127.0.0.1:{port}/#/")
    entry = page.locator(".dock textarea")
    mode = page.locator(".dock").get_by_role("combobox", name="mode", exact=True)
    mode.select_option("plan")
    entry.fill("half a task")
    entry.blur()
    page.keyboard.press("Control+K")
    if pick == "Review":
        page.get_by_role("dialog", name="Switch session").get_by_role("button", name=pick).click()
    else:
        page.keyboard.type(">new session")
        expect(page.get_by_role("dialog", name="Commands").get_by_role("option")).to_have_count(1)
        page.keyboard.press("Enter")
    expect(entry).to_be_focused()
    expect(entry).to_have_value("/review agent6/run-0" if pick == "Review" else "half a task")
    expect(mode).to_have_value("plan")
    assert not errors


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


def test_a_long_dialog_title_wraps_beside_its_close_button(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """On a phone a long config key pushed the Close button out of its dialog.

    The dialog opened scrolled sideways, cutting off the key's start.
    The key wraps, at a dot first.
    """
    from playwright.sync_api import expect

    page, errors = open_page(port)
    page.set_viewport_size({"width": 390, "height": 844})
    page.goto(f"http://127.0.0.1:{port}/#/config")
    key = "review.min_remaining_budget_fraction"
    page.get_by_text(key, exact=True).click()
    dialog = page.get_by_role("dialog", name=key)
    expect(dialog).to_be_visible()
    wide, room, scrolled = dialog.evaluate("d => [d.scrollWidth, d.clientWidth, d.scrollLeft]")
    assert wide <= room and scrolled == 0
    close, box = dialog.get_by_role("button", name="Close").bounding_box(), dialog.bounding_box()
    assert close is not None and box is not None
    assert close["x"] + close["width"] <= box["x"] + box["width"]
    first, rest = dialog.locator("h2").evaluate("""h => { const r = document.createRange();
      r.selectNodeContents(h.firstChild);
      return [r.getBoundingClientRect().top, h.lastChild.getBoundingClientRect().top]; }""")
    assert rest > first  # "review." ends the first line
    assert not errors


def test_a_dialog_keeps_its_close_button_in_view_while_it_scrolls(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """On a short screen the keyboard list scrolled its Close button out of the dialog."""
    from playwright.sync_api import expect

    page, errors = open_page(port)
    page.set_viewport_size({"width": 390, "height": 480})
    page.goto(f"http://127.0.0.1:{port}/#/")
    expect(page.locator(".dock textarea")).to_be_visible()
    page.keyboard.press("?")
    dialog = page.get_by_role("dialog", name="Keyboard shortcuts")
    close = dialog.get_by_role("button", name="Close")
    before = close.bounding_box()
    assert dialog.evaluate("d => { d.scrollTop = d.scrollHeight; return d.scrollTop; }") > 0
    after, box = close.bounding_box(), dialog.bounding_box()
    assert before is not None and after is not None and box is not None
    assert after["y"] == before["y"] and after["y"] >= box["y"]
    assert not errors


@pytest.mark.parametrize(
    ("hash_", "files", "asker", "title", "action", "opens_on", "post"),
    [
        ("#/", [], "Clear saved asks", "Delete every saved ask?", "Delete", "Cancel", "rm_asks"),
        ("#/machines", ["code-fixer"], "Run code-fixer", "Run code-fixer?", "Run", "Run", "run"),
    ],
)
def test_a_question_acts_only_on_its_action_button(
    open_page: Callable[[int], tuple[Page, list[str]]],
    port: int,
    hash_: str,
    files: list[str],
    asker: str,
    title: str,
    action: str,
    opens_on: str,
    post: str,
) -> None:
    """A native confirm() blocked the page, and a browser that suppresses it answers no unasked.

    Esc, Cancel and a click outside answer no and post nothing, the focus back on the button that
    asked; only the action posts. A destructive question opens on Cancel, any other on its action.
    """
    from playwright.sync_api import Route, expect

    page, errors = open_page(port)
    posts: list[str] = []

    def answer(route: Route) -> None:
        posts.append(route.request.url.rsplit("/", 1)[1])
        route.fulfill(json={"ok": True, "message": "done"})

    page.route(f"**/api/*/{post}", answer)
    if files:
        mfs = [{"name": name, "path": f"machines/{name}.yaml"} for name in files]
        page.route("**/api/hub", lambda route: route.fulfill(json={"machine_files": mfs}))
    page.goto(f"http://127.0.0.1:{port}/{hash_}")
    button = page.get_by_role("button", name=asker, exact=True)
    question = page.get_by_role("alertdialog", name=title)
    for decline in ("Escape", "Cancel", "outside"):
        button.click()
        expect(question.get_by_role("button", name=opens_on, exact=True)).to_be_focused()
        if decline == "Escape":
            page.keyboard.press("Escape")
        elif decline == "Cancel":
            question.get_by_role("button", name="Cancel", exact=True).click()
        else:
            page.mouse.click(5, 5)
        expect(question).to_have_count(0)
        expect(button).to_be_focused()
    assert posts == []
    button.click()
    question.get_by_role("button", name=action, exact=True).click()
    expect(question).to_have_count(0)
    expect(page.locator(".toast").first).to_be_visible()
    assert posts == [post]
    assert not errors


def test_a_held_enter_answers_no_question(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """The repeats of the Enter that opened a question pressed its action: a paid run started."""
    from playwright.sync_api import expect

    page, errors = open_page(port)
    posts: list[str] = []
    files = {"machine_files": [{"name": "code-fixer", "path": "machines/code-fixer.yaml"}]}
    page.route("**/api/hub", lambda route: route.fulfill(json=files))
    page.route(
        "**/api/machine/run",
        lambda route: (posts.append(route.request.method), route.fulfill(json={"ok": True})),
    )
    page.goto(f"http://127.0.0.1:{port}/#/machines")
    page.get_by_role("button", name="Run code-fixer", exact=True).focus()
    page.keyboard.down("Enter")
    run = page.get_by_role("alertdialog", name="Run code-fixer?").get_by_role("button", name="Run")
    expect(run).to_be_focused()
    page.keyboard.down("Enter")  # held: the keyboard repeats it
    page.keyboard.down("Enter")
    page.keyboard.up("Enter")
    expect(run).to_be_focused()
    assert posts == []
    page.keyboard.press("Enter")
    expect(run).to_have_count(0)
    expect(page.locator(".toast").first).to_contain_text("started code-fixer")
    assert posts == ["POST"]
    assert not errors
