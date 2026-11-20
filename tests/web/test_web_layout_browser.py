from __future__ import annotations

import json
import pathlib
from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest

from agent6 import paths
from tests.web import signin, test_web_browser

if TYPE_CHECKING:
    from playwright.sync_api import Browser, Locator, Page

pytestmark = pytest.mark.browser
browser = test_web_browser.browser
open_page = test_web_browser.open_page
dirty_git_port = test_web_browser.dirty_git_port
git_port = test_web_browser.git_port
port = test_web_browser.port


def _width(element: Locator) -> float:
    return float(element.evaluate("e => e.getBoundingClientRect().width"))


def _drag(page: Page, handle: Locator, distance: int) -> None:
    box = handle.bounding_box()
    assert box is not None
    x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    page.mouse.move(x, y)
    page.mouse.down()
    page.mouse.move(x + distance, y, steps=5)
    page.mouse.up()


@pytest.mark.parametrize("width", [1280, 390])
@pytest.mark.parametrize("theme", ["dark", "light"])
def test_navigation_icons_follow_bar_order_size_and_theme(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, width: int, theme: str
) -> None:
    """Desktop and phone navigation keep readable labels and inherit each theme's colour."""
    from playwright.sync_api import expect

    page, errors = open_page(port)
    page.add_init_script(f"localStorage.setItem('a6-theme', '{theme}')")
    page.set_viewport_size({"width": width, "height": 900})
    page.goto(f"http://127.0.0.1:{port}/")
    bar = page.locator(".rail-nav" if width == 1280 else "nav.tabs")
    expect(bar).to_be_visible()
    expect(bar.locator("a")).to_have_text(["Sessions", "Git", "Files", "State machines", "Config"])
    for link in bar.locator("a").all():
        svg = link.locator("svg")
        expect(svg).to_be_visible()
        assert _width(svg) == 20
        assert svg.evaluate("e => getComputedStyle(e).stroke") == link.evaluate(
            "e => getComputedStyle(e).color"
        )
    assert not errors


def test_the_theme_buttons_name_the_theme_they_switch_to(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """Each theme button's name follows a press and a change of the system's scheme.

    Both buttons were named "theme", whichever theme a press would bring.
    """
    from playwright.sync_api import expect

    page, errors = open_page(port)
    page.emulate_media(color_scheme="dark")
    page.goto(f"http://127.0.0.1:{port}/#/")
    rail, header = page.locator("aside.rail .theme-toggle"), page.locator("header .theme-toggle")
    expect(rail).to_have_accessible_name("Switch to light theme")
    rail.click()
    expect(rail).to_have_accessible_name("Switch to dark theme")
    expect(header).to_have_attribute("aria-label", "Switch to dark theme")
    rail.click()  # back to the system's scheme, which it follows from here
    expect(rail).to_have_accessible_name("Switch to light theme")
    page.emulate_media(color_scheme="light")
    expect(rail).to_have_accessible_name("Switch to dark theme")
    expect(header).to_have_attribute("aria-label", "Switch to dark theme")
    assert not errors


def test_the_collapse_button_steps_the_rail_through_labels_icons_and_hidden(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """The collapse button folds the rail to its icons, then hides it; the header's button shows it.

    Each button carries aria-expanded and is named by what a press does; the pressed button that
    leaves with the rail hands the focus to the other one, and a reload keeps the state. The rail
    and the theme row keep their icons in the nav links' column. The rail had two states and could
    not leave the screen.
    """
    from playwright.sync_api import expect

    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/")
    toggle, show = page.locator("#rail-toggle"), page.locator("#rail-show")
    rail, labels = page.locator("aside.rail"), page.locator("aside.rail .rail-nav a span")
    icons = page.locator("aside.rail .ic")
    lefts = "icons => icons.map(i => i.getBoundingClientRect().left)"
    expect(toggle).to_have_attribute("aria-expanded", "true")
    expect(toggle).to_have_accessible_name("Collapse the sidebar")
    expect(labels.first).to_be_visible()
    expect(show).to_be_hidden()
    expect(icons).to_have_count(7)
    assert len(set(icons.evaluate_all(lefts))) == 1
    toggle.click()
    expect(toggle).to_have_attribute("aria-expanded", "false")
    expect(toggle).to_have_accessible_name("Hide the sidebar")
    expect(labels.first).to_be_hidden()
    assert len(set(icons.evaluate_all(lefts))) == 1
    page.reload()
    expect(toggle).to_have_accessible_name("Hide the sidebar")
    toggle.focus()
    page.keyboard.press("Enter")
    expect(rail).to_be_hidden()
    expect(show).to_be_focused()
    expect(show).to_have_attribute("aria-expanded", "false")
    expect(show).to_have_accessible_name("Show the sidebar")
    expect(page.locator("header .brand")).to_be_visible()
    page.reload()
    expect(page.locator("#crumb")).to_have_text("Sessions")
    expect(rail).to_be_hidden()
    show.click()
    expect(labels.first).to_be_visible()
    expect(toggle).to_be_focused()
    expect(toggle).to_have_accessible_name("Collapse the sidebar")
    expect(show).to_be_hidden()
    page.reload()
    expect(toggle).to_have_attribute("aria-expanded", "true")
    assert not errors


def test_a_rail_under_1024px_wide_shows_its_icons_and_hides(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """Under 1024px the rail is its icon strip, and a press hides it; its labels never show there.

    Its 216px of labels left an 820px tablet's run 258px of conversation beside the details.
    """
    from playwright.sync_api import expect

    page, errors = open_page(port)
    page.set_viewport_size({"width": 820, "height": 900})
    page.goto(f"http://127.0.0.1:{port}/#/")
    toggle, show = page.locator("#rail-toggle"), page.locator("#rail-show")
    content = page.locator(".content")
    expect(toggle).to_have_accessible_name("Hide the sidebar")
    expect(toggle).to_have_attribute("aria-expanded", "false")
    expect(page.locator("aside.rail .rail-label").first).to_be_hidden()
    assert content.evaluate("e => e.getBoundingClientRect().left") == 64
    toggle.click()
    expect(page.locator("aside.rail")).to_be_hidden()
    assert content.evaluate("e => e.getBoundingClientRect().left") == 0
    show.click()
    expect(toggle).to_have_accessible_name("Hide the sidebar")
    assert content.evaluate("e => e.getBoundingClientRect().left") == 64
    page.set_viewport_size({"width": 1280, "height": 900})
    expect(toggle).to_have_accessible_name("Collapse the sidebar")
    expect(page.locator("aside.rail .rail-label").first).to_be_visible()
    assert not errors


def test_the_open_pages_nav_link_is_the_current_page(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """The open page's link in the rail and the tab bar carries aria-current="page"."""
    from playwright.sync_api import expect

    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/")
    current = page.locator("[aria-current=page]")
    expect(current).to_have_text(["Sessions", "Sessions"])
    page.locator("aside.rail a[data-tab='config']").click()
    expect(current).to_have_text(["Config", "Config"])
    assert not errors


@pytest.mark.parametrize("route", ["#/session/pane-run-AAA111", "#/git", "#/files"])
def test_a_desktop_pane_fills_the_screen_under_a_header_of_any_height(
    open_page: Callable[[int], tuple[Page, list[str]]],
    git_port: int,
    tmp_path: pathlib.Path,
    route: str,
) -> None:
    """The run, git and files panes end at the screen's bottom edge, whatever the header's height.

    Each pane restated the header's height (100vh - 42px), so a taller header pushed the pane,
    and the composer at its foot, below the screen.
    """
    from playwright.sync_api import expect

    session = paths.state_dir(tmp_path) / "sessions" / "runs" / "pane-run-AAA111"
    session.mkdir(parents=True)
    start = {"type": "session.start", "mode": "run", "user_task": "Fill the pane"}
    (session / "logs.jsonl").write_text(json.dumps(start) + "\n", encoding="utf-8")
    page, errors = open_page(git_port)
    page.goto(f"http://127.0.0.1:{git_port}/{route}")
    pane = page.locator(".run-app, .git-app, .files-app")
    expect(pane).to_be_visible()
    page.add_style_tag(content="header { padding-block: 30px; }")
    header, box = page.locator("header").bounding_box(), pane.bounding_box()
    assert header is not None and box is not None
    assert box["y"] == pytest.approx(header["y"] + header["height"], abs=0.5)
    assert box["y"] + box["height"] == pytest.approx(900, abs=0.5)
    assert page.evaluate("document.scrollingElement.scrollHeight") == 900
    assert not errors


def test_a_touch_phones_page_starts_below_its_header(browser: Browser, port: int) -> None:
    """The page clears the fixed header, which takes the touch screen's taller controls.

    The page reserved a fixed 47px while the header grew to 53px with its 40px buttons, so
    the first 6px of every page and every overlay hid under the header.
    """
    from playwright.sync_api import expect

    context = browser.new_context(
        viewport={"width": 390, "height": 844}, has_touch=True, is_mobile=True
    )
    context.add_init_script(f"localStorage.setItem('a6-token', {json.dumps(signin.token(port))})")
    try:
        page = context.new_page()
        page.goto(f"http://127.0.0.1:{port}/#/")
        expect(page.get_by_role("button", name="Start", exact=True)).to_be_visible()
        header = page.locator("header").bounding_box()
        assert header is not None and header["height"] == 53
        main_top = "parseFloat(getComputedStyle(document.querySelector('main')).paddingTop)"
        assert page.evaluate(main_top) == header["height"]
    finally:
        context.close()


def test_a_long_line_in_a_phone_widget_never_widens_the_page(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, tmp_path: pathlib.Path
) -> None:
    """A tool call's or a background command's long line stays inside its card on a phone.

    The phone's cards flow at their natural size, so the line set the grid column's width, or
    overflowed a card that does not clip: the page widened and pushed the header and the tab
    bar off-screen.
    """
    from playwright.sync_api import expect

    session = paths.state_dir(tmp_path) / "sessions" / "runs" / "wide-line-AAA111"
    session.mkdir(parents=True)
    rows = [
        {"type": "session.start", "mode": "run", "user_task": "Read a deep file"},
        {"type": "tool.call", "name": "read_file", "args_preview": {"path": "src/" + "a" * 400}},
        {"type": "tool.result", "name": "read_file", "ok": True, "summary": "12 bytes"},
        {"type": "session.stop", "reason": "finish_session", "verification": "passed"},
    ]
    (session / "logs.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    shell = session / "shells" / "bg1"
    shell.mkdir(parents=True)
    (shell / "meta.json").write_text(json.dumps({"command": "curl https://" + "a" * 400}))
    page, errors = open_page(port)
    page.set_viewport_size({"width": 390, "height": 844})
    page.goto(f"http://127.0.0.1:{port}/#/session/{session.name}")
    for widget, line in (("Tool calls", ".card-tools .args"), ("Background shells", ".shell")):
        page.locator(".wmenu-btn").click()
        page.locator(".wmenu").get_by_role("button", name=widget, exact=True).click()
        expect(page.locator(line)).to_be_visible()
        assert page.evaluate("document.scrollingElement.scrollWidth") == 390, widget
    assert not errors


def test_a_composer_entry_rests_at_one_line_and_grows_to_a_cap(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """The docked entry is one field tall at rest and grows with each typed line.

    It stops at 40% of the screen, shrinks back when emptied, and keeps Start on its bottom
    edge. It was two lines tall at rest, and only a drag grip above it made it taller.
    """
    from playwright.sync_api import expect

    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/")
    entry = page.locator(".dock textarea")
    expect(entry).to_be_visible()
    start = page.get_by_role("button", name="Start", exact=True)

    def box(locator: Locator) -> tuple[float, float]:
        b = locator.bounding_box()
        assert b is not None
        return b["height"], b["y"] + b["height"]

    assert box(entry)[0] == 32
    entry.fill("one\ntwo\nthree\nfour")
    assert box(entry)[0] == 32 + 3 * 21
    assert box(start)[1] == box(entry)[1]
    entry.fill("\n".join(["line"] * 60))
    assert box(entry)[0] == pytest.approx(900 * 0.4, abs=0.5)
    assert box(start)[1] == box(entry)[1]
    entry.fill("")
    assert box(entry)[0] == 32
    assert not errors


def test_a_live_conversation_follows_its_tail_while_a_steer_grows(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, tmp_path: pathlib.Path
) -> None:
    """A new item lands in view while a steer of several lines is typed under the conversation.

    The growing entry shrank the conversation box, so the box read as scrolled away from its
    tail and stopped following: the new item landed below the box's visible part.
    """
    import os

    from playwright.sync_api import expect

    from agent6.sessions import process

    session = paths.state_dir(tmp_path) / "sessions" / "runs" / "follow-live-AAA111"
    session.mkdir(parents=True)
    process.write_worker_pid(session, os.getpid())
    rows: list[dict[str, object]] = [
        {"type": "session.start", "mode": "run", "user_task": "Follow the tail"}
    ]
    for i in range(40):
        path = {"path": f"f{i}.py"}
        rows.append({"type": "tool.call", "name": "read_file", "args_preview": path})
        rows.append({"type": "tool.result", "name": "read_file", "ok": True, "summary": "1 byte"})
    log = session / "logs.jsonl"
    log.write_text("".join(json.dumps(row) + "\n" for row in rows))
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/session/{session.name}")
    # A paint renders the items and scrolls to the tail in one task.
    in_box = """text => {
      const box = document.querySelector('.card-conv .conv-box').getBoundingClientRect();
      const item = [...document.querySelectorAll('.card-conv .ci')]
        .find(e => e.textContent.includes(text));
      const r = item.getBoundingClientRect();
      return r.top >= box.top && r.bottom <= box.bottom;
    }"""
    items = page.locator(".card-conv .ci")
    expect(items.filter(has_text="f39.py")).to_be_attached()
    assert page.evaluate(in_box, "f39.py")
    page.locator(".run-app .composer textarea").fill("check\nthe\ntests\ntoo")
    new = {"type": "tool.call", "name": "read_file", "args_preview": {"path": "new.py"}}
    with log.open("a") as stream:
        stream.write(json.dumps(new) + "\n")
    expect(items.filter(has_text="new.py")).to_be_attached()
    assert page.evaluate(in_box, "new.py")
    assert not errors


@pytest.mark.parametrize(("width", "height"), [(1280, 900), (390, 844)])
def test_the_last_session_row_scrolls_clear_of_the_dock(
    open_page: Callable[[int], tuple[Page, list[str]]],
    port: int,
    tmp_path: pathlib.Path,
    width: int,
    height: int,
) -> None:
    """At the end of the Sessions page the last row sits above the dock, however tall it grows.

    The page reserved a fixed space under a fixed dock: on a phone the dock outgrew it and hid
    the last row, and a grown entry hid it on a desktop too.
    """
    from playwright.sync_api import expect

    runs = paths.state_dir(tmp_path) / "sessions" / "runs"
    for i in range(24):
        run = runs / f"row-{i:02d}-AAAAAA"
        run.mkdir(parents=True)
        start = {"type": "session.start", "mode": "run", "user_task": f"Task {i}"}
        (run / "logs.jsonl").write_text(json.dumps(start) + "\n", encoding="utf-8")
    page, errors = open_page(port)
    page.set_viewport_size({"width": width, "height": height})
    page.goto(f"http://127.0.0.1:{port}/#/")
    expect(page.locator(".item")).to_have_count(24)
    # Scroll and measure in one task: the model list landing later may grow the dock.
    gap_at_end = """() => {
      window.scrollTo(0, document.scrollingElement.scrollHeight);
      const last = [...document.querySelectorAll('.item')].at(-1).getBoundingClientRect();
      return document.querySelector('.dock').getBoundingClientRect().top - last.bottom;
    }"""
    for text in ("", "\n".join(["line"] * 60)):
        page.locator(".dock textarea").fill(text)
        assert page.evaluate(gap_at_end) >= 0
    assert not errors


@pytest.mark.parametrize(("width", "height"), [(640, 400), (512, 384), (390, 500), (390, 600)])
def test_a_short_screens_dock_fits_between_the_bars_and_covers_no_row(
    open_page: Callable[[int], tuple[Page, list[str]]],
    port: int,
    tmp_path: pathlib.Path,
    width: int,
    height: int,
) -> None:
    """On a short screen the open dock slid under the header and hid the row the keyboard reached.

    A window at 200% zoom or a phone under its keyboard: a message rose off the top with the dock.
    At 600px the sticky dock filled the room; at the page's end a message covered Start.
    """
    from playwright.sync_api import expect

    runs = paths.state_dir(tmp_path) / "sessions" / "runs"
    for i in range(24):
        run = runs / f"row-{i:02d}-AAAAAA"
        run.mkdir(parents=True)
        start = {"type": "session.start", "mode": "run", "user_task": f"Task {i}"}
        (run / "logs.jsonl").write_text(json.dumps(start) + "\n", encoding="utf-8")
    page, errors = open_page(port)
    page.set_viewport_size({"width": width, "height": height})
    page.goto(f"http://127.0.0.1:{port}/#/")
    expect(page.locator(".item")).to_have_count(24)
    page.evaluate("clearInterval(hubTimer)")  # a refresh must not repaint mid-walk
    summary = page.locator(".dock .picker-summary")
    summary.click()
    page.locator(".dock .pickers").get_by_role("button", name="advanced", exact=True).click()
    page.locator(".dock textarea").fill("\n".join(f"step {n}" for n in range(12)))  # 40dvh tall
    # Scroll and hit-test in one task: what a click at each control's centre would reach.
    reached = """() => {
      window.scrollTo(0, document.scrollingElement.scrollHeight);
      return ['.dock .picker-summary', '.dock button.primary'].map(s => {
        const b = document.querySelector(s).getBoundingClientRect();
        return !!document.elementFromPoint(b.x + 8, b.y + b.height / 2)?.closest(s);
      });
    }"""
    assert page.evaluate(reached) == [True, True]
    summary.focus()
    clear = """() => {
      const row = document.activeElement.closest('.item').getBoundingClientRect();
      const dock = document.querySelector('.dock').getBoundingClientRect();
      const tabs = document.querySelector('nav.tabs').getBoundingClientRect();
      return row.top >= document.querySelector('header').getBoundingClientRect().bottom
        && row.bottom <= Math.min(dock.top, tabs.top);
    }"""
    for _ in range(3):
        page.keyboard.press("Shift+Tab")
        assert page.evaluate(clear)
    # A message raised at the page's end shows between the bars, off the dock's Start.
    message = """() => {
      window.scrollTo(0, document.scrollingElement.scrollHeight);
      toast('probe', true);
      const t = document.querySelector('.toast').getBoundingClientRect();
      const go = document.querySelector('.dock button.primary').getBoundingClientRect();
      return [t.top >= document.querySelector('header').getBoundingClientRect().bottom,
        t.bottom <= document.querySelector('nav.tabs').getBoundingClientRect().top,
        t.bottom <= go.top || t.top >= go.bottom];
    }"""
    assert page.evaluate(message) == [True, True, True]
    assert not errors


def test_a_dock_at_the_pages_end_covers_no_height(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """A dock that stopped sticking on a short screen still read as covering its height."""
    from playwright.sync_api import expect

    page, errors = open_page(port)
    page.set_viewport_size({"width": 390, "height": 844})
    page.goto(f"http://127.0.0.1:{port}/#/")
    expect(page.locator(".dock")).to_be_visible()
    covers = "getComputedStyle(document.documentElement).getPropertyValue('--dock-h') === "
    height = "document.querySelector('.dock').getBoundingClientRect().height + 'px'"
    page.wait_for_function(covers + height)
    page.set_viewport_size({"width": 390, "height": 500})  # the dock stays at the page's end
    page.wait_for_function(covers + "'0px'")
    page.set_viewport_size({"width": 390, "height": 844})
    page.wait_for_function(covers + height)
    assert not errors


def test_every_expand_control_draws_the_folds_chevron(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """A fold, a Config section heading and a Files folder drew three kinds of arrow.

    The class .chevron gives any of them the fold's, turned down while it, its row or its details
    is open.
    """
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/")
    page.wait_for_selector(".dock .pickers button.fold")
    page.evaluate("""() => document.body.insertAdjacentHTML('beforeend',
      '<details open><summary class="chevron">open</summary></details>'
      + '<details><summary class="chevron">closed</summary></details>'
      + '<div aria-expanded="true"><span class="chevron"></span></div>'
      + '<div aria-expanded="false"><span class="chevron"></span></div>')""")
    drawn = """es => es.map(e => { const s = getComputedStyle(e, '::before');
      return [s.width, s.height, s.borderRightWidth, s.rotate]; })"""
    fold = page.locator(".dock .pickers button.fold").evaluate(f"e => ({drawn})([e])[0]")
    assert fold[3] == "-45deg"
    turns = ["45deg", "-45deg", "45deg", "-45deg"]
    assert page.locator(".chevron").evaluate_all(drawn) == [[*fold[:3], t] for t in turns]
    assert not errors


@pytest.mark.parametrize(("width", "height"), [(390, 844), (820, 1000)])
def test_a_narrow_commit_message_takes_the_whole_row(
    open_page: Callable[[int], tuple[Page, list[str]]],
    dirty_git_port: int,
    width: int,
    height: int,
) -> None:
    """On a phone or in a narrow side panel the commit message spans its row, the button under it.

    Beside Commit staged, the one-line message cut its placeholder short on a phone, and in an
    820px window's side panel it shrank to a column a few characters wide.
    """
    from playwright.sync_api import expect

    page, errors = open_page(dirty_git_port)
    page.set_viewport_size({"width": width, "height": height})
    page.goto(f"http://127.0.0.1:{dirty_git_port}/#/git/worktree")
    message = page.locator(".git-commit textarea")
    expect(message).to_have_attribute("placeholder", "commit message for 1 file staged…")
    row, box = page.locator(".git-commit").bounding_box(), message.bounding_box()
    commit = page.get_by_role("button", name="Commit staged", exact=True).bounding_box()
    assert row is not None and box is not None and commit is not None
    assert box["width"] == row["width"]
    assert commit["y"] >= box["y"] + box["height"]
    assert not errors


def test_files_tree_width_survives_reload_and_keyboard_resize(
    open_page: Callable[[int], tuple[Page, list[str]]], git_port: int
) -> None:
    """Dragging changes the split, and a reload retains drag and arrow-key widths."""
    from playwright.sync_api import expect

    page, errors = open_page(git_port)
    page.goto(f"http://127.0.0.1:{git_port}/#/files/HEAD/b.txt")
    tree = page.locator(".files-tree")
    expect(tree.locator("[data-path='b.txt']")).to_be_visible()
    before = _width(tree)
    handle = page.get_by_role("separator", name="File tree width")
    expect(handle).to_be_visible()
    _drag(page, handle, 120)
    assert _width(tree) == pytest.approx(before + 120, abs=1)
    page.reload()
    expect(tree.locator("[data-path='b.txt']")).to_be_visible()
    assert _width(tree) == pytest.approx(before + 120, abs=1)
    handle.focus()
    handle.press("ArrowLeft")
    assert _width(tree) == pytest.approx(before + 100, abs=1)
    handle.press("ArrowRight")
    assert _width(tree) == pytest.approx(before + 120, abs=1)
    expect(handle).to_have_attribute("aria-valuenow", str(round(_width(tree))))
    page.reload()
    expect(tree.locator("[data-path='b.txt']")).to_be_visible()
    assert _width(tree) == pytest.approx(before + 120, abs=1)
    assert not errors


def test_files_collapse_is_remembered_but_does_not_hide_the_phone_tree(
    open_page: Callable[[int], tuple[Page, list[str]]], git_port: int
) -> None:
    """The desktop toggle frees the file pane without changing the phone's file overlay."""
    from playwright.sync_api import expect

    page, errors = open_page(git_port)
    page.goto(f"http://127.0.0.1:{git_port}/#/files/HEAD/b.txt")
    tree = page.locator(".files-tree")
    pane = page.locator(".files-view")
    expect(pane.locator(".sl").first).to_be_visible()
    before = _width(tree)
    hide = page.get_by_role("button", name="Hide tree", exact=True)
    expect(hide).to_be_visible()
    hide.press("Enter")
    expect(tree).to_be_hidden()
    assert _width(pane) == pytest.approx(_width(page.locator(".files-body")), abs=1)
    page.reload()
    expect(pane.locator(".sl").first).to_be_visible()
    expect(tree).to_be_hidden()
    show = page.get_by_role("button", name="Show tree", exact=True)
    expect(show).to_have_attribute("aria-expanded", "false")
    show.press("Space")
    expect(tree).to_be_visible()
    assert _width(tree) == pytest.approx(before, abs=1)
    hide.click()
    page.set_viewport_size({"width": 390, "height": 844})
    expect(show).to_be_hidden()
    expect(page.get_by_role("separator", name="File tree width")).to_be_hidden()
    assert _width(pane) == pytest.approx(390, abs=1)
    pane.get_by_role("button", name="\u2039 Files", exact=True).click()
    expect(pane).to_be_hidden()
    expect(tree.locator("[data-path='b.txt']")).to_be_visible()
    tree.locator("[data-path='b.txt']").click()
    expect(pane.locator(".sl").first).to_be_visible()
    page.set_viewport_size({"width": 1280, "height": 900})
    expect(tree).to_be_hidden()
    expect(show).to_be_visible()
    assert not errors


@pytest.mark.parametrize("saved", ["100000px", "-20px", "not a width"])
def test_files_restored_width_fits_the_current_container(
    open_page: Callable[[int], tuple[Page, list[str]]], git_port: int, saved: str
) -> None:
    """An old or malformed width cannot squeeze the file out of a narrower container."""
    from playwright.sync_api import expect

    page, errors = open_page(git_port)
    page.add_init_script(f"localStorage.setItem('a6-files-tree-w', {json.dumps(saved)})")
    page.set_viewport_size({"width": 900, "height": 900})
    page.goto(f"http://127.0.0.1:{git_port}/#/files/HEAD/b.txt")
    expect(page.get_by_role("separator", name="File tree width")).to_be_visible()
    tree, pane = page.locator(".files-tree"), page.locator(".files-view")
    assert _width(tree) >= 180
    assert _width(pane) >= 240
    page.set_viewport_size({"width": 800, "height": 900})
    assert _width(tree) >= 180
    assert _width(pane) >= 240
    assert _width(tree) + _width(pane) <= _width(page.locator(".files-body")) + 0.5
    assert not errors


def test_files_layout_works_when_browser_storage_is_blocked(
    browser: Browser, git_port: int
) -> None:
    """Unavailable storage must not prevent sign-in, resizing or collapsing the Files tree."""
    from playwright.sync_api import expect

    context = browser.new_context(viewport={"width": 1280, "height": 900})
    context.add_init_script("""
Object.defineProperty(window, 'localStorage', {
  get() { throw new DOMException('storage blocked', 'SecurityError'); }
});
""")
    try:
        page = context.new_page()
        errors: list[str] = []
        page.on("pageerror", lambda exc: errors.append(str(exc)))
        page.goto(f"http://127.0.0.1:{git_port}/#/files/HEAD/b.txt")
        page.locator("#login input").fill(signin.SECRET)
        page.get_by_role("button", name="Sign in", exact=True).click()
        tree = page.locator(".files-tree")
        expect(tree.locator("[data-path='b.txt']")).to_be_visible()
        before = _width(tree)
        _drag(page, page.get_by_role("separator", name="File tree width"), 80)
        assert _width(tree) == pytest.approx(before + 80, abs=1)
        page.get_by_role("button", name="Hide tree", exact=True).click()
        expect(tree).to_be_hidden()
        page.get_by_role("button", name="Show tree", exact=True).click()
        expect(tree).to_be_visible()
        assert not errors
    finally:
        context.close()


def test_run_drawer_keeps_drag_resize_reload_and_reset(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, tmp_path: pathlib.Path
) -> None:
    """Sharing the divider retains the drawer's saved width and double-click reset."""
    from playwright.sync_api import expect

    run = paths.state_dir(tmp_path) / "sessions" / "runs" / "layout-run"
    run.mkdir(parents=True)
    (run / "logs.jsonl").write_text(
        json.dumps({"type": "session.start", "mode": "run", "user_task": "Check the drawer"})
        + "\n",
        encoding="utf-8",
    )
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/session/layout-run")
    drawer = page.locator(".drawer")
    expect(drawer).to_be_visible()
    before = _width(drawer)
    handle = page.get_by_role("separator", name="Details width")
    expect(handle).to_be_visible()
    _drag(page, handle, 100)
    assert _width(drawer) == pytest.approx(before + 100, abs=1)
    page.reload()
    expect(drawer).to_be_visible()
    assert _width(drawer) == pytest.approx(before + 100, abs=1)
    handle.press("ArrowLeft")
    assert _width(drawer) == pytest.approx(before + 80, abs=1)
    handle.dblclick()
    assert _width(drawer) == pytest.approx(before, abs=1)
    page.reload()
    expect(drawer).to_be_visible()
    assert _width(drawer) == pytest.approx(before, abs=1)
    assert not errors
