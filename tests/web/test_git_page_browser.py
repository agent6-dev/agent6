# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The Git page driven in a browser: its layout, rows, keyboard, focus and waiting states."""

from __future__ import annotations

import pathlib
import re
from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest

from agent6.ui.web import git_model
from tests.unit.test_git_history import _git
from tests.web import test_web_browser

if TYPE_CHECKING:
    from playwright.sync_api import Page, Route

pytestmark = pytest.mark.browser
browser = test_web_browser.browser
open_page = test_web_browser.open_page
port = test_web_browser.port
git_port = test_web_browser.git_port
dirty_git_port = test_web_browser.dirty_git_port

OpenPage = Callable[[int], tuple["Page", list[str]]]
# A computed style the Open in Files link shares with the mini buttons beside it.
SKIN = """e => { const s = getComputedStyle(e);
  return [s.color, s.backgroundColor, s.borderTopColor, s.borderTopLeftRadius, s.fontSize,
    Math.round(e.getBoundingClientRect().height)]; }"""


def _held(page: Page, pattern: str) -> list[Route]:
    """Hold every request matching `pattern` until the test continues it."""
    held: list[Route] = []

    def hold(route: Route) -> None:
        held.append(route)

    page.route(pattern, hold)
    return held


def _first(page: Page, held: list[Route]) -> Route:
    """Wait for the first held request to arrive."""
    while not held:
        page.wait_for_timeout(50)
    return held[0]


def _default_font(page: Page, px: int) -> None:
    """Set Chromium's default font size, as a reader's browser setting does."""
    cdp = page.context.new_cdp_session(page)
    cdp.send("Page.enable")
    cdp.send("Page.setFontSizes", {"fontSizes": {"standard": px, "fixed": round(px * 13 / 16)}})


def _long_history(path: pathlib.Path) -> str:
    """Add twenty empty commits, then one that adds forty files; return the last one's sha."""
    for n in range(20):
        _git(path, "commit", "-q", "--allow-empty", "-m", f"empty {n}")
    for n in range(40):
        (path / f"f{n:02d}.txt").write_text(f"{n}\n", encoding="utf-8")
    _git(path, "add", "-A")
    _git(path, "commit", "-q", "-m", "forty files")
    return _git(path, "rev-parse", "HEAD")


def test_a_graph_row_shows_its_first_ref_and_counts_the_rest(
    open_page: OpenPage, git_port: int, tmp_path: pathlib.Path
) -> None:
    """A row's ref badges hid its subject; one badge in git's order hid HEAD's branch.

    The row shows HEAD's ref, else a branch, a tag, a remote-tracking branch, an agent6 run ref.
    """
    page, errors = open_page(git_port)
    page.goto(f"http://127.0.0.1:{git_port}/#/git")

    def badges(subject: str) -> tuple[list[str], str | None]:
        row = page.locator(".gc-row", has_text=subject)
        return row.locator(".gref").all_inner_texts(), row.locator(".more").get_attribute("title")

    page.locator(".gc-row").first.wait_for()
    assert badges("merge feature") == (["main", "+2"], "v2, origin/main")
    assert badges("feature: rename") == (["feature", "+1"], "agent6/run-1/head")
    _git(tmp_path, "checkout", "-q", "-b", "topic")  # HEAD on a branch git lists after main
    page.reload()
    page.locator(".gc-row").first.wait_for()
    assert badges("merge feature") == (["topic", "+3"], "main, v2, origin/main")
    assert errors == [], errors


def test_a_narrow_graph_rows_badge_leaves_its_subject_room(
    open_page: OpenPage, git_port: int, tmp_path: pathlib.Path
) -> None:
    """At 820px a long branch name and the ref count left a row's subject 20px of its column."""
    _git(tmp_path, "branch", "a-long-descriptive-branch-name", "feature")
    page, errors = open_page(git_port)
    page.set_viewport_size({"width": 820, "height": 1180})
    page.goto(f"http://127.0.0.1:{git_port}/#/git")
    row = page.locator(".gc-row", has_text="feature: rename")
    shown, column = row.evaluate(
        "r => { const s = r.querySelector('.gc-subj').getBoundingClientRect();"
        " return [s.right - r.querySelector('.gc-text').getBoundingClientRect().left, s.width]; }"
    )
    assert shown >= column / 3, (shown, column)
    assert errors == [], errors


def test_arrow_keys_move_between_graph_rows_and_tab_stops_at_one(
    open_page: OpenPage, git_port: int
) -> None:
    """Arrow keys scrolled the graph under a focused row, and every row was a tab stop.

    A row's name lists the refs, subject, author and date it shows; its lanes are hidden.
    """
    from playwright.sync_api import expect

    page, errors = open_page(git_port)
    page.goto(f"http://127.0.0.1:{git_port}/#/git")
    rows = page.locator(".gc-row")
    rows.first.wait_for()
    expect(page.locator(".git-graph [tabindex='0']")).to_have_count(1)
    rows.first.focus()
    page.keyboard.press("ArrowDown")
    expect(rows.nth(1)).to_be_focused()
    page.keyboard.press("End")
    expect(rows.last).to_be_focused()
    page.keyboard.press("Home")
    expect(rows.first).to_be_focused()
    expect(page.locator(".git-graph [tabindex='0']")).to_have_count(1)
    # HEAD is the merge, where main, origin/main and v2 point.
    name = rows.first.get_attribute("aria-label") or ""
    assert name.startswith("main, v2, origin/main, merge feature, "), name
    assert rows.first.locator("svg").get_attribute("aria-hidden") == "true"
    assert errors == [], errors


def test_arrow_keys_keep_a_phones_rows_in_view(
    open_page: OpenPage, git_port: int, tmp_path: pathlib.Path
) -> None:
    """On a phone End and Home left the focused graph row under the tab bar or the top bar.

    A file row scrolled its button into view and left the rest of the row and its ring outside.
    """
    sha = _long_history(tmp_path)
    page, errors = open_page(git_port)
    page.set_viewport_size({"width": 390, "height": 600})
    page.goto(f"http://127.0.0.1:{git_port}/#/git")
    box = (
        "e => { const r = (e.closest('.gf') || e).getBoundingClientRect();"
        " return [r.top, r.bottom]; }"
    )
    top = page.locator("header").evaluate("e => e.getBoundingClientRect().bottom")
    bottom = page.locator("nav.tabs").evaluate("e => e.getBoundingClientRect().top")
    rows = page.locator(".gc-row")
    rows.first.focus()
    page.keyboard.press("End")
    assert rows.last.evaluate(box)[1] <= bottom
    page.keyboard.press("Home")
    assert rows.first.evaluate(box)[0] >= top
    page.goto(f"http://127.0.0.1:{git_port}/#/git/commit/{sha}")
    files = page.locator(".git-side .gf-open")
    files.first.focus()
    page.keyboard.press("End")
    side = page.locator(".git-side").evaluate("e => e.getBoundingClientRect().bottom")
    assert files.last.evaluate(box)[1] <= side
    assert errors == [], errors


def test_a_working_tree_rows_buttons_sit_beside_its_path_not_inside_a_button(
    open_page: OpenPage, dirty_git_port: int
) -> None:
    """A working-tree row was a button holding Stage, Unstage and Discard: nested controls."""
    from playwright.sync_api import expect

    page, errors = open_page(dirty_git_port)
    page.goto(f"http://127.0.0.1:{dirty_git_port}/#/git/worktree")
    row = page.locator(".git-side .gf[data-path='b.txt']")
    row.get_by_role("button", name="Stage", exact=True).wait_for()
    expect(page.locator(".git-side [role=button] button")).to_have_count(0)
    expect(row.get_by_role("button", name="M b.txt", exact=True)).to_have_count(1)
    row.click(position={"x": 4, "y": 4})  # the row's corner, away from the path
    page.wait_for_selector(".git-diff .diff")
    assert errors == [], errors


def test_hovering_a_file_rows_letter_names_its_state(
    open_page: OpenPage, dirty_git_port: int, tmp_path: pathlib.Path
) -> None:
    """The row's button covered the status letter, so hovering U or M showed no word."""
    _git(tmp_path, "add", "b.txt")
    (tmp_path / "b.txt").write_text("one\ntwo\nthree\nfour\nfive\n", encoding="utf-8")
    page, errors = open_page(dirty_git_port)
    page.goto(f"http://127.0.0.1:{dirty_git_port}/#/git/worktree")
    page.wait_for_selector(".git-side .gf")
    tooltip = (
        "p => { const b = [...document.querySelectorAll('.git-side .gf')]"
        ".find(r => r.dataset.path === p).querySelector('.gf-st').getBoundingClientRect();"
        " return document.elementFromPoint(b.x + b.width / 2, b.y + b.height / 2)"
        ".closest('[title]')?.title; }"
    )
    assert page.evaluate(tooltip, "new.txt") == "untracked"
    assert page.evaluate(tooltip, "b.txt") == "modified, partly staged"
    assert errors == [], errors


def test_a_phones_commit_panel_keeps_its_back_button_in_view(
    open_page: OpenPage, git_port: int
) -> None:
    """The back button scrolled away with a long commit panel on a phone."""
    page, errors = open_page(git_port)
    page.set_viewport_size({"width": 390, "height": 360})
    page.goto(f"http://127.0.0.1:{git_port}/#/git")
    page.locator(".gc-row").first.click()
    side = page.locator(".git-side.open")
    page.wait_for_selector(".git-side .gf")
    assert side.evaluate("s => { s.scrollTo(0, s.scrollHeight); return s.scrollTop > 0; }")
    back = page.get_by_role("button", name="\u2039 graph", exact=True)
    panel, button = side.bounding_box(), back.bounding_box()
    assert panel and button and button["y"] >= panel["y"], (panel, button)
    assert errors == [], errors


def test_a_file_row_the_arrow_keys_reach_stays_below_the_panels_head(
    open_page: OpenPage, git_port: int, tmp_path: pathlib.Path
) -> None:
    """Home in a long file list scrolled the first row under the side panel's sticky head."""
    sha = _long_history(tmp_path)
    page, errors = open_page(git_port)
    page.goto(f"http://127.0.0.1:{git_port}/#/git/commit/{sha}")
    files = page.locator(".git-side .gf-open")
    files.first.focus()
    page.keyboard.press("End")
    page.keyboard.press("Home")
    head = page.locator(".git-side-head").evaluate("e => e.getBoundingClientRect().bottom")
    top = files.first.evaluate("e => e.closest('.gf').getBoundingClientRect().top")
    assert top >= head, (top, head)
    assert errors == [], errors


def test_the_focus_follows_a_file_into_its_diff_and_back(
    open_page: OpenPage, git_port: int
) -> None:
    """Opening or closing a file's diff, or switching its mode, dropped the focus to the page."""
    from playwright.sync_api import expect

    page, errors = open_page(git_port)
    page.goto(f"http://127.0.0.1:{git_port}/#/git")
    page.locator(".gc-row").first.click()  # the merge: b.txt and img.bin
    page.locator(".git-side .gf[data-path='b.txt']").click()
    page.wait_for_selector(".git-diff .diff")
    head = page.locator(".git-diff-head")
    expect(head.get_by_role("button", name="\u2039 files", exact=True)).to_be_focused()
    head.get_by_role("button", name="split", exact=True).click()
    page.wait_for_selector(".git-diff .ds-row")
    expect(head.get_by_role("button", name="split", exact=True)).to_be_focused()
    head.get_by_role("button", name="\u2039 files", exact=True).click()
    focused = "document.activeElement.closest('.gf')?.dataset.path"
    page.wait_for_function(f"{focused} === 'b.txt'")
    expect(page.locator(".git-side .gf.sel")).to_have_attribute("data-path", "b.txt")
    assert errors == [], errors


def test_load_more_hands_the_focus_to_the_first_row_it_adds(
    open_page: OpenPage, git_port: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Load more rebuilt itself when its page came in, and the focus it held dropped to the page."""
    from playwright.sync_api import expect

    monkeypatch.setattr(git_model, "GRAPH_PAGE", 2)
    page, errors = open_page(git_port)
    page.goto(f"http://127.0.0.1:{git_port}/#/git")
    page.get_by_role("button", name="Load more", exact=True).focus()
    page.keyboard.press("Enter")
    expect(page.locator(".gc-row").nth(2)).to_be_focused()
    assert errors == [], errors


def test_closing_a_phones_panel_returns_the_focus_to_the_graph(
    open_page: OpenPage, git_port: int
) -> None:
    """Closing a phone's commit panel hid the focused back button and dropped the focus."""
    from playwright.sync_api import expect

    page, errors = open_page(git_port)
    page.set_viewport_size({"width": 390, "height": 844})
    page.goto(f"http://127.0.0.1:{git_port}/#/git")
    rows = page.locator(".gc-row")
    rows.nth(1).click()
    page.get_by_role("button", name="\u2039 graph", exact=True).click()
    expect(rows.nth(1)).to_be_focused()
    assert errors == [], errors


def test_a_graph_that_fails_to_load_reads_as_an_error(open_page: OpenPage, git_port: int) -> None:
    """A graph request that failed left 'no commits yet' and '0 commits', as an empty repository."""
    from playwright.sync_api import expect

    page, errors = open_page(git_port)
    page.goto(f"http://127.0.0.1:{git_port}/#/git")
    page.locator(".gc-row").first.wait_for()
    page.route(
        "**/api/git/graph?*", lambda route: route.fulfill(status=500, json={"error": "it broke"})
    )
    page.get_by_role("button", name="refresh", exact=True).click()
    expect(page.locator(".git-graph .err")).to_have_text("error: it broke")
    expect(page.locator(".git-graph")).not_to_contain_text("no commits yet")
    expect(page.locator(".git-bar .git-counts").first).to_have_text("5 of 7 refs")
    assert [e for e in errors if "status of 500" not in e] == [], errors


def test_refresh_fetch_and_load_more_wait_while_their_requests_run(
    open_page: OpenPage, git_port: int, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refresh and Load more stayed live and the graph read idle while it loaded; Fetch dimmed."""
    from playwright.sync_api import expect

    monkeypatch.setattr(git_model, "GRAPH_PAGE", 2)
    _git(tmp_path, "remote", "add", "up", str(tmp_path))
    page, errors = open_page(git_port)
    page.goto(f"http://127.0.0.1:{git_port}/#/git")
    graph, more = page.locator(".git-graph"), page.locator(".git-more")
    more.wait_for()
    held = _held(page, "**/api/git/graph*")
    more.click()
    request = _first(page, held)
    expect(more).to_have_text("Loading…")
    expect(more).to_be_disabled()
    expect(graph).to_have_attribute("aria-busy", "true")
    request.continue_()
    expect(graph).to_have_attribute("aria-busy", "false")
    held.clear()
    refresh = page.get_by_role("button", name="refresh", exact=True)
    refresh.click()
    request = _first(page, held)
    expect(refresh).to_be_disabled()
    expect(graph).to_have_attribute("aria-busy", "true")
    request.continue_()
    page.unroute("**/api/git/graph*")
    expect(refresh).to_be_enabled()
    expect(graph).to_have_attribute("aria-busy", "false")
    fetching = _held(page, "**/api/git/fetch")
    page.get_by_role("button", name="Fetch", exact=True).click()
    fetch = _first(page, fetching)
    expect(page.get_by_role("button", name="Fetching…", exact=True)).to_be_disabled()
    expect(page.get_by_role("button", name="Fetch PR", exact=True)).to_be_disabled()
    fetch.fulfill(json={"ok": True, "message": "fetched up"})
    expect(page.get_by_role("button", name="Fetch", exact=True)).to_be_enabled()
    assert errors == [], errors


def test_typing_in_the_toolbars_fields_keeps_the_toolbar_still(
    open_page: OpenPage, git_port: int, tmp_path: pathlib.Path
) -> None:
    """The new-branch field grew with each key until it took the next row and moved the graph."""
    _git(tmp_path, "remote", "add", "up", str(tmp_path))
    page, errors = open_page(git_port)
    page.goto(f"http://127.0.0.1:{git_port}/#/git")
    name, pull = page.get_by_placeholder("new branch…"), page.get_by_placeholder("PR #")
    graph = page.locator(".git-graph")
    page.locator(".gc-row").first.wait_for()
    before = name.bounding_box(), pull.bounding_box(), graph.bounding_box()
    name.press_sequentially("feature/a-fairly-long-branch-name-typed-here")
    pull.press_sequentially("1234567")
    assert (name.bounding_box(), pull.bounding_box(), graph.bounding_box()) == before
    assert errors == [], errors


def test_create_fetch_pr_and_commit_wait_for_their_input(
    open_page: OpenPage, dirty_git_port: int, tmp_path: pathlib.Path
) -> None:
    """Create ignored an empty name; Fetch PR and Commit staged answered it with an error toast."""
    from playwright.sync_api import expect

    _git(tmp_path, "remote", "add", "up", str(tmp_path))
    page, errors = open_page(dirty_git_port)
    page.goto(f"http://127.0.0.1:{dirty_git_port}/#/git/worktree")
    create = page.get_by_role("button", name="Create", exact=True)
    expect(create).to_be_disabled()
    page.get_by_placeholder("new branch…").fill("topic")
    expect(create).to_be_enabled()
    pull = page.get_by_role("button", name="Fetch PR", exact=True)
    expect(pull).to_be_disabled()
    page.get_by_placeholder("PR #").fill("#0")
    expect(pull).to_be_disabled()
    page.get_by_placeholder("PR #").fill("#12")
    expect(pull).to_be_enabled()
    commit = page.get_by_role("button", name="Commit staged", exact=True)
    expect(commit).to_be_disabled()  # the rename is staged; the message is empty
    page.locator(".git-commit textarea").fill("rename f to g")
    expect(commit).to_be_enabled()
    assert errors == [], errors


def test_create_from_the_keyboard_hands_its_focus_to_the_name_field(
    open_page: OpenPage, git_port: int
) -> None:
    """Create disabled itself while it held the focus, and the focus dropped to the page."""
    from playwright.sync_api import expect

    page, errors = open_page(git_port)
    page.goto(f"http://127.0.0.1:{git_port}/#/git")
    name = page.get_by_placeholder("new branch…")
    name.fill("topic")
    page.get_by_role("button", name="Create", exact=True).focus()
    page.keyboard.press("Enter")
    expect(name).to_be_focused()
    expect(page.locator(".git-branch select").first).to_have_value("topic")  # re-read
    expect(name).to_be_focused()
    assert errors == [], errors


def test_a_write_from_the_keyboard_keeps_the_focus_on_the_page(
    open_page: OpenPage, dirty_git_port: int, tmp_path: pathlib.Path
) -> None:
    """Switch, Stage and Commit staged pressed from the keyboard dropped the focus to the page.

    Switch disabled itself once its branch was checked out; a write repainted the working-tree
    panel and the button that held the focus with it.
    """
    from playwright.sync_api import expect

    _git(tmp_path, "branch", "twin")  # at HEAD, so the uncommitted files move with the switch
    page, errors = open_page(dirty_git_port)
    page.goto(f"http://127.0.0.1:{dirty_git_port}/#/git/worktree")
    side = page.locator(".git-side")
    side.locator(".gf").first.wait_for()
    branches = page.locator(".git-branch select").first
    branches.select_option("twin")
    page.get_by_role("button", name="Switch", exact=True).focus()
    page.keyboard.press("Enter")
    expect(page.locator("#toasts")).to_contain_text("on branch twin")
    expect(page.get_by_role("button", name="Switch", exact=True)).to_be_disabled()
    expect(branches).to_be_focused()
    row = side.locator(".gf[data-path='new.txt']")
    row.get_by_role("button", name="Stage", exact=True).focus()
    page.keyboard.press("Enter")
    expect(row.get_by_role("button", name="Unstage", exact=True)).to_be_visible()
    expect(row.locator(".gf-open")).to_be_focused()
    side.locator(".git-commit textarea").fill("add new.txt")
    side.get_by_role("button", name="Commit staged", exact=True).focus()
    page.keyboard.press("Enter")
    expect(side.locator(".gf[data-path='new.txt']")).to_have_count(0)
    expect(side.get_by_role("heading", name="Working tree")).to_be_focused()
    assert errors == [], errors


def test_a_new_side_panel_takes_the_focus_it_replaces_or_covers(
    open_page: OpenPage, git_port: int
) -> None:
    """On a phone, a row opened from the keyboard left the focus on the graph under the panel.

    A parent sha opened from the keyboard rebuilt the panel and dropped the focus to the page.
    """
    from playwright.sync_api import expect

    page, errors = open_page(git_port)
    page.set_viewport_size({"width": 390, "height": 844})
    page.goto(f"http://127.0.0.1:{git_port}/#/git")
    side, heading = page.locator(".git-side"), page.get_by_role("heading", name="Commit")
    page.locator(".gc-row", has_text="root commit").focus()
    page.keyboard.press("Enter")
    expect(heading).to_be_focused()
    expect(side.locator(".gd-subject")).to_contain_text("root commit")
    expect(heading).to_be_focused()  # the head stays while the commit loads
    page.set_viewport_size({"width": 1280, "height": 900})
    page.locator(".gc-row", has_text="merge feature").click()
    side.locator(".gd-meta button.mini").first.focus()  # the first parent
    page.keyboard.press("Enter")
    expect(side.locator(".gd-subject")).to_contain_text("main: drop d.txt")
    expect(heading).to_be_focused()
    assert errors == [], errors


def test_the_compare_pickers_reached_option_stays_in_view_and_is_named(
    open_page: OpenPage, git_port: int, tmp_path: pathlib.Path
) -> None:
    """ArrowDown took the highlight past the list's edge, unnamed; the swap button read '⇄'."""
    from playwright.sync_api import expect

    for n in range(24):
        _git(tmp_path, "tag", f"t{n:02d}")
    page, errors = open_page(git_port)
    page.goto(f"http://127.0.0.1:{git_port}/#/git")
    page.get_by_role("button", name="Compare", exact=True).click()
    base = page.get_by_label("base", exact=True)
    base.click()
    page.locator(".gpick-list .ac-item").nth(20).wait_for()
    for _ in range(18):
        base.press("ArrowDown")
    reached = page.locator(".gpick-list .ac-item.on")
    shown, box = page.locator(".gpick-list").bounding_box(), reached.bounding_box()
    assert shown and box, (shown, box)
    assert shown["y"] <= box["y"] <= shown["y"] + shown["height"] - box["height"], (shown, box)
    expect(base).to_have_attribute("aria-activedescendant", reached.get_attribute("id") or "-")
    expect(reached).to_have_attribute("aria-selected", "true")
    expect(page.get_by_role("button", name="swap base and head", exact=True)).to_be_visible()
    assert errors == [], errors


def test_a_binary_files_diff_offers_no_text_controls(
    open_page: OpenPage, git_port: int, tmp_path: pathlib.Path
) -> None:
    """A PNG's diff offered the four modes, full context and blame, which do nothing for it."""
    from playwright.sync_api import expect

    sha = _git(tmp_path, "rev-parse", "main^1")  # the commit that adds logo.png
    page, errors = open_page(git_port)
    page.goto(f"http://127.0.0.1:{git_port}/#/git/commit/{sha}?path=logo.png")
    page.wait_for_selector(".git-img img")
    tools = page.locator(".git-diff-tools")
    expect(tools.get_by_role("link", name="Open in Files", exact=True)).to_be_visible()
    for name in ("inline", "split", "old", "new", "Show full context", "blame"):
        expect(tools.get_by_role("button", name=name, exact=True)).to_have_count(0)
    assert errors == [], errors


def test_open_in_files_wears_the_mini_buttons_skin(
    open_page: OpenPage, git_port: int, tmp_path: pathlib.Path
) -> None:
    """Open in Files drew accent text on no fill beside the filled mini buttons."""
    sha = _git(tmp_path, "rev-parse", "main")
    page, errors = open_page(git_port)
    page.goto(f"http://127.0.0.1:{git_port}/#/git/commit/{sha}?path=b.txt")
    page.wait_for_selector(".git-diff .diff")
    tools = page.locator(".git-diff-tools")
    link = tools.get_by_role("link", name="Open in Files", exact=True)
    button = tools.get_by_role("button", name="blame", exact=True)
    assert link.evaluate(SKIN) == button.evaluate(SKIN)
    assert errors == [], errors


def test_a_selected_merges_hollow_dot_takes_its_rows_colour(
    open_page: OpenPage, git_port: int
) -> None:
    """A merge's hollow dot kept the page colour: a dark disc on the selected row."""
    from playwright.sync_api import expect

    page, errors = open_page(git_port)
    page.goto(f"http://127.0.0.1:{git_port}/#/git")
    row = page.locator(".gc-row").first  # the merge
    row.click()
    expect(row).to_have_class(re.compile(r"\bsel\b"))
    fill = row.locator("circle.merge").evaluate("d => getComputedStyle(d).fill")
    assert fill == row.evaluate("r => getComputedStyle(r).backgroundColor")
    assert errors == [], errors


def test_a_disabled_ref_in_the_filter_reads_not_allowed(open_page: OpenPage, git_port: int) -> None:
    """A ref of an unticked group kept the pointer cursor while its checkbox was disabled."""
    page, errors = open_page(git_port)
    page.goto(f"http://127.0.0.1:{git_port}/#/git")
    page.get_by_role("button", name="Refs ▾").click()
    off = page.locator(".gfilter-item.off").first  # the agent6 run refs' group starts unticked
    off.wait_for()
    assert off.evaluate("e => getComputedStyle(e).cursor") == "not-allowed"
    assert errors == [], errors


def test_a_handle_between_the_graph_and_the_panel_resizes_the_graph(
    open_page: OpenPage, git_port: int
) -> None:
    """The graph and the side panel kept a fixed split, where the Files tab offers a handle."""
    from playwright.sync_api import expect

    page, errors = open_page(git_port)
    page.goto(f"http://127.0.0.1:{git_port}/#/git")
    handle = page.get_by_role("separator", name="Graph width", exact=True)
    expect(handle).to_have_count(1)
    graph = page.locator(".git-graph")
    before = graph.bounding_box()
    handle.focus()
    page.keyboard.press("ArrowRight")
    page.keyboard.press("ArrowRight")
    after = graph.bounding_box()
    assert before and after and round(after["width"] - before["width"]) == 40, (before, after)
    page.locator(".gc-row").first.click()
    page.locator(".git-side .gf").first.click()
    page.wait_for_selector(".git-diff .diff")
    expect(handle).to_be_hidden()
    assert errors == [], errors


def test_a_lane_is_one_row_tall_at_a_larger_default_font(
    open_page: OpenPage, git_port: int
) -> None:
    """At a 20px default font each row was 37.5px and its lanes 30px, so every line broke."""
    page, errors = open_page(git_port)
    _default_font(page, 20)
    page.goto(f"http://127.0.0.1:{git_port}/#/git")
    page.wait_for_selector(".gc-row")
    boxes = page.locator(".gc-row").evaluate_all(
        """rows => rows.map(r => { const a = r.getBoundingClientRect();
          const s = r.querySelector('svg').getBoundingClientRect();
          return [a.height, s.top - a.top, s.height]; })"""
    )
    assert len(boxes) == 4 and all(b == [37.5, 0, 37.5] for b in boxes), boxes
    assert errors == [], errors


def test_the_graph_handle_stops_where_the_stylesheet_stops_the_graph(
    open_page: OpenPage, git_port: int
) -> None:
    """At a 20px default font the handle offered 320px of a 400px graph and stored unseen widths."""
    from playwright.sync_api import expect

    page, errors = open_page(git_port)
    _default_font(page, 20)
    page.goto(f"http://127.0.0.1:{git_port}/#/git")
    graph = page.locator(".git-graph")
    handle = page.get_by_role("separator", name="Graph width", exact=True)
    handle.focus()
    expect(handle).to_have_attribute("aria-valuemin", "400")
    stored = "() => localStorage.getItem('a6-git-graph-w')"
    drawn = "g => Math.round(g.getBoundingClientRect().width) + 'px'"
    for key in ("ArrowLeft", "ArrowRight"):
        for _ in range(40):
            page.keyboard.press(key)
        assert page.evaluate(stored) == graph.evaluate(drawn), key
    expect(handle).to_have_attribute("aria-valuemax", graph.evaluate(drawn)[:-2])
    assert errors == [], errors


def test_the_git_toolbar_controls_share_one_height_and_fit_their_values(
    open_page: OpenPage, git_port: int
) -> None:
    """Every toolbar control is one height, and the branch select fits its branch, not the row."""
    page, errors = open_page(git_port)
    page.goto(f"http://127.0.0.1:{git_port}/#/git")
    page.wait_for_selector(".gc-row")
    heights = (
        page.locator(".git-bar")
        .locator("button, select, input")
        .evaluate_all(
            "els => els.filter(e => e.offsetParent)"
            ".map(e => Math.round(e.getBoundingClientRect().height))"
        )
    )
    assert len(heights) >= 7 and len(set(heights)) == 1, heights
    select = page.locator(".git-branch select").first
    assert select.evaluate("e => e.getBoundingClientRect().width") < 200
    assert errors == [], errors


def test_the_fetch_row_hides_in_a_repository_without_a_remote(
    open_page: OpenPage, git_port: int
) -> None:
    """With no remote to fetch from, the toolbar shows no Fetch or Fetch PR."""
    from playwright.sync_api import expect

    page, errors = open_page(git_port)
    page.goto(f"http://127.0.0.1:{git_port}/#/git")
    page.wait_for_selector(".gc-row")
    expect(page.locator(".git-fetch")).to_be_hidden()
    expect(page.get_by_role("button", name="Fetch PR")).to_be_hidden()
    assert errors == [], errors


def test_the_refs_filter_hangs_under_its_button_and_escape_refocuses_it(
    open_page: OpenPage, git_port: int
) -> None:
    """The Refs filter opens just under Refs, which reads open; Escape puts focus back on Refs."""
    from playwright.sync_api import expect

    page, errors = open_page(git_port)
    page.goto(f"http://127.0.0.1:{git_port}/#/git")
    page.wait_for_selector(".gc-row")
    refs = page.get_by_role("button", name="Refs ▾")
    refs.click()
    expect(page.get_by_role("searchbox")).to_be_focused()
    expect(refs).to_have_attribute("aria-expanded", "true")
    button, popup = refs.bounding_box(), page.locator(".gfilter").bounding_box()
    assert button and popup
    assert 0 <= popup["y"] - (button["y"] + button["height"]) <= 8
    assert abs(popup["x"] - button["x"]) <= 1
    page.keyboard.press("Escape")
    expect(page.locator(".gfilter")).to_be_hidden()
    expect(refs).to_be_focused()
    expect(refs).to_have_attribute("aria-expanded", "false")
    assert errors == [], errors


def test_the_git_panes_lay_out_by_their_own_width(open_page: OpenPage, git_port: int) -> None:
    """At 820px the graph drops its author column and the compare puts head under base."""
    from playwright.sync_api import expect

    page, errors = open_page(git_port)
    page.set_viewport_size({"width": 820, "height": 900})
    page.goto(f"http://127.0.0.1:{git_port}/#/git")
    page.wait_for_selector(".gc-row")
    expect(page.locator(".gc-author").first).to_be_hidden()
    page.get_by_role("button", name="Compare", exact=True).click()
    base = page.get_by_label("base", exact=True).bounding_box()
    head = page.get_by_label("head", exact=True).bounding_box()
    assert base and head and head["y"] >= base["y"] + base["height"]
    assert errors == [], errors


def test_graph_lanes_read_at_three_to_one_in_the_light_theme(
    open_page: OpenPage, git_port: int
) -> None:
    """Every lane reads at 3:1 on the light page; the dark theme's pastels read 1.9 to 2.3:1."""
    page, errors = open_page(git_port)
    page.emulate_media(color_scheme="light")
    page.goto(f"http://127.0.0.1:{git_port}/#/git")
    page.wait_for_selector(".gc-row")
    ratios = page.evaluate(
        """() => {
          const lum = c => c.match(/[\\d.]+/g).slice(0, 3).map(v => v / 255)
            .map(v => v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4)
            .reduce((sum, v, i) => sum + v * [0.2126, 0.7152, 0.0722][i], 0);
          const bg = lum(getComputedStyle(document.body).backgroundColor);
          return [...document.querySelectorAll('.gc-lanes :is(path, circle)')].map(e => {
            const l = lum(getComputedStyle(e).stroke);
            return (Math.max(l, bg) + 0.05) / (Math.min(l, bg) + 0.05);
          });
        }"""
    )
    assert len(ratios) > 4 and min(ratios) >= 3, ratios
    assert errors == [], errors


def test_a_wrapped_file_path_breaks_after_a_slash(open_page: OpenPage, port: int) -> None:
    """A rename's two paths wrap between folders; Chromium cut 'lane_watch.p' from its 'y'."""
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/")
    page.wait_for_selector("#view")
    lines = page.evaluate(
        """() => {
          const box = el('div'); box.style.width = '180px'; document.body.appendChild(box);
          const p = filePath({path: 'src/agent6/app/child_wait.py',
            old_path: 'src/agent6/app/lane_watch.py'}, false);
          box.appendChild(p);
          const lines = [''];
          let top = null;
          const walk = document.createTreeWalker(p, NodeFilter.SHOW_TEXT);
          for (let n = walk.nextNode(); n; n = walk.nextNode()) {
            for (let i = 0; i < n.length; i++) {
              const r = document.createRange(); r.setStart(n, i); r.setEnd(n, i + 1);
              const y = r.getBoundingClientRect().top;
              if (top !== null && y > top + 2) lines.push('');
              top = y; lines[lines.length - 1] += n.data[i];
            }
          }
          return lines;
        }"""
    )
    assert len(lines) > 2, lines
    assert all(line.endswith(("/", " ")) for line in lines[:-1]), lines
    assert errors == [], errors


def test_a_long_ref_name_shows_whole_in_the_refs_filter(
    open_page: OpenPage, git_port: int, tmp_path: pathlib.Path
) -> None:
    """A name past the graph badge's cap shows whole in the filter, its sha in the shas' column."""
    name = "topic/a-long-branch-name-the-row-has-room-for"
    _git(tmp_path, "branch", name)
    page, errors = open_page(git_port)
    page.goto(f"http://127.0.0.1:{git_port}/#/git")
    page.wait_for_selector(".gc-row")
    page.get_by_role("button", name="Refs ▾").click()
    badge = page.locator(".gfilter-item .gref", has_text=name)
    assert badge.get_attribute("title") == name + " · branch"
    assert badge.evaluate("b => b.scrollWidth <= b.clientWidth")
    shas = page.locator(".gfilter-item .gref + .mono").evaluate_all(
        "ss => ss.map(s => Math.round(s.getBoundingClientRect().right))"
    )
    assert len(set(shas)) == 1, shas
    assert errors == [], errors


def test_the_side_and_diff_heads_line_up_with_the_page(open_page: OpenPage, git_port: int) -> None:
    """The side panel's heading starts where its content does, the diff's where the toolbar does."""
    page, errors = open_page(git_port)
    page.goto(f"http://127.0.0.1:{git_port}/#/git")
    page.wait_for_selector(".gc-row")
    page.locator(".gc-row").first.click()
    page.wait_for_selector(".git-side .gf")
    left = "e => Math.round(e.getBoundingClientRect().left)"
    assert page.locator(".git-side-head h3").evaluate(left) == page.locator(".gd").evaluate(left)
    toolbar = page.locator(".git-bar button").first.evaluate(left)
    page.locator(".git-side .gf .gf-path").first.click()
    page.wait_for_selector(".git-diff .diff")
    assert page.locator(".git-diff-head button").first.evaluate(left) == toolbar
    assert page.locator(".git-diff-body .gf-patch").evaluate(left) == toolbar
    assert errors == [], errors


def test_a_commits_parent_label_sits_on_its_sha_buttons_baseline(
    open_page: OpenPage, git_port: int
) -> None:
    """The 'parents' label and its sha buttons share a baseline; the label sat 3px higher."""
    page, errors = open_page(git_port)
    page.goto(f"http://127.0.0.1:{git_port}/#/git/commit/main")
    page.wait_for_selector(".gd-meta button")
    baselines = page.evaluate(
        """() => {
          const label = [...document.querySelectorAll('.gd-meta .k')]
            .find(k => k.textContent === 'parents');
          const at = e => { const probe = el('span'); probe.style.display = 'inline-block';
            e.appendChild(probe); return probe.getBoundingClientRect().top; };
          return [at(label), at(label.nextElementSibling.querySelector('button'))];
        }"""
    )
    assert abs(baselines[0] - baselines[1]) < 0.5, baselines
    assert errors == [], errors


def test_the_compare_hint_reads_as_a_note(open_page: OpenPage, git_port: int) -> None:
    """The line under the pickers is small muted text, as quiet as the line it explains."""
    page, errors = open_page(git_port)
    page.goto(f"http://127.0.0.1:{git_port}/#/git")
    page.wait_for_selector(".gc-row")
    page.get_by_role("button", name="Compare", exact=True).click()
    style = "e => [getComputedStyle(e).fontSize, getComputedStyle(e).color]"
    hint = page.locator(".gcmp .hint").evaluate(style)
    muted = page.locator(".gcmp-result .empty").evaluate(style)
    assert hint == ["12px", muted[1]], (hint, muted)
    assert errors == [], errors


def test_the_compare_pickers_placeholders_fit_their_fields(
    open_page: OpenPage, git_port: int
) -> None:
    """Side by side at 1440px, each picker's placeholder fits; both were cut mid-word."""
    page, errors = open_page(git_port)
    page.set_viewport_size({"width": 1440, "height": 900})
    page.goto(f"http://127.0.0.1:{git_port}/#/git")
    page.wait_for_selector(".gc-row")
    page.get_by_role("button", name="Compare", exact=True).click()
    fits = page.locator(".gcmp-pick input").evaluate_all(
        """inputs => inputs.map(i => {
          const s = getComputedStyle(i), text = document.createElement('canvas').getContext('2d');
          text.font = s.font;
          const room = i.clientWidth - parseFloat(s.paddingLeft) - parseFloat(s.paddingRight);
          return text.measureText(i.placeholder).width <= room;
        })"""
    )
    assert fits == [True, True]
    assert errors == [], errors


def test_a_commits_sha_and_dates_never_break_mid_token_on_a_phone(
    open_page: OpenPage, git_port: int
) -> None:
    """The full sha keeps one line and a date wraps only before its zone; both broke mid-token."""
    page, errors = open_page(git_port)
    page.set_viewport_size({"width": 390, "height": 844})
    page.goto(f"http://127.0.0.1:{git_port}/#/git/commit/main")
    page.wait_for_selector(".gd-meta button")
    value = "xpath=following-sibling::*[1]"
    sha = page.locator(".gd-meta .k", has_text="commit").locator(value)
    assert sha.get_attribute("title") == sha.inner_text() and len(sha.inner_text()) == 40
    assert sha.evaluate("e => e.scrollHeight <= parseFloat(getComputedStyle(e).lineHeight) + 1")
    date_lines = (
        page.locator(".gd-meta .k", has_text="author")
        .locator(value)
        .evaluate(
            """e => {
          const walk = document.createTreeWalker(e, NodeFilter.SHOW_TEXT);
          for (let n = walk.nextNode(); n; n = walk.nextNode()) {
            const at = n.data.search(/\\d{4}-\\d\\d-\\d\\d/);
            if (at < 0) continue;
            const range = document.createRange();
            range.setStart(n, at); range.setEnd(n, at + 19);
            return range.getClientRects().length;
          }
        }"""
        )
    )
    assert date_lines == 1
    assert errors == [], errors


def test_a_commit_body_wraps_a_hard_wrapped_item_once(
    open_page: OpenPage, git_port: int, tmp_path: pathlib.Path
) -> None:
    """An item git wrapped reads as one block whose lines hang under its text.

    The panel kept git's hard wraps and wrapped each line again into ragged fragments.
    """
    item = (
        "an item whose text the author wrapped at seventy-two columns, as the\n"
        "  convention asks, and which then went on for a while longer"
    )
    _git(tmp_path, "commit", "-q", "--allow-empty", "-m", "subject\n\n- " + item)
    sha = _git(tmp_path, "rev-parse", "HEAD")
    page, errors = open_page(git_port)
    page.goto(f"http://127.0.0.1:{git_port}/#/git/commit/{sha}")
    page.wait_for_selector(".gd-body")
    assert "as the convention asks" in page.locator(".gd-body").inner_text()
    lefts = page.locator(".gd-body .gd-line").first.evaluate(
        """line => { const range = document.createRange(); range.selectNodeContents(line.lastChild);
          return [...range.getClientRects()].map(r => Math.round(r.left)); }"""
    )
    assert len(lefts) > 1 and len(set(lefts)) == 1, lefts
    assert errors == [], errors


def test_a_phone_presses_inline_when_split_has_no_room(open_page: OpenPage, git_port: int) -> None:
    """With split chosen, a phone shows inline pressed and split disabled; split read pressed."""
    from playwright.sync_api import expect

    page, errors = open_page(git_port)
    page.add_init_script("localStorage.setItem('a6-diff-mode', 'split')")
    page.set_viewport_size({"width": 390, "height": 844})
    page.goto(f"http://127.0.0.1:{git_port}/#/git/commit/feature?path=b.txt")
    expect(page.locator(".git-diff .diff")).to_be_visible()
    expect(page.locator(".git-diff .gview button.active")).to_have_text("inline")
    expect(page.locator(".git-diff .gview button", has_text="split")).to_be_disabled()
    expect(page.locator(".git-diff")).to_contain_text("split needs a wider screen")
    assert page.evaluate("localStorage.getItem('a6-diff-mode')") == "split"
    assert errors == [], errors


@pytest.mark.parametrize("width", [390, 820])
def test_a_narrow_diff_head_keeps_its_controls_on_one_line(
    open_page: OpenPage, git_port: int, width: int
) -> None:
    """Under the path the diff's controls took three rows on a phone, and two at 820px."""
    from playwright.sync_api import expect

    page, errors = open_page(git_port)
    page.set_viewport_size({"width": width, "height": 844})
    page.goto(f"http://127.0.0.1:{git_port}/#/git/commit/feature?path=b.txt")
    expect(page.locator(".git-diff .diff-blame")).to_be_enabled()
    controls = page.locator(".git-diff-head").locator("button, a")
    tops = controls.evaluate_all("es => es.map(e => Math.round(e.getBoundingClientRect().top))")
    assert len(tops) == 10 and len(set(tops[1:])) == 1, tops  # past the back button
    assert errors == [], errors


def test_the_working_tree_row_joins_head_with_a_dashed_line(
    open_page: OpenPage, dirty_git_port: int
) -> None:
    """A dashed line leaves the working tree's dot and enters HEAD's; the row floated, unjoined."""
    page, errors = open_page(dirty_git_port)
    page.goto(f"http://127.0.0.1:{dirty_git_port}/#/git")
    page.wait_for_selector(".gc-wt")
    head = page.locator(".gc-row:not(.gc-wt)").first  # HEAD is the newest commit here
    assert head.locator("circle").get_attribute("r") == "5"
    assert page.locator(".gc-wt path.wt").count() == 1
    assert head.locator("path.wt").count() == 1
    assert errors == [], errors


def test_the_working_tree_rows_say_untracked_and_staged(
    open_page: OpenPage, dirty_git_port: int, tmp_path: pathlib.Path
) -> None:
    """An untracked file reads U, not A, and a row says when its changes are staged."""
    _git(tmp_path, "add", "b.txt")
    (tmp_path / "b.txt").write_text("one\ntwo\nthree\nfour\nfive\n", encoding="utf-8")
    page, errors = open_page(dirty_git_port)
    page.goto(f"http://127.0.0.1:{dirty_git_port}/#/git/worktree")
    page.wait_for_selector(".git-side .gf")

    def row(path: str) -> tuple[str, str]:
        found = page.locator(f".git-side .gf[data-path='{path}']")
        index = found.locator(".gf-index")
        return found.locator(".gf-st").inner_text(), index.inner_text() if index.count() else ""

    assert row("new.txt") == ("U", "")
    assert row("dir/g.txt") == ("R", "staged")
    assert row("b.txt") == ("M", "partly staged")
    assert row("logo.png") == ("M", "")
    assert errors == [], errors


def test_the_working_tree_rows_keep_one_slot_per_verb(
    open_page: OpenPage, dirty_git_port: int
) -> None:
    """Each verb's button sits at one x down the list, and a narrow panel never squeezes a path."""
    page, errors = open_page(dirty_git_port)
    for width in (1280, 820):
        page.set_viewport_size({"width": width, "height": 900})
        page.goto(f"http://127.0.0.1:{dirty_git_port}/#/git/worktree")
        page.wait_for_selector(".git-side .gf .gf-acts")
        lefts = page.locator(".git-side .gf-acts button").evaluate_all(
            "bs => bs.map(b => [b.textContent, Math.round(b.getBoundingClientRect().left)])"
        )
        slots: dict[str, set[int]] = {}
        for verb, left in lefts:
            slots.setdefault(verb, set()).add(left)
        assert len(slots) == 3 and all(len(x) == 1 for x in slots.values()), (width, slots)
        paths = page.locator(".git-side .gf-path").evaluate_all(
            "ps => ps.map(p => Math.round(p.getBoundingClientRect().width))"
        )
        assert min(paths) > 100, (width, paths)
    assert errors == [], errors


def test_an_added_images_one_side_takes_the_whole_width(
    open_page: OpenPage, git_port: int, tmp_path: pathlib.Path
) -> None:
    """An added image's only figure spans the diff; it took half and left the other half empty."""
    sha = _git(tmp_path, "rev-parse", "main^1")  # the commit that adds logo.png
    page, errors = open_page(git_port)
    page.goto(f"http://127.0.0.1:{git_port}/#/git/commit/{sha}?path=logo.png")
    page.wait_for_selector(".git-img img")
    wrap = page.locator(".git-img").bounding_box()
    figure = page.locator(".git-img figure").bounding_box()
    assert wrap and figure and figure["width"] > 0.9 * wrap["width"]
    assert errors == [], errors
