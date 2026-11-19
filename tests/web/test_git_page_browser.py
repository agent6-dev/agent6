# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The Git page driven in a browser: its rows, keyboard, focus and waiting states."""

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
