# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The rendered web UI driven in a browser: the forms and views the HTTP tests cannot reach.

Marked `browser`: Playwright's Chromium runs them, and they skip where it is not installed.
"""

from __future__ import annotations

import json
import pathlib
import re
import threading
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING

import pytest

from agent6 import paths
from agent6.ui.web import server as web_server
from tests.unit.test_git_history import build_history, make_dirty
from tests.web import signin

if TYPE_CHECKING:
    from playwright.sync_api import Browser, BrowserContext, Page

pytestmark = pytest.mark.browser


@pytest.fixture(scope="module")
def browser() -> Iterator[Browser]:
    """One Chromium for the module."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        chromium = pw.chromium.launch()
        try:
            yield chromium
        finally:
            chromium.close()


def _serve(cwd: pathlib.Path) -> Iterator[int]:
    srv = web_server.WebServer(("127.0.0.1", 0), cwd, "", secret=signin.SECRET)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield int(srv.server_address[1])
    finally:
        srv.shutdown()
        srv.server_close()


@pytest.fixture
def port(tmp_path: pathlib.Path) -> Iterator[int]:
    """A server on an ephemeral port, serving from an empty directory."""
    yield from _serve(tmp_path)


@pytest.fixture
def git_port(tmp_path: pathlib.Path) -> Iterator[int]:
    """A server reading a repository with a merge, a rename, a binary file and tags."""
    build_history(tmp_path)
    yield from _serve(tmp_path)


@pytest.fixture
def dirty_git_port(tmp_path: pathlib.Path) -> Iterator[int]:
    """A server reading that repository with five uncommitted files, the PNG among them."""
    build_history(tmp_path)
    make_dirty(tmp_path)
    yield from _serve(tmp_path)


@pytest.fixture
def files_port(tmp_path: pathlib.Path) -> Iterator[int]:
    """The dirty repository with an untracked Python file under `src/`."""
    build_history(tmp_path)
    make_dirty(tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "tool.py").write_text('def f(x):\n    return "hi"\n', encoding="utf-8")
    yield from _serve(tmp_path)


@pytest.fixture
def open_page(browser: Browser) -> Iterator[Callable[[int], tuple[Page, list[str]]]]:
    """Open a signed-in desktop page on a server's port, collecting console and page errors."""
    contexts: list[BrowserContext] = []

    def _open(port: int) -> tuple[Page, list[str]]:
        context = browser.new_context(viewport={"width": 1280, "height": 900})
        contexts.append(context)
        token = json.dumps(signin.token(port))
        context.add_init_script(f"localStorage.setItem('a6-token', {token})")
        page = context.new_page()
        errors: list[str] = []
        page.on("pageerror", lambda exc: errors.append(f"pageerror: {exc}"))
        page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
        return page, errors

    try:
        yield _open
    finally:
        for context in contexts:
            context.close()


def test_the_hub_opens_as_a_review_queue_and_remembers_activity(
    port: int, tmp_path: pathlib.Path, open_page: Callable[[int], tuple[Page, list[str]]]
) -> None:
    """The hub lists pending review work under group headings; Activity is a remembered choice."""
    from playwright.sync_api import expect

    from tests.unit import test_review_queue as queue_tests

    def write(session_id: str, rows: list[dict[str, object]]) -> None:
        run = paths.state_dir(tmp_path) / "sessions" / "runs" / session_id
        run.mkdir(parents=True)
        (run / "logs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    blocked: list[dict[str, object]] = [
        queue_tests._START,
        *queue_tests._panel_rows(block=1),
        queue_tests._END,
    ]
    newest: list[dict[str, object]] = [
        {
            "type": "session.start",
            "ts": "2026-10-25T12:00:00+00:00",
            "mode": "run",
            "user_task": "n",
        }
    ]
    write("run-block", blocked)
    write("run-newest", newest)
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/")
    items = page.locator(".card .list .item")
    expect(items.first).to_contain_text("run-block")
    headings = page.locator(".card .list .group")
    expect(headings).to_have_count(2)
    expect(headings.first).to_have_text("Blocked")
    expect(items.first).to_contain_text("1 open block finding")
    page.get_by_role("button", name="Activity").click()
    expect(items.first).to_contain_text("run-newest")
    expect(headings).to_have_count(0)
    page.reload()
    expect(page.locator(".card .list .item").first).to_contain_text("run-newest")
    assert page.evaluate("localStorage.getItem('a6-hub-sort')") == "activity"
    assert not errors


def test_a_stale_hub_refresh_never_overrides_the_sort_toggle(
    port: int, tmp_path: pathlib.Path, open_page: Callable[[int], tuple[Page, list[str]]]
) -> None:
    """A review-order refresh in flight when the operator picks Activity lands unpainted."""
    from playwright.sync_api import Route, expect

    from tests.unit import test_review_queue as queue_tests

    run = paths.state_dir(tmp_path) / "sessions" / "runs"
    for session_id, rows in (
        ("run-block", [queue_tests._START, *queue_tests._panel_rows(block=1), queue_tests._END]),
        ("run-newest", [{**queue_tests._START, "ts": "2026-10-25T12:00:00+00:00"}]),
    ):
        (run / session_id).mkdir(parents=True)
        (run / session_id / "logs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    page, errors = open_page(port)
    held: list[Route] = []
    painted: list[bool] = []

    def hold(route: Route) -> None:
        if painted and "sort=review" in route.request.url:
            held.append(route)
        else:
            route.continue_()

    page.route("**/api/hub*", hold)
    page.goto(f"http://127.0.0.1:{port}/")
    items = page.locator(".card .list .item")
    headings = page.locator(".card .list .group")
    expect(items.first).to_contain_text("run-block")
    painted.append(True)
    page.evaluate("document.dispatchEvent(new Event('visibilitychange'))")
    page.get_by_role("button", name="Activity").click()
    expect(items.first).to_contain_text("run-newest")
    assert len(held) == 1
    with page.expect_response(lambda r: "sort=review" in r.url):
        held[0].continue_()
    # Read once, without retrying: the next poll would repaint the right order and hide a slip.
    page.wait_for_timeout(300)
    assert "run-newest" in items.first.inner_text() and headings.count() == 0
    assert not errors


def test_the_provider_form_writes_the_chosen_deployment(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """The rendered add-provider form posts the deployment the operator picked.

    The HTTP tests post hand-written payloads; this drives the form the browser builds from
    the entry-table registry, so a field the form drops or misnames shows here.
    """
    cfg_path = paths.global_config_path()
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(
        '[providers.az]\napi_format = "openai"\ndeployment = "direct"\n'
        'base_url = "https://res.openai.azure.com"\n'
        'extra_query = { "api-version" = "2024-10-21" }\n',
        encoding="utf-8",
    )
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/config")
    page.get_by_role("button", name=re.compile(r"^Add provider", re.I)).click()
    dialog = page.get_by_role("dialog")
    dialog.locator("input[name='name']").fill("az")
    dialog.locator("select[name='api_format']").select_option("openai")
    dialog.locator("select[name='deployment']").select_option("azure")
    dialog.get_by_role("button", name="Add", exact=True).click()
    page.wait_for_function("() => !document.querySelector('[role=dialog]')")
    assert errors == [], errors
    cfg = cfg_path.read_text(encoding="utf-8")
    assert 'deployment = "azure"' in cfg and "api-version" in cfg, cfg


def test_a_files_diff_fills_the_git_tab(
    open_page: Callable[[int], tuple[Page, list[str]]], git_port: int
) -> None:
    """A click on a changed file opens its patch over the whole tab; the back button returns.

    The graph and the side panel hide while the diff shows; the HTTP tests see only the JSON.
    """
    page, errors = open_page(git_port)
    page.goto(f"http://127.0.0.1:{git_port}/#/git")
    page.wait_for_selector(".gc-row")
    assert page.locator(".gc-wt").count() == 0  # a clean tree has no working-tree row
    page.locator(".gc-row").first.click()
    page.locator(".gf").first.click()
    page.wait_for_selector(".git-diff .diff")
    body = page.locator(".git-body")
    assert "diffing" in (body.get_attribute("class") or "")
    assert not page.locator(".git-graph").is_visible()
    assert page.locator(".git-diff-pos").inner_text().startswith("1 / ")
    page.locator(".git-diff-head button").first.click()  # the "files" back button
    assert "diffing" not in (body.get_attribute("class") or "")
    assert page.locator(".git-graph").is_visible()
    assert errors == [], errors


def test_the_working_tree_row_opens_its_uncommitted_files(
    open_page: Callable[[int], tuple[Page, list[str]]], dirty_git_port: int
) -> None:
    """A dirty tree tops the graph with a row that opens its files like a commit's."""
    page, errors = open_page(dirty_git_port)
    page.goto(f"http://127.0.0.1:{dirty_git_port}/#/git")
    row = page.locator(".gc-wt")
    row.wait_for()
    assert "Working tree" in row.inner_text() and "5 uncommitted files" in row.inner_text()
    row.click()
    page.wait_for_selector(".git-side .gf")
    assert page.locator(".git-side .gf").count() == 5
    page.locator(".git-side .gf", has_text="new.txt").click()
    page.wait_for_selector(".git-diff .diff")
    assert "+hello" in page.locator(".git-diff .diff").inner_text()
    assert errors == [], errors


def test_an_images_diff_shows_it_before_and_after(
    open_page: Callable[[int], tuple[Page, list[str]]], dirty_git_port: int
) -> None:
    """A changed PNG shows both versions above git's binary line, each with its dimensions."""
    page, errors = open_page(dirty_git_port)
    page.goto(f"http://127.0.0.1:{dirty_git_port}/#/git")
    page.locator(".gc-wt").click()
    page.locator(".git-side .gf", has_text="logo.png").click()
    page.wait_for_function("() => document.querySelectorAll('.git-img img').length === 2")
    page.wait_for_function("() => document.body.innerText.includes('2\u00d72')")
    text = page.locator(".git-img").inner_text()
    assert "before" in text and "1\u00d71" in text and "after" in text and "2\u00d72" in text
    assert "Binary files" in page.locator(".git-diff .diff").inner_text()
    assert errors == [], errors


def test_a_changed_file_list_reads_as_a_tree_that_folds(
    open_page: Callable[[int], tuple[Page, list[str]]], dirty_git_port: int
) -> None:
    """The list | tree toggle groups the files under their folders.

    A folder folds on a click, the choice is kept in the browser, and a file in the tree
    opens its diff as in the list.
    """
    page, errors = open_page(dirty_git_port)
    page.goto(f"http://127.0.0.1:{dirty_git_port}/#/git")
    page.locator(".gc-wt").click()
    page.wait_for_selector(".git-side .gf")
    page.locator(".git-side .gview button", has_text="tree").click()
    folder = page.locator(".git-side .gtree-dir", has_text="dir")
    folder.wait_for()
    assert "1 file" in folder.inner_text()
    leaf = page.locator(".git-side [data-path='dir/g.txt']")
    assert leaf.is_visible()
    folder.click()
    assert not leaf.is_visible()
    folder.click()
    leaf.click()
    page.wait_for_selector(".git-diff .diff")
    assert "rename from dir/f.txt" in page.locator(".git-diff .diff").inner_text()
    assert page.evaluate("localStorage.getItem('a6-files-view')") == "tree"
    assert errors == [], errors


def test_the_diff_view_switches_between_inline_split_old_and_new(
    open_page: Callable[[int], tuple[Page, list[str]]], dirty_git_port: int
) -> None:
    """Every rendered line carries its numbers; split pairs the sides; new marks the changes.

    The choice is kept in the browser.
    """
    page, errors = open_page(dirty_git_port)
    page.goto(f"http://127.0.0.1:{dirty_git_port}/#/git")
    page.locator(".gc-wt").click()
    page.locator(".git-side [data-path='b.txt']").click()
    page.wait_for_selector(".git-diff .diff")
    assert page.locator(".git-diff .diff .dl.add[data-new='4']").count() == 1
    assert page.locator(".git-diff .diff .dl[data-old='3'][data-new='3']").count() == 1
    page.locator(".git-diff .gview button", has_text="split").click()
    page.wait_for_selector(".git-diff .ds-row")
    added = page.locator(".git-diff .ds-row[data-new='4']")
    assert added.count() == 1
    assert added.locator(".ds-cell").nth(0).inner_text().strip() == ""
    assert "four" in added.locator(".ds-cell").nth(1).inner_text()
    ctx = page.locator(".git-diff .ds-row[data-old='3']")
    assert "three" in ctx.locator(".ds-cell").nth(0).inner_text()
    assert "three" in ctx.locator(".ds-cell").nth(1).inner_text()
    # The gutter is drawn by the stylesheet, so a blank line still has a height.
    drawn = page.evaluate(
        "getComputedStyle(document.querySelector(\".ds-row[data-old='3'] .ds-cell .g\"),"
        " '::before').content"
    )
    assert drawn == '"3"', drawn
    page.locator(".git-diff .gview button", has_text="new").click()
    page.wait_for_selector(".git-diff .src .sl[data-line='4']")
    assert page.locator(".git-diff .src .sl").count() == 4
    assert "changed" in (
        page.locator(".git-diff .src .sl[data-line='4']").get_attribute("class") or ""
    )
    assert "changed" not in (
        page.locator(".git-diff .src .sl[data-line='1']").get_attribute("class") or ""
    )
    assert page.evaluate("localStorage.getItem('a6-diff-mode')") == "new"
    page.locator(".git-diff .gview button", has_text="old").click()
    page.wait_for_selector(".git-diff .src .sl[data-line='3']")
    assert page.locator(".git-diff .src .sl").count() == 3
    assert errors == [], errors


def test_blame_opens_the_lines_commit_in_git(
    open_page: Callable[[int], tuple[Page, list[str]]], dirty_git_port: int
) -> None:
    """A working-tree blame marks uncommitted lines and opens a committed line's panel."""
    page, errors = open_page(dirty_git_port)
    page.goto(f"http://127.0.0.1:{dirty_git_port}/#/files/WORKTREE/b.txt")
    page.locator(".files-blame").click()
    page.wait_for_selector(".sl[data-line='4'] .blame-cell")
    assert "not committed" in page.locator(".sl[data-line='4'] .blame-cell").inner_text()
    assert page.locator(".files-view .sl").count() == 4
    page.locator(".sl[data-line='1'] .blame-commit").click()
    page.wait_for_selector(".git-side .gd-subject")
    assert "root commit" in page.locator(".git-side .gd-subject").inner_text()
    assert page.locator(".gc-row.sel").count() == 1
    assert errors == [], errors


def test_git_and_files_deep_links_survive_reload_and_back(
    open_page: Callable[[int], tuple[Page, list[str]]], git_port: int
) -> None:
    """Encoded file paths, commit and compare hashes select their panes on entry and Back."""
    page, errors = open_page(git_port)
    page.goto(f"http://127.0.0.1:{git_port}/#/files/HEAD/b.txt")
    page.wait_for_selector(".files-view .sl[data-line='3']")
    page.reload()
    page.wait_for_selector(".files-view .sl[data-line='3']")
    page.goto(f"http://127.0.0.1:{git_port}/#/git/commit/feature")
    page.wait_for_selector(".git-side .gd-subject")
    assert "feature: rename" in page.locator(".git-side .gd-subject").inner_text()
    assert page.locator(".gc-row.sel").count() == 1
    page.locator(".gc-row").first.click()
    page.wait_for_function(
        "() => document.querySelector('.gd-subject')?.textContent.includes('merge feature')"
    )
    page.go_back()
    page.wait_for_function(
        "() => document.querySelector('.gd-subject')?.textContent.includes('feature: rename')"
    )
    page.goto(f"http://127.0.0.1:{git_port}/#/git/compare/v1..v2")
    page.wait_for_selector(".gcmp-result .gf")
    assert page.locator(".gcmp-result .gf").count() == 5
    assert errors == [], errors


def test_files_marks_worktree_states_and_sums_folders(
    open_page: Callable[[int], tuple[Page, list[str]]], files_port: int
) -> None:
    """Working-tree marks distinguish tracked, untracked and modified files, not commits."""
    page, errors = open_page(files_port)
    page.goto(f"http://127.0.0.1:{files_port}/#/files")
    page.wait_for_selector(".files-tree [data-path='b.txt']")
    assert page.locator(".files-tree .gf-st").count() == 0
    page.locator(".files-bar .gpick input").click()
    page.locator(".gpick-list .ac-item", has_text="working tree").click()
    page.wait_for_selector(".files-tree [data-path='src/tool.py']")
    for path, mark in (
        ("b.txt", "M"),
        ("img.bin", "H"),
        ("new.txt", "A"),
        ("src/tool.py", "A"),
        ("src", "A"),
        ("dir", "M"),
    ):
        assert page.locator(f".files-tree [data-path='{path}'] .gf-st").inner_text() == mark
    assert "1 untracked" in (
        page.locator(".files-tree [data-path='src'] .gf-st").get_attribute("title") or ""
    )
    page.locator(".files-bar .gpick input").fill("HEAD")
    page.locator(".files-bar .gpick input").press("Enter")
    page.wait_for_selector(".files-tree [data-path='dir/f.txt']")
    assert page.locator(".files-tree .gf-st").count() == 0
    assert errors == [], errors


def test_the_files_tab_opens_a_file_highlighted(
    open_page: Callable[[int], tuple[Page, list[str]]], files_port: int
) -> None:
    """The Files tab lists the working tree's files as a tree and shows one with its spans.

    HEAD is the revision to start; the picker's "working tree" lists the untracked file
    too, and the filter narrows the tree.
    """
    page, errors = open_page(files_port)
    page.goto(f"http://127.0.0.1:{files_port}/#/files")
    page.wait_for_selector(".files-tree [data-path='b.txt']")
    assert page.locator(".files-tree [data-path='src/tool.py']").count() == 0  # not at HEAD
    page.locator(".files-bar .gpick input").click()
    page.locator(".gpick-list .ac-item", has_text="working tree").click()
    page.wait_for_selector(".files-tree [data-path='src/tool.py']")
    assert page.locator(".files-tree .gtree-dir[data-path='src']").count() == 1
    page.locator(".files-bar input.filter").fill("tool")
    assert page.locator(".files-tree [data-path='b.txt']").count() == 0
    page.locator(".files-tree [data-path='src/tool.py']").click()
    page.wait_for_selector(".files-view .src .sl[data-line='2']")
    first = page.locator(".files-view .src .sl[data-line='1']")
    assert "def" in first.inner_text() and first.locator(".hl-keyword").count() >= 1
    assert page.locator(".files-view .src .sl").count() == 2
    assert "python" in page.locator(".files-view").inner_text()
    assert errors == [], errors


def test_back_from_a_compare_link_returns_to_the_graph(
    open_page: Callable[[int], tuple[Page, list[str]]], git_port: int
) -> None:
    """Back from a comparison lands on the bare Git tab; the restore pushed its hash again."""
    page, errors = open_page(git_port)
    base = f"http://127.0.0.1:{git_port}/"
    page.goto(base + "#/git")
    page.wait_for_selector(".gc-row")
    page.goto(base + "#/git/compare/v1..v2")
    page.wait_for_selector(".gcmp-result .gf")
    page.go_back()
    page.wait_for_selector(".git-side .empty", timeout=3000)
    assert page.evaluate("location.hash") == "#/git"
    assert page.locator(".gcmp-result").count() == 0
    assert errors == [], errors


def test_the_worktree_pane_has_a_hash_that_survives_reload(
    open_page: Callable[[int], tuple[Page, list[str]]], dirty_git_port: int
) -> None:
    """The working-tree row publishes `#/git/worktree`, and a reload reopens that pane."""
    page, errors = open_page(dirty_git_port)
    page.goto(f"http://127.0.0.1:{dirty_git_port}/#/git")
    page.locator(".gc-wt").click()
    page.wait_for_selector(".git-side .gd-subject")
    assert page.evaluate("location.hash") == "#/git/worktree"
    page.reload()
    page.wait_for_selector(".git-side .gd-subject")
    assert "Uncommitted changes" in page.locator(".git-side .gd-subject").inner_text()
    assert errors == [], errors
