from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import pytest

from agent6.ui.web import git_model
from tests.web import test_files_preview_browser, test_review_evidence, test_web_browser

if TYPE_CHECKING:
    from playwright.sync_api import Page, Route

pytestmark = pytest.mark.browser
browser = test_files_preview_browser.browser
open_page = test_web_browser.open_page
dirty_git_port = test_web_browser.dirty_git_port
evidence = test_review_evidence.evidence
git_port = test_web_browser.git_port


def test_compare_counts_commits_on_each_side(
    open_page: Callable[[int], tuple[Page, list[str]]], dirty_git_port: int
) -> None:
    """Compare counts commits rather than changed files or rows on the graph page."""
    from playwright import sync_api

    page, errors = open_page(dirty_git_port)
    for revisions, count in [
        ("v1..main", "3 commits"),
        ("v1..feature", "1 commit"),
        ("main..feature", "2 behind"),
        ("feature..main~1", "1 ahead, 1 behind"),
        ("main..HEAD", "0 commits"),
        ("v1..WORKTREE", "3 commits and uncommitted changes"),
    ]:
        page.goto(f"http://127.0.0.1:{dirty_git_port}/#/git/compare/{revisions}")
        sync_api.expect(page.locator(".gcmp-result").get_by_text(count, exact=True)).to_be_visible()
    assert not errors


def test_compare_graph_pages_both_sides_outside_the_ref_filter(
    open_page: Callable[[int], tuple[Page, list[str]]],
    dirty_git_port: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A one-commit page still reaches both histories and stops before their common parent."""
    from playwright import sync_api

    monkeypatch.setattr(git_model, "GRAPH_PAGE", 1)
    page, errors = open_page(dirty_git_port)
    page.goto(
        f"http://127.0.0.1:{dirty_git_port}/#/git/compare/main~1..refs%2Fagent6%2Frun-2%2Fhead"
    )
    toolbar = page.locator(".git-bar")
    sync_api.expect(toolbar).to_contain_text("Compared commits")
    more = page.get_by_role("button", name="Load more")
    for _ in range(2):
        more.click()
    rows = page.locator(".gc-row .gc-text")
    sync_api.expect(rows).to_have_count(3)
    assert set(rows.all_text_contents()) == {
        "chain-only commit",
        "feature: rename a.txt and grow it",
        "main: drop d.txt, add dir/f.txt",
    }
    sync_api.expect(more).to_have_count(0)
    page.get_by_role("button", name="Show all commits").click()
    sync_api.expect(rows.last).to_have_text("merge feature")
    sync_api.expect(page.get_by_role("heading", name="Compare")).to_be_visible()
    page.get_by_role("button", name="Close Compare").click()
    page.get_by_role("button", name="Compare", exact=True).click()
    sync_api.expect(toolbar).to_contain_text("Compared commits")
    page.set_viewport_size({"width": 390, "height": 844})
    page.get_by_role("button", name="Graph", exact=True).press("Enter")
    sync_api.expect(rows.last).to_have_text("merge feature")
    sync_api.expect(page.locator('.gc-row[tabindex="0"]')).to_be_focused()
    assert page.url.endswith("#/git")
    assert not errors


def test_reopening_a_review_comparison_keeps_the_tree_identities(
    open_page: Callable[[int], tuple[Page, list[str]]], evidence: tuple[int, dict[str, Any]]
) -> None:
    """A recorded tree has no commit range, including after closing and reopening Compare."""
    from playwright import sync_api

    port, revs = evidence
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/git/compare/{revs['base']}..{revs['head']}?panel=panel")
    summary = page.locator(".gcmp-sum")
    sync_api.expect(summary).to_contain_text("reviewed head")
    page.get_by_role("button", name="Close Compare").click()
    page.get_by_role("button", name="Compare", exact=True).click()
    sync_api.expect(summary).to_contain_text("reviewed head")
    sync_api.expect(summary).not_to_contain_text("commit")
    sync_api.expect(page.get_by_role("button", name="Show all commits")).to_be_hidden()
    assert not errors


def test_close_compare_returns_focus_when_the_first_graph_page_arrives(
    open_page: Callable[[int], tuple[Page, list[str]]], git_port: int
) -> None:
    """Graph restores focus even when the operator closes Compare before the first page arrives."""
    from playwright import sync_api

    page, errors = open_page(git_port)
    held: list[Route] = []

    def hold(route: Route) -> None:
        held.append(route)

    page.route("**/api/git/graph?*", hold)
    page.set_viewport_size({"width": 390, "height": 844})
    page.goto(f"http://127.0.0.1:{git_port}/#/git")
    page.get_by_role("button", name="Compare", exact=True).press("Enter")
    page.get_by_role("button", name="Graph", exact=True).press("Enter")
    assert len(held) == 1
    held[0].continue_()
    sync_api.expect(page.locator('.gc-row[tabindex="0"]')).to_be_focused()
    assert not errors
