from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import pytest

from agent6.ui.web import git_model
from tests.web import test_review_evidence, test_web_browser

if TYPE_CHECKING:
    from playwright.sync_api import Page

pytestmark = pytest.mark.browser
evidence = test_review_evidence.evidence
browser = test_web_browser.browser
open_page = test_web_browser.open_page


def test_pre_existing_badge_opens_base_lines_and_blame_without_changing_severity(
    open_page: Callable[[int], tuple[Page, list[str]]],
    evidence: tuple[int, dict[str, Any]],
) -> None:
    from playwright.sync_api import expect

    port, revs = evidence
    page, errors = open_page(port)
    url = f"http://127.0.0.1:{port}/#/session/review-run"
    page.goto(url)
    older = page.locator('.card-findings .finding[data-finding="rename"]')
    changed = page.locator('.card-findings .finding[data-finding="head"]')
    expect(older).to_contain_text("block · open")
    expect(changed).to_contain_text("block · open")
    assert changed.locator(".finding-origin").count() == 0
    badge = older.get_by_role("link", name="Pre-existing line", exact=True)
    expect(badge).to_be_visible()
    expect(badge).to_have_attribute(
        "title", "The cited lines exist at the base. This does not date the defect."
    )
    badge.click()
    expect(page.locator('.files-text [data-line="2"].finding-selected')).to_contain_text("second")
    page.reload()
    expect(page.locator('.files-text [data-line="2"].finding-selected')).to_contain_text("second")
    page.goto(url)
    older.get_by_role("link", name="Blame " + revs["base"][:7], exact=True).click()
    expect(page).to_have_url(f"http://127.0.0.1:{port}/#/git/commit/{revs['base']}")
    expect(page.locator(".git-app")).to_contain_text("base")
    page.goto(url)
    expect(older).to_contain_text("block · open")
    expect(changed).to_contain_text("block · open")
    assert not errors


@pytest.mark.parametrize("mode", ["inline", "split", "old", "new", "narrow", "truncated"])
def test_reviewed_diff_deep_link_reload_and_exact_file(
    open_page: Callable[[int], tuple[Page, list[str]]],
    evidence: tuple[int, dict[str, Any]],
    mode: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from playwright.sync_api import expect

    port, revs = evidence
    page, errors = open_page(port)
    diff_mode = "split" if mode in {"narrow", "truncated"} else mode
    page.add_init_script(f"localStorage.setItem('a6-diff-mode', '{diff_mode}')")
    if mode == "narrow":
        page.set_viewport_size({"width": 390, "height": 844})
    if mode == "truncated":
        monkeypatch.setattr(git_model, "DIFF_MAX_BYTES", 30)
    finding, path, side = (
        ("head", "new.txt", "head") if mode == "new" else ("rename", "old.txt", "base")
    )
    url = (
        f"http://127.0.0.1:{port}/#/git/compare/{revs['base']}..{revs['head']}"
        f"?panel=panel&finding={finding}&path={path}&side={side}"
    )
    page.goto(url)
    card = page.locator(f'.git-diff .finding[data-finding="{finding}"]')
    expect(card).to_be_visible()
    expect(card).to_contain_text("<img src=x onerror=alert(1)>")
    assert card.locator("img,script").count() == 0
    assert card.evaluate("e => !!e.closest('[data-old],[data-new],[data-line]')") == (
        mode != "truncated"
    )
    if mode == "narrow":
        expect(page.locator(".git-diff")).to_contain_text("showing inline")
    if mode == "truncated":
        expect(card).to_contain_text("outside the displayed lines")
        page.get_by_role("button", name="Show the whole patch").click()
        expect(page.locator('.git-diff [data-old="2"] .finding')).to_be_visible()
    page.reload()
    expect(card).to_be_visible()
    card.get_by_role("link", name="Reviewed file", exact=True).click()
    expect(page.locator(".files-text")).to_contain_text("second")
    assert "side=" + side in page.url
    page.reload()
    expect(page.locator(".files-text")).to_contain_text("second")
    assert not errors


@pytest.mark.parametrize(
    ("finding", "path", "side", "text", "anchored"),
    [
        ("delete", "gone.txt", "base", "deleted", True),
        ("file", "context.txt", "head", "unchanged context", False),
    ],
)
def test_deletion_and_file_level_findings_keep_their_reviewed_context(
    open_page: Callable[[int], tuple[Page, list[str]]],
    evidence: tuple[int, dict[str, Any]],
    finding: str,
    path: str,
    side: str,
    text: str,
    anchored: bool,
) -> None:
    from playwright.sync_api import expect

    port, revs = evidence
    page, errors = open_page(port)
    page.goto(
        f"http://127.0.0.1:{port}/#/git/compare/{revs['base']}..{revs['head']}"
        f"?panel=panel&finding={finding}&path={path}&side={side}"
    )
    card = page.locator(f'.git-diff .finding[data-finding="{finding}"]')
    expect(card).to_be_visible()
    assert card.evaluate("e => !!e.closest('[data-old],[data-new],[data-line]')") == anchored
    card.get_by_role("link", name="Reviewed file", exact=True).click()
    expect(page.locator(".files-text")).to_contain_text(text)
    assert not errors


def test_findings_card_resolves_with_literal_reason_and_reopens(
    open_page: Callable[[int], tuple[Page, list[str]]],
    evidence: tuple[int, dict[str, Any]],
) -> None:
    from playwright.sync_api import expect

    port, _ = evidence
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/session/review-run")
    card = page.locator('.card-findings .finding[data-finding="rename"]')
    expect(card).to_be_visible()
    card.get_by_role("button", name="Resolve", exact=True).click()
    card.get_by_label("Reason").fill("<b>operator conclusion</b>")
    card.get_by_role("button", name="Submit resolution").click()
    page.locator(".card-findings").get_by_label("Finding status").select_option("all")
    card.get_by_role("button", name="Reopen", exact=True).click()
    card.get_by_label("Reason").fill("Still reproducible")
    card.get_by_role("button", name="Submit reopen").click()
    page.reload()
    expect(card).to_contain_text("Still reproducible")
    expect(card).to_contain_text("<b>operator conclusion</b>")
    assert card.locator("b").count() == 0
    assert not errors
