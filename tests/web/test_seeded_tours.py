from __future__ import annotations

import pathlib
import re
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from playwright.sync_api import Page

pytestmark = pytest.mark.browser


def _stop(page: Page, tmp_path: pathlib.Path, name: str) -> None:
    """Keep the rendered stop and its text beside the test's server log."""
    screenshot = tmp_path / f"{name}.png"
    page.screenshot(path=str(screenshot), full_page=True)
    (tmp_path / f"{name}.txt").write_text(page.locator("body").inner_text(), encoding="utf-8")
    print(f"Tour screenshot: {screenshot}")


def test_hub_and_session_tour(tour_page: Page, tmp_path: pathlib.Path) -> None:
    """Recorded runs keep their status, task, graph, budget, conversation and resume model."""
    from playwright.sync_api import expect

    page = tour_page
    expect(page.locator(".item")).to_have_count(7)
    _stop(page, tmp_path, "01-hub")
    for name, status in (
        ("ready-rowan-A5P972", "passed"),
        ("merry-fountain-B7XAJS", "passed"),
        ("upright-fox-5B71YT", "passed"),
        ("willing-glen-9ZYWWB", "stale"),
        ("friendly-crane-1X3ER0", "failed · max turns"),
        ("tidy-river-165YS6", "passed"),
        ("thoughtful-comet-1TQASW", "passed"),
    ):
        expect(page.locator(".item").filter(has_text=name)).to_contain_text(status)
    page.locator(".item").filter(has_text="merry-fountain-B7XAJS").click()
    expect(page.locator(".card-conv .ci").first).to_be_visible()
    expect(page.locator(".card-head")).to_contain_text("Add percentile(xs, p) to src/stats.py")
    tasks = page.locator(".card-tasks .tree")
    expect(tasks).to_contain_text("✓ Add percentile(xs, p) function to src/stats.py")
    expect(tasks).to_contain_text("✓ Add variance(xs) function (sample variance) to src/stats.py")
    expect(tasks).to_contain_text("✓ Add tests for percentile(xs, p) and variance(xs)")
    expect(page.locator(".card-budget")).to_contain_text("usage: 0pt + $0.11 / $10.00")
    expect(page.locator(".card-budget")).to_contain_text("tokens: in 28839 · out 54982")
    expect(page.locator(".card-conv")).to_contain_text("Completed adding percentile(xs, p)")
    _stop(page, tmp_path, "02-session")
    tasks.scroll_into_view_if_needed()
    _stop(page, tmp_path, "03-task-graph")
    page.locator(".card-budget").scroll_into_view_if_needed()
    _stop(page, tmp_path, "04-budget")
    model = page.locator(".composer select").nth(1)
    expect(model).to_be_visible()
    expect(model.locator("option:checked")).to_have_text(
        "openrouter/openai/gpt-4.1-mini (as recorded)"
    )
    model.focus()
    _stop(page, tmp_path, "05-resume-model")


def test_git_and_files_tour(tour_page: Page, tmp_path: pathlib.Path) -> None:
    """The bundled history opens commit files, both diff layouts, Markdown source and blame."""
    from playwright.sync_api import expect

    page = tour_page
    page.locator("aside.rail a[data-tab='git']").click()
    expect(page.locator(".gc-row:not(.gc-wt)").first).to_be_visible()
    _stop(page, tmp_path, "01-git-graph")
    expect(
        page.locator(".gc-row").filter(has_text="Add the stats module and its tests")
    ).to_be_visible()
    page.locator(".gc-row").filter(
        has_text="Fix the failing median test, then add mode(xs)"
    ).click()
    expect(page.locator(".gf")).to_have_count(2)
    expect(page.locator(".gf").filter(has_text="src/stats.py")).to_be_visible()
    expect(page.locator(".gf").filter(has_text="tests/test_stats.py")).to_be_visible()
    _stop(page, tmp_path, "02-commit-files")
    # Wait for the fetch before starting the DOM assertion's shorter deadline.
    with page.expect_response("**/api/git/diff?*") as diff:
        page.locator(".gf").filter(has_text="src/stats.py").click()
    assert diff.value.ok, diff.value.text()
    diff.value.finished()
    expect(page.locator(".git-diff .add").filter(has_text="def mode(xs)")).to_be_visible()
    expect(page.locator(".git-diff .del").first).to_be_visible()
    expect(page.locator(".git-diff .dl.add .diff-marker").first).to_have_text("+")
    _stop(page, tmp_path, "03-inline-diff")
    page.locator(".git-diff").get_by_role("button", name="split", exact=True).click()
    expect(page.locator(".git-diff .ds-row .add").filter(has_text="def mode(xs)")).to_be_visible()
    expect(page.locator(".git-diff .ds-row .del").first).to_be_visible()
    _stop(page, tmp_path, "04-split-diff")
    page.locator("aside.rail a[data-tab='files']").click()
    page.locator(".gtree [data-path='README.md']").click()
    expect(page.locator(".md-view")).to_be_visible()
    expect(page.locator(".md-view h1")).to_have_text(re.compile("stats", re.IGNORECASE))
    _stop(page, tmp_path, "05-readme-rendered")
    page.locator(".files-md-mode").get_by_role("button", name="source", exact=True).click()
    expect(page.locator("pre.src")).to_be_visible()
    expect(page.locator('pre.src [data-line="1"]')).to_contain_text("# stats")
    gutters = page.locator("pre.src .g")
    expect(gutters).to_have_count(5)
    for index in range(5):
        expect(gutters.nth(index)).to_be_visible()
        expect(gutters.nth(index)).to_have_attribute("data-n", str(index + 1))
    _stop(page, tmp_path, "06-readme-source")
    page.locator("button.files-blame").click()
    expect(page.locator(".blame-cell").first).to_be_visible()
    expect(page.locator(".blame-cell").first).to_contain_text("agent6 demo")
    expect(page.locator(".blame-cell").first).to_contain_text("2026-10-27")
    _stop(page, tmp_path, "07-readme-blame")


def test_hub_and_findings_tour(tour_page: Page, tmp_path: pathlib.Path) -> None:
    """The hub's reviewed session retains two open warnings and the recorded receipt."""
    from playwright.sync_api import expect

    page = tour_page
    reviewed = page.locator(".item").filter(has_text="merry-fountain-B7XAJS")
    expect(reviewed).to_contain_text("open findings: 2 warn")
    expect(reviewed).to_contain_text("passed · unmerged")
    _stop(page, tmp_path, "01-hub")
    reviewed.click()
    findings = page.locator(".card-findings")
    expect(findings.locator(".finding-counts")).to_be_visible()
    expect(findings.locator(".finding-title")).to_have_text(
        [
            "warn · open · Percentile tests only exercise sorted inputs",
            "warn · open · Consider explicitly rejecting empty percentile input",
        ]
    )
    for index in range(2):
        finding = findings.locator(".finding").nth(index)
        finding.scroll_into_view_if_needed()
        expect(finding).to_be_visible()
        _stop(page, tmp_path, f"0{index + 2}-open-warning")
    with page.expect_response("**/api/session/merry-fountain-B7XAJS/receipt") as response:
        findings.get_by_role("button", name="Receipt", exact=True).click()
    assert response.value.ok, response.value.text()
    response.value.finished()
    expect(findings.locator(".receipt > pre")).to_be_visible()
    receipt = findings.locator(".receipt > pre")
    expect(receipt).to_contain_text("Receipt merry-fountain-B7XAJS")
    expect(receipt).to_contain_text("Final tree passed")
    expect(receipt).to_contain_text("python3 -m unittest discover -s tests -t .")
    expect(receipt).to_contain_text("revert")
    _stop(page, tmp_path, "04-receipt")


def test_resume_offers_recorded_model(tour_page: Page) -> None:
    """A recorded model remains selectable even when the config default has changed."""
    from playwright.sync_api import expect

    tour_page.locator(".item").filter(has_text="merry-fountain-B7XAJS").click()
    model = tour_page.locator(".composer select").nth(1)
    expect(model.locator("option:checked")).to_have_text(
        "openrouter/openai/gpt-4.1-mini (as recorded)"
    )
