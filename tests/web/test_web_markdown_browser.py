from __future__ import annotations

import pathlib
from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest

from agent6.ui.web import git_model
from tests.unit import test_git_history
from tests.web import test_web_browser

if TYPE_CHECKING:
    from playwright.sync_api import Page

pytestmark = pytest.mark.browser
browser = test_web_browser.browser
open_page = test_web_browser.open_page
git_port = test_web_browser.git_port


@pytest.fixture
def markdown_repo(git_port: int, tmp_path: pathlib.Path) -> tuple[int, str]:
    """A committed guide with a table of contents, a sibling document and a relative image."""
    docs = tmp_path / "docs"
    docs.mkdir()
    source = (
        "# Workspace guide\n\n[details](#details) · [next](next.markdown#next-step)\n\n"
        "![Logo](../logo.png)\n\n"
        "<script>window.markdownExecuted = true</script>\n\n"
        "Inline <img src=x onerror=alert(1)>\n\n"
        "| Name | Value |\n| --- | --- |\n| safe | ~~old~~ |\n\n"
        "> A quoted note.\n\n- First\n- Second\n\n"
        "```python\nprint('hello')\n```\n\n"
        + "A paragraph to scroll past.\n\n" * 30
        + "## Details\n\nThe final section.\n"
    )
    (docs / "guide #%.md").write_text(source, encoding="utf-8")
    (docs / "next.markdown").write_text("# Next step\n\nA second document.\n", encoding="utf-8")
    test_git_history._git(tmp_path, "add", "docs")
    test_git_history._git(tmp_path, "commit", "-m", "guides")
    sha = test_git_history._git(tmp_path, "rev-parse", "HEAD")
    (docs / "next.markdown").write_text("# Changed on disk\n", encoding="utf-8")
    return git_port, sha


@pytest.mark.parametrize("theme", ["dark", "light"])
@pytest.mark.parametrize("width", [1280, 390])
def test_markdown_opens_rendered_and_remembers_both_toggle_choices(
    markdown_repo: tuple[int, str],
    open_page: Callable[[int], tuple[Page, list[str]]],
    theme: str,
    width: int,
) -> None:
    """Markdown starts rendered; source and rendered choices survive a reload in either theme."""
    from playwright.sync_api import expect

    port, sha = markdown_repo
    page, errors = open_page(port)
    page.add_init_script(f"localStorage.setItem('a6-theme', '{theme}')")
    page.set_viewport_size({"width": width, "height": 900})
    image_auth: list[str] = []
    page.on(
        "request",
        lambda request: (
            image_auth.append(request.headers.get("authorization", ""))
            if "/api/git/blob?" in request.url
            else None
        ),
    )
    page.goto(f"http://127.0.0.1:{port}/#/files/HEAD/docs/guide%20%23%25.md")
    heading = page.locator(".md-view h1")
    expect(heading).to_have_text("Workspace guide")
    expect(heading).to_be_visible()
    source = page.get_by_role("button", name="source", exact=True)
    rendered = page.get_by_role("button", name="rendered", exact=True)
    expect(rendered).to_have_attribute("aria-pressed", "true")
    expect(page.locator(".files-view .src")).to_be_hidden()
    expect(page.locator(".files-blame")).to_be_hidden()
    expect(page.locator(".md-view table")).to_be_visible()
    expect(page.locator(".md-view s")).to_have_text("old")
    expect(page.locator(".md-view blockquote")).to_contain_text("A quoted note.")
    expect(page.locator(".md-view li")).to_have_count(2)
    assert page.locator(".md-view pre").evaluate(
        "e => getComputedStyle(e).fontFamily === "
        "getComputedStyle(document.querySelector('.src')).fontFamily"
    )
    assert page.locator(".md-view pre").evaluate(
        "e => getComputedStyle(e).backgroundColor === "
        "getComputedStyle(document.querySelector('header')).backgroundColor"
    )
    expect(page.locator(".md-view script, .md-view [onerror]")).to_have_count(0)
    expect(page.locator(".md-view")).to_contain_text(
        "<script>window.markdownExecuted = true</script>"
    )
    assert page.evaluate("window.markdownExecuted") is None
    assert (
        page.locator(".md-view img").get_attribute("data-blob-src")
        == f"/api/git/blob?rev={sha}&path=logo.png"
    )
    page.wait_for_function("document.querySelector('.md-view img')?.naturalWidth === 1")
    assert image_auth and all(value.startswith("Bearer ") for value in image_auth)
    assert page.locator(".md-view img").evaluate("e => e.src.startsWith('data:image/png;')")
    assert page.locator(".files-view").evaluate("e => e.scrollWidth <= e.clientWidth")
    source.click()
    expect(page.locator(".files-view .src")).to_be_visible()
    expect(page.locator(".files-view .src")).to_contain_text("# Workspace guide")
    expect(heading).to_be_hidden()
    expect(page.locator(".files-blame")).to_be_visible()
    page.reload()
    expect(source).to_have_attribute("aria-pressed", "true")
    expect(page.locator(".files-view .src")).to_be_visible()
    expect(heading).to_be_hidden()
    rendered.click()
    expect(heading).to_be_visible()
    expect(page.locator(".files-view .src")).to_be_hidden()
    page.reload()
    expect(heading).to_be_visible()
    expect(rendered).to_have_attribute("aria-pressed", "true")
    assert not errors


def test_markdown_toc_and_relative_links_keep_the_files_route_and_revision(
    markdown_repo: tuple[int, str], open_page: Callable[[int], tuple[Page, list[str]]]
) -> None:
    """TOC clicks scroll to headings, and linked documents stay at the committed revision."""
    from playwright.sync_api import expect

    port, sha = markdown_repo
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/files/HEAD/docs/guide%20%23%25.md")
    page.get_by_role("link", name="details", exact=True).click()
    expect(page).to_have_url(
        f"http://127.0.0.1:{port}/#/files/{sha}/docs/guide%20%23%25.md?anchor=details"
    )
    expect(page.locator(".md-view #details")).to_be_in_viewport()
    page.reload()
    expect(page.locator(".md-view #details")).to_be_in_viewport()
    page.get_by_role("link", name="next", exact=True).click()
    expect(page.locator(".md-view h1")).to_have_text("Next step")
    expect(page).to_have_url(
        f"http://127.0.0.1:{port}/#/files/{sha}/docs/next.markdown?anchor=next-step"
    )
    assert not errors


def test_truncated_markdown_keeps_the_source_note_until_opened_whole(
    markdown_repo: tuple[int, str],
    open_page: Callable[[int], tuple[Page, list[str]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A capped document shows source and its existing whole-file action, not partial HTML."""
    from playwright.sync_api import expect

    monkeypatch.setattr(git_model, "TEXT_MAX_BYTES", 100)
    port, _ = markdown_repo
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/files/HEAD/docs/guide%20%23%25.md")
    expect(page.locator(".files-view .src")).to_be_visible()
    expect(page.locator(".files-text")).to_contain_text("the file is cut at")
    expect(page.get_by_role("button", name="rendered", exact=True)).to_have_count(0)
    page.get_by_role("button", name="Show the whole file", exact=True).click()
    expect(page.locator(".md-view h1")).to_have_text("Workspace guide")
    expect(page.get_by_role("button", name="rendered", exact=True)).to_have_attribute(
        "aria-pressed", "true"
    )
    assert not errors
