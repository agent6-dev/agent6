from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest

from agent6 import paths
from tests.web import captures, test_files_preview_browser, test_web_browser

if TYPE_CHECKING:
    from playwright.sync_api import Page, Route

pytestmark = pytest.mark.browser
browser = test_files_preview_browser.browser
open_page = test_web_browser.open_page
port = test_web_browser.port


@pytest.mark.parametrize("width", [390, 1440])
def test_config_keeps_its_filter_and_names_its_preset(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, width: int
) -> None:
    """The filter stays reachable while scrolling and the preset button states its value."""
    from playwright.sync_api import expect

    page, errors = open_page(port)
    page.set_viewport_size({"width": width, "height": 900})
    page.goto(f"http://127.0.0.1:{port}/#/config")
    expect(page.get_by_role("button", name="Preset: none", exact=True)).to_be_visible()
    page.locator(".cfg-section").last.scroll_into_view_if_needed()
    expect(page.locator(".cfg-bar input.filter")).to_be_in_viewport()
    captures.save(page, "config-sticky")
    assert not errors


def test_config_focus_clears_the_wrapped_toolbar(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """Shift+Tab keeps the focused row below the toolbar as the buttons wrap."""
    from playwright import sync_api

    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/config")
    rows = page.locator(".cfg-section", has=page.locator("#cfg-section-sandbox")).locator(
        "tr[data-key]"
    )
    current = rows.nth(3).get_by_role("button")
    for width, height in [(1280, 960), (640, 480)]:  # the CSS viewport at 100% and 200% zoom
        page.set_viewport_size({"width": width, "height": height})
        current.focus()
        rows.nth(3).evaluate("""r => window.scrollBy(0, r.getBoundingClientRect().top
            - document.querySelector('.cfg-bar').getBoundingClientRect().bottom - 2)""")
        page.keyboard.press("Shift+Tab")
        sync_api.expect(rows.nth(2).get_by_role("button")).to_be_focused()
        assert rows.nth(2).evaluate("""r => r.getBoundingClientRect().top >=
            document.querySelector('.cfg-bar').getBoundingClientRect().bottom""")
    assert not errors


@pytest.mark.parametrize("width", [390, 1440])
def test_config_marks_required_fields_and_pending_saves(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, width: int
) -> None:
    """Add names its required field and Save says Saving while the response is pending."""
    from playwright.sync_api import expect

    page, errors = open_page(port)
    page.set_viewport_size({"width": width, "height": 900})
    page.goto(f"http://127.0.0.1:{port}/#/config")
    page.get_by_role("button", name="Add preset…", exact=True).click()
    expect(page.get_by_label("name (required)", exact=True)).to_have_attribute("required", "")
    page.keyboard.press("Escape")
    page.get_by_role("button", name="Add provider…", exact=True).click()
    expect(page.get_by_label("name (required)", exact=True)).to_have_attribute("name", "name")
    page.keyboard.press("Escape")
    page.get_by_role("button", name="edit web.port", exact=True).click()
    held: list[Route] = []

    def hold(route: Route) -> None:
        held.append(route)

    page.route("**/api/config", hold)
    page.get_by_role("button", name="Save", exact=True).click()
    expect(page.get_by_role("button", name="Saving…", exact=True)).to_be_disabled()
    captures.save(page, "config-saving")
    assert held
    held.pop().fulfill(status=422, json={"error": "Rejected for this test"})
    expect(page.get_by_role("button", name="Save", exact=True)).to_be_enabled()
    assert [e for e in errors if "422" not in e] == []


def test_cancelling_removal_returns_to_the_editor_draft(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """Cancelling a provider or reviewer removal keeps the value, layer and focus."""
    from playwright import sync_api

    config = paths.global_config_path()
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        '[providers.local]\nbackend = "openai"\nbase_url = "http://localhost:1"\n'
        '[review]\nreviewers = [{focus = "tests"}]\n'
    )
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/config")
    for key, action in [
        ("providers.local.base_url", "Remove provider"),
        ("review.reviewers", "Remove reviewer 1: tests"),
    ]:
        page.get_by_role("button", name=f"edit {key}", exact=True).click()
        editor = page.get_by_role("dialog", name=key)
        field = editor.get_by_role("textbox")
        layer = editor.get_by_role("checkbox")
        field.fill("unsaved draft")
        layer.check()
        remove = editor.get_by_role("button", name=action, exact=True)
        question = page.get_by_role("alertdialog")
        for answer in ["Cancel", "Escape"]:
            remove.click()
            cancel = question.get_by_role("button", name="Cancel")
            sync_api.expect(cancel).to_be_focused()
            if answer == "Escape":
                page.keyboard.press("Escape")
            else:
                cancel.click()
            sync_api.expect(field).to_have_value("unsaved draft")
            sync_api.expect(layer).to_be_checked()
            sync_api.expect(remove).to_be_focused()
        remove.click()
        question.get_by_role("button", name="Remove", exact=True).click()
        sync_api.expect(page.locator("dialog[open]")).to_have_count(0)
    assert "localhost" not in config.read_text()
    assert "tests" not in config.read_text()
    assert not errors


@pytest.mark.parametrize("lose_reload", [False, True])
def test_a_lost_removal_reply_requires_a_fresh_reviewer_confirmation(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, lose_reload: bool
) -> None:
    """A lost reply cannot replay an index after the next reviewer takes its place."""
    from playwright import sync_api

    config = paths.global_config_path()
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text('[review]\nreviewers = [{focus = "tests"}, {focus = "security"}]\n')
    page, _ = open_page(port)

    def lose_reply(route: Route) -> None:
        assert route.fetch().ok
        route.abort()

    page.route("**/api/config/reviewer/remove", lose_reply, times=1)
    page.goto(f"http://127.0.0.1:{port}/#/config")
    edit = page.get_by_role("button", name="edit review.reviewers", exact=True)
    edit.click()
    page.get_by_role("button", name="Remove reviewer 1: tests").click()
    question = page.get_by_role("alertdialog")
    if lose_reload:
        page.route("**/api/config", lambda route: route.abort(), times=1)
    question.get_by_role("button", name="Remove", exact=True).click()
    sync_api.expect(question).to_have_count(0)
    if lose_reload:
        sync_api.expect(page.get_by_role("heading", name="Could not reload Config")).to_be_visible()
        sync_api.expect(edit).to_have_count(0)
        page.get_by_role("button", name="Try again").click()
    edit.click()
    page.get_by_role("button", name="Remove reviewer 1: security").click()
    sync_api.expect(question).to_have_accessible_name("Remove reviewer 1: security?")
    question.get_by_role("button", name="Cancel").click()
    assert "tests" not in config.read_text() and "security" in config.read_text()
