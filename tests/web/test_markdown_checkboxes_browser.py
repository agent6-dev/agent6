from __future__ import annotations

import pathlib
from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest

from tests.unit import test_git_history
from tests.web import captures, test_files_edit, test_files_preview_browser, test_web_browser

if TYPE_CHECKING:
    from playwright.sync_api import Page

pytestmark = pytest.mark.browser
served = test_files_edit.served
browser = test_files_preview_browser.browser
open_page = test_web_browser.open_page


@pytest.mark.parametrize("width", [390, 1440])
def test_each_checkbox_save_requires_confirmation_and_the_displayed_digest(
    served: tuple[pathlib.Path, int],
    open_page: Callable[[int], tuple[Page, list[str]]],
    width: int,
) -> None:
    """A checkbox changes only its marker, after Save and only on the displayed bytes."""
    from playwright import sync_api

    repo, port = served
    path = repo / "dir" / "plan.md"
    source = "# Release checklist 📋\r\n\n- [ ] Review the notes\n- [X] Run the tests\r\n"
    path.write_bytes(source.encode())
    page, errors = open_page(port)
    page.set_viewport_size({"width": width, "height": 800})
    page.goto(f"http://127.0.0.1:{port}/#/files/WORKTREE/dir/plan.md")
    frame = page.frame_locator(".files-preview iframe")
    box = frame.get_by_role("checkbox", name="Review the notes")
    sync_api.expect(box).to_be_enabled()
    sync_api.expect(frame.get_by_role("checkbox", name="Run the tests")).to_be_checked()
    dialog = page.get_by_role("dialog", name="Save checkbox change?")
    box.click()
    sync_api.expect(dialog).to_contain_text("dir/plan.md")
    sync_api.expect(dialog).to_contain_text(f"{len(source.encode())} bytes")
    assert page.context.browser
    captures.save(page, "files-save-confirm-" + page.context.browser.browser_type.name)
    assert path.read_bytes() == source.encode()
    dialog.get_by_role("button", name="Cancel", exact=True).click()
    sync_api.expect(box).not_to_be_checked()
    box.click()
    with page.expect_response("**/api/files/text") as save:
        dialog.get_by_role("button", name="Save", exact=True).click()
    assert save.value.ok
    checked = source.replace("[ ]", "[x]")
    assert path.read_bytes() == checked.encode()
    sync_api.expect(box).to_be_checked()
    box.click()
    sync_api.expect(dialog).to_be_visible()
    path.write_text("Changed elsewhere\n")
    with page.expect_response("**/api/files/text") as stale:
        dialog.get_by_role("button", name="Save", exact=True).click()
    assert stale.value.status == 409
    sync_api.expect(dialog).to_contain_text("changed on disk")
    assert path.read_text() == "Changed elsewhere\n"
    expected = "Failed to load resource: the server responded with a status of 409 (Conflict)"
    assert errors == ([expected] if page.context.browser.browser_type.name == "chromium" else [])


def test_task_checkboxes_replace_only_their_own_bullets(
    served: tuple[pathlib.Path, int], open_page: Callable[[int], tuple[Page, list[str]]]
) -> None:
    """A task has no second marker, even in a loose list or under an ordinary bullet."""
    from playwright import sync_api

    repo, port = served
    (repo / "plan.md").write_text(
        "- Parent\n  - [x] Nested task\n\n1. [ ] Numbered task\n\n"
        "- [ ] Loose task\n\n  Another paragraph.\n"
    )
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/files/WORKTREE/plan.md")
    frame = page.frame_locator(".files-preview iframe")
    for task in ("Nested task", "Loose task"):
        sync_api.expect(frame.get_by_role("checkbox", name=task)).to_be_visible()
        sync_api.expect(frame.locator("li", has_text=task).last).to_have_css(
            "list-style-type", "none"
        )
    sync_api.expect(frame.locator("li", has_text="Parent")).to_have_css("list-style-type", "disc")
    sync_api.expect(frame.locator("ol > li")).to_have_css("list-style-type", "decimal")
    assert not errors


def test_only_the_open_working_markdown_can_request_its_own_save(
    served: tuple[pathlib.Path, int], open_page: Callable[[int], tuple[Page, list[str]]]
) -> None:
    """Another window, another file name, HTML and historical markdown get no save dialog."""
    from playwright import sync_api

    repo, port = served
    (repo / "plan.md").write_text("- [ ] Review\n")
    (repo / "page.html").write_text("<h1>Page</h1>")
    test_git_history._git(repo, "add", ".")
    test_git_history._git(repo, "commit", "-qm", "checklist")
    page, errors = open_page(port)
    for rev, name in (("WORKTREE", "plan.md"), ("HEAD", "plan.md"), ("WORKTREE", "page.html")):
        page.goto(f"http://127.0.0.1:{port}/#/files/{rev}/{name}")
        frame = page.frame_locator(".files-preview iframe")
        if name.endswith(".html"):
            page.get_by_role("button", name="Preview", exact=True).click()
            sync_api.expect(frame.get_by_role("heading", name="Page")).to_be_visible()
        elif rev == "HEAD":
            sync_api.expect(frame.get_by_role("checkbox")).to_be_disabled()
        else:
            sync_api.expect(frame.get_by_role("checkbox")).to_be_enabled()
        proposed = {"type": "preview-save", "name": name, "text": "not saved"}
        page.evaluate("m => window.postMessage(m, '*')", proposed)
        if name == "plan.md" and rev == "WORKTREE":
            proposed["name"] = "../other.md"
        frame.locator("body").evaluate("(_, m) => parent.postMessage(m, '*')", proposed)
        page.evaluate("() => new Promise(r => setTimeout(r, 60))")
        sync_api.expect(page.get_by_role("dialog")).to_have_count(0)
    assert not errors
