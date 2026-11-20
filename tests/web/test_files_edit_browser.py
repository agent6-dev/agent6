from __future__ import annotations

import pathlib
from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest

from tests.web import captures, test_files_edit, test_web_browser

if TYPE_CHECKING:
    from playwright.sync_api import Page

pytestmark = pytest.mark.browser
served = test_files_edit.served
open_page = test_web_browser.open_page


@pytest.mark.parametrize("width", [390, 1440])
@pytest.mark.parametrize("choice", ["load", "overwrite", "copy"])
def test_edit_keeps_the_draft_and_offers_three_conflict_choices(
    served: tuple[pathlib.Path, int],
    open_page: Callable[[int], tuple[Page, list[str]]],
    width: int,
    choice: str,
) -> None:
    """A disk change during editing cannot replace the draft or be silently overwritten."""
    from playwright.sync_api import expect

    repo, port = served
    page, errors = open_page(port)
    page.set_viewport_size({"width": width, "height": 900})
    page.goto(f"http://127.0.0.1:{port}/#/files/WORKTREE/plan.md")
    page.get_by_role("button", name="Edit", exact=True).click()
    dialog = page.get_by_role("dialog", name="Edit plan.md")
    editor = dialog.get_by_role("textbox", name="File text", exact=True)
    expect(editor).to_have_value("# Plan\n")
    editor.fill("# My plan\n\n- [ ] Review the result\n")
    if choice == "load":
        captures.save(page, "files-edit")
    (repo / "plan.md").write_text("# Disk version\n", encoding="utf-8")
    expect(dialog.get_by_text("The file changed on disk.", exact=True)).to_be_visible(timeout=8000)
    expect(editor).to_have_value("# My plan\n\n- [ ] Review the result\n")
    for label in ("Load new file", "Overwrite with mine", "Save as new file"):
        expect(dialog.get_by_role("button", name=label, exact=True)).to_be_visible()
    if choice == "load":
        captures.save(page, "files-edit-conflict")
        dialog.get_by_role("button", name="Load new file", exact=True).click()
        expect(editor).to_have_value("# Disk version\n")
        editor.fill("saved after reload\n")
        dialog.get_by_role("button", name="Save", exact=True).click()
        expected = "saved after reload\n"
    elif choice == "overwrite":
        dialog.get_by_role("button", name="Overwrite with mine", exact=True).click()
        expected = "# My plan\n\n- [ ] Review the result\n"
    else:
        dialog.get_by_role("textbox", name="New file name", exact=True).fill("copy.md")
        dialog.get_by_role("button", name="Save as new file", exact=True).click()
        expected = "# Disk version\n"
    expect(dialog).not_to_be_visible()
    assert (repo / "plan.md").read_text() == expected
    if choice == "copy":
        assert (repo / "copy.md").read_text() == "# My plan\n\n- [ ] Review the result\n"
        expect(page).to_have_url(f"http://127.0.0.1:{port}/#/files/WORKTREE/copy.md")
    assert not errors


def test_overwrite_refuses_a_second_disk_change(
    served: tuple[pathlib.Path, int], open_page: Callable[[int], tuple[Page, list[str]]]
) -> None:
    """Overwrite acknowledges only the disk version shown, not every later write."""
    from playwright.sync_api import expect

    repo, port = served
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/files/WORKTREE/plan.md")
    page.get_by_role("button", name="Edit", exact=True).click()
    dialog = page.get_by_role("dialog", name="Edit plan.md")
    editor = dialog.get_by_role("textbox", name="File text", exact=True)
    expect(editor).to_have_value("# Plan\n")
    editor.fill("mine\n")
    (repo / "plan.md").write_text("second\n", encoding="utf-8")
    expect(dialog.get_by_role("textbox", name="Disk version")).to_have_value(
        "second\n", timeout=8000
    )
    (repo / "plan.md").write_text("third\n", encoding="utf-8")
    dialog.get_by_role("button", name="Overwrite with mine").click()
    expect(dialog.get_by_role("textbox", name="Disk version")).to_have_value("third\n")
    expect(editor).to_have_value("mine\n")
    assert (repo / "plan.md").read_text() == "third\n"
    dialog.get_by_role("textbox", name="New file name").fill("../escape.md")
    dialog.get_by_role("button", name="Save as new file").click()
    expect(
        dialog.get_by_text("Use a new file name in this folder, without slashes or a leading dot.")
    ).to_be_visible()
    assert not (repo.parent / "escape.md").exists()
    assert all("409" in error for error in errors), errors


def test_edit_loads_the_whole_file_and_escape_keeps_unsaved_text(
    served: tuple[pathlib.Path, int], open_page: Callable[[int], tuple[Page, list[str]]]
) -> None:
    """The text preview's truncation cannot truncate a save from the editor."""
    from playwright.sync_api import expect

    repo, port = served
    content = "word" * 300_000 + "\n"
    (repo / "long.txt").write_text(content, encoding="utf-8")
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/files/WORKTREE/long.txt")
    page.get_by_role("button", name="Edit", exact=True).click()
    dialog = page.get_by_role("dialog", name="Edit long.txt")
    editor = dialog.get_by_role("textbox", name="File text", exact=True)
    expect(editor).to_be_enabled()
    assert editor.input_value() == content
    editor.fill("draft\n")
    page.keyboard.press("Escape")
    expect(dialog).to_be_visible()
    expect(editor).to_have_value("draft\n")
    dialog.get_by_role("button", name="Cancel", exact=True).click()
    assert (repo / "long.txt").read_text() == content
    assert not errors


def test_cancel_and_historical_files_never_write(
    served: tuple[pathlib.Path, int], open_page: Callable[[int], tuple[Page, list[str]]]
) -> None:
    """Cancel discards only the draft; a committed revision has no Edit button."""
    from playwright.sync_api import expect

    repo, port = served
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/files/WORKTREE/plan.md")
    page.get_by_role("button", name="Edit", exact=True).click()
    dialog = page.get_by_role("dialog", name="Edit plan.md")
    dialog.get_by_role("textbox", name="File text", exact=True).fill("discard\n")
    dialog.get_by_role("button", name="Cancel", exact=True).click()
    expect(dialog).not_to_be_visible()
    assert (repo / "plan.md").read_text() == "# Plan\n"
    page.goto(f"http://127.0.0.1:{port}/#/files/HEAD/b.txt")
    expect(page.locator(".files-source [data-line='1']")).to_be_visible()
    expect(page.get_by_role("button", name="Edit", exact=True)).to_have_count(0)
    assert not errors
