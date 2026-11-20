from __future__ import annotations

import json
import pathlib
from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest

from tests.web import test_files_edit, test_files_preview_browser, test_web_browser

if TYPE_CHECKING:
    from playwright.sync_api import Page

pytestmark = pytest.mark.browser
served = test_files_edit.served
browser = test_files_preview_browser.browser
open_page = test_web_browser.open_page


@pytest.mark.parametrize("scripts", [False, True])
def test_only_the_confirmed_form_answers_are_written(
    served: tuple[pathlib.Path, int],
    open_page: Callable[[int], tuple[Page, list[str]]],
    scripts: bool,
) -> None:
    """The form proposes a sibling file with scripts off or on; Save checks its displayed digest."""
    from playwright import sync_api

    repo, port = served
    (repo / "survey.html").write_text(
        '<h1>Book survey</h1><form method="post"><input name="pick" value="A">'
        '<input name="pick" value="B"><button>Submit answers</button></form>'
    )
    answers = repo / "survey.answers.json"
    page, _errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/files/WORKTREE/survey.html")
    page.get_by_role("button", name="Preview", exact=True).click()
    iframe = page.locator(".files-preview iframe")
    if scripts:
        page.get_by_role("button", name="Run scripts", exact=True).click()
        page.get_by_role("alertdialog").get_by_role(
            "button", name="Run scripts", exact=True
        ).click()
        sync_api.expect(iframe).to_have_attribute("sandbox", "allow-forms allow-scripts")
    submit = iframe.content_frame.get_by_role("button", name="Submit answers")
    submit.click()
    dialog = page.get_by_role("dialog", name="Save form answers?")
    sync_api.expect(dialog).to_contain_text("Create survey.answers.json (20 bytes)")
    assert not answers.exists()
    dialog.get_by_role("button", name="Cancel", exact=True).click()
    assert not answers.exists()
    submit.click()
    with page.expect_response("**/api/files/text") as saving:
        dialog.get_by_role("button", name="Save", exact=True).click()
    assert saving.value.ok
    assert json.loads(answers.read_text()) == {"pick": ["A", "B"]}
    submit.click()
    sync_api.expect(dialog).to_contain_text("Replace survey.answers.json (20 bytes)")
    answers.write_text("changed after submit")
    dialog.get_by_role("button", name="Save", exact=True).click()
    sync_api.expect(dialog.locator(".err[role=status]")).to_contain_text("file changed on disk")
    assert answers.read_text() == "changed after submit"
