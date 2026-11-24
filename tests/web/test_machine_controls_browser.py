from __future__ import annotations

import pathlib
from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest

from agent6 import paths
from tests.web import test_files_preview_browser, test_web_browser

if TYPE_CHECKING:
    from playwright.sync_api import Page

pytestmark = pytest.mark.browser
browser = test_files_preview_browser.browser
open_page = test_web_browser.open_page
port = test_web_browser.port
OpenPage = Callable[[int], tuple["Page", list[str]]]


def test_machine_description_survives_navigation_until_create(
    open_page: OpenPage, port: int, tmp_path: pathlib.Path
) -> None:
    """Returning to State machines keeps its description separate from the Sessions draft."""
    from playwright import sync_api

    draft = paths.state_dir(tmp_path) / "sessions" / "machines" / "sort-rows"
    draft.mkdir(parents=True)
    (draft / "logs.jsonl").write_text('{"type":"session.start","mode":"run"}\n')
    page, errors = open_page(port)
    base = f"http://127.0.0.1:{port}/#/"
    page.goto(base)
    task = page.get_by_label("Task or question")
    task.fill("a different task")
    page.goto(base + "machines")
    field = page.get_by_label("Machine description")
    create = page.get_by_role("button", name="Create machine", exact=True)
    field.fill("sort the rows")
    page.goto(base)
    sync_api.expect(task).to_have_value("a different task")
    page.go_back()
    sync_api.expect(field).to_have_value("sort the rows")
    url = "**/api/machine/create"
    page.route(url, lambda route: route.fulfill(json={"ok": False, "error": "not created"}))
    create.click()
    sync_api.expect(page.get_by_text("not created", exact=True)).to_be_visible()
    page.reload()
    sync_api.expect(field).to_have_value("sort the rows")
    page.route(url, lambda route: route.fulfill(json={"draft": draft.name}))
    create.click()
    sync_api.expect(page).to_have_url(base + "draft/" + draft.name)
    page.go_back()
    sync_api.expect(field).to_be_empty()
    sync_api.expect(create).to_be_disabled()
    assert not errors
