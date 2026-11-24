from __future__ import annotations

import json
import pathlib
from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest

from agent6 import paths
from tests.web import test_files_preview_browser, test_machine_browser, test_web_browser

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


def test_machine_detail_waits_for_something_to_toggle(
    open_page: OpenPage, port: int, tmp_path: pathlib.Path
) -> None:
    """An empty conversation has no detail button. Live thinking alone makes it useful."""
    from playwright import sync_api

    machine = test_machine_browser._machine(
        tmp_path, "working", test_machine_browser.LOOP, [], [], alive=True
    )
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/machine/working")
    card = page.locator(".conv-card")
    sync_api.expect(card.locator(".conv-empty")).to_be_visible()
    button = card.locator(".card-head-row button")
    sync_api.expect(button).to_be_hidden()
    call = {"actor": "worker", "model_call_id": "c1"}
    with (machine / "states/0000-work/logs.jsonl").open("a") as log:
        for row in [
            {"type": "session.start", "mode": "run", "user_task": "Do the work."},
            {"type": "model.call", "model": "m", **call},
            {"type": "model.thinking_delta", "text": "Look up the next state.", **call},
        ]:
            log.write(json.dumps(row) + "\n")
    sync_api.expect(card.locator(".conv-live")).to_contain_text("thinking…")
    button.click()
    sync_api.expect(card.locator(".conv-live")).to_contain_text("Look up the next state.")
    button.click()
    sync_api.expect(button).to_have_accessible_name("detail: hidden")
    sync_api.expect(button).to_be_visible()
    assert not errors
