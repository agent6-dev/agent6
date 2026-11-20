from __future__ import annotations

import pathlib
from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest

from tests.web import captures, session_logs, test_machine_browser, test_web_browser

if TYPE_CHECKING:
    from playwright.sync_api import Page

pytestmark = pytest.mark.browser
open_page = test_web_browser.open_page
port = test_web_browser.port


@pytest.mark.parametrize("width", [390, 1440])
def test_composers_have_visible_associated_labels(
    open_page: Callable[[int], tuple[Page, list[str]]],
    port: int,
    tmp_path: pathlib.Path,
    width: int,
) -> None:
    """Composer names survive typing because labels, not placeholders, name the fields."""
    from playwright.sync_api import expect

    session_logs.write_run(tmp_path, "labels-AAA111", [], live=True)
    test_machine_browser._machine(
        tmp_path, "labels", test_machine_browser.LOOP, [], test_machine_browser._talk(1), alive=True
    )
    page, errors = open_page(port)
    page.set_viewport_size({"width": width, "height": 900})
    for route, name in [
        ("/", "Task or question"),
        ("/machines", "Machine description"),
        ("/session/labels-AAA111", "Message to the run"),
        ("/machine/labels", "Message to the machine"),
    ]:
        page.goto(f"http://127.0.0.1:{port}/#{route}")
        expect(page.get_by_label(name, exact=True)).to_have_count(1)
        label = page.locator("label", has_text=name)
        expect(label).to_be_visible()
        label.click()
        expect(page.get_by_label(name, exact=True)).to_be_focused()
        captures.save(page, "label-" + route.replace("/", "-"))
    assert not errors
