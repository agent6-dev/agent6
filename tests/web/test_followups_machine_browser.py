from __future__ import annotations

import pathlib
from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest

from tests.web import captures, test_machine_browser, test_web_browser

if TYPE_CHECKING:
    from playwright.sync_api import Page

pytestmark = pytest.mark.browser
open_page = test_web_browser.open_page
port = test_web_browser.port


@pytest.mark.parametrize("width", [390, 1440])
def test_ended_machine_replaces_its_composer_with_a_summary(
    open_page: Callable[[int], tuple[Page, list[str]]],
    port: int,
    tmp_path: pathlib.Path,
    width: int,
) -> None:
    """An ended machine has no disabled input or send buttons in its dock."""
    from playwright.sync_api import expect

    test_machine_browser._ended(tmp_path)
    page, errors = open_page(port)
    page.set_viewport_size({"width": width, "height": 900})
    page.goto(f"http://127.0.0.1:{port}/#/machine/ended")
    expect(page.locator(".machine-app .dock textarea")).to_be_hidden()
    expect(page.locator(".machine-app .dock button:visible")).to_have_count(0)
    expect(page.locator(".machine-app .dock .refusal")).to_contain_text("ended")
    captures.save(page, "machine-ended")
    assert not errors
