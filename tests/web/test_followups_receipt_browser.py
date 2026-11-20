from __future__ import annotations

import pathlib
import types
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import pytest

from tests.unit import test_web_demo
from tests.web import captures, session_logs, test_review_evidence, test_web_browser

if TYPE_CHECKING:
    from playwright.sync_api import Page

pytestmark = pytest.mark.browser
open_page = test_web_browser.open_page
evidence = test_review_evidence.evidence
port = test_web_browser.port
web_demo = test_web_demo.web_demo


@pytest.mark.parametrize("width", [390, 1440])
def test_receipt_opens_in_a_wide_dialog(
    open_page: Callable[[int], tuple[Page, list[str]]],
    evidence: tuple[int, dict[str, Any]],
    width: int,
) -> None:
    """Receipt uses a modal outside the narrow drawer and returns focus on Escape."""
    from playwright.sync_api import expect

    port, _ = evidence
    page, errors = open_page(port)
    page.set_viewport_size({"width": width, "height": 900})
    page.goto(f"http://127.0.0.1:{port}/#/session/review-run")
    if width == 390:
        page.get_by_role("button", name="Widgets", exact=True).click()
        page.locator(".wmenu").get_by_role("button", name="Findings", exact=True).click()
    button = page.get_by_role("button", name="Receipt", exact=True)
    button.click()
    dialog = page.get_by_role("dialog", name="Receipt", exact=True)
    expect(dialog.locator(".receipt > pre")).to_contain_text("Receipt review-run")
    box = dialog.bounding_box()
    assert box and box["width"] == min(760, width - 32)
    captures.save(page, "receipt-dialog")
    page.keyboard.press("Escape")
    expect(dialog).to_have_count(0)
    expect(button).to_be_focused()
    assert not errors


@pytest.mark.parametrize("width", [390, 1440])
def test_recorded_tour_finds_the_receipt_in_its_dialog(
    open_page: Callable[[int], tuple[Page, list[str]]],
    port: int,
    tmp_path: pathlib.Path,
    web_demo: types.ModuleType,
    width: int,
) -> None:
    """The documentation tour waits for the receipt outside the Findings card."""
    from playwright.sync_api import expect

    session_logs.write_run(tmp_path, "merry-fountain-demo", [])
    page, errors = open_page(port)
    page.set_viewport_size({"width": width, "height": 900})
    page.add_init_script(web_demo.OVERLAY_INIT_SCRIPT)
    page.set_default_timeout(3000)
    page.goto(f"http://127.0.0.1:{port}/#/")
    web_demo.drive_findings(page, "phone" if width == 390 else "desktop")
    expect(page.get_by_role("dialog", name="Receipt")).to_contain_text("merry-fountain-demo")
    assert not errors
