from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import pytest

from tests.web import geometry, test_files_preview_browser, test_review_evidence, test_web_browser

if TYPE_CHECKING:
    from playwright.sync_api import Page

pytestmark = pytest.mark.browser
browser = test_files_preview_browser.browser
open_page = test_web_browser.open_page
evidence = test_review_evidence.evidence
OpenPage = Callable[[int], tuple["Page", list[str]]]


def test_reviewed_source_shares_the_heading_gutter(
    open_page: OpenPage, evidence: tuple[int, dict[str, Any]]
) -> None:
    """The reviewed bytes and their heading start at the same page inset."""
    port, revs = evidence
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/files/{revs['base']}/old.txt?panel=panel&side=base")
    heading, source = geometry.boxes(
        page.get_by_role("heading", name="base · old.txt"), page.get_by_text("Reviewed tree:")
    )
    assert heading["x"] == pytest.approx(source["x"], abs=0.5)
    assert not errors
