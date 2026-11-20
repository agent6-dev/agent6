from __future__ import annotations

import pathlib
from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest

from tests.web import captures, session_logs, test_web_browser

if TYPE_CHECKING:
    from playwright.sync_api import Page

pytestmark = pytest.mark.browser
open_page = test_web_browser.open_page
git_port = test_web_browser.git_port


@pytest.mark.parametrize("width", [390, 1440])
def test_control_icons_use_labelled_svg_not_font_glyphs(
    open_page: Callable[[int], tuple[Page, list[str]]],
    git_port: int,
    tmp_path: pathlib.Path,
    width: int,
) -> None:
    """Stop, refresh, swap and back controls use the same SVG stroke as navigation."""
    from playwright.sync_api import expect

    session_logs.write_run(tmp_path, "icons-AAA111", [], live=True)
    page, errors = open_page(git_port)
    page.set_viewport_size({"width": width, "height": 900})
    for route, name in [
        ("/session/icons-AAA111", "Stop now"),
        ("/git", "refresh"),
        ("/git/compare/HEAD~1..HEAD", "swap base and head"),
    ]:
        page.goto(f"http://127.0.0.1:{git_port}/#{route}")
        button = page.get_by_role("button", name=name, exact=True)
        expect(button.locator("svg[aria-hidden=true]")).to_have_count(1)
        assert button.locator("svg").get_attribute("stroke") == "currentColor"
        if route == "/git":
            expect(page.locator(".gc-row").first).to_be_visible()
        elif route.startswith("/git/compare/"):
            expect(page.locator(".gf-open").first).to_be_visible()
        captures.save(page, "icons-" + name.replace(" ", "-"))
    assert not errors
