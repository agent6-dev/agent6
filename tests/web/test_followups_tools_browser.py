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
port = test_web_browser.port


@pytest.mark.parametrize("width", [390, 1440])
def test_tool_rows_open_to_selectable_arguments(
    open_page: Callable[[int], tuple[Page, list[str]]],
    port: int,
    tmp_path: pathlib.Path,
    width: int,
) -> None:
    """Tool details open with the keyboard and expose text beyond the clipped preview."""
    from playwright.sync_api import expect

    tail = "a long argument " * 20 + "END_ARGUMENT"
    session_logs.write_run(
        tmp_path,
        "tools-AAA111",
        [
            {"type": "tool.call", "name": "run_command", "args_preview": {"argv": [tail]}},
            {
                "type": "tool.result",
                "name": "run_command",
                "ok": True,
                "summary": "first\nlast",
                "exit_code": 3,
                "duration_s": 0.7501234567,
            },
        ],
    )
    page, errors = open_page(port)
    page.set_viewport_size({"width": width, "height": 844 if width == 390 else 900})
    page.goto(f"http://127.0.0.1:{port}/#/session/tools-AAA111")
    if width == 390:
        page.get_by_role("button", name="Widgets", exact=True).click()
        page.locator(".wmenu").get_by_role("button", name="Tool calls", exact=True).click()
    call = page.locator(".card-tools details")
    expect(call).to_have_count(1)
    call.locator("summary").focus()
    page.keyboard.press("Enter")
    expect(call.locator("pre")).to_contain_text("END_ARGUMENT")
    expect(call.locator("pre")).to_contain_text("first\nlast")
    expect(call.locator("pre")).to_contain_text("exit 3 in 0.75s")
    call.locator("pre").evaluate("e => getSelection().selectAllChildren(e)")
    assert "END_ARGUMENT" in page.evaluate("getSelection().toString()")
    captures.save(page, "tool-details")
    assert not errors
