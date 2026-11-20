from __future__ import annotations

import pathlib
from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest

from tests.web import session_logs, test_web_browser

if TYPE_CHECKING:
    from playwright import sync_api

pytestmark = pytest.mark.browser
open_page = test_web_browser.open_page
port = test_web_browser.port


def test_an_open_hub_sets_aside_a_running_sessions_pack(
    tmp_path: pathlib.Path, port: int, open_page: Callable[[int], tuple[sync_api.Page, list[str]]]
) -> None:
    """The next signed-in session read keeps working and warns about the pack."""
    session_logs.write_run(tmp_path, "active", [], live=True)
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/")
    page.locator(".dock textarea").wait_for()
    directory = tmp_path / ".agent6"
    directory.mkdir(exist_ok=True)
    (directory / "pack.json").write_text("{}", encoding="utf-8")
    page.goto(f"http://127.0.0.1:{port}/#/session/active")
    page.get_by_text("Do not run agent6 unpack", exact=False).first.wait_for(timeout=15000)
    assert (directory / "pack.invalid" / "pack.json.invalid").is_file()
    assert not errors
