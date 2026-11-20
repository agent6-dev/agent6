from __future__ import annotations

import json
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


@pytest.mark.parametrize("width, height", [(1440, 900), (390, 844)])
def test_long_run_values_wrap_inside_the_overview(
    open_page: Callable[[int], tuple[sync_api.Page, list[str]]],
    port: int,
    tmp_path: pathlib.Path,
    width: int,
    height: int,
) -> None:
    """A long ID or model name must wrap rather than disappear at the card's edge."""
    from playwright import sync_api

    model = "reader/" + "multilingual" * 8
    session = session_logs.write_run(tmp_path, "import-" + "inventory" * 10, [])
    manifest = {"mode": "run", "models": {"driver": {"provider": "local", "model": model}}}
    (session / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    page, errors = open_page(port)
    page.set_viewport_size({"width": width, "height": height})
    page.goto(f"http://127.0.0.1:{port}/#/session/{session.name}")
    if width == 390:
        page.get_by_role("button", name="Widgets", exact=True).click()
        page.get_by_role("button", name="Overview", exact=True).click()
    grid = page.locator(".card-head .kv")
    for text in (session.name, model):
        value = grid.get_by_text(text, exact=False)
        sync_api.expect(value).to_be_visible()
        lines, right, edge = value.evaluate("""v => {
          const r = document.createRange();
          r.selectNodeContents(v);
          return [r.getClientRects().length, r.getBoundingClientRect().right,
                  v.closest('.kv').getBoundingClientRect().right];
        }""")
        assert right <= edge, text
        assert lines > 1, text
    assert not errors
