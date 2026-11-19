"""The Config page in a browser: the settings grid, its bar, its editors and their states."""

from __future__ import annotations

import pathlib
import threading
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING

import pytest

from agent6 import paths
from agent6.ui.web import server as web_server
from tests.web import signin, test_web_browser

if TYPE_CHECKING:
    from playwright.sync_api import Page

pytestmark = pytest.mark.browser
browser = test_web_browser.browser
open_page = test_web_browser.open_page
port = test_web_browser.port

READY = '[data-key="agent6.config_version"]'


@pytest.fixture
def flag_port(tmp_path: pathlib.Path) -> Iterator[int]:
    """A server started with `--config FILE`: the leaf that file sets reads `flag (FILE)`."""
    flag = tmp_path / "operator-projects" / "a-monorepo-checkout" / "review-heavy-overrides.toml"
    flag.parent.mkdir(parents=True)
    flag.write_text('[sandbox]\nnetwork = "host"\n', encoding="utf-8")
    srv = web_server.WebServer(("127.0.0.1", 0), tmp_path, "", (flag,), secret=signin.SECRET)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield int(srv.server_address[1])
    finally:
        srv.shutdown()
        srv.server_close()


def _config(page: Page, port: int, width: int = 1280) -> None:
    page.set_viewport_size({"width": width, "height": 900})
    page.goto(f"http://127.0.0.1:{port}/#/config")
    page.locator(READY).wait_for()


@pytest.mark.parametrize("width", [1024, 390])
def test_the_config_bar_keeps_every_button_on_the_card(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, width: int
) -> None:
    """The filter and eight buttons shared one line that never wrapped.

    Below 1440px the last buttons sat past the card's edge, cut off, and the filter shrank to a
    sliver too narrow to show what was typed.
    """
    page, errors = open_page(port)
    _config(page, port, width)
    bar = page.locator(".cfg-bar")
    card = bar.evaluate("e => e.closest('.card').getBoundingClientRect().right")
    rights = bar.locator("button").evaluate_all(
        "buttons => buttons.map(b => b.getBoundingClientRect().right)"
    )
    assert len(rights) == 8
    assert max(rights) <= card
    assert bar.locator("input.filter").evaluate("e => e.offsetWidth") == bar.evaluate(
        "e => e.clientWidth"
    )
    assert not errors


# Per visible row: where its value starts, where its source ends, and whether the source overflows.
COLUMNS = """() => [...document.querySelectorAll('.cfg tr[data-key]')].filter(r => r.offsetParent)
  .map(r => [Math.round(r.cells[1].getBoundingClientRect().left),
             Math.round(r.cells[2].getBoundingClientRect().right),
             r.cells[2].scrollWidth > r.cells[2].clientWidth])"""


@pytest.mark.parametrize("width", [1440, 1024, 390])
def test_every_config_section_lines_its_columns_up(
    open_page: Callable[[int], tuple[Page, list[str]]], flag_port: int, width: int
) -> None:
    """Each section's table sized its columns from its own rows.

    The value and source columns started at a different x in every section and moved again
    when Modified only hid rows; a `--config FILE` source never wrapped, so it widened its
    column on a desktop and pushed a phone's rows past the card.
    """
    page, errors = open_page(flag_port)
    _config(page, flag_port, width)
    assert page.locator('[data-key="sandbox.network"] .src').inner_text().startswith("flag (")
    rows = page.evaluate(COLUMNS)
    columns = {(value, source) for value, source, _ in rows}
    assert len(columns) == 1
    assert not any(overflow for *_, overflow in rows)
    page.get_by_role("button", name="Modified only").click()
    modified = page.evaluate(COLUMNS)
    assert 0 < len(modified) < len(rows)
    assert {(value, source) for value, source, _ in modified} == columns
    assert page.evaluate("document.scrollingElement.scrollWidth") == width
    assert not errors


# The character before each line break in every visible settings cell.
BREAKS = r"""() => [...document.querySelectorAll('.cfg td')].filter(td => td.offsetParent)
.flatMap(td => {
  const walker = document.createTreeWalker(td, NodeFilter.SHOW_TEXT);
  let text = '', top = null; const before = [];
  for (let node = walker.nextNode(); node; node = walker.nextNode()) {
    for (let i = 0; i < node.length; i++) {
      const range = document.createRange(); range.setStart(node, i); range.setEnd(node, i + 1);
      const box = [...range.getClientRects()].find(rect => rect.width > 0);
      if (box && top !== null && box.top > top + 2) before.push(text.slice(-1));
      if (box) top = box.top;
      text += node.data[i];
    }
  }
  return before;
})"""


def test_a_long_config_key_or_value_wraps_between_its_parts(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """Keys broke anywhere (`providers.company-gateway-eu-west-produ` over `ction.base_url`).

    Chromium also split a URL value inside a word; every break now falls after a dot, a slash,
    a hyphen or a space.
    """
    config = paths.global_config_path()
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        "[providers.company-gateway-eu-west-production]\n"
        'backend = "openai"\n'
        'base_url = "https://llm-gateway.internal.example.com/openrouter/api/v1/chat/completions"\n',
        encoding="utf-8",
    )
    page, errors = open_page(port)
    _config(page, port, 1024)
    before = page.evaluate(BREAKS)
    assert len(before) >= 12
    assert set(before) <= set("./- ")
    assert not errors
