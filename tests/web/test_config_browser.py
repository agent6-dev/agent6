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


def test_a_config_row_lights_under_the_pointer(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """A row opens its editor on a click, yet only the cursor changed under the pointer."""
    page, errors = open_page(port)
    _config(page, port)
    row = page.locator('[data-key="web.port"]')
    rest = row.evaluate("e => getComputedStyle(e).backgroundColor")
    row.hover()
    assert row.evaluate("e => getComputedStyle(e).backgroundColor") != rest
    assert not errors


def test_an_unset_effort_opens_its_editor_on_the_value_its_row_shows(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """An effort no layer sets matched none of its choices, so its editor opened on a blank select.

    The row read `"off"  (adaptive)` meanwhile.
    """
    config = paths.global_config_path()
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        '[providers.anthropic]\nbackend = "anthropic"\n\n'
        '[models.worker]\nmodel = "anthropic/claude-sonnet-4-6"\n',
        encoding="utf-8",
    )
    page, errors = open_page(port)
    _config(page, port)
    row = page.locator('[data-key="models.worker.effort"]')
    shown = " ".join(row.locator(".val").inner_text().split())
    assert shown == '"off" (adaptive)'
    row.click()
    select = page.get_by_role("dialog").locator("select.field")
    assert select.evaluate("e => e.selectedIndex") == 0
    option = select.locator("option").first
    assert option.inner_text() == shown
    assert option.is_disabled()
    assert not errors


def test_an_add_dialog_shows_each_field_description_whole_under_the_field(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """Add provider and Add MCP server held each field's description in its placeholder.

    The placeholder was cut at the field's edge with its backticks raw, and went once typing
    started.
    """
    page, errors = open_page(port)
    _config(page, port)
    page.get_by_role("button", name="Add provider…").click()
    dialog = page.get_by_role("dialog")
    field = dialog.locator('input[name="base_url"]')
    field.fill("https://llm.example.com/v1")
    assert field.get_attribute("placeholder") is None
    described = dialog.locator(f"#{field.get_attribute('aria-describedby')}")
    assert described.is_visible()
    assert "https://api.anthropic.com/v1" in described.locator("code").all_inner_texts()
    assert "`" not in described.inner_text()
    assert described.evaluate("e => e.scrollWidth <= e.clientWidth")
    assert not errors


def test_a_long_list_value_shows_whole_in_its_editor(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """A list, table or string was edited in a one-line input that showed only a long value's tail.

    Enter still saves; Shift+Enter breaks the line.
    """
    command = (
        '["sh", "-c", "uv run ruff check && uv run ruff format --check && uv run pyright'
        ' && uv run tach check && uv run pytest"]'
    )
    config = paths.global_config_path()
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(f"[harness.verify]\ncommand = {command}\n", encoding="utf-8")
    page, errors = open_page(port)
    _config(page, port)
    page.locator('[data-key="harness.verify.command"]').click()
    dialog = page.get_by_role("dialog")
    field = dialog.locator(".field")
    assert field.input_value() == command
    hidden = "e => [e.scrollWidth - e.clientWidth, e.scrollHeight - e.clientHeight, e.scrollLeft]"
    assert field.evaluate(hidden) == [0, 0, 0]
    field.press("Shift+Enter")
    assert field.input_value() == command + "\n"
    assert dialog.is_visible()
    field.press("Enter")
    dialog.wait_for(state="hidden")
    assert not errors
