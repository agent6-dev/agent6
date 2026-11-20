from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest

from tests.web import captures, test_config_browser, test_web_browser

if TYPE_CHECKING:
    from playwright.sync_api import Page

pytestmark = pytest.mark.browser
open_page = test_web_browser.open_page
port = test_web_browser.port


@pytest.mark.parametrize("width", [390, 1440])
def test_config_paths_wrap_after_slashes(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, width: int
) -> None:
    """Path components that fit the column stay whole when the value wraps."""
    from playwright.sync_api import expect

    path = "/alpha-long-folder/beta-long-folder/gamma-long-folder/delta-long-folder" * 3
    test_config_browser._global_config(f'[sandbox]\nextra_read_paths = ["{path}"]\n')
    page, errors = open_page(port)
    page.set_viewport_size({"width": width, "height": 900})
    page.goto(f"http://127.0.0.1:{port}/#/config")
    page.locator(".cfg-bar input.filter").fill("extra_read_paths")
    value = page.locator('[data-key="sandbox.extra_read_paths"] .val')
    expect(value).to_contain_text(path)
    lines = value.evaluate("""p => {
      const lines = [], range = document.createRange(), walker = document.createTreeWalker(p, 4);
      let top = null;
      for (let node; (node = walker.nextNode());) for (let i = 0; i < node.length; i++) {
        range.setStart(node, i); range.setEnd(node, i + 1);
        const y = Math.round(range.getBoundingClientRect().top);
        if (y !== top) lines.push('');
        top = y; lines[lines.length - 1] += node.data[i];
      }
      return lines;
    }""")
    assert len(lines) > 1 and all(line.endswith("/") for line in lines[:-1]), lines
    captures.save(page, "config-paths")
    assert not errors
