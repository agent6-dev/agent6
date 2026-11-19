"""The source view the Git and Files pages share, in a browser: its lines and what they carry."""

from __future__ import annotations

import pathlib
from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest

from tests.unit.test_git_history import _git
from tests.web import test_web_browser

if TYPE_CHECKING:
    from playwright.sync_api import Page

pytestmark = pytest.mark.browser
browser = test_web_browser.browser
open_page = test_web_browser.open_page
git_port = test_web_browser.git_port

# Each line box of a row's code starts at the code column: the leftmost fragment per line box.
_LINE_STARTS = """row => {
  const code = row.querySelector('.source-code'), range = document.createRange();
  range.selectNodeContents(code);
  const starts = {};
  for (const r of range.getClientRects()) {
    const top = Math.round(r.top);
    starts[top] = Math.min(starts[top] ?? Infinity, r.left);
  }
  return {code: code.getBoundingClientRect().left, starts: Object.values(starts),
    gutter: row.querySelector('.g').getBoundingClientRect().right};
}"""


@pytest.fixture
def long_file(git_port: int, tmp_path: pathlib.Path) -> tuple[int, str, str]:
    """`long.txt`: 60 lines, line 2 blank and line 3 long; a second commit changes 3 and 50."""
    lines = [f"value_{i} = {i}" for i in range(1, 61)]
    lines[1] = ""
    lines[2] = "long = '" + "word " * 80 + "'"
    (tmp_path / "long.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _git(tmp_path, "add", "long.txt")
    _git(tmp_path, "commit", "-m", "long file")
    base = _git(tmp_path, "rev-parse", "HEAD")
    lines[2] = "long = '" + "other " * 80 + "'"
    lines[49] = "value_50 = 'changed'"
    (tmp_path / "long.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _git(tmp_path, "commit", "-am", "change lines 3 and 50")
    return git_port, base, _git(tmp_path, "rev-parse", "HEAD")


def _starts_in_code_column(page: Page, selector: str) -> None:
    geometry = page.locator(selector).evaluate(_LINE_STARTS)
    assert len(geometry["starts"]) > 1, "the line did not wrap"
    assert all(abs(x - geometry["code"]) < 1 for x in geometry["starts"]), geometry
    assert geometry["code"] > geometry["gutter"], geometry


def _select_lines_1_to_3(page: Page) -> str:
    return page.evaluate("""() => {
      const first = document.querySelector("[data-line='1'] .source-code");
      const third = document.querySelector("[data-line='3'] .source-code").firstChild;
      const range = document.createRange(); range.setStart(first, 0); range.setEnd(third, 4);
      const selection = getSelection(); selection.removeAllRanges(); selection.addRange(range);
      return selection.toString();
    }""")


def test_a_file_line_wraps_in_its_code_column_and_a_copy_keeps_blank_lines(
    open_page: Callable[[int], tuple[Page, list[str]]], long_file: tuple[int, str, str]
) -> None:
    """Wrapped text ran back under the line numbers, and a copy in blame mode lost blank lines."""
    port, _, _ = long_file
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/files/HEAD/long.txt")
    page.wait_for_selector("[data-line='60']")
    _starts_in_code_column(page, "[data-line='3']")
    assert _select_lines_1_to_3(page) == "value_1 = 1\n\nlong"
    page.locator(".files-blame").click()
    page.wait_for_selector("[data-line='50'] .blame-commit")
    assert _select_lines_1_to_3(page) == "value_1 = 1\n\nlong"
    assert not errors


@pytest.mark.parametrize("mode", ["inline", "split"])
def test_a_diff_line_wraps_in_its_code_column(
    open_page: Callable[[int], tuple[Page, list[str]]],
    long_file: tuple[int, str, str],
    mode: str,
) -> None:
    """Wrapped diff text ran back under the line numbers in every mode."""
    port, base, head = long_file
    page, errors = open_page(port)
    page.add_init_script(f"localStorage.setItem('a6-diff-mode', '{mode}')")
    page.goto(f"http://127.0.0.1:{port}/#/git/compare/{base}..{head}?path=long.txt")
    row = ".dl[data-new='3']" if mode == "inline" else ".ds-row[data-new='3'] > :nth-child(2)"
    page.wait_for_selector(row)
    _starts_in_code_column(page, row)
    assert not errors
