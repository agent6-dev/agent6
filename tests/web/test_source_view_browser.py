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


def test_blame_leads_each_line_and_the_code_wraps_beside_it(
    open_page: Callable[[int], tuple[Page, list[str]]], long_file: tuple[int, str, str]
) -> None:
    """Blame sat after the longest line, past the right edge of the file pane."""
    port, _, _ = long_file
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/files/HEAD/long.txt")
    page.locator(".files-blame").click()
    page.wait_for_selector("[data-line='50'] .blame-commit")
    row = page.locator("[data-line='1']")
    assert row.evaluate("r => r.firstElementChild.classList.contains('blame-cell')")
    assert row.evaluate(
        "r => r.querySelector('.blame-cell').getBoundingClientRect().right"
        " <= r.querySelector('.g').getBoundingClientRect().left"
    )
    assert page.locator(".files-text").evaluate("e => e.scrollWidth <= e.clientWidth + 1")
    _starts_in_code_column(page, "[data-line='3']")
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


def test_the_blame_card_opens_beside_its_line_near_the_bottom_edge(
    open_page: Callable[[int], tuple[Page, list[str]]], long_file: tuple[int, str, str]
) -> None:
    """Near the bottom of the window the card slid up over the line it describes, squeezed."""
    from playwright.sync_api import expect

    port, _, _ = long_file
    page, errors = open_page(port)
    page.set_viewport_size({"width": 1280, "height": 600})
    page.goto(f"http://127.0.0.1:{port}/#/files/HEAD/long.txt")
    page.locator(".files-blame").click()
    link = page.locator("[data-line='50'] .blame-commit")
    link.wait_for()
    link.evaluate("""a => {
      const pane = a.closest('.files-text');
      pane.scrollTop += a.getBoundingClientRect().bottom - (innerHeight - 20);
      a.focus({preventScroll: true});
    }""")
    card = page.locator(".blame-card:visible")
    expect(card.locator(".blame-title")).to_have_text("change lines 3 and 50")
    box = card.bounding_box()
    line = link.bounding_box()
    assert box and line
    assert box["y"] + box["height"] <= line["y"] + 1 or box["y"] >= line["y"] + line["height"] - 1
    assert box["y"] + box["height"] <= 600 and box["x"] + box["width"] <= 1280
    assert card.locator(".mono").evaluate("e => e.getClientRects().length") == 1
    assert not errors


def test_the_blame_card_text_can_be_selected(
    open_page: Callable[[int], tuple[Page, list[str]]], long_file: tuple[int, str, str]
) -> None:
    """The card took the blame cell's no-select, so a double-click on its sha selected nothing."""
    from playwright.sync_api import expect

    port, _, head = long_file
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/files/HEAD/long.txt")
    page.locator(".files-blame").click()
    link = page.locator("[data-line='50'] .blame-commit")
    link.wait_for()
    link.focus()
    sha = page.locator(".blame-card:visible .mono")
    expect(sha).to_have_text(head)
    sha.dblclick()
    assert page.evaluate("getSelection().toString()") == head
    assert not errors


def test_a_drag_in_the_split_view_selects_one_side(
    open_page: Callable[[int], tuple[Page, list[str]]], long_file: tuple[int, str, str]
) -> None:
    """A drag down the new column selected the old column too, so a copy interleaved them."""
    port, base, head = long_file
    page, errors = open_page(port)
    page.add_init_script("localStorage.setItem('a6-diff-mode', 'split')")
    page.goto(f"http://127.0.0.1:{port}/#/git/compare/{base}..{head}?path=long.txt")
    start = page.locator(".ds-row[data-new='48'] > :nth-child(2) .source-code").bounding_box()
    end = page.locator(".ds-row[data-new='52'] > :nth-child(2) .source-code").bounding_box()
    assert start and end
    page.mouse.move(start["x"] + 2, start["y"] + start["height"] / 2)
    page.mouse.down()
    page.mouse.move(end["x"] + 60, end["y"] + end["height"] / 2, steps=8)
    page.mouse.up()
    copied = page.evaluate("getSelection().toString()")
    assert "value_50 = 'changed'" in copied
    assert "value_50 = 50" not in copied, copied
    assert not errors


def test_line_numbers_quote_from_the_keyboard(
    open_page: Callable[[int], tuple[Page, list[str]]], long_file: tuple[int, str, str]
) -> None:
    """A click on a line number quoted its line, but no key reached the numbers."""
    from playwright.sync_api import expect

    port, base, head = long_file
    page, errors = open_page(port)
    page.add_init_script("localStorage.setItem('a6-diff-mode', 'inline')")
    page.goto(f"http://127.0.0.1:{port}/#/git/compare/{base}..{head}?path=long.txt")
    page.wait_for_selector(".dl[data-new='50']")
    page.locator(".diff-blame").focus()
    page.keyboard.press("Tab")
    focused = page.locator(":focus")
    expect(focused).to_have_attribute("role", "button")
    expect(focused).to_have_attribute("aria-label", "Quote line 1, base side")
    page.keyboard.press("ArrowDown")
    expect(focused).to_have_attribute("aria-label", "Quote line 1, head side")
    page.keyboard.press("Enter")
    task = page.evaluate("window.pendingNewWork.task")
    assert task.startswith("long.txt:1 (head side, commit ") and "> value_1 = 1\n" in task
    assert page.locator(".git-diff .g[tabindex='0']").count() == 1
    assert not errors


def test_old_and_new_modes_tint_changed_lines_by_side(
    open_page: Callable[[int], tuple[Page, list[str]]], long_file: tuple[int, str, str]
) -> None:
    """Old and new marked their lines khaki while inline and split used red and green."""
    port, base, head = long_file
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/git/compare/{base}..{head}?path=long.txt")
    background = "e => getComputedStyle(e).backgroundColor"
    tints: dict[str, str] = {}
    for mode, selector in (
        ("inline", ".dl.del"),
        ("inline", ".dl.add"),
        ("old", ".sl.changed"),
        ("new", ".sl.changed"),
    ):
        page.evaluate(f"localStorage.setItem('a6-diff-mode', '{mode}')")
        page.reload()
        page.wait_for_selector(selector)
        tints[mode + selector] = page.locator(selector).first.evaluate(background)
    assert tints["old.sl.changed"] == tints["inline.dl.del"], tints
    assert tints["new.sl.changed"] == tints["inline.dl.add"], tints
    assert not errors


def test_a_tinted_diff_line_keeps_its_code_in_the_text_colour(
    open_page: Callable[[int], tuple[Page, list[str]]], long_file: tuple[int, str, str]
) -> None:
    """Inline coloured a changed line's code green or red on its tint, where split did not."""
    port, base, head = long_file
    page, errors = open_page(port)
    page.add_init_script("localStorage.setItem('a6-diff-mode', 'inline')")
    page.goto(f"http://127.0.0.1:{port}/#/git/compare/{base}..{head}?path=long.txt")
    page.wait_for_selector(".dl.add")
    colours = page.evaluate("""() => {
      const of = s => getComputedStyle(document.querySelector(s)).color;
      const probe = document.createElement('span'); document.body.appendChild(probe);
      const token = v => { probe.style.color = v; return getComputedStyle(probe).color; };
      return {code: of('.dl.add .source-code'), marker: of('.dl.add .diff-marker'),
        deleted: of('.dl.del .diff-marker'), text: token('var(--text)'), ok: token('var(--ok)'),
        err: token('var(--err)')};
    }""")
    assert colours["code"] == colours["text"], colours
    assert colours["marker"] == colours["ok"] and colours["deleted"] == colours["err"], colours
    assert not errors
