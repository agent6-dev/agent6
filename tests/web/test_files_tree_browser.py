"""The Files page in a browser: its tree, its bar and the open file's head."""

from __future__ import annotations

import pathlib
import re
from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest

from agent6.ui.web import git_model
from tests.unit.test_git_history import _git
from tests.web import test_web_browser

if TYPE_CHECKING:
    from playwright.sync_api import Page

pytestmark = pytest.mark.browser
browser = test_web_browser.browser
open_page = test_web_browser.open_page
git_port = test_web_browser.git_port
files_port = test_web_browser.files_port

DEEP = "deep/alpha-folder-name/beta-folder-name/gamma"
LEAF = DEEP + "/a-long-file-name-for-a-narrow-tree.txt"
# Where a row's first item, the caret, starts: a folded caret is turned, so its own box is not.
_CONTENT_LEFT = "r => r.getBoundingClientRect().left + parseFloat(getComputedStyle(r).paddingLeft)"


@pytest.fixture
def big_tree(git_port: int, tmp_path: pathlib.Path) -> int:
    """Over 200 files, so the tree starts folded, and a deep folder with long names."""
    (tmp_path / "many").mkdir()
    for i in range(210):
        (tmp_path / "many" / f"f{i:03}.txt").write_text(f"{i}\n", encoding="utf-8")
    (tmp_path / DEEP / "delta").mkdir(parents=True)
    (tmp_path / LEAF).write_text("leaf\n", encoding="utf-8")
    (tmp_path / DEEP / "delta" / "x.txt").write_text("x\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-m", "a big tree")
    return git_port


def _row(path: str) -> str:
    return f".files-tree [data-path='{path}']"


def _open_deep(page: Page) -> None:
    """Open the folders down to the long file by clicking them."""
    parts = DEEP.split("/")
    for depth in range(1, len(parts) + 1):
        row = page.locator(_row("/".join(parts[:depth])))
        if row.get_attribute("aria-expanded") == "false":
            row.click()


def test_a_deep_link_reveals_its_file_and_a_filter_edit_keeps_open_folders(
    open_page: Callable[[int], tuple[Page, list[str]]], big_tree: int
) -> None:
    """A linked file's row stayed in folded folders, and any filter edit folded every folder."""
    from playwright.sync_api import expect

    page, errors = open_page(big_tree)
    page.goto(f"http://127.0.0.1:{big_tree}/#/files/HEAD/{LEAF}")
    page.wait_for_selector("[data-line='1']")
    expect(page.locator(_row(LEAF))).to_be_in_viewport()
    expect(page.locator(_row("deep"))).to_have_attribute("aria-expanded", "true")
    folder = page.locator(_row("many"))
    folder.click()
    expect(folder).to_have_attribute("aria-expanded", "true")
    search = page.locator(".files-bar input.filter")
    search.click()
    page.keyboard.type("x")
    page.keyboard.press("Backspace")
    expect(folder).to_have_attribute("aria-expanded", "true")
    folder.click()
    search.fill("f.txt")
    page.locator(_row("dir/f.txt")).click()
    page.wait_for_selector(".files-view [data-line='1']")
    search.fill("")
    expect(page.locator(_row("dir/f.txt"))).to_be_visible()
    expect(folder).to_have_attribute("aria-expanded", "false")
    assert not errors


def test_the_tree_is_one_tab_stop_that_the_arrow_keys_walk(
    open_page: Callable[[int], tuple[Page, list[str]]], big_tree: int
) -> None:
    """Every row was its own Tab stop with no arrow keys and no tree semantics."""
    from playwright.sync_api import expect

    page, errors = open_page(big_tree)
    page.goto(f"http://127.0.0.1:{big_tree}/#/files/HEAD")
    page.wait_for_selector(_row("deep"))
    expect(page.locator(".files-tree [role=tree]")).to_have_count(1)
    page.locator(".files-bar input.filter").focus()
    page.keyboard.press("Tab")
    focused = page.locator(":focus")
    expect(focused).to_have_attribute("role", "treeitem")
    expect(focused).to_have_attribute("data-path", "deep")
    assert page.locator(".files-tree [tabindex='0']").count() == 1
    page.keyboard.press("ArrowRight")
    expect(page.locator(_row("deep"))).to_have_attribute("aria-expanded", "true")
    page.keyboard.press("ArrowRight")
    expect(focused).to_have_attribute("data-path", "deep/alpha-folder-name")
    expect(focused).to_have_attribute("aria-level", "2")
    page.keyboard.press("ArrowLeft")
    expect(focused).to_have_attribute("data-path", "deep")
    page.keyboard.press("ArrowLeft")
    expect(page.locator(_row("deep"))).to_have_attribute("aria-expanded", "false")
    page.keyboard.press("ArrowDown")
    expect(focused).to_have_attribute("data-path", "dir")
    page.keyboard.press("End")
    expect(focused).to_have_attribute("data-path", "logo.png")
    page.keyboard.press("Enter")
    page.wait_for_selector(".files-view img")
    expect(page.locator(_row("logo.png"))).to_have_attribute("aria-selected", "true")
    assert page.locator(".files-tree [tabindex='0']").count() == 1
    assert not errors


def test_tab_reaches_the_tree_when_its_open_file_is_folded_away(
    open_page: Callable[[int], tuple[Page, list[str]]], big_tree: int
) -> None:
    """A repaint left the folded-away open file the only Tab stop, so Tab skipped the tree."""
    from playwright.sync_api import expect

    page, errors = open_page(big_tree)
    page.goto(f"http://127.0.0.1:{big_tree}/#/files/HEAD/{LEAF}")
    page.wait_for_selector("[data-line='1']")
    page.locator(_row("deep")).click()
    expect(page.locator(_row("deep"))).to_have_attribute("aria-expanded", "false")
    page.locator(".files-bar input.filter").click()
    page.keyboard.type("x")
    page.keyboard.press("Backspace")
    page.keyboard.press("Tab")
    expect(page.locator(":focus")).to_have_attribute("data-path", "deep")
    assert not errors


def test_a_narrow_tree_keeps_each_name_on_one_line(
    open_page: Callable[[int], tuple[Page, list[str]]], big_tree: int
) -> None:
    """At the tree's narrowest, names broke down to a character per line beside the notes."""
    page, errors = open_page(big_tree)
    page.add_init_script("localStorage.setItem('a6-files-tree-w', '180px')")
    page.goto(f"http://127.0.0.1:{big_tree}/#/files/HEAD")
    page.wait_for_selector(_row("deep"))
    _open_deep(page)
    heights = page.locator(".files-tree .gtree-dir:visible, .files-tree .gf:visible").evaluate_all(
        "rows => rows.map(r => r.getBoundingClientRect().height)"
    )
    assert heights and max(heights) < 1.5 * min(heights), heights
    assert page.locator(".files-tree .gtree-note:visible").count() == 0
    assert page.locator(_row(LEAF)).get_attribute("title") == LEAF
    assert page.locator(_row(DEEP)).get_attribute("title") == DEEP
    assert not errors


def test_tree_files_line_up_with_their_sibling_folders(
    open_page: Callable[[int], tuple[Page, list[str]]], big_tree: int
) -> None:
    """File names sat left of their sibling folders' names, children left of their parent's."""
    page, errors = open_page(big_tree)
    page.goto(f"http://127.0.0.1:{big_tree}/#/files/HEAD")
    page.wait_for_selector(_row("deep"))
    _open_deep(page)
    left = "e => e.getBoundingClientRect().left"
    file_name = page.locator(_row(LEAF) + " .gf-path").evaluate(left)
    folder = _row(DEEP + "/delta")
    assert page.locator(folder + " .gtree-name").evaluate(left) == pytest.approx(file_name, abs=1)
    parent_name = page.locator(_row(DEEP) + " .gtree-name").evaluate(left)
    assert page.locator(folder).evaluate(_CONTENT_LEFT) == pytest.approx(parent_name, abs=1)
    assert not errors


@pytest.mark.parametrize("width", [1280, 390])
def test_tree_rows_start_at_the_page_inset(
    open_page: Callable[[int], tuple[Page, list[str]]], big_tree: int, width: int
) -> None:
    """The tree's text sat 4px from its edge while the bar above kept the page inset."""
    page, errors = open_page(big_tree)
    page.set_viewport_size({"width": width, "height": 900})
    page.goto(f"http://127.0.0.1:{big_tree}/#/files/HEAD")
    page.wait_for_selector(_row("deep"))
    bar = page.locator(".files-bar").evaluate(
        "b => [...b.children].find(c => c.offsetParent).getBoundingClientRect().left"
    )
    assert page.locator(_row("deep")).evaluate(_CONTENT_LEFT) == pytest.approx(bar, abs=1)
    assert not errors


@pytest.mark.parametrize("width", [1280, 820, 390])
def test_the_revision_list_opens_under_its_field(
    open_page: Callable[[int], tuple[Page, list[str]]],
    big_tree: int,
    tmp_path: pathlib.Path,
    width: int,
) -> None:
    """The list spanned the bar, opened under the filter on a phone, and cut names to its field."""
    _git(tmp_path, "branch", "a-rather-long-branch-for-the-list")
    page, errors = open_page(big_tree)
    page.set_viewport_size({"width": width, "height": 900})
    page.goto(f"http://127.0.0.1:{big_tree}/#/files/HEAD")
    field = page.get_by_role("combobox", name="revision")
    field.click()
    listing = page.locator(".gpick-list")
    listing.wait_for()
    box, under = field.bounding_box(), listing.bounding_box()
    bar = page.locator(".files-bar").bounding_box()
    assert box and under and bar
    assert under["x"] == pytest.approx(box["x"], abs=1)
    assert 0 <= under["y"] - (box["y"] + box["height"]) <= 6
    room = bar["x"] + bar["width"] - box["x"]
    assert under["width"] == pytest.approx(min(max(box["width"], 480), room), abs=1)
    cut = listing.locator(".gref, .gref + span").evaluate_all(
        "refs => refs.filter(r => r.scrollWidth > r.clientWidth + 1).map(r => r.textContent)"
    )
    assert cut == []
    assert not errors


def test_a_revision_list_note_shows_its_whole_text(
    open_page: Callable[[int], tuple[Page, list[str]]], big_tree: int
) -> None:
    """A note row in the revision list, such as a long error, was cut mid-word with no ellipsis."""
    from playwright.sync_api import expect

    reason = "git log failed: " + "a reason that keeps going " * 8
    page, errors = open_page(big_tree)
    page.route(
        "**/api/git/search*",
        lambda route: route.fulfill(status=500, json={"error": reason.strip()}),
    )
    page.goto(f"http://127.0.0.1:{big_tree}/#/files/HEAD")
    page.get_by_role("combobox", name="revision").click()
    note = page.locator(".gpick-list .ac-item.err")
    expect(note).to_have_text(reason.strip())
    assert note.evaluate("e => e.scrollWidth <= e.clientWidth")
    assert [e for e in errors if "500" not in e] == []  # the failed search's response


def test_the_revision_field_drops_text_it_did_not_apply(
    open_page: Callable[[int], tuple[Page, list[str]]], big_tree: int
) -> None:
    """Typed text stayed in the revision field after Escape or a click away, naming no listing."""
    from playwright.sync_api import expect

    page, errors = open_page(big_tree)
    page.goto(f"http://127.0.0.1:{big_tree}/#/files/HEAD")
    page.wait_for_selector(_row("deep"))
    field = page.get_by_role("combobox", name="revision")
    field.click()
    page.keyboard.type("mai")
    page.keyboard.press("Escape")
    expect(field).to_have_value("HEAD")
    page.keyboard.press("ArrowDown")  # the whole list again, not a search for the label
    expect(page.locator(".gpick-list .ac-item", has_text="working tree")).to_be_visible()
    page.keyboard.type("mai")
    page.keyboard.press("Tab")
    expect(field).to_have_value("HEAD")
    field.click()
    page.keyboard.type("mai")
    page.mouse.click(1200, 850)  # the empty pane, clear of the list
    expect(field).to_have_value("HEAD")
    assert page.url.endswith("#/files/HEAD")
    assert not errors


def test_a_revision_from_the_hash_reads_as_the_picker_names_it(
    open_page: Callable[[int], tuple[Page, list[str]]], git_port: int, tmp_path: pathlib.Path
) -> None:
    """After a reload a picked commit read as its 40-character sha, and a ref as its full name."""
    from playwright.sync_api import expect

    sha = _git(tmp_path, "rev-parse", "HEAD")
    page, errors = open_page(git_port)
    field = page.get_by_role("combobox", name="revision")
    for hash_, label in ((sha, sha[:7]), ("refs%2Fheads%2Ffeature", "feature")):
        page.goto(f"http://127.0.0.1:{git_port}/#/files/{hash_}")
        page.reload()
        expect(field).to_have_value(label)
        expect(page.locator(_row("b.txt"))).to_be_visible()
    assert not errors


def test_the_file_head_keeps_one_height(
    open_page: Callable[[int], tuple[Page, list[str]]], big_tree: int
) -> None:
    """The head was 7px shorter for an image, and for a text file until its text arrived."""
    page, errors = open_page(big_tree)
    page.goto(f"http://127.0.0.1:{big_tree}/#/files/HEAD/logo.png")
    page.wait_for_selector(".files-view img")
    head = page.locator(".files-head")
    image = head.evaluate("e => e.getBoundingClientRect().height")
    page.locator(_row("b.txt")).click()
    page.wait_for_selector(".files-blame")
    assert head.evaluate("e => e.getBoundingClientRect().height") == pytest.approx(image, abs=0.5)
    assert not errors


@pytest.mark.parametrize("width", [820, 390])
def test_a_narrow_file_head_keeps_its_height_while_the_text_loads(
    open_page: Callable[[int], tuple[Page, list[str]]],
    big_tree: int,
    tmp_path: pathlib.Path,
    width: int,
) -> None:
    """A narrow head grew as its text or Markdown buttons loaded; an image's head was shorter."""
    (tmp_path / "notes.md").write_text("# Notes\n", encoding="utf-8")
    _git(tmp_path, "add", "notes.md")
    _git(tmp_path, "commit", "-m", "notes")
    page, errors = open_page(big_tree)
    page.set_viewport_size({"width": width, "height": 900})
    page.add_init_script("""
      window.__heads = [];
      new MutationObserver(() => {
        const head = document.querySelector('.files-head');
        if (head) window.__heads.push(head.getBoundingClientRect().height);
      }).observe(document, {subtree: true, childList: true, characterData: true});
    """)
    held: dict[str, float] = {}
    for path, loaded in (("b.txt", ".files-blame:enabled"), ("notes.md", ".md-view h1")):
        page.goto(f"http://127.0.0.1:{big_tree}/#/files/HEAD/{path}")
        page.reload()  # a fresh document, so the heights recorded are this file's alone
        page.wait_for_selector(loaded)
        heights = page.evaluate("window.__heads")
        assert heights and len(set(heights)) == 1, (path, heights)
        held[path] = heights[0]
    page.goto(f"http://127.0.0.1:{big_tree}/#/files/HEAD/logo.png")
    page.wait_for_selector(".files-view img")
    assert page.locator(".files-head").evaluate(
        "e => e.getBoundingClientRect().height"
    ) == pytest.approx(held["b.txt"], abs=0.5)
    assert not errors


def test_the_file_note_shares_the_path_row_where_it_fits(
    open_page: Callable[[int], tuple[Page, list[str]]], big_tree: int
) -> None:
    """At 1024px every file head gave its note a row of its own, though it fit beside the path."""
    page, errors = open_page(big_tree)
    page.set_viewport_size({"width": 1024, "height": 768})
    page.goto(f"http://127.0.0.1:{big_tree}/#/files/HEAD/b.txt")
    page.wait_for_selector(".files-blame:enabled")
    middle = "e => { const r = e.getBoundingClientRect(); return r.top + r.height / 2; }"
    note = page.locator(".files-head .files-note").evaluate(middle)
    assert note == pytest.approx(page.locator(".files-head .gf-path").evaluate(middle), abs=1)
    assert not errors


def test_the_file_pane_draws_its_focus_ring_inside(
    open_page: Callable[[int], tuple[Page, list[str]]], git_port: int, tmp_path: pathlib.Path
) -> None:
    """The scrolling file pane's ring sat outside it, cut off at the screen's edges."""
    (tmp_path / "long.txt").write_text("".join(f"line {i}\n" for i in range(200)), encoding="utf-8")
    _git(tmp_path, "add", "long.txt")
    _git(tmp_path, "commit", "-m", "a long file")
    page, errors = open_page(git_port)
    page.goto(f"http://127.0.0.1:{git_port}/#/files/HEAD/long.txt")
    page.wait_for_selector(".files-blame:enabled")
    page.locator(".files-blame").focus()
    page.keyboard.press("Tab")
    ring = page.locator(".files-text").evaluate(
        "e => e === document.activeElement && e.matches(':focus-visible')"
        " && parseFloat(getComputedStyle(e).outlineOffset)"
        " + parseFloat(getComputedStyle(e).outlineWidth)"
    )
    assert ring is not False and ring <= 0, ring
    assert not errors


def test_the_tree_and_the_file_meet_at_one_hairline(
    open_page: Callable[[int], tuple[Page, list[str]]], big_tree: int
) -> None:
    """Tree rows stopped 6px short of the divider, and a hidden tree left a doubled line."""
    page, errors = open_page(big_tree)
    page.goto(f"http://127.0.0.1:{big_tree}/#/files/HEAD/b.txt")
    page.wait_for_selector(".files-blame:enabled")
    left = "e => e.getBoundingClientRect().left"
    row = page.locator(_row("deep")).evaluate("e => e.getBoundingClientRect().right")
    assert page.locator(".files-head").evaluate(left) - row == pytest.approx(1, abs=0.5)
    page.get_by_role("button", name="Hide tree", exact=True).click()
    rail = page.locator("aside.rail").evaluate("e => e.getBoundingClientRect().right")
    assert page.locator(".files-head").evaluate(left) == pytest.approx(rail, abs=0.5)
    assert not errors


@pytest.mark.parametrize(("width", "line"), [(1280, "0px"), (390, "1px")])
def test_one_hairline_tops_the_tree(
    open_page: Callable[[int], tuple[Page, list[str]]], big_tree: int, width: int, line: str
) -> None:
    """The first row drew its own line under the body's edge, a 2px line over the tree."""
    page, errors = open_page(big_tree)
    page.set_viewport_size({"width": width, "height": 900})
    page.goto(f"http://127.0.0.1:{big_tree}/#/files/HEAD")
    page.wait_for_selector(_row("deep"))
    top = page.locator(_row("deep")).evaluate("e => getComputedStyle(e).borderTopWidth")
    assert top == line
    assert not errors


def test_the_tree_insets_its_messages(
    open_page: Callable[[int], tuple[Page, list[str]]],
    big_tree: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cut-list note sat on the tree's edge, and an error flush under the tree's top."""
    from playwright.sync_api import expect

    monkeypatch.setattr(git_model, "TREE_MAX", 5)
    page, errors = open_page(big_tree)
    page.goto(f"http://127.0.0.1:{big_tree}/#/files/HEAD")
    text = (
        "e => { const r = document.createRange(); r.selectNodeContents(e);"
        " return r.getBoundingClientRect(); }"
    )
    note = page.locator(".files-tree > .gf-note").evaluate(text)
    row = page.locator(".files-tree [data-path]").first.evaluate(_CONTENT_LEFT)
    assert note["x"] == pytest.approx(row, abs=1)
    page.goto(f"http://127.0.0.1:{big_tree}/#/files/no-such-revision")
    message = page.locator(".files-tree > .err")
    expect(message).to_contain_text("unknown revision")
    top = page.locator(".files-tree").evaluate("e => e.getBoundingClientRect().top")
    assert message.evaluate(text)["y"] - top >= 16
    assert [e for e in errors if "422" not in e] == []  # the unknown revision's response


def test_a_failed_listing_shows_no_count(
    open_page: Callable[[int], tuple[Page, list[str]]], big_tree: int
) -> None:
    """The count kept the last listing's total beside a revision that failed to list."""
    from playwright.sync_api import expect

    page, errors = open_page(big_tree)
    page.goto(f"http://127.0.0.1:{big_tree}/#/files/HEAD")
    counts = page.locator(".files-counts")
    expect(counts).to_have_text(re.compile(r"^\d+ files$"))
    page.get_by_role("combobox", name="revision").fill("zz-unknown")
    page.keyboard.press("Enter")
    expect(page.locator(".files-tree .err")).to_contain_text("unknown revision")
    expect(counts).to_have_text("")
    assert [e for e in errors if "422" not in e] == []  # the unknown revision's response


def test_the_bar_holds_still_while_filtering_and_toggling(
    open_page: Callable[[int], tuple[Page, list[str]]], big_tree: int
) -> None:
    """The count resized the fields as it changed, and the tree toggle grew 6px on Show tree."""
    page, errors = open_page(big_tree)
    page.goto(f"http://127.0.0.1:{big_tree}/#/files/HEAD")
    page.wait_for_selector(_row("deep"))
    boxes = (
        "() => [...document.querySelectorAll('.files-bar > *')].filter(c => c.offsetParent)"
        ".map(c => { const r = c.getBoundingClientRect(); return [r.left, r.width]; }).flat()"
    )
    still = page.evaluate(boxes)
    page.locator(".files-bar input.filter").fill("f1")
    page.wait_for_function("document.querySelector('.files-counts').textContent.includes(' of ')")
    assert page.evaluate(boxes) == pytest.approx(still, abs=0.5)
    page.get_by_role("button", name="Hide tree", exact=True).click()
    assert page.evaluate(boxes) == pytest.approx(still, abs=0.5)
    assert not errors


def test_the_bar_holds_still_when_the_listing_arrives(
    open_page: Callable[[int], tuple[Page, list[str]]], big_tree: int
) -> None:
    """The fields shrank 54px when the count arrived beside them, on every load."""
    from playwright.sync_api import Route

    page, errors = open_page(big_tree)
    held: list[Route] = []

    def hold(route: Route) -> None:
        held.append(route)

    page.route("**/api/git/tree*", hold)
    page.goto(f"http://127.0.0.1:{big_tree}/#/files/HEAD")
    page.wait_for_selector(".files-tree .empty")
    boxes = (
        "() => [...document.querySelectorAll('.files-bar .gpick, .files-bar input.filter')]"
        ".map(c => { const r = c.getBoundingClientRect(); return [r.left, r.width]; }).flat()"
    )
    loading = page.evaluate(boxes)
    while not held:
        page.wait_for_timeout(50)
    held[0].continue_()
    page.wait_for_selector(_row("deep"))
    assert page.evaluate(boxes) == pytest.approx(loading, abs=0.5)
    page.locator(".files-bar input.filter").fill("f1")
    gap = page.evaluate(
        "() => { const f = document.querySelector('.files-bar input.filter');"
        " const end = f.getBoundingClientRect().right"
        " - parseFloat(getComputedStyle(f).paddingRight);"
        " return document.querySelector('.files-counts').getBoundingClientRect().left - end; }"
    )
    assert gap > 0, "the typed text runs under the count"
    assert not errors


def test_the_empty_pane_says_how_to_open_a_file(
    open_page: Callable[[int], tuple[Page, list[str]]], big_tree: int
) -> None:
    """With the tree hidden and no file open, the pane still said to click a file."""
    from playwright.sync_api import expect

    page, errors = open_page(big_tree)
    page.goto(f"http://127.0.0.1:{big_tree}/#/files/HEAD")
    page.wait_for_selector(_row("deep"))
    pane = page.locator(".files-view .empty")
    expect(pane).to_have_text("click a file to read it")
    page.get_by_role("button", name="Hide tree", exact=True).click()
    expect(pane).to_have_text("show the tree to pick a file")
    page.reload()
    expect(pane).to_have_text("show the tree to pick a file")
    page.get_by_role("button", name="Show tree", exact=True).click()
    expect(pane).to_have_text("click a file to read it")
    assert not errors


def test_the_pane_offers_a_file_only_beside_a_listing_with_files(
    open_page: Callable[[int], tuple[Page, list[str]]], big_tree: int
) -> None:
    """The pane said to click a file while the tree loaded, beside a failed listing or no match."""
    from playwright.sync_api import Route, expect

    page, errors = open_page(big_tree)
    held: list[Route] = []

    def hold(route: Route) -> None:
        held.append(route)

    page.route("**/api/git/tree*", hold)
    page.goto(f"http://127.0.0.1:{big_tree}/#/files/HEAD")
    prompt = page.locator(".files-view .empty")
    while not held:
        page.wait_for_timeout(50)
    expect(prompt).to_have_count(0)
    held.pop().continue_()
    expect(prompt).to_have_text("click a file to read it")
    search = page.locator(".files-bar input.filter")
    search.fill("no-such-file")
    expect(page.locator(".files-tree .empty")).to_have_text("no file matches")
    expect(prompt).to_be_hidden()
    search.fill("")
    expect(prompt).to_be_visible()
    page.unroute("**/api/git/tree*", hold)
    page.route("**/api/git/tree*", lambda route: route.fulfill(status=500, json={"error": "nope"}))
    page.reload()
    expect(page.locator(".files-tree .err")).to_contain_text("nope")
    expect(prompt).to_have_count(0)
    assert [e for e in errors if "500" not in e] == []  # the failed listing's response


def test_a_failed_read_offers_to_read_again(
    open_page: Callable[[int], tuple[Page, list[str]]], big_tree: int
) -> None:
    """A failed listing or file read left only red text, with no way to read it again."""
    from playwright.sync_api import Route, expect

    page, errors = open_page(big_tree)
    failing = {"tree": 1, "text": 1}

    def fail_once(route: Route) -> None:
        kind = "tree" if "/api/git/tree" in route.request.url else "text"
        if failing[kind]:
            failing[kind] -= 1
            route.fulfill(status=500, json={"error": "busy"})
        else:
            route.continue_()

    page.route("**/api/git/tree*", fail_once)
    page.route("**/api/git/text*", fail_once)
    page.goto(f"http://127.0.0.1:{big_tree}/#/files/HEAD/b.txt")
    tree, text = page.locator(".files-tree"), page.locator(".files-text")
    expect(tree.locator(".err")).to_contain_text("busy")
    tree.get_by_role("button", name="Try again", exact=True).click()
    expect(text.locator(".err")).to_contain_text("busy")
    text.get_by_role("button", name="Try again", exact=True).click()
    expect(text.locator("[data-line='1']")).to_be_visible()
    page.goto(f"http://127.0.0.1:{big_tree}/#/files/no-such-revision")
    expect(tree.locator(".err")).to_contain_text("unknown revision")
    expect(page.get_by_role("button", name="Try again", exact=True)).to_have_count(0)
    assert [e for e in errors if "500" not in e and "422" not in e] == []  # the failed reads


def test_the_pane_says_loading_and_failure_in_one_style(
    open_page: Callable[[int], tuple[Page, list[str]]], big_tree: int
) -> None:
    """A failed text load was 14px red text flush on the head, unlike every other pane note."""
    from playwright.sync_api import Route, expect

    page, errors = open_page(big_tree)
    held: list[Route] = []

    def hold(route: Route) -> None:
        held.append(route)

    page.route("**/api/git/text*", hold)
    page.goto(f"http://127.0.0.1:{big_tree}/#/files/HEAD/b.txt")
    expect(page.locator(".files-text > .gf-note")).to_have_text("loading…")
    while not held:
        page.wait_for_timeout(50)
    held.pop().fulfill(status=500, json={"error": "nope"})
    expect(page.locator(".files-text > .gf-note.err")).to_contain_text("nope")
    assert [e for e in errors if "500" not in e] == []  # the failed read's response


def test_a_file_open_on_a_phone_keeps_the_focus_out_of_the_tree_under_it(
    open_page: Callable[[int], tuple[Page, list[str]]], big_tree: int
) -> None:
    """On a phone the focus stayed on the tapped row under the file, and Tab walked the tree."""
    page, errors = open_page(big_tree)
    page.set_viewport_size({"width": 390, "height": 844})
    page.goto(f"http://127.0.0.1:{big_tree}/#/files/HEAD")
    page.locator(_row("b.txt")).click()
    page.wait_for_selector(".files-view [data-line='1']")
    assert page.evaluate("document.activeElement.classList.contains('files-back')")
    covered = "!!document.activeElement.closest('.files-tree, .files-bar')"
    for key in ("Shift+Tab", "Tab", "Tab", "Tab"):
        page.keyboard.press(key)
        assert not page.evaluate(covered), key
    page.locator(".files-back").click()
    assert page.evaluate("document.activeElement.dataset.path") == "b.txt"
    assert not errors


def test_a_long_file_path_wraps_after_a_slash(
    open_page: Callable[[int], tuple[Page, list[str]]], git_port: int, tmp_path: pathlib.Path
) -> None:
    """Chromium broke the head's path inside a name, or started a line with its slash."""
    leaf = "alpha_folder/beta_folder/gamma_folder/delta_folder/a_long_file_name.txt"
    (tmp_path / leaf).parent.mkdir(parents=True)
    (tmp_path / leaf).write_text("leaf\n", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-m", "a long path")
    page, errors = open_page(git_port)
    page.set_viewport_size({"width": 390, "height": 844})
    page.goto(f"http://127.0.0.1:{git_port}/#/files/HEAD/{leaf}")
    page.wait_for_selector(".files-view [data-line='1']")
    lines = page.locator(".files-head .gf-path").evaluate("""p => {
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
    assert len(lines) > 1 and "".join(lines) == leaf, lines
    assert all(line.endswith("/") for line in lines[:-1]), lines
    assert not errors


def test_back_between_files_keeps_the_tree_as_it_is(
    open_page: Callable[[int], tuple[Page, list[str]]], big_tree: int
) -> None:
    """Back to a file rebuilt the page: the filter emptied and the tree loaded and painted anew.

    In place, Tab then entered the tree at the file left, not the file shown.
    """
    from playwright.sync_api import expect

    page, errors = open_page(big_tree)
    listings: list[str] = []
    page.on("request", lambda r: listings.append(r.url) if "/api/git/tree" in r.url else None)
    page.goto(f"http://127.0.0.1:{big_tree}/#/files/HEAD")
    search = page.locator(".files-bar input.filter")
    search.fill("f10")
    head = page.locator(".files-head .gf-path")
    for name in ("f100.txt", "f101.txt"):
        page.locator(_row("many/" + name)).click()
        expect(head).to_have_text("many/" + name)
    tree = page.locator(".files-tree .gtree")
    tree.evaluate("e => e.dataset.painted = 'once'")
    for step, name in ((page.go_back, "f100.txt"), (page.go_forward, "f101.txt")):
        step()
        expect(head).to_have_text("many/" + name)
        expect(page.locator(_row("many/" + name))).to_have_attribute("aria-selected", "true")
        search.focus()
        page.keyboard.press("Tab")
        expect(page.locator(_row("many/" + name))).to_be_focused()
    page.go_back()
    page.go_back()  # the bare listing the page opened on
    expect(head).to_have_count(0)
    expect(page.locator(".files-tree [aria-selected=true]")).to_have_count(0)
    search.focus()
    page.keyboard.press("Tab")
    expect(page.locator(_row("many"))).to_be_focused()  # with no file shown, the first row
    expect(search).to_have_value("f10")
    expect(tree).to_have_attribute("data-painted", "once")
    expect(page.locator(".files-app")).not_to_have_attribute("aria-busy", "true")
    assert len(listings) == 1, listings
    assert not errors


def test_a_revision_change_keeps_the_open_file_where_it_is_listed(
    open_page: Callable[[int], tuple[Page, list[str]]], files_port: int
) -> None:
    """Picking another revision closed the open file, though the new revision listed it."""
    from playwright.sync_api import expect

    def pick(label: str) -> None:
        page.locator(".files-bar .gpick input").click()
        page.locator(".gpick-list .ac-item", has_text=label).click()

    base = f"http://127.0.0.1:{files_port}/#/files/"
    page, errors = open_page(files_port)
    page.goto(base + "HEAD/b.txt")
    page.wait_for_selector(".files-view [data-line='3']")
    pick("working tree")
    expect(page).to_have_url(base + "WORKTREE/b.txt")
    expect(page.locator(".files-view [data-line='4']")).to_contain_text("four")
    expect(page.locator(_row("b.txt"))).to_have_attribute("aria-selected", "true")
    page.goto(base + "HEAD/dir/f.txt")
    expect(page.locator(".files-head .gf-path")).to_have_text("dir/f.txt")
    pick("working tree")  # the rename to dir/g.txt is staged
    expect(page).to_have_url(base + "WORKTREE")
    expect(page.locator(".files-view .empty")).to_have_text("click a file to read it")
    assert not errors


def test_an_image_states_its_facts_once(
    open_page: Callable[[int], tuple[Page, list[str]]], big_tree: int
) -> None:
    """An image's size showed in the head and again in a caption above the image."""
    from playwright.sync_api import expect

    page, errors = open_page(big_tree)
    page.goto(f"http://127.0.0.1:{big_tree}/#/files/HEAD/logo.png")
    page.wait_for_selector(".files-view img")
    expect(page.locator(".files-head .files-note")).to_have_text("69 B · 1\u00d71")
    expect(page.locator(".files-view figcaption")).to_have_count(0)
    assert not errors


def test_an_svg_states_no_pixel_size_it_does_not_have(
    open_page: Callable[[int], tuple[Page, list[str]]], git_port: int, tmp_path: pathlib.Path
) -> None:
    """A size-less SVG read 150x150 in Chromium and 300x300 in Firefox: the browser's guess."""
    from playwright.sync_api import expect

    svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 96 96"><circle r="9"/></svg>'
    (tmp_path / "mark.svg").write_text(svg, encoding="utf-8")
    _git(tmp_path, "add", "mark.svg")
    _git(tmp_path, "commit", "-m", "a mark")
    page, errors = open_page(git_port)
    page.goto(f"http://127.0.0.1:{git_port}/#/files/HEAD/mark.svg")
    page.wait_for_selector(".files-view img")
    expect(page.locator(".files-head .files-note")).to_have_text(f"{len(svg)} B")
    assert not errors


def test_an_image_the_browser_cannot_show_says_so_in_a_note(
    open_page: Callable[[int], tuple[Page, list[str]]], big_tree: int
) -> None:
    """An image that failed to decode showed a broken image on a checkerboard strip."""
    from playwright.sync_api import expect

    page, errors = open_page(big_tree)
    page.route(
        "**/api/git/blob*",
        lambda route: route.fulfill(status=200, content_type="image/png", body=b"not a png"),
    )
    page.goto(f"http://127.0.0.1:{big_tree}/#/files/HEAD/logo.png")
    text = page.locator(".files-text")
    expect(text.locator(".gf-note")).to_have_text("the browser cannot show this image")
    expect(text.locator("img, figure")).to_have_count(0)
    expect(page.locator(".files-head .files-note")).to_have_text("69 B")
    assert not errors


def test_an_image_says_it_is_loading(
    open_page: Callable[[int], tuple[Page, list[str]]], big_tree: int
) -> None:
    """An image's pane was an empty box until its image arrived."""
    from playwright.sync_api import expect

    page, errors = open_page(big_tree)
    page.add_init_script("""
      const fetchNow = window.fetch;
      window.fetch = (url, init) => String(url).includes('/api/git/blob')
        ? new Promise(r => setTimeout(r, 1500)).then(() => fetchNow(url, init))
        : fetchNow(url, init);
    """)
    page.goto(f"http://127.0.0.1:{big_tree}/#/files/HEAD/logo.png")
    text = page.locator(".files-text")
    expect(text).to_have_text("loading…")
    page.wait_for_selector(".files-view img")
    expect(text).not_to_contain_text("loading…")
    assert not errors


def test_a_symlink_links_its_target_when_the_revision_lists_it(
    open_page: Callable[[int], tuple[Page, list[str]]], git_port: int, tmp_path: pathlib.Path
) -> None:
    """A symlink's target was plain text, though the tree listed the file it names."""
    from playwright.sync_api import expect

    (tmp_path / "dir" / "guide.txt").symlink_to("../b.txt")
    (tmp_path / "outside").symlink_to("../elsewhere.txt")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-m", "two symlinks")
    base = f"http://127.0.0.1:{git_port}/#/files/HEAD/"
    page, errors = open_page(git_port)
    page.goto(base + "outside")
    expect(page.locator(".files-text")).to_have_text("symlink to ../elsewhere.txt")
    expect(page.locator(".files-text a")).to_have_count(0)
    page.goto(base + "dir/guide.txt")
    target = page.get_by_role("link", name="../b.txt", exact=True)
    expect(target).to_have_attribute("href", "#/files/HEAD/b.txt")
    target.click()
    expect(page.locator(".files-head .gf-path")).to_have_text("b.txt")
    assert not errors
