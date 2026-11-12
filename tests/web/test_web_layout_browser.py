from __future__ import annotations

import json
import pathlib
from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest

from agent6 import paths
from tests.web import signin, test_web_browser

if TYPE_CHECKING:
    from playwright.sync_api import Browser, Locator, Page

pytestmark = pytest.mark.browser
browser = test_web_browser.browser
open_page = test_web_browser.open_page
git_port = test_web_browser.git_port
port = test_web_browser.port


def _width(element: Locator) -> float:
    return float(element.evaluate("e => e.getBoundingClientRect().width"))


def _drag(page: Page, handle: Locator, distance: int) -> None:
    box = handle.bounding_box()
    assert box is not None
    x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    page.mouse.move(x, y)
    page.mouse.down()
    page.mouse.move(x + distance, y, steps=5)
    page.mouse.up()


@pytest.mark.parametrize("width", [1280, 390])
@pytest.mark.parametrize("theme", ["dark", "light"])
def test_navigation_icons_follow_bar_order_size_and_theme(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, width: int, theme: str
) -> None:
    """Desktop and phone navigation keep readable labels and inherit each theme's colour."""
    from playwright.sync_api import expect

    page, errors = open_page(port)
    page.add_init_script(f"localStorage.setItem('a6-theme', '{theme}')")
    page.set_viewport_size({"width": width, "height": 900})
    page.goto(f"http://127.0.0.1:{port}/")
    bar = page.locator(".rail-nav" if width == 1280 else "nav.tabs")
    expect(bar).to_be_visible()
    expect(bar.locator("a")).to_have_text(["Sessions", "Git", "Files", "State machines", "Config"])
    for link in bar.locator("a").all():
        svg = link.locator("svg")
        expect(svg).to_be_visible()
        assert _width(svg) == 20
        assert svg.evaluate("e => getComputedStyle(e).stroke") == link.evaluate(
            "e => getComputedStyle(e).color"
        )
    assert not errors


def test_files_tree_width_survives_reload_and_keyboard_resize(
    open_page: Callable[[int], tuple[Page, list[str]]], git_port: int
) -> None:
    """Dragging changes the split, and a reload retains drag and arrow-key widths."""
    from playwright.sync_api import expect

    page, errors = open_page(git_port)
    page.goto(f"http://127.0.0.1:{git_port}/#/files/HEAD/b.txt")
    tree = page.locator(".files-tree")
    expect(tree.locator("[data-path='b.txt']")).to_be_visible()
    before = _width(tree)
    handle = page.get_by_role("separator", name="File tree width")
    expect(handle).to_be_visible()
    _drag(page, handle, 120)
    assert _width(tree) == pytest.approx(before + 120, abs=1)
    page.reload()
    expect(tree.locator("[data-path='b.txt']")).to_be_visible()
    assert _width(tree) == pytest.approx(before + 120, abs=1)
    handle.focus()
    handle.press("ArrowLeft")
    assert _width(tree) == pytest.approx(before + 100, abs=1)
    handle.press("ArrowRight")
    assert _width(tree) == pytest.approx(before + 120, abs=1)
    expect(handle).to_have_attribute("aria-valuenow", str(round(_width(tree))))
    page.reload()
    expect(tree.locator("[data-path='b.txt']")).to_be_visible()
    assert _width(tree) == pytest.approx(before + 120, abs=1)
    assert not errors


def test_files_collapse_is_remembered_but_does_not_hide_the_phone_tree(
    open_page: Callable[[int], tuple[Page, list[str]]], git_port: int
) -> None:
    """The desktop toggle frees the file pane without changing the phone's file overlay."""
    from playwright.sync_api import expect

    page, errors = open_page(git_port)
    page.goto(f"http://127.0.0.1:{git_port}/#/files/HEAD/b.txt")
    tree = page.locator(".files-tree")
    pane = page.locator(".files-view")
    expect(pane.locator(".sl").first).to_be_visible()
    before = _width(tree)
    hide = page.get_by_role("button", name="Hide tree", exact=True)
    expect(hide).to_be_visible()
    hide.press("Enter")
    expect(tree).to_be_hidden()
    assert _width(pane) == pytest.approx(_width(page.locator(".files-body")), abs=1)
    page.reload()
    expect(pane.locator(".sl").first).to_be_visible()
    expect(tree).to_be_hidden()
    show = page.get_by_role("button", name="Show tree", exact=True)
    expect(show).to_have_attribute("aria-expanded", "false")
    show.press("Space")
    expect(tree).to_be_visible()
    assert _width(tree) == pytest.approx(before, abs=1)
    hide.click()
    page.set_viewport_size({"width": 390, "height": 844})
    expect(show).to_be_hidden()
    expect(page.get_by_role("separator", name="File tree width")).to_be_hidden()
    assert _width(pane) == pytest.approx(390, abs=1)
    pane.get_by_role("button", name="\u2039 files", exact=True).click()
    expect(pane).to_be_hidden()
    expect(tree.locator("[data-path='b.txt']")).to_be_visible()
    tree.locator("[data-path='b.txt']").click()
    expect(pane.locator(".sl").first).to_be_visible()
    page.set_viewport_size({"width": 1280, "height": 900})
    expect(tree).to_be_hidden()
    expect(show).to_be_visible()
    assert not errors


@pytest.mark.parametrize("saved", ["100000px", "-20px", "not a width"])
def test_files_restored_width_fits_the_current_container(
    open_page: Callable[[int], tuple[Page, list[str]]], git_port: int, saved: str
) -> None:
    """An old or malformed width cannot squeeze the file out of a narrower container."""
    from playwright.sync_api import expect

    page, errors = open_page(git_port)
    page.add_init_script(f"localStorage.setItem('a6-files-tree-w', {json.dumps(saved)})")
    page.set_viewport_size({"width": 900, "height": 900})
    page.goto(f"http://127.0.0.1:{git_port}/#/files/HEAD/b.txt")
    expect(page.get_by_role("separator", name="File tree width")).to_be_visible()
    tree, pane = page.locator(".files-tree"), page.locator(".files-view")
    assert _width(tree) >= 180
    assert _width(pane) >= 240
    page.set_viewport_size({"width": 800, "height": 900})
    assert _width(tree) >= 180
    assert _width(pane) >= 240
    assert _width(tree) + _width(pane) < _width(page.locator(".files-body"))
    assert not errors


def test_files_layout_works_when_browser_storage_is_blocked(
    browser: Browser, git_port: int
) -> None:
    """Unavailable storage must not prevent sign-in, resizing or collapsing the Files tree."""
    from playwright.sync_api import expect

    context = browser.new_context(viewport={"width": 1280, "height": 900})
    context.add_init_script("""
Object.defineProperty(window, 'localStorage', {
  get() { throw new DOMException('storage blocked', 'SecurityError'); }
});
""")
    try:
        page = context.new_page()
        errors: list[str] = []
        page.on("pageerror", lambda exc: errors.append(str(exc)))
        page.goto(f"http://127.0.0.1:{git_port}/#/files/HEAD/b.txt")
        page.locator("#login input").fill(signin.SECRET)
        page.get_by_role("button", name="Sign in", exact=True).click()
        tree = page.locator(".files-tree")
        expect(tree.locator("[data-path='b.txt']")).to_be_visible()
        before = _width(tree)
        _drag(page, page.get_by_role("separator", name="File tree width"), 80)
        assert _width(tree) == pytest.approx(before + 80, abs=1)
        page.get_by_role("button", name="Hide tree", exact=True).click()
        expect(tree).to_be_hidden()
        page.get_by_role("button", name="Show tree", exact=True).click()
        expect(tree).to_be_visible()
        assert not errors
    finally:
        context.close()


def test_run_drawer_keeps_drag_resize_reload_and_reset(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, tmp_path: pathlib.Path
) -> None:
    """Sharing the divider retains the drawer's saved width and double-click reset."""
    from playwright.sync_api import expect

    run = paths.state_dir(tmp_path) / "sessions" / "runs" / "layout-run"
    run.mkdir(parents=True)
    (run / "logs.jsonl").write_text(
        json.dumps({"type": "session.start", "mode": "run", "user_task": "Check the drawer"})
        + "\n",
        encoding="utf-8",
    )
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/session/layout-run")
    drawer = page.locator(".drawer")
    expect(drawer).to_be_visible()
    before = _width(drawer)
    handle = page.get_by_role("separator", name="Details width")
    expect(handle).to_be_visible()
    _drag(page, handle, 100)
    assert _width(drawer) == pytest.approx(before + 100, abs=1)
    page.reload()
    expect(drawer).to_be_visible()
    assert _width(drawer) == pytest.approx(before + 100, abs=1)
    handle.press("ArrowLeft")
    assert _width(drawer) == pytest.approx(before + 80, abs=1)
    handle.dblclick()
    assert _width(drawer) == pytest.approx(before, abs=1)
    page.reload()
    expect(drawer).to_be_visible()
    assert _width(drawer) == pytest.approx(before, abs=1)
    assert not errors
