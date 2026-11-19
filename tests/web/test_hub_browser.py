"""The Sessions and Machines lists driven in a browser: rows, their refresh and the composer."""

from __future__ import annotations

import json
import pathlib
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING

import pytest

from agent6 import paths
from tests.unit import test_sessions_show_group
from tests.web import test_web_browser

if TYPE_CHECKING:
    from playwright.sync_api import Page

pytestmark = pytest.mark.browser
browser = test_web_browser.browser
open_page = test_web_browser.open_page
port = test_web_browser.port
group = test_sessions_show_group.group

STOP: dict[str, object] = {
    "type": "session.stop",
    "reason": "finish_session",
    "verification": "passed",
}
# Keyboard focus on a row: its title link, or the row where the row is the control.
FOCUS_ROW = "e => (e.querySelector('a[href]') || e).focus()"


def _run(repo: pathlib.Path, sid: str, task: str, *events: dict[str, object]) -> pathlib.Path:
    """Record a run that started on `task`, then `events`; return its journal."""
    run = paths.state_dir(repo) / "sessions" / "runs" / sid
    run.mkdir(parents=True, exist_ok=True)
    log = run / "logs.jsonl"
    start = {"type": "session.start", "mode": "run", "user_task": task}
    log.write_text("".join(json.dumps(e) + "\n" for e in (start, *events)), encoding="utf-8")
    return log


def _refresh(page: Page) -> None:
    """Run the hub's list refresh now, as a tab coming back into view does."""
    with page.expect_response(lambda r: r.url.endswith("/api/hub")):
        page.evaluate("document.dispatchEvent(new Event('visibilitychange'))")
    page.wait_for_timeout(300)


def test_a_refresh_keeps_the_focus_a_selection_and_a_press_in_the_list(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, tmp_path: pathlib.Path
) -> None:
    """The 4 s refresh rebuilt every row: the focus fell to the page, a selection and a click died.

    A row whose data changed is painted afresh and keeps the focus on the same control.
    """
    from playwright.sync_api import expect

    _run(tmp_path, "alpha-AAAAAA", "Alpha task", STOP)
    beta = _run(tmp_path, "beta-BBBBBB", "Beta task")
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/")
    rows = page.locator(".item")
    alpha, other = rows.filter(has_text="alpha-AAAAAA"), rows.filter(has_text="beta-BBBBBB")
    expect(other).to_contain_text("Beta task")
    focused = "() => document.activeElement.closest('.item')?.textContent || ''"
    alpha.evaluate(FOCUS_ROW)
    beta.write_text(beta.read_text() + json.dumps(STOP) + "\n", encoding="utf-8")
    _refresh(page)
    expect(other).to_contain_text("passed")
    assert "alpha-AAAAAA" in page.evaluate(focused)
    # The focused row itself changes: its fresh paint takes the focus.
    other.evaluate(FOCUS_ROW)
    spent = {"type": "budget.update", "usd_total": 1.5}
    beta.write_text(beta.read_text() + json.dumps(spent) + "\n", encoding="utf-8")
    _refresh(page)
    expect(other).to_contain_text("$1.50")
    assert "beta-BBBBBB" in page.evaluate(focused)
    # A selection waits out the refresh.
    page.evaluate("""() => { const r = document.createRange();
      r.selectNodeContents([...document.querySelectorAll('.item .sub')].at(-1));
      getSelection().removeAllRanges(); getSelection().addRange(r); }""")
    selected = page.evaluate("String(getSelection())")
    _run(tmp_path, "gamma-CCCCCC", "Gamma task")
    _refresh(page)
    assert page.evaluate("String(getSelection())") == selected
    # A press held across a refresh still opens its row.
    box = alpha.bounding_box()
    assert box
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    page.mouse.down()
    _refresh(page)
    page.mouse.up()
    expect(page).to_have_url(f"http://127.0.0.1:{port}/#/session/alpha-AAAAAA")
    assert not errors


@pytest.fixture
def fan_port(group: pathlib.Path) -> Iterator[int]:
    """A server over a fan-out group, `fan`, with two children."""
    yield from test_web_browser._serve(group)  # pyright: ignore[reportPrivateUsage]


def test_a_drag_that_selects_text_in_a_row_keeps_the_page(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, tmp_path: pathlib.Path
) -> None:
    """A drag that selected a row's id opened the session, so no id in the list could be copied."""
    from playwright.sync_api import expect

    _run(tmp_path, "alpha-AAAAAA", "Alpha task", STOP)
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/")
    sub = page.locator(".item .sub").first
    expect(sub).to_contain_text("alpha-AAAAAA")
    box = sub.bounding_box()
    assert box
    page.mouse.move(box["x"] + 2, box["y"] + box["height"] / 2)
    page.mouse.down()
    page.mouse.move(box["x"] + 160, box["y"] + box["height"] / 2, steps=8)
    page.mouse.up()
    page.wait_for_timeout(300)
    assert page.evaluate("String(getSelection())")
    expect(page).to_have_url(f"http://127.0.0.1:{port}/#/")
    page.mouse.click(box["x"] + 40, box["y"] + box["height"] / 2)  # a plain click still opens it
    expect(page).to_have_url(f"http://127.0.0.1:{port}/#/session/alpha-AAAAAA")
    assert not errors


def test_a_groups_children_are_rows_of_their_own(
    open_page: Callable[[int], tuple[Page, list[str]]], fan_port: int
) -> None:
    """Expanded children sat inside the group's text column, nested in its row.

    The group's status floated to the middle of the group, hovering a child lit the group, and
    Enter on a child opened the group. Each row is now a title link beside its controls, never a
    button holding buttons.
    """
    from playwright.sync_api import expect

    page, errors = open_page(fan_port)
    page.goto(f"http://127.0.0.1:{fan_port}/#/")
    parent = page.locator(".item").filter(has_text="fan · ")
    expect(parent).to_have_count(1)
    expect(page.locator(".list [role=button]")).to_have_count(0)
    expect(parent.get_by_role("link")).to_have_attribute("href", "#/session/fan")
    parent.get_by_role("button", name="children: 2 · show").click()
    child = page.locator(".item").filter(has_text="fan-l1")
    expect(child).to_be_visible()
    above, under = parent.bounding_box(), child.bounding_box()
    assert above and under
    assert above["y"] + above["height"] <= under["y"], (above, under)
    child.hover()
    lit = "e => getComputedStyle(e).backgroundColor"
    assert parent.evaluate(lit) != child.evaluate(lit)
    child.get_by_role("link").focus()
    page.keyboard.press("Enter")
    expect(page).to_have_url(f"http://127.0.0.1:{fan_port}/#/session/fan-l1")
    assert not errors
