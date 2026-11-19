"""The Sessions and Machines lists driven in a browser: rows, their refresh and the composer."""

from __future__ import annotations

import json
import pathlib
import subprocess
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


def _manifest(repo: pathlib.Path, sid: str, **fields: object) -> None:
    """Write a run's manifest with `fields`."""
    run = paths.state_dir(repo) / "sessions" / "runs" / sid
    run.mkdir(parents=True, exist_ok=True)
    manifest = {"version": 3, "session_id": sid, "mode": "run", **fields}
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


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


def test_a_rows_status_leads_its_sub_line_and_its_text_keeps_the_width(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, tmp_path: pathlib.Path
) -> None:
    """A row's status and buttons sat beside its text, so a phone left the text a sliver.

    The dots stood wherever each label's length put them, and a squeezed sub line broke inside
    an id. The status now leads the first sub line and the buttons wrap under the text.
    """
    from playwright.sync_api import expect

    def git(*args: str) -> None:
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)

    git("init", "-q", "-b", "master")
    git("config", "user.name", "t")
    git("config", "user.email", "t@example.invalid")
    git("commit", "-q", "--allow-empty", "-m", "base")
    git("branch", "agent6/review-BBBBBB")
    reason = "the checkout has uncommitted changes"
    _manifest(tmp_path, "parked-AAAAAA", parked_task="Bump textual", parked_reason=reason)
    _manifest(tmp_path, "review-BBBBBB", run_branch="agent6/review-BBBBBB")
    _run(tmp_path, "review-BBBBBB", "Review me", STOP)
    _run(tmp_path, "short-CCCCCC", "Short", STOP)
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/")
    rows = page.locator(".item")
    expect(rows).to_have_count(3)
    expect(rows.filter(has_text="parked-AAAAAA")).to_contain_text(reason)
    expect(rows.get_by_role("button", name="Review")).to_have_count(1)
    lefts = "ps => ps.map(p => Math.round(p.getBoundingClientRect().left))"
    assert len(set(page.locator(".item .pill").evaluate_all(lefts))) == 1
    page.set_viewport_size({"width": 390, "height": 844})
    text = "rows => rows.map(r => r.querySelector('.grow').getBoundingClientRect().width)"
    assert min(rows.evaluate_all(text)) >= 300
    # Each id sits on one line: the range over it draws a single box.
    whole = """rows => rows.map(r => {
      const sub = r.querySelector('.sub'), id = r.querySelector('a.title').hash.split('/').pop();
      const walk = document.createTreeWalker(sub, NodeFilter.SHOW_TEXT);
      for (let n = walk.nextNode(); n; n = walk.nextNode()) {
        const at = n.data.indexOf(id);
        if (at < 0) continue;
        const range = document.createRange();
        range.setStart(n, at); range.setEnd(n, at + id.length);
        return [...range.getClientRects()].filter(b => b.width).length;
      }
      return 0;
    })"""
    assert rows.evaluate_all(whole) == [1, 1, 1]
    assert not errors


def test_space_opens_a_focused_row_as_enter_does(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, tmp_path: pathlib.Path
) -> None:
    """Space on a row's title link scrolled the page, though `?` lists Space as opening a row."""
    from playwright.sync_api import expect

    _run(tmp_path, "alpha-AAAAAA", "Alpha task", STOP)
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/")
    page.locator(".item a.title").first.focus()
    page.keyboard.press(" ")
    expect(page).to_have_url(f"http://127.0.0.1:{port}/#/session/alpha-AAAAAA")
    assert not errors


def test_a_repeated_start_or_prune_sends_one_request(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """Enter twice while a Start was in flight started two paid sessions; Prune sent twice too."""
    from playwright.sync_api import Route, expect

    page, errors = open_page(port)
    held: list[Route] = []

    def hold(route: Route) -> None:
        held.append(route)

    for url in ("**/api/new", "**/api/sessions/prune"):
        page.route(url, hold)
    page.goto(f"http://127.0.0.1:{port}/#/")
    task = page.locator(".dock textarea")
    task.fill("probe")
    task.press("Enter")
    task.press("Enter")
    page.get_by_role("button", name="Prune merged runs", exact=True).dblclick()
    page.wait_for_timeout(500)
    assert [r.request.url.rsplit("/", 1)[1] for r in held] == ["new", "prune"]
    for route in held:
        route.fulfill(json={"ok": False, "error": "held"})
    expect(page.locator(".toast")).to_have_count(2)
    expect(page.get_by_role("button", name="Start", exact=True)).to_be_enabled()
    assert not errors


def test_the_folds_say_whether_they_are_open_and_keep_the_focus(
    open_page: Callable[[int], tuple[Page, list[str]]], fan_port: int
) -> None:
    """The children toggle was a 40px button with no open state, and Enter on it lost the focus.

    The advanced fold was an 18px line on the entry's top edge, and its spend cap stood taller
    than the plan dropdown beside it.
    """
    from playwright.sync_api import expect

    page, errors = open_page(fan_port)
    page.goto(f"http://127.0.0.1:{fan_port}/#/")
    height = "e => e.getBoundingClientRect().height"
    show = page.get_by_role("button", name="children: 2 · show")
    expect(show).to_have_attribute("aria-expanded", "false")
    show.focus()
    page.keyboard.press("Enter")
    hide = page.get_by_role("button", name="children: 2 · hide")
    expect(hide).to_be_focused()
    expect(hide).to_have_attribute("aria-expanded", "true")
    assert 24 <= hide.evaluate(height) < 32
    advanced = page.locator(".dock .pickers").get_by_role("button", name="advanced", exact=True)
    expect(advanced).to_have_attribute("aria-expanded", "false")
    advanced.click()
    expect(advanced).to_have_attribute("aria-expanded", "true")
    fields = page.locator(".dock select[title^='seed the run'], .dock input[type=number]")
    expect(fields).to_have_count(2)
    boxes = fields.evaluate_all("es => es.map(e => [e.getBoundingClientRect().y, e.offsetHeight])")
    assert boxes[0] == boxes[1], boxes
    assert not errors


def test_a_disabled_picker_reads_disabled_with_its_label(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """A plan greys worktree and from plan: Firefox drew both, and both labels, as enabled."""
    from playwright.sync_api import expect

    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/")
    page.locator(".dock select").first.select_option("plan")
    worktree = page.locator(".dock input[type=checkbox]")
    expect(worktree).to_be_disabled()
    opacity = "es => es.map(e => +getComputedStyle(e).opacity)"
    labels = page.locator(".dock .pick", has=page.locator(":disabled")).locator("> span")
    expect(labels).to_have_text(["worktree", "from plan"])
    assert max(worktree.evaluate_all(opacity) + labels.evaluate_all(opacity)) < 1
    assert not errors


def test_a_suggestion_popup_opens_on_its_entry_at_the_entrys_width(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """The directive popup opened 10px left of the entry and over the pickers, cut at 480px."""
    from playwright.sync_api import expect

    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/")
    entry = page.locator(".dock textarea")
    entry.fill("/")
    popup = page.locator(".ac-pop")
    expect(popup).to_contain_text("/parallel")
    above, under = popup.bounding_box(), entry.bounding_box()
    assert above and under
    assert above["x"] == under["x"], (above, under)
    assert 0 <= under["y"] - (above["y"] + above["height"]) <= 8, (above, under)
    whole = "rows => rows.every(r => r.scrollWidth <= r.clientWidth)"
    assert popup.locator(".ac-item").evaluate_all(whole)
    assert not errors
