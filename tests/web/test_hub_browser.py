"""The Sessions and Machines lists driven in a browser: rows, their refresh and the composer."""

from __future__ import annotations

import json
import pathlib
import re
import subprocess
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING

import pytest

from agent6 import paths
from tests import review_helpers
from tests.unit import test_sessions_show_group
from tests.web import test_web_actions, test_web_browser

if TYPE_CHECKING:
    from playwright.sync_api import Locator, Page

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


def _run(
    repo: pathlib.Path, sid: str, task: str, *events: dict[str, object], mode: str = "run"
) -> pathlib.Path:
    """Record a session in `mode` that started on `task`, then `events`; return its journal."""
    run = paths.state_dir(repo) / "sessions" / f"{mode}s" / sid
    run.mkdir(parents=True, exist_ok=True)
    log = run / "logs.jsonl"
    start = {"type": "session.start", "mode": mode, "user_task": task}
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
    """A drag across a row's id or title opened the row, so no id, task or name could be copied."""
    from playwright.sync_api import expect

    def drag(text: Locator) -> str:
        """Drag across the start of `text`; return the selection."""
        page.evaluate("getSelection().removeAllRanges()")
        box = text.bounding_box()
        assert box
        page.mouse.move(box["x"] + 2, box["y"] + box["height"] / 2)
        page.mouse.down()
        page.mouse.move(box["x"] + 160, box["y"] + box["height"] / 2, steps=8)
        page.mouse.up()
        page.wait_for_timeout(300)
        return page.evaluate("String(getSelection())")

    _run(tmp_path, "alpha-AAAAAA", "Alpha task", STOP)
    test_web_actions._ended_machine(tmp_path, "tiny")  # pyright: ignore[reportPrivateUsage]
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/machines")
    assert drag(page.locator(".item a.title")) == "tiny"
    expect(page).to_have_url(f"http://127.0.0.1:{port}/#/machines")
    page.goto(f"http://127.0.0.1:{port}/#/")
    sub = page.locator(".item .sub").first
    expect(sub).to_contain_text("alpha-AAAAAA")
    assert drag(sub)
    assert drag(page.locator(".item a.title")).startswith("Alpha")
    expect(page).to_have_url(f"http://127.0.0.1:{port}/#/")
    box = sub.bounding_box()
    assert box
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


def test_a_list_and_its_rows_read_as_a_list(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, tmp_path: pathlib.Path
) -> None:
    """A list's rows were bare boxes: a screen reader heard no list, no count and no rows."""
    from playwright.sync_api import expect

    _run(tmp_path, "alpha-AAAAAA", "Alpha task", STOP)
    _run(tmp_path, "beta-BBBBBB", "Beta task", STOP)
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/")
    expect(page.get_by_role("list").get_by_role("listitem")).to_have_count(2)
    page.goto(f"http://127.0.0.1:{port}/#/machines")
    expect(page.get_by_role("list").get_by_role("listitem")).to_have_count(1)  # the empty line
    assert not errors


def test_a_list_is_one_tab_stop_whose_arrows_move_between_its_rows(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, tmp_path: pathlib.Path
) -> None:
    """The Sessions list took none of the keys the keyboard list promises: ↑ ↓, Home and End.

    Every row and each of its buttons was a Tab stop. The list is one Tab stop, the focused row's
    title and buttons, through a refresh that repaints a row.
    """
    from playwright.sync_api import expect

    for args in (
        ("init", "-q", "-b", "master"),
        ("config", "user.name", "t"),
        ("config", "user.email", "t@example.invalid"),
        ("commit", "-q", "--allow-empty", "-m", "base"),
        ("branch", "agent6/review-BBBBBB"),
    ):
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True)
    _manifest(tmp_path, "review-BBBBBB", run_branch="agent6/review-BBBBBB")
    _run(tmp_path, "review-BBBBBB", "Review me", STOP)
    _run(tmp_path, "alpha-AAAAAA", "Alpha task", STOP)
    gamma = _run(tmp_path, "gamma-CCCCCC", "Gamma task")
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/")
    expect(page.get_by_role("list").get_by_role("listitem")).to_have_count(3)
    titles = page.locator(".item a.title")
    order = titles.all_text_contents()
    titles.first.focus()
    for key, at in (("ArrowDown", 1), ("End", 2), ("Home", 0), ("ArrowUp", 0)):
        page.keyboard.press(key)
        expect(titles.nth(at)).to_be_focused()
    for _ in range(order.index("Review me")):
        page.keyboard.press("ArrowDown")
    buttons = page.locator(".item", has_text="Review me").get_by_role("button")
    stop = ["Review me", *buttons.all_text_contents()]
    stops = """() => [...document.querySelectorAll('.list a, .list button')]
      .filter(c => c.tabIndex >= 0).map(c => c.textContent)"""
    assert page.evaluate(stops) == stop
    gamma.write_text(gamma.read_text() + json.dumps(STOP) + "\n", encoding="utf-8")
    _refresh(page)
    expect(page.locator(".item", has_text="gamma-CCCCCC")).to_contain_text("passed")
    assert page.evaluate(stops) == stop
    for at in range(buttons.count()):
        page.keyboard.press("Tab")
        expect(buttons.nth(at)).to_be_focused()
    page.keyboard.press("Tab")
    assert page.evaluate("!document.activeElement.closest('.list')")
    assert not errors


def test_a_pressed_button_keeps_the_focus_through_the_repaint_it_starts(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """Prune and a row's Merge repainted the page and left the focus on the page body."""
    from playwright.sync_api import expect

    row = {
        "session_id": "run-A",
        "mode": "run",
        "status": "passed",
        "status_label": "passed",
        "status_level": "ok",
        "task": "Merge me",
        "branch": "agent6/run-A",
        "unmerged": True,
    }
    page, errors = open_page(port)
    page.route("**/api/hub", lambda route: route.fulfill(json={"sessions": [row]}))
    page.route("**/api/sessions/prune", lambda route: route.fulfill(json={"ok": True}))
    page.route("**/api/session/run-A/merge", lambda route: route.fulfill(json={"ok": True}))
    page.goto(f"http://127.0.0.1:{port}/#/")
    expect(page.get_by_role("button", name="Merge", exact=True)).to_be_visible()
    page.evaluate("clearInterval(hubTimer)")  # only the action may repaint the page
    for name in ("Prune merged runs", "Merge"):
        button = page.get_by_role("button", name=name, exact=True)
        pressed = button.element_handle()
        button.focus()
        page.keyboard.press("Enter")
        if name == "Merge":
            question = page.get_by_role("alertdialog", name="Merge run run-A?")
            expect(question.get_by_role("button", name="Merge")).to_be_focused()
            page.keyboard.press("Enter")
        page.wait_for_function("b => !b.isConnected", arg=pressed)  # the page repainted
        expect(button).to_be_focused()
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


def test_a_phones_pickers_fold_behind_one_line_that_names_their_picks(
    open_page: Callable[[int], tuple[Page, list[str]]],
    port: int,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On a phone the new-work pickers fold behind one line: the mode, preset and model it runs.

    The rows took four of the dock's lines over the entry, and the model was cut mid-word. A tap
    opens them and a second closes them; each run option set joins the line, which wraps rather
    than cutting one, and a desktop keeps the rows.
    """
    from playwright.sync_api import expect

    _run(tmp_path, "plan-AAAAAA", "Plan it", STOP, mode="plan")
    config = tmp_path / "config" / "agent6"
    config.mkdir(parents=True)
    (config / "config.toml").write_text(
        '[providers.o]\nbackend = "openai"\nbase_url = "https://x/v1"\n'
        '[models.worker]\nmodel = "o/vendor/configured"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config.parent))
    page, errors = open_page(port)
    page.set_viewport_size({"width": 390, "height": 844})
    page.goto(f"http://127.0.0.1:{port}/#/")
    summary = page.locator(".dock").get_by_role("button", name=re.compile("^run · "))
    pickers = page.locator(".dock .pickers").first
    expect(summary).to_have_text("run · preset: none · configured")
    expect(summary).to_have_attribute("aria-expanded", "false")
    expect(pickers).to_be_hidden()
    summary.click()
    expect(summary).to_have_attribute("aria-expanded", "true")
    expect(pickers).to_be_visible()
    page.locator(".dock input[type=checkbox]").check()
    expect(summary).to_have_text("run · preset: none · configured · worktree")
    page.locator(".dock .pickers").get_by_role("button", name="advanced", exact=True).click()
    page.locator(".dock select[title^='seed the run']").select_option("plan-AAAAAA")
    page.locator(".dock input[type=number]").fill("2.5")
    expect(summary).to_have_text(
        "run · preset: none · configured · worktree · from plan: plan-AAAAAA · max USD: 2.5"
    )
    summary.click()
    expect(pickers).to_be_hidden()
    assert summary.evaluate("e => e.firstChild.scrollWidth <= e.firstChild.clientWidth")
    page.set_viewport_size({"width": 1280, "height": 900})
    expect(summary).to_be_hidden()
    expect(pickers).to_be_visible()
    assert not errors


def test_a_phones_summary_line_says_when_no_model_resolves(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """With no model configured, a phone's folded line read "run · preset: none · none"."""
    from playwright.sync_api import expect

    page, errors = open_page(port)
    page.set_viewport_size({"width": 390, "height": 844})
    page.goto(f"http://127.0.0.1:{port}/#/")
    expect(page.locator(".dock .picker-summary")).to_have_text("run · preset: none · no model")
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


def test_a_failed_first_load_offers_retry_and_recovers_on_its_own(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, tmp_path: pathlib.Path
) -> None:
    """A failed first load showed one red line and stayed so after the server answered again.

    The card says how often the page retries on its own.
    """
    from playwright.sync_api import Route, expect

    _run(tmp_path, "alpha-AAAAAA", "Alpha task", STOP)
    page, errors = open_page(port)

    def fail(route: Route) -> None:
        route.fulfill(status=500, json={"error": "hub listing failed: probe"})

    page.route("**/api/hub", fail)
    page.goto(f"http://127.0.0.1:{port}/#/")
    expect(page.locator("main")).to_contain_text("hub listing failed: probe")
    expect(page.locator(".problem")).to_contain_text("retrying every 4 s")
    expect(page.get_by_role("button", name="Try again")).to_be_visible()
    page.unroute("**/api/hub", fail)
    expect(page.locator(".item")).to_contain_text("Alpha task", timeout=10000)
    errors[:] = [e for e in errors if "500" not in e]  # the failed loads the test answered
    assert not errors


@pytest.mark.parametrize(("width", "height"), [(1280, 900), (390, 844)])
def test_the_hubs_text_starts_at_one_left_edge(
    open_page: Callable[[int], tuple[Page, list[str]]],
    port: int,
    tmp_path: pathlib.Path,
    width: int,
    height: int,
) -> None:
    """The toolbar's text, the rows' text and the composer started at three left edges."""
    from playwright.sync_api import expect

    _run(tmp_path, "alpha-AAAAAA", "Alpha task", STOP)
    page, errors = open_page(port)
    page.set_viewport_size({"width": width, "height": height})
    page.goto(f"http://127.0.0.1:{port}/#/")
    expect(page.locator(".item")).to_have_count(1)
    left = """sel => {
      const e = document.querySelector(sel), range = document.createRange();
      const text = [...e.childNodes].find(n => n.nodeType === 3 && n.data.trim());
      if (!text) return Math.round(e.getBoundingClientRect().left);
      range.selectNodeContents(text);
      return Math.round(range.getBoundingClientRect().left);
    }"""
    pickers = ".dock .pick > span" if width > 780 else ".dock .picker-summary"  # a phone folds them
    places = [".toolbar button", ".item a.title", pickers, ".dock textarea"]
    assert len({page.evaluate(left, sel) for sel in places}) == 1
    assert not errors


def test_a_selection_holds_back_only_its_own_row(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, tmp_path: pathlib.Path
) -> None:
    """A text selection anywhere in the list held back the whole refresh, so other rows went stale.

    The row holding the selection keeps its paint until the selection goes.
    """
    from playwright.sync_api import expect

    alpha = _run(tmp_path, "alpha-AAAAAA", "Alpha task", STOP)
    beta = _run(tmp_path, "beta-BBBBBB", "Beta task", STOP)
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/")
    rows = page.locator(".item")
    first, second = rows.filter(has_text="alpha-AAAAAA"), rows.filter(has_text="beta-BBBBBB")
    expect(second).to_contain_text("Beta task")
    second.locator(".sub").first.evaluate(
        """e => { const r = document.createRange(); r.selectNodeContents(e);
        getSelection().removeAllRanges(); getSelection().addRange(r); }"""
    )
    for log, usd in ((alpha, 1.5), (beta, 2.5)):
        spent = {"type": "budget.update", "usd_total": usd}
        log.write_text(log.read_text() + json.dumps(spent) + "\n", encoding="utf-8")
    _refresh(page)
    expect(first).to_contain_text("$1.50")
    expect(second).not_to_contain_text("$2.50")
    assert "beta-BBBBBB" in page.evaluate("String(getSelection())")
    page.evaluate("getSelection().removeAllRanges()")
    _refresh(page)
    expect(second).to_contain_text("$2.50")
    assert not errors


def test_a_picker_is_as_wide_as_its_chosen_option(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, tmp_path: pathlib.Path
) -> None:
    """The 'from plan' dropdown took its longest plan's width, 460px, to show 'none'."""
    from playwright.sync_api import expect

    task = "Plan moving the argument parsing onto subcommand modules, one module per verb"
    _run(tmp_path, "plan-AAAAAA", task, STOP, mode="plan")
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/")
    page.locator(".dock .pickers").get_by_role("button", name="advanced", exact=True).click()
    seed = page.locator(".dock select[title^='seed the run']")
    expect(seed.locator("option")).to_have_count(2)
    width = "e => e.getBoundingClientRect().width"
    resting = seed.evaluate(width)
    seed.select_option("plan-AAAAAA")
    assert resting < 100 < seed.evaluate(width), resting
    assert not errors


def test_a_model_list_that_fails_to_load_leaves_the_config_default(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """A failed /api/routes left the model dropdown reading "…", as if it were still resolving."""
    from playwright.sync_api import expect

    page, errors = open_page(port)
    page.route("**/api/routes*", lambda route: route.abort())
    page.goto(f"http://127.0.0.1:{port}/#/")
    model = page.locator(".dock select[title^='the model']")
    expect(model).to_be_enabled()
    expect(model.locator("option")).to_have_text(["config default"])
    expect(page.locator(".toast.bad")).to_contain_text("could not list the models: ")
    errors[:] = [e for e in errors if "ERR_FAILED" not in e]  # the request the test aborted
    assert not errors


def test_the_model_picker_says_it_is_resolving_until_its_routes_arrive(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """The model dropdown painted empty and 40px wide until /api/routes answered."""
    from playwright.sync_api import Route, expect

    page, errors = open_page(port)
    held: list[Route] = []

    def hold(route: Route) -> None:
        held.append(route)

    page.route("**/api/routes*", hold)
    page.goto(f"http://127.0.0.1:{port}/#/")
    model = page.locator(".dock select[title^='the model']")
    expect(model.locator("option")).to_have_text(["…"])
    expect(model).to_be_disabled()
    for route in held:
        route.continue_()
    expect(model).to_be_enabled()
    expect(model.locator("option").first).to_contain_text("(")
    assert not errors


def test_start_and_create_machine_wait_for_something_to_send(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, tmp_path: pathlib.Path
) -> None:
    """Start on an empty entry and Create machine with no description read enabled, did nothing."""
    from playwright.sync_api import expect

    _run(tmp_path, "plan-AAAAAA", "Plan it", STOP, mode="plan")
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/")
    start = page.get_by_role("button", name="Start", exact=True)
    entry = page.locator(".dock textarea")
    expect(start).to_be_disabled()
    entry.fill("probe")
    expect(start).to_be_enabled()
    entry.fill("  ")
    expect(start).to_be_disabled()
    page.locator(".dock .pickers").get_by_role("button", name="advanced", exact=True).click()
    page.locator(".dock select[title^='seed the run']").select_option("plan-AAAAAA")
    expect(start).to_be_enabled()  # a run from a plan needs no task
    page.locator(".dock select").first.select_option("plan")
    expect(start).to_be_disabled()
    page.goto(f"http://127.0.0.1:{port}/#/machines")
    create = page.get_by_role("button", name="Create machine")
    expect(create).to_be_disabled()
    page.locator(".dock textarea").fill("watch CI")
    expect(create).to_be_enabled()
    assert not errors


def test_review_fills_the_entry_on_screen_and_focuses_it(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, tmp_path: pathlib.Path
) -> None:
    """Review fetched and repainted the whole hub to fill the entry, then left the focus nowhere."""
    from playwright.sync_api import Request, expect

    for args in (
        ("init", "-q", "-b", "master"),
        ("config", "user.name", "t"),
        ("config", "user.email", "t@example.invalid"),
        ("commit", "-q", "--allow-empty", "-m", "base"),
        ("branch", "agent6/review-BBBBBB"),
    ):
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True)
    _manifest(tmp_path, "review-BBBBBB", run_branch="agent6/review-BBBBBB")
    _run(tmp_path, "review-BBBBBB", "Review me", STOP)
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/")
    entry = page.locator(".dock textarea")
    entry.fill("a draft")
    page.evaluate("clearInterval(hubTimer)")  # only the click may fetch the hub
    fetched: list[str] = []

    def hub(request: Request) -> None:
        if request.url.endswith("/api/hub"):
            fetched.append(request.url)

    page.on("request", hub)
    page.get_by_role("button", name="Review").click()
    expect(entry).to_have_value("/review agent6/review-BBBBBB")
    expect(entry).to_be_focused()
    assert not fetched
    assert not errors


def test_a_missing_worktree_is_named_once_in_its_row(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, tmp_path: pathlib.Path
) -> None:
    """A run whose worktree is gone printed the worktree's full path twice in its row."""
    from playwright.sync_api import expect

    gone = tmp_path / "worktrees" / "gone-AAAAAA"
    _manifest(tmp_path, "gone-AAAAAA", worktree=str(gone))
    _run(tmp_path, "gone-AAAAAA", "Gone task", STOP)
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/")
    row = page.locator(".item")
    expect(row).to_contain_text("not found; resume needs it")
    assert row.inner_text().count(str(gone)) == 1
    assert not errors


def test_the_parallel_model_popup_is_a_listbox_the_entry_points_into(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """The /parallel model popup had no roles: a screen reader heard no list and no highlight."""
    from playwright.sync_api import Route, expect

    def models(route: Route) -> None:
        route.fulfill(json={"values": ["prov/one", "prov/two"]})

    page, errors = open_page(port)
    page.route("**/api/config/suggest/parallel.models", models)
    page.goto(f"http://127.0.0.1:{port}/#/")
    entry = page.locator(".dock textarea")
    entry.fill("/parallel ")
    popup = page.locator(".ac-pop")
    options = popup.get_by_role("option")
    expect(options).to_have_text(["prov/one", "prov/two"])
    expect(popup).to_have_attribute("role", "listbox")
    expect(entry).to_have_attribute("aria-controls", popup.get_attribute("id") or "")
    expect(entry).to_have_attribute("aria-autocomplete", "list")
    entry.press("ArrowDown")
    expect(options.first).to_have_attribute("aria-selected", "true")
    expect(entry).to_have_attribute(
        "aria-activedescendant", options.first.get_attribute("id") or ""
    )
    entry.press("Escape")
    expect(popup).to_have_count(0)
    assert entry.get_attribute("aria-activedescendant") is None
    assert not errors


def test_the_directive_popup_is_a_listbox_the_entry_points_into(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """The directive popup had no roles: a screen reader heard no list and no highlight.

    The entry points at the /parallel popup that takes over once Tab completes the directive.
    """
    from playwright.sync_api import Route, expect

    def models(route: Route) -> None:
        route.fulfill(json={"values": ["prov/one", "prov/two"]})

    page, errors = open_page(port)
    page.route("**/api/config/suggest/parallel.models", models)
    page.goto(f"http://127.0.0.1:{port}/#/")
    entry = page.locator(".dock textarea")
    popup = page.locator(".ac-pop")
    expect(entry).to_have_attribute("aria-autocomplete", "list")
    # The second pass finds the models fetched: the /parallel popup opens before the other closes.
    for _ in range(2):
        entry.fill("/")
        options = popup.get_by_role("option")
        expect(options).to_have_count(2)
        expect(popup).to_have_attribute("role", "listbox")
        expect(entry).to_have_attribute("aria-controls", popup.get_attribute("id") or "")
        entry.press("ArrowDown")
        expect(options.first).to_have_attribute("aria-selected", "true")
        expect(entry).to_have_attribute(
            "aria-activedescendant", options.first.get_attribute("id") or ""
        )
        entry.press("Tab")
        expect(options).to_have_text(["prov/one", "prov/two"])
        expect(entry).to_have_attribute("aria-controls", popup.get_attribute("id") or "")
        entry.press("Escape")
        expect(popup).to_have_count(0)
        assert entry.get_attribute("aria-controls") is None
    assert not errors


def test_typing_in_the_dock_or_pressing_review_keeps_the_list_still(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, tmp_path: pathlib.Path
) -> None:
    """Chromium scrolled the list on every key typed in the dock, and Review scrolled its row away.

    Chromium scrolls a focused entry out of html's scroll-padding, and the dock's entry sat in it.
    """
    from playwright.sync_api import expect

    for args in (
        ("init", "-q", "-b", "master"),
        ("config", "user.name", "t"),
        ("config", "user.email", "t@example.invalid"),
        ("commit", "-q", "--allow-empty", "-m", "base"),
        ("branch", "agent6/review-BBBBBB"),
    ):
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True)
    for n in range(24):
        _run(tmp_path, f"row{n:02d}-AAAAAA", f"Task {n}", STOP)
    _manifest(tmp_path, "review-BBBBBB", run_branch="agent6/review-BBBBBB")
    _run(tmp_path, "review-BBBBBB", "Review me", STOP)
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/")
    expect(page.locator(".item")).to_have_count(25)
    page.evaluate("clearInterval(hubTimer)")  # a refresh must not repaint mid-check
    # A pointer's click on the middle of a box, with no scroll into view first.
    middle = """e => { const b = e.getBoundingClientRect();
      return [b.x + b.width / 2, b.y + b.height / 2]; }"""
    entry = page.locator(".dock textarea")
    page.mouse.click(*entry.evaluate(middle))
    page.keyboard.type("hello")
    assert page.evaluate("scrollY") == 0
    review = page.get_by_role("button", name="Review")
    review.evaluate("b => scrollBy(0, b.getBoundingClientRect().top - innerHeight / 3)")
    resting = page.evaluate("scrollY")
    page.mouse.click(*review.evaluate(middle))
    expect(entry).to_have_value("/review agent6/review-BBBBBB")
    assert page.evaluate("scrollY") == resting
    assert not errors


def test_a_row_the_keyboard_reaches_scrolls_clear_of_the_dock(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, tmp_path: pathlib.Path
) -> None:
    """Tabbing down a long list left focused rows under the docked composer."""
    from playwright.sync_api import expect

    for n in range(24):
        _run(tmp_path, f"row{n:02d}-AAAAAA", f"Task {n}", STOP)
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/")
    expect(page.locator(".item")).to_have_count(24)
    page.evaluate("clearInterval(hubTimer)")  # a refresh must not repaint mid-walk
    page.locator(".item a.title").first.focus()
    page.keyboard.press("Shift+Tab")
    under = """() => {
      const row = document.activeElement.closest('.item');
      const dock = document.querySelector('.composer.dock').getBoundingClientRect();
      return row ? row.getBoundingClientRect().bottom - dock.top : -1;
    }"""
    overlaps = []
    for _ in range(24):
        page.keyboard.press("Tab")
        overlaps.append(page.evaluate(under))
    assert max(overlaps) <= 0, overlaps
    assert not errors


def test_a_long_machine_file_name_wraps_from_its_buttons_start(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, tmp_path: pathlib.Path
) -> None:
    """A machine file name too long for a phone centred its two lines in its Run button."""
    name = "release-notes-drafter-for-the-quarterly-platform-changelog.asm.toml"
    (tmp_path / "machines").mkdir()
    (tmp_path / "machines" / name).write_text("", encoding="utf-8")
    page, errors = open_page(port)
    page.set_viewport_size({"width": 390, "height": 844})
    page.goto(f"http://127.0.0.1:{port}/#/machines")
    lines = page.get_by_role("button", name=name).evaluate(
        """b => { const r = document.createRange(); r.selectNodeContents(b);
        return [...r.getClientRects()].map(x => Math.round(x.left)); }"""
    )
    assert len(lines) > 1 and len(set(lines)) == 1, lines
    assert not errors


def test_an_instance_row_goes_by_the_instances_name(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, tmp_path: pathlib.Path
) -> None:
    """Instance rows led with their machine's name, so two instances of one machine read alike.

    The name is the instance's identity, so a long one wraps whole on a phone.
    """
    from playwright.sync_api import expect

    long = "weekly-dependency-upgrade-and-security-advisory-sweep"
    for name in ("tiny", "tiny-two", long):
        test_web_actions._ended_machine(tmp_path, name)  # pyright: ignore[reportPrivateUsage]
    page, errors = open_page(port)
    page.set_viewport_size({"width": 390, "height": 844})
    page.goto(f"http://127.0.0.1:{port}/#/machines")
    rows = page.locator(".item")
    expect(rows.locator("a.title")).to_have_text([long, "tiny-two", "tiny"])  # newest first
    expect(rows.nth(1).locator(".sub")).to_contain_text("machine tiny")
    expect(rows.last.locator(".sub")).not_to_contain_text("machine")
    assert rows.first.locator("a.title").evaluate("e => e.scrollWidth <= e.clientWidth")
    assert not errors


def test_an_unreadable_instance_names_no_state(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, tmp_path: pathlib.Path
) -> None:
    """An unreadable instance's row read "at ?" for the state its journal could not tell."""
    from playwright.sync_api import expect

    test_web_actions._ended_machine(tmp_path, "broken")  # pyright: ignore[reportPrivateUsage]
    journal = paths.state_dir(tmp_path) / "machines" / "broken" / "journal.jsonl"
    journal.write_text(journal.read_text(encoding="utf-8") + '{"type": "step"}\n')
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/machines")
    sub = page.locator(".item .sub")
    expect(sub).to_contain_text("unreadable")
    expect(sub).not_to_contain_text(re.compile(r"· at\b"))
    assert not errors


def test_a_row_names_its_mode_once(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, tmp_path: pathlib.Path
) -> None:
    """A plan's row read "plan · stale · plan", and a draft's "machine · stale · draft"."""
    from playwright.sync_api import expect

    _run(tmp_path, "steady-AAAAAA", "Lay it out", mode="plan")
    _run(tmp_path, "brisk-BBBBBB", "Author a watcher", mode="machine")
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/")
    sub = page.locator(".item .sub").first
    expect(sub).to_contain_text("plan · ")
    assert sub.inner_text().count("plan") == 1
    page.goto(f"http://127.0.0.1:{port}/#/machines")
    draft = page.locator(".card", has=page.get_by_role("heading", name="Drafts")).locator(".sub")
    expect(draft).to_contain_text("brisk-BBBBBB")
    expect(draft).not_to_contain_text("machine")
    expect(draft).not_to_contain_text("draft")
    assert not errors


def test_open_findings_read_as_part_of_the_status(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int, tmp_path: pathlib.Path
) -> None:
    """The open findings sat on a muted line of their own, as quiet as the date line."""
    from playwright.sync_api import expect

    findings = review_helpers.panel_rows(block=1, warn=2)
    _run(tmp_path, "blocked-AAAAAA", "Blocked task", *findings, review_helpers.END)
    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/")
    status = page.locator(".item .sub .pill")
    expect(status).to_contain_text("· open findings: 1 block, 2 warn")
    sub = page.locator(".item .sub")
    expect(sub).to_have_count(1)
    color = "e => getComputedStyle(e).color"
    assert status.evaluate(color) != sub.evaluate(color)  # the status's text colour, not muted
    assert not errors


def test_an_empty_list_says_how_to_start_where_a_rows_text_starts(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """An empty list was a bare centred line that never pointed at the composer below it."""
    from playwright.sync_api import expect

    page, errors = open_page(port)
    left = """e => {
      const range = document.createRange();
      range.selectNodeContents([...e.childNodes].find(n => n.nodeType === 3 && n.data.trim()));
      return Math.round(range.getBoundingClientRect().left);
    }"""
    page.goto(f"http://127.0.0.1:{port}/#/")
    empty = page.locator(".list > .empty")
    expect(empty).to_have_text("no sessions yet (describe a task below to start one)")
    assert empty.evaluate(left) == page.locator(".dock .pick > span").first.evaluate(left)
    page.goto(f"http://127.0.0.1:{port}/#/machines")
    expect(empty).to_have_text("no machines yet (describe one below to create it)")
    assert empty.evaluate(left) == page.locator(".card h2").first.evaluate(left)
    assert not errors


def test_the_notifications_button_says_whether_they_are_on(
    open_page: Callable[[int], tuple[Page, list[str]]], port: int
) -> None:
    """The bell read "🔔 Notifications" whether the browser had them on, blocked or never asked.

    A click in either state says where its setting lives. Headless Chromium refuses
    notifications, so a stub stands in for a browser that granted them.
    """
    from playwright.sync_api import expect

    page, errors = open_page(port)
    page.goto(f"http://127.0.0.1:{port}/#/")
    bell = page.locator(".toolbar").get_by_role("button", name="Notifications blocked")
    expect(bell).to_have_attribute(
        "title", "allow notifications for this page in the browser's site settings"
    )
    bell.click()
    expect(page.locator(".toast.bad")).to_contain_text("notifications are blocked")
    page, granted = open_page(port)
    page.add_init_script(
        "Object.defineProperty(Notification, 'permission', { get: () => 'granted' });"
        " Notification.requestPermission = async () => 'granted';"
    )
    page.goto(f"http://127.0.0.1:{port}/#/")
    page.locator(".toolbar").get_by_role("button", name="Notifications on").click()
    expect(page.locator(".toast")).to_contain_text(
        "notifications are on: the browser's site settings turn them off"
    )
    assert not errors + granted
